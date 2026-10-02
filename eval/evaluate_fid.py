#!/usr/bin/env python3
"""
Compute FID, FVD, and E-FID for generated video evaluation.

FID uses InceptionV3 features from video frames.
FVD uses Kinetics-pretrained R3D-18 video features as a practical FVD-style
feature extractor.
E-FID uses InceptionV3 features from face-cropped video frames.
"""

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from tqdm import tqdm


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
INCEPTION_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
INCEPTION_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
VIDEO_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
VIDEO_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)


def collect_videos(directory):
    paths = []
    for root, _, files in os.walk(directory):
        for name in files:
            if Path(name).suffix.lower() in VIDEO_EXTS:
                paths.append(os.path.join(root, name))
    return sorted(paths)


def sample_indices(length, count):
    if count is None or length <= count:
        return np.arange(length)
    return np.linspace(0, length - 1, count, dtype=int)


def load_video_frames(path, max_frames=None, resize=None, as_float=True):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize is not None:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_LANCZOS4)
        frames.append(frame)
    cap.release()

    if not frames:
        dtype = np.float32 if as_float else np.uint8
        return np.zeros((0, 1, 1, 3), dtype=dtype)

    indices = sample_indices(len(frames), max_frames)
    frames = [frames[i] for i in indices]
    array = np.stack(frames, axis=0)
    if as_float:
        array = array.astype(np.float32) / 255.0
    return array


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    offset = np.eye(sigma1.shape[0]) * eps
    covmean, _ = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset), disp=False)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            max_imag = np.max(np.abs(covmean.imag))
            print(f"Warning: sqrtm returned a large imaginary component: {max_imag:.3e}")
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def compute_stats(features):
    return np.mean(features, axis=0), np.cov(features, rowvar=False)


def compute_frechet_from_features(real_features, gen_features, metric_name):
    if len(real_features) < 2 or len(gen_features) < 2:
        print(f"[{metric_name}] Not enough feature samples; returning nan.")
        return float("nan")
    real_mu, real_sigma = compute_stats(real_features)
    gen_mu, gen_sigma = compute_stats(gen_features)
    return frechet_distance(real_mu, real_sigma, gen_mu, gen_sigma)


class InceptionFeatureExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        from torchvision.models import Inception_V3_Weights, inception_v3

        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        model.fc = nn.Identity()
        model.eval()
        self.model = model.to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, x):
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        mean = INCEPTION_MEAN.to(device=x.device, dtype=x.dtype)
        std = INCEPTION_STD.to(device=x.device, dtype=x.dtype)
        x = (x - mean) / std
        features = self.model(x)
        if isinstance(features, tuple):
            features = features[0]
        return features


class VideoFeatureExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        from torchvision.models.video import R3D_18_Weights, r3d_18

        model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        model.fc = nn.Identity()
        model.eval()
        self.model = model.to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, clips):
        clips = F.interpolate(
            clips,
            size=(clips.shape[2], 112, 112),
            mode="trilinear",
            align_corners=False,
        )
        mean = VIDEO_MEAN.to(device=clips.device, dtype=clips.dtype)
        std = VIDEO_STD.to(device=clips.device, dtype=clips.dtype)
        clips = (clips - mean) / std
        return self.model(clips)


def init_face_detector(confidence):
    try:
        import mediapipe as mp
    except ImportError:
        return None
    return mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=confidence,
    )


def center_crop_square(frame):
    height, width = frame.shape[:2]
    side = min(height, width)
    y0 = (height - side) // 2
    x0 = (width - side) // 2
    return frame[y0:y0 + side, x0:x0 + side]


def crop_face(frame_rgb, detector, margin=0.2, output_size=299):
    if detector is None:
        crop = center_crop_square(frame_rgb)
        return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LANCZOS4)

    result = detector.process(frame_rgb)
    if not result.detections:
        crop = center_crop_square(frame_rgb)
        return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LANCZOS4)

    height, width = frame_rgb.shape[:2]
    box = result.detections[0].location_data.relative_bounding_box
    x1 = int((box.xmin - margin * box.width) * width)
    y1 = int((box.ymin - margin * box.height) * height)
    x2 = int((box.xmin + (1.0 + margin) * box.width) * width)
    y2 = int((box.ymin + (1.0 + margin) * box.height) * height)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)

    if x2 <= x1 or y2 <= y1:
        crop = center_crop_square(frame_rgb)
    else:
        crop = frame_rgb[y1:y2, x1:x2]
    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LANCZOS4)


def extract_inception_features(
    video_paths,
    extractor,
    batch_size,
    frames_per_video,
    pool_per_video,
    desc,
    face_mode=False,
    face_confidence=0.5,
    face_margin=0.2,
):
    all_features = []
    detector = init_face_detector(face_confidence) if face_mode else None
    if face_mode and detector is None:
        print("[E-FID] mediapipe is unavailable; using center crops.")

    for video_path in tqdm(video_paths, desc=desc):
        frames = load_video_frames(
            video_path,
            max_frames=frames_per_video,
            resize=None if face_mode else (299, 299),
            as_float=False,
        )
        if len(frames) == 0:
            continue

        if face_mode:
            frames = np.stack([
                crop_face(frame, detector, margin=face_margin, output_size=299)
                for frame in frames
            ], axis=0)

        frames = frames.astype(np.float32) / 255.0
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).to(extractor.device)

        video_features = []
        for start in range(0, len(tensor), batch_size):
            batch = tensor[start:start + batch_size]
            video_features.append(extractor(batch).cpu().numpy())
        video_features = np.concatenate(video_features, axis=0)
        if pool_per_video:
            all_features.append(video_features.mean(axis=0, keepdims=True))
        else:
            all_features.append(video_features)

    if not all_features:
        return np.zeros((0, 2048), dtype=np.float32)
    return np.concatenate(all_features, axis=0)


def extract_video_clips(frames, clip_len, stride):
    if len(frames) == 0:
        return np.zeros((0, clip_len, 112, 112, 3), dtype=np.float32)
    if len(frames) < clip_len:
        pad = np.repeat(frames[-1:], clip_len - len(frames), axis=0)
        frames = np.concatenate([frames, pad], axis=0)

    clips = []
    last_start = len(frames) - clip_len
    for start in range(0, last_start + 1, stride):
        clips.append(frames[start:start + clip_len])
    if not clips:
        clips.append(frames[:clip_len])
    return np.stack(clips, axis=0)


def extract_fvd_features(
    video_paths,
    extractor,
    batch_size,
    clip_len,
    clip_stride,
    max_clips_per_video,
    pool_per_video,
    desc,
):
    all_features = []
    for video_path in tqdm(video_paths, desc=desc):
        max_frames = None
        if max_clips_per_video is not None:
            max_frames = clip_len + max(0, max_clips_per_video - 1) * clip_stride
        frames = load_video_frames(
            video_path,
            max_frames=max_frames,
            resize=(112, 112),
            as_float=True,
        )
        clips = extract_video_clips(frames, clip_len=clip_len, stride=clip_stride)
        if max_clips_per_video is not None and len(clips) > max_clips_per_video:
            indices = sample_indices(len(clips), max_clips_per_video)
            clips = clips[indices]

        tensor = torch.from_numpy(clips).permute(0, 4, 1, 2, 3).to(extractor.device)
        video_features = []
        for start in range(0, len(tensor), batch_size):
            batch = tensor[start:start + batch_size]
            video_features.append(extractor(batch).cpu().numpy())
        video_features = np.concatenate(video_features, axis=0)
        if pool_per_video:
            all_features.append(video_features.mean(axis=0, keepdims=True))
        else:
            all_features.append(video_features)

    if not all_features:
        return np.zeros((0, 512), dtype=np.float32)
    return np.concatenate(all_features, axis=0)


def compute_fid(real_paths, gen_paths, args):
    extractor = InceptionFeatureExtractor(args.device)
    real_features = extract_inception_features(
        real_paths,
        extractor,
        args.batch_size,
        args.frames_per_video,
        args.pool_per_video,
        "FID reference",
    )
    gen_features = extract_inception_features(
        gen_paths,
        extractor,
        args.batch_size,
        args.frames_per_video,
        args.pool_per_video,
        "FID generated",
    )
    return compute_frechet_from_features(real_features, gen_features, "FID")


def compute_efid(real_paths, gen_paths, args):
    extractor = InceptionFeatureExtractor(args.device)
    real_features = extract_inception_features(
        real_paths,
        extractor,
        args.batch_size,
        args.frames_per_video,
        args.pool_per_video,
        "E-FID reference",
        face_mode=True,
        face_confidence=args.face_confidence,
        face_margin=args.face_margin,
    )
    gen_features = extract_inception_features(
        gen_paths,
        extractor,
        args.batch_size,
        args.frames_per_video,
        args.pool_per_video,
        "E-FID generated",
        face_mode=True,
        face_confidence=args.face_confidence,
        face_margin=args.face_margin,
    )
    return compute_frechet_from_features(real_features, gen_features, "E-FID")


def compute_fvd(real_paths, gen_paths, args):
    extractor = VideoFeatureExtractor(args.device)
    real_features = extract_fvd_features(
        real_paths,
        extractor,
        args.video_batch_size,
        args.clip_len,
        args.clip_stride,
        args.max_clips_per_video,
        args.pool_per_video,
        "FVD reference",
    )
    gen_features = extract_fvd_features(
        gen_paths,
        extractor,
        args.video_batch_size,
        args.clip_len,
        args.clip_stride,
        args.max_clips_per_video,
        args.pool_per_video,
        "FVD generated",
    )
    return compute_frechet_from_features(real_features, gen_features, "FVD")


def save_results_csv(results, output_csv):
    with open(output_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for metric, value in results.items():
            writer.writerow([metric, "nan" if np.isnan(value) else f"{value:.6f}"])
    print(f"Saved: {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute FID, FVD, and E-FID between reference and generated videos."
    )
    parser.add_argument("--ref_dir", required=True, help="Reference video directory.")
    parser.add_argument("--gen_dir", required=True, help="Generated video directory.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["fid", "fvd", "efid"],
        choices=["fid", "fvd", "efid"],
        help="Metrics to compute.",
    )
    parser.add_argument("--output_csv", default=None, help="Optional CSV output path.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--video_batch_size", type=int, default=8)
    parser.add_argument("--frames_per_video", type=int, default=None)
    parser.add_argument("--pool_per_video", action="store_true")
    parser.add_argument("--clip_len", type=int, default=16)
    parser.add_argument("--clip_stride", type=int, default=16)
    parser.add_argument("--max_clips_per_video", type=int, default=None)
    parser.add_argument("--face_confidence", type=float, default=0.5)
    parser.add_argument("--face_margin", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    real_paths = collect_videos(args.ref_dir)
    gen_paths = collect_videos(args.gen_dir)
    print(f"Reference videos: {len(real_paths)}")
    print(f"Generated videos: {len(gen_paths)}")
    if not real_paths or not gen_paths:
        raise RuntimeError("No videos found in one of the input directories.")

    results = {}
    if "fid" in args.metrics:
        print("\nComputing FID")
        results["FID"] = compute_fid(real_paths, gen_paths, args)
    if "fvd" in args.metrics:
        print("\nComputing FVD")
        results["FVD"] = compute_fvd(real_paths, gen_paths, args)
    if "efid" in args.metrics:
        print("\nComputing E-FID")
        results["E-FID"] = compute_efid(real_paths, gen_paths, args)

    print("\nResults")
    for metric, value in results.items():
        value_text = "nan" if np.isnan(value) else f"{value:.6f}"
        print(f"{metric}: {value_text}")

    if args.output_csv:
        save_results_csv(results, args.output_csv)


if __name__ == "__main__":
    main()
