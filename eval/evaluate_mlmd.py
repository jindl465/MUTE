#!/usr/bin/env python3
"""
Compute M-LMD (Mouth Landmark Distance) between reference and generated videos.

The metric follows the EAT-style protocol:
  1. Decode each video to frames at a fixed resolution.
  2. Detect 20 mouth landmarks with MediaPipe FaceLandmarker.
  3. Center each mouth landmark set at its own centroid.
  4. Average per-point Euclidean distances over valid frames and videos.
"""

import argparse
import csv
import math
import os
import subprocess
import tempfile
from glob import glob
from pathlib import Path

import mediapipe as mp
import numpy as np
from tqdm import tqdm


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

LIP_IDS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    146, 91, 181, 84, 17, 314, 405, 321, 375,
]

VIDEO_KEY_SUFFIXES = (
    "_adv_inference",
    "_inference",
    "_adv",
    "_gt",
)


def project_root():
    return Path(__file__).resolve().parents[1]


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return project_root() / path


def run_ffmpeg(args):
    subprocess.check_call(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args]
    )


def build_landmarker(model_path):
    base_options = mp.tasks.BaseOptions
    face_landmarker = mp.tasks.vision.FaceLandmarker
    face_landmarker_options = mp.tasks.vision.FaceLandmarkerOptions
    running_mode = mp.tasks.vision.RunningMode
    options = face_landmarker_options(
        base_options=base_options(model_asset_path=str(model_path)),
        running_mode=running_mode.IMAGE,
    )
    return face_landmarker.create_from_options(options)


def extract_frames(video_path, out_dir, size):
    os.makedirs(out_dir, exist_ok=True)
    run_ffmpeg([
        "-i", str(video_path),
        "-vf", f"scale={size}:{size}",
        os.path.join(out_dir, "%06d.png"),
    ])


def get_mouth_landmarks(landmarker, frame_dir):
    frame_files = sorted(glob(os.path.join(frame_dir, "*.png")))
    results = []
    for frame_path in frame_files:
        image = mp.Image.create_from_file(frame_path)
        height, width = image.height, image.width
        detected = landmarker.detect(image)
        if not detected.face_landmarks or len(detected.face_landmarks) != 1:
            results.append(None)
            continue
        landmarks = np.array([
            [point.x * width, point.y * height]
            for point in detected.face_landmarks[0]
        ])
        results.append(landmarks[LIP_IDS])
    return results


def center_landmarks(landmarks):
    centered = landmarks.copy()
    centered[:, 0] -= centered[:, 0].mean()
    centered[:, 1] -= centered[:, 1].mean()
    return centered


def euclidean_distance(point_a, point_b):
    return math.sqrt(
        (point_a[0] - point_b[0]) ** 2 +
        (point_a[1] - point_b[1]) ** 2
    )


def resample_landmarks(landmarks, target_len):
    if target_len <= 0:
        return []
    indices = np.linspace(0, len(landmarks) - 1, target_len).round().astype(int)
    return [landmarks[index] for index in indices]


def compute_mlmd_for_pair(landmarker, ref_path, gen_path, eval_size):
    with tempfile.TemporaryDirectory(prefix="mlmd_eval_") as temp_dir:
        ref_dir = os.path.join(temp_dir, "ref")
        gen_dir = os.path.join(temp_dir, "gen")
        extract_frames(ref_path, ref_dir, size=eval_size)
        extract_frames(gen_path, gen_dir, size=eval_size)
        ref_landmarks = get_mouth_landmarks(landmarker, ref_dir)
        gen_landmarks = get_mouth_landmarks(landmarker, gen_dir)

    target_len = min(len(ref_landmarks), len(gen_landmarks))
    if target_len == 0:
        return None, 0
    if len(ref_landmarks) != target_len:
        ref_landmarks = resample_landmarks(ref_landmarks, target_len)
    if len(gen_landmarks) != target_len:
        gen_landmarks = resample_landmarks(gen_landmarks, target_len)

    total_distance = 0.0
    valid_frames = 0
    for ref_lm, gen_lm in zip(ref_landmarks, gen_landmarks):
        if ref_lm is None or gen_lm is None:
            continue
        ref_centered = center_landmarks(ref_lm)
        gen_centered = center_landmarks(gen_lm)
        for index in range(len(LIP_IDS)):
            total_distance += euclidean_distance(
                ref_centered[index],
                gen_centered[index],
            )
        valid_frames += 1

    if valid_frames == 0:
        return None, 0
    return float(total_distance / (valid_frames * len(LIP_IDS))), valid_frames


def canonical_video_key(name):
    key = name
    changed = True
    while changed:
        changed = False
        for suffix in VIDEO_KEY_SUFFIXES:
            if key.endswith(suffix):
                key = key[:-len(suffix)]
                changed = True
                break
    return key


def infer_video_key(video_path):
    path = Path(video_path)
    stem = canonical_video_key(path.stem)
    parent = canonical_video_key(path.parent.name)
    if parent.isdigit():
        return parent
    return stem


def find_videos(root):
    videos = {}
    for dirpath, _, filenames in os.walk(root):
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            if path.suffix.lower() not in VIDEO_EXTS:
                continue
            key = infer_video_key(path)
            if key in videos and Path(videos[key]).resolve() != path.resolve():
                raise ValueError(
                    f"Duplicate normalized video key '{key}' under {root}:\n"
                    f"  {videos[key]}\n"
                    f"  {path}"
                )
            videos[key] = str(path)
    return videos


def match_videos(ref_dir, gen_dir):
    ref_videos = find_videos(ref_dir)
    gen_videos = find_videos(gen_dir)
    common = sorted(set(ref_videos) & set(gen_videos))
    if not common:
        raise ValueError(
            "No matching video keys found.\n"
            f"Reference keys, first 5: {sorted(ref_videos)[:5]}\n"
            f"Generated keys, first 5: {sorted(gen_videos)[:5]}"
        )
    return [(ref_videos[key], gen_videos[key], key) for key in common]


def write_csv(rows, mean_mlmd, output_csv):
    with open(output_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video", "mlmd", "valid_frames"])
        writer.writerows(rows)
        writer.writerow(["MEAN", f"{mean_mlmd:.6f}", ""])
    print(f"Saved: {output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute M-LMD between video directories.")
    parser.add_argument("--ref_dir", required=True, help="Reference video directory.")
    parser.add_argument("--gen_dir", required=True, help="Generated video directory.")
    parser.add_argument(
        "--model_path",
        default="pretrained_models/face_analysis/models/face_landmarker_v2_with_blendshapes.task",
        help="Path to MediaPipe FaceLandmarker .task file.",
    )
    parser.add_argument("--output_csv", default=None, help="Optional CSV output path.")
    parser.add_argument(
        "--eval_size",
        type=int,
        default=256,
        help="Frame resize resolution before landmark detection.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = resolve_path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Landmark model not found: {model_path}. "
            "Place it under pretrained_models/face_analysis/models/ or pass --model_path."
        )

    pairs = match_videos(args.ref_dir, args.gen_dir)
    print(f"Matched video pairs: {len(pairs)}")
    landmarker = build_landmarker(model_path)

    rows = []
    per_video_scores = []
    try:
        for ref_path, gen_path, key in tqdm(pairs, desc="M-LMD"):
            score, valid_frames = compute_mlmd_for_pair(
                landmarker,
                ref_path,
                gen_path,
                eval_size=args.eval_size,
            )
            if score is None:
                rows.append([key, "nan", 0])
                continue
            rows.append([key, f"{score:.6f}", valid_frames])
            per_video_scores.append(score)
    finally:
        landmarker.close()

    if not per_video_scores:
        raise RuntimeError("No valid video pairs could be evaluated.")

    mean_mlmd = float(np.mean(per_video_scores))
    print(f"M-LMD videos: {len(per_video_scores)}")
    print(f"M-LMD mean: {mean_mlmd:.6f}")
    if args.output_csv:
        write_csv(rows, mean_mlmd, args.output_csv)


if __name__ == "__main__":
    main()
