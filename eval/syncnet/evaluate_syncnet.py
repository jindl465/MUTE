#!/usr/bin/env python3
"""
Evaluate audio-video synchronization with SyncNet.

The script supports either the audio embedded in each video or a matching
external wav file. It crops the dominant face track before running SyncNet,
matching the evaluation flow used in the experiments.
"""

import argparse
import csv
import glob
import os
import subprocess
import tempfile
from shutil import rmtree
from types import SimpleNamespace

import cv2
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

from detectors import S3FD
from SyncNetInstance import SyncNetInstance


VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def run_ffmpeg(args):
    subprocess.check_call(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args]
    )


def infer_speaker_candidates(video_path):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    stem = stem.replace("_inference", "")
    parent = os.path.basename(os.path.dirname(video_path))
    candidates = []

    def add(value):
        if value and value not in candidates:
            candidates.append(value)

    add(stem)
    parts = [p for p in stem.split("_") if p]
    if parts:
        add(parts[0])
        add(parts[-1])
    if parent:
        add(parent)
    return candidates


def infer_primary_speaker(video_path):
    candidates = infer_speaker_candidates(video_path)
    return candidates[1] if len(candidates) > 1 else candidates[0]


def bbox_iou(box_a, box_b):
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    denom = float(area_a + area_b - inter_area)
    return 0.0 if denom <= 0 else inter_area / denom


def track_faces(scene_faces, num_failed_det=25, min_track=10):
    tracks = []
    while True:
        track = []
        for frame_faces in scene_faces:
            for face in list(frame_faces):
                if not track:
                    track.append(face)
                    frame_faces.remove(face)
                elif face["frame"] - track[-1]["frame"] <= num_failed_det:
                    if bbox_iou(face["bbox"], track[-1]["bbox"]) > 0.5:
                        track.append(face)
                        frame_faces.remove(face)
                        continue
                else:
                    break

        if not track:
            break
        if len(track) > min_track:
            frame_ids = np.array([f["frame"] for f in track])
            boxes = np.array([np.array(f["bbox"]) for f in track])
            frame_i = np.arange(frame_ids[0], frame_ids[-1] + 1)
            boxes_i = []
            for coord in range(4):
                boxes_i.append(interp1d(frame_ids, boxes[:, coord])(frame_i))
            boxes_i = np.stack(boxes_i, axis=1)
            mean_w = np.mean(boxes_i[:, 2] - boxes_i[:, 0])
            mean_h = np.mean(boxes_i[:, 3] - boxes_i[:, 1])
            if max(mean_w, mean_h) > 50:
                tracks.append({"frame": frame_i, "bbox": boxes_i})
    return tracks


def crop_video(frames, track, crop_base, audio_wav, frame_rate=25, crop_scale=0.40):
    temp_avi = crop_base + "_video.avi"
    out_avi = crop_base + ".avi"
    audio_trim = crop_base + "_audio.wav"
    writer = cv2.VideoWriter(
        temp_avi,
        cv2.VideoWriter_fourcc(*"XVID"),
        frame_rate,
        (224, 224),
    )

    dets = {"x": [], "y": [], "s": []}
    for det in track["bbox"]:
        dets["s"].append(max(det[3] - det[1], det[2] - det[0]) / 2)
        dets["y"].append((det[1] + det[3]) / 2)
        dets["x"].append((det[0] + det[2]) / 2)

    dets["s"] = signal.medfilt(dets["s"], kernel_size=13)
    dets["x"] = signal.medfilt(dets["x"], kernel_size=13)
    dets["y"] = signal.medfilt(dets["y"], kernel_size=13)

    for idx, frame_idx in enumerate(track["frame"]):
        scale = crop_scale
        box_size = dets["s"][idx]
        pad = int(box_size * (1 + 2 * scale))
        image = frames[int(frame_idx)]
        frame = np.pad(
            image,
            ((pad, pad), (pad, pad), (0, 0)),
            "constant",
            constant_values=(110, 110),
        )
        mid_y = dets["y"][idx] + pad
        mid_x = dets["x"][idx] + pad
        face = frame[
            int(mid_y - box_size):int(mid_y + box_size * (1 + 2 * scale)),
            int(mid_x - box_size * (1 + scale)):int(mid_x + box_size * (1 + scale)),
        ]
        writer.write(cv2.resize(face, (224, 224)))
    writer.release()

    audio_start = float(track["frame"][0]) / frame_rate
    audio_end = float(track["frame"][-1] + 1) / frame_rate
    run_ffmpeg([
        "-i", audio_wav,
        "-ss", f"{audio_start:.3f}",
        "-to", f"{audio_end:.3f}",
        "-ac", "1",
        "-ar", "16000",
        audio_trim,
    ])
    run_ffmpeg(["-i", temp_avi, "-i", audio_trim, "-c:v", "copy", "-c:a", "copy", out_avi])

    for path in (temp_avi, audio_trim):
        if os.path.exists(path):
            os.remove(path)
    return out_avi


def make_full_frame_video(frames, out_base, audio_wav, frame_rate=25):
    temp_avi = out_base + "_video.avi"
    out_avi = out_base + ".avi"
    writer = cv2.VideoWriter(
        temp_avi,
        cv2.VideoWriter_fourcc(*"XVID"),
        frame_rate,
        (224, 224),
    )
    for frame in frames:
        writer.write(cv2.resize(frame, (224, 224)))
    writer.release()
    run_ffmpeg(["-i", temp_avi, "-i", audio_wav, "-c:v", "copy", "-c:a", "copy", out_avi])
    if os.path.exists(temp_avi):
        os.remove(temp_avi)
    return out_avi


def resolve_external_audio(video_path, audio_root):
    if not audio_root:
        return None

    stem = os.path.splitext(os.path.basename(video_path))[0]
    speaker_candidates = infer_speaker_candidates(video_path)
    wav_names = []

    def add_name(name):
        if name and name not in wav_names:
            wav_names.append(name)

    add_name(stem.replace("_inference", "") + ".wav")
    add_name(stem + ".wav")
    for speaker in speaker_candidates:
        add_name(f"{speaker}.wav")

    candidates = []
    for speaker in speaker_candidates:
        for wav_name in wav_names:
            candidates.append(os.path.join(audio_root, speaker, wav_name))
    for wav_name in wav_names:
        candidates.append(os.path.join(audio_root, wav_name))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    for wav_name in wav_names:
        matches = sorted(glob.glob(os.path.join(audio_root, "**", wav_name), recursive=True))
        if matches:
            return matches[0]
    return None


def collect_videos(directory, recursive=False):
    pattern = os.path.join(directory, "**", "*") if recursive else os.path.join(directory, "*")
    return sorted(
        path for path in glob.glob(pattern, recursive=recursive)
        if os.path.splitext(path)[1].lower() in VIDEO_EXTS
    )


def evaluate_video(video_path, syncnet, face_detector, args, work_dir, audio_source=None):
    if os.path.exists(work_dir):
        rmtree(work_dir)
    os.makedirs(work_dir)

    avi_path = os.path.join(work_dir, "video.avi")
    audio_path = os.path.join(work_dir, "audio.wav")
    frames_dir = os.path.join(work_dir, "frames")
    crop_dir = os.path.join(work_dir, "crops")
    os.makedirs(frames_dir)
    os.makedirs(crop_dir)

    try:
        run_ffmpeg(["-i", video_path, "-qscale:v", "2", "-an", "-r", "25", avi_path])
        run_ffmpeg(["-i", avi_path, "-qscale:v", "2", "-threads", "1", "-f", "image2", os.path.join(frames_dir, "%06d.jpg")])
        audio_input = audio_source if audio_source is not None else video_path
        run_ffmpeg(["-i", audio_input, "-ac", "1", "-vn", "-acodec", "pcm_s16le", "-ar", "16000", audio_path])

        frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
        if not frame_files:
            return None, None
        frames = [cv2.imread(path) for path in frame_files]

        detections = []
        for frame_idx, frame in enumerate(frames):
            image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_detector.detect_faces(image_np, conf_th=0.9, scales=[0.25])
            detections.append([
                {"frame": frame_idx, "bbox": box[:-1].tolist(), "conf": box[-1]}
                for box in boxes
            ])

        tracks = track_faces(detections, num_failed_det=25, min_track=10)
        if tracks:
            best_track = max(tracks, key=lambda item: len(item["frame"]))
            eval_video_path = crop_video(
                frames,
                best_track,
                os.path.join(crop_dir, "crop_00000"),
                audio_path,
            )
        else:
            eval_video_path = make_full_frame_video(
                frames,
                os.path.join(crop_dir, "full_frame"),
                audio_path,
            )

        sync_opt = SimpleNamespace(
            initial_model=args.initial_model,
            batch_size=args.batch_size,
            vshift=args.vshift,
            tmp_dir=work_dir,
            reference="synceval",
        )
        os.makedirs(os.path.join(work_dir, sync_opt.reference), exist_ok=True)
        offset, confidence, _ = syncnet.evaluate(sync_opt, videofile=eval_video_path)
        return float(confidence), int(offset)
    finally:
        if os.path.exists(work_dir):
            rmtree(work_dir)


def iter_video_records(args):
    if args.mode == "flat":
        label = os.path.basename(os.path.normpath(args.data_root))
        for video_path in collect_videos(args.data_root, recursive=False):
            yield label, infer_primary_speaker(video_path), video_path
        return

    if args.mode == "clean":
        for speaker in sorted(os.listdir(args.data_root)):
            speaker_dir = os.path.join(args.data_root, speaker)
            if not os.path.isdir(speaker_dir):
                continue
            for video_path in collect_videos(speaker_dir, recursive=True):
                yield "clean", speaker, video_path
        return

    methods = [args.method] if args.method else sorted(os.listdir(args.data_root))
    for method in methods:
        method_dir = os.path.join(args.data_root, method)
        if not os.path.isdir(method_dir):
            continue
        direct_videos = collect_videos(method_dir, recursive=False)
        for video_path in direct_videos:
            yield method, infer_primary_speaker(video_path), video_path
        for speaker in sorted(os.listdir(method_dir)):
            speaker_dir = os.path.join(method_dir, speaker)
            if not os.path.isdir(speaker_dir):
                continue
            for video_path in collect_videos(speaker_dir, recursive=True):
                yield method, speaker, video_path


def write_results(rows, output_csv):
    fields = ["Method", "Speaker", "File", "Confidence", "AV_Offset"]
    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    if not rows:
        return
    confidences = np.array([row["Confidence"] for row in rows], dtype=float)
    offsets = np.array([row["AV_Offset"] for row in rows], dtype=float)
    summary_path = output_csv.replace(".csv", "_summary.csv")
    with open(summary_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Total_Videos", "Mean_Confidence", "Std_Confidence", "Mean_Offset"],
        )
        writer.writeheader()
        writer.writerow({
            "Total_Videos": len(rows),
            "Mean_Confidence": float(np.mean(confidences)),
            "Std_Confidence": float(np.std(confidences)),
            "Mean_Offset": float(np.mean(offsets)),
        })
    print(f"Detailed results: {output_csv}")
    print(f"Summary results: {summary_path}")


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Evaluate videos with SyncNet.")
    parser.add_argument("--initial_model", default=os.path.join(script_dir, "data", "syncnet_v2.model"))
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--vshift", type=int, default=15)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--audio_root", default=None)
    parser.add_argument("--mode", choices=["method", "clean", "flat"], required=True)
    parser.add_argument("--method", default=None)
    parser.add_argument("--output_csv", default="syncnet_results.csv")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.initial_model):
        raise FileNotFoundError(
            f"SyncNet checkpoint not found: {args.initial_model}. "
            "Place it under eval/syncnet/data/ or pass --initial_model."
        )

    syncnet = SyncNetInstance()
    syncnet.loadParameters(args.initial_model)
    face_detector = S3FD(device=args.device)

    rows = []
    with tempfile.TemporaryDirectory(prefix="syncnet_eval_") as temp_root:
        for index, (method, speaker, video_path) in enumerate(iter_video_records(args), start=1):
            print(f"[{index}] {method}/{speaker}/{os.path.basename(video_path)}")
            audio_source = resolve_external_audio(video_path, args.audio_root)
            if args.audio_root and audio_source is None:
                print("  matching external audio not found; skipping")
                continue
            confidence, offset = evaluate_video(
                video_path,
                syncnet,
                face_detector,
                args,
                os.path.join(temp_root, "work"),
                audio_source=audio_source,
            )
            if confidence is None:
                print("  failed")
                continue
            print(f"  confidence={confidence:.3f}, offset={offset}")
            rows.append({
                "Method": method,
                "Speaker": speaker,
                "File": os.path.basename(video_path),
                "Confidence": confidence,
                "AV_Offset": offset,
            })

    write_results(rows, args.output_csv)


if __name__ == "__main__":
    main()
