# Evaluation

This folder contains evaluation scripts used with the attack outputs.

SyncNet and M-LMD call the `ffmpeg` executable. Install ffmpeg separately with
your system package manager before running those evaluations.

## Evaluation Inputs

Generated videos are expected to be organized either as a flat directory of
video files or as method/speaker subdirectories. The scripts match samples by
relative path and file stem where applicable.

Reference and generated video folders typically look like:

```text
reference_videos/
  speaker_id/
    sample_id.mp4

generated_videos/
  speaker_id/
    sample_id.mp4
```

SyncNet can use the audio embedded in each video. If the generated videos do
not contain the target audio track, pass `--audio_root /path/to/wav_dataset`.

## FID / FVD / E-FID

```bash
python eval/evaluate_fid.py \
  --ref_dir /path/to/reference_videos \
  --gen_dir /path/to/generated_videos \
  --metrics fid fvd efid \
  --output_csv fid_results.csv
```

Use `--metrics fid`, `--metrics fvd`, or `--metrics efid` to run only one
metric. E-FID applies the FID computation to face crops.

## SyncNet

```bash
python eval/syncnet/evaluate_syncnet.py \
  --data_root /path/to/generated_videos \
  --mode method \
  --method METHOD_NAME \
  --output_csv syncnet_results.csv
```

Required SyncNet checkpoints are not included:

```text
eval/syncnet/data/syncnet_v2.model
eval/syncnet/detectors/s3fd/weights/sfd_face.pth
```

See `eval/syncnet/README.md` for the SyncNet-specific notes.

## M-LMD

```bash
python eval/evaluate_mlmd.py \
  --ref_dir /path/to/reference_videos \
  --gen_dir /path/to/generated_videos \
  --output_csv mlmd_results.csv
```

The default landmark model path is
`pretrained_models/face_analysis/models/face_landmarker_v2_with_blendshapes.task`.

## Checkpoints

Evaluation checkpoints and datasets are not included in this code-only
submission. Place them at the default paths shown above, or pass explicit paths
with the corresponding command-line arguments when available.
