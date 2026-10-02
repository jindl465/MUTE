# SyncNet Evaluation

This directory contains the SyncNet evaluation code used for audio-video
synchronization scores.

Required checkpoints are not included:

```text
eval/syncnet/data/syncnet_v2.model
eval/syncnet/detectors/s3fd/weights/sfd_face.pth
```

Run from the submission folder:

```bash
python eval/syncnet/evaluate_syncnet.py \
  --data_root /path/to/generated_videos \
  --mode method \
  --method METHOD_NAME \
  --output_csv syncnet_results.csv
```

To evaluate a generated video against external audio instead of the video's
embedded audio, pass `--audio_root /path/to/wav_dataset`.
