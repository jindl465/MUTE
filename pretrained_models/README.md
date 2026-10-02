Model checkpoints are not included in this code-only submission folder because
they are large and may have separate redistribution terms.

Place or symlink the required assets here:

```text
pretrained_models/
  stable-diffusion-v1-5/
  sd-vae-ft-mse/
  face_analysis/
    models/
      face_landmarker_v2_with_blendshapes.task
  motion_module/
    mm_sd_v15_v2.ckpt
  hallo/
    net.pth
  wav2vec/
    wav2vec2-base-960h/
```

The attack reads the Hallo, Stable Diffusion, VAE, face-analysis, motion-module,
and wav2vec paths from `hallo/configs/train/stage2.yaml`. Update that file if
your local checkpoint layout differs.

Keep this directory as a local checkpoint mount point. The submitted archive
should include this README, not the checkpoint binaries.
