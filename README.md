# MUTE: Multi-Level Alignment Uncoupling Against Talking-Head Exploitation for Voice Protection

Official PyTorch implementation of **MUTE**, accepted to **NeurIPS 2026**.

**Donghyun Kim\*, Jin Hong\*, Seungmin Kim\*, Dain Kim, Junseok Kwon, Daeseon Choi**
<br>\* Equal contribution

<p>
  <a href="https://jindl465.github.io/website/MUTE/"><img src="https://img.shields.io/badge/Project-Page-1772d0?style=flat-square" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2611.11989"><img src="https://img.shields.io/badge/Paper-PDF-b31b1b?style=flat-square" alt="Paper"></a>
  <img src="https://img.shields.io/badge/NeurIPS-2026-6a3fa0?style=flat-square" alt="NeurIPS 2026">
</p>

🎬 **Project page:** https://jindl465.github.io/website/MUTE/
<br>Side-by-side video comparisons with the baselines, black-box transfer, robustness to purification, combination with image protection, and an interactive version of the user study.

## Overview

<p align="center">
  <img src="assets/overview.png" alt="Overview of the MUTE framework" width="100%">
</p>

MUTE protects speech from being exploited by audio-driven talking-head
generators. It adds a small, L∞-bounded perturbation to the waveform that
breaks the audio-visual alignment these models rely on, so that a talking-head
model driven by the protected audio no longer produces lip motion that follows
the speech, while the audio itself stays close to the original.

MUTE uncouples the alignment at two levels:

- **Alignment-level Disruption** perturbs the audio-visual cross-attention of
  the talking-head model (`L_attn`).
- **Representation-level Degradation** perturbs the temporal audio embeddings
  (`L_emb`).

The perturbation is constrained to the high-energy regions of the clean
speech in the STFT domain (`L_stft`), which keeps it robust to
post-processing, denoising-based purification, and adaptive attacks. It is
optimized with a two-phase PGD procedure against a frozen, pretrained
[Hallo](https://github.com/fudan-generative-vision/hallo) model, so no training
is involved. The protection transfers to black-box talking-head models (Hallo2,
Audio2Head, SadTalker, and Ditto) and can be combined with image-based
protections.

## Repository Structure

```text
MUTE/
  assets/                      # figures used in this README
  protect/
    protect_audio_multi.py     # main entry point: protects every speaker folder in a dataset
    protect_audio.py           # two-phase PGD optimization, L_stft, hyperparameters
    audio_loss.py              # L_emb, L_attn, and the attention hooks
    visualize_lip_attn.py      # attention capture used to build the target distribution
    protect_hallo.py           # minimal Hallo runtime wrapper
    configs/attack/base.yaml   # config for the single-speaker entry point
  hallo/                       # Hallo model code, portrait preprocessing, runtime config
  eval/
    evaluate_fid.py            # FID, FVD, E-FID
    evaluate_mlmd.py           # M-LMD (mouth landmark distance)
    syncnet/                   # SyncNet audio-visual synchronization
  example_data/0002/           # one portrait and one WAV for a quick end-to-end run
  pretrained_models/           # place the pretrained checkpoints here (not tracked)
  requirements.txt
```

## Installation

The code requires Python 3.10 or later and an NVIDIA GPU (the experiments in
the paper were run on an RTX 4090). `requirements.txt` pins PyTorch 2.2.2 built
for CUDA 12.1.

```bash
git clone https://github.com/jindl465/MUTE.git
cd MUTE

conda create -n mute python=3.10
conda activate mute
pip install -r requirements.txt
```

The SyncNet and M-LMD evaluation scripts call the `ffmpeg` executable. Install
it with your system package manager if it is not already available, for example
`sudo apt-get install ffmpeg`.

## Pretrained Models

MUTE runs against the pretrained Hallo model and the components it depends on.
All of them are distributed in the official Hallo model repository,
[`fudan-generative-ai/hallo`](https://huggingface.co/fudan-generative-ai/hallo).
Download them into `pretrained_models/` from the repository root:

```bash
huggingface-cli download fudan-generative-ai/hallo \
  --local-dir pretrained_models \
  --exclude README.md .gitattributes
```

The protection code reads the following files:

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

These paths match the defaults in `hallo/configs/train/stage2.yaml`. If your
checkpoints live elsewhere, symlink them into `pretrained_models/` or edit that
file. The checkpoints are subject to the licenses of their original sources.

## Data Preparation

Prepare one folder per speaker, each with one portrait and one or more speech
clips:

```text
dataset/
  speaker_id/
    portrait.png
    audio.wav
    ...
```

- The portrait can be `.png`, `.jpg`, or `.jpeg`. If a folder contains several
  images, the first one in alphabetical order is used.
- Every `.wav` file in the folder is protected. Audio is loaded at 16 kHz; mono
  WAV files are recommended.
- The target attention distribution is computed once per speaker, from the
  first WAV file, and reused for the remaining files of that speaker.

The paper reports results on two datasets:

- **TalkingHead-1KH**: 50 samples from the preprocessed TalkingHead-1KH data
  released by [Silencer](https://github.com/yuangan/silencer). Download it
  through the link in the Silencer repository.
- **CelebV-HQ**: 50 high-quality videos selected by CLIP-IQA score, with the
  audio trimmed to 5-second segments.

The datasets are not redistributed here; please obtain them from the sources
above.

## Quick Start

`example_data/0002` contains one portrait and one WAV file. Once the pretrained
models are in place, run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python protect/protect_audio_multi.py \
  --dataset_dir example_data \
  --output_dir outputs/example \
  --speaker_ids 0002
```

This runs the full pipeline: model loading, portrait preprocessing, target
attention construction, and the two-phase PGD optimization. The protected audio
is written to:

```text
outputs/example/
  0002/
    audio_emb_frame_l1+kl+stft_mask_cons_eps0.004_steps200_200_score/
      audio_adv.wav
```

## Protecting Your Own Data

```bash
CUDA_VISIBLE_DEVICES=0 python protect/protect_audio_multi.py \
  --dataset_dir /path/to/dataset \
  --output_dir /path/to/output
```

| Argument | Default | Description |
| --- | --- | --- |
| `--dataset_dir` | `dataset` | Root directory with one sub-folder per speaker. |
| `--output_dir` | `outputs/mute_attack` | Root directory for the protected audio. |
| `--min_timesteps` | `200` | Lower bound of the diffusion timestep used by `L_attn`. The timestep is drawn from `[min_timesteps, min_timesteps + 100)`. |
| `--speaker_ids` | all | Process only the listed speaker folders. |
| `--num_speakers` | all | Process only the first *N* speaker folders. |
| `--device` | `cuda` | Device for the portrait tensor. Leave at the default (see the note below). |

Notes:

- Run the script from the repository root. The Hallo config and the checkpoint
  paths are resolved relative to the working directory.
- The models are loaded on the default CUDA device. To choose a GPU, set
  `CUDA_VISIBLE_DEVICES` rather than `--device`.
- For each input `NAME.wav` the script writes
  `<output_dir>/<speaker_id>/<run_tag>/NAME_adv.wav`, a 16 kHz float WAV file.
  The run tag encodes the active losses, the perturbation budget, the number of
  steps, and `--min_timesteps`.
- The runtime Hallo config of the run is saved to
  `<output_dir>/<run_tag>/_stage2_audio.yaml`, and portrait preprocessing
  results are cached under `.cache/`.
- Set `SAVE_GT_TARGET_ATTN_MAPS=1` to additionally save visualizations of the
  clean, target, and final adversarial attention maps next to the output audio.

`protect/protect_audio.py` also provides a single-speaker entry point configured
through Hydra (`protect/configs/attack/base.yaml`):

```bash
python protect/protect_audio.py \
  attack.img_path=/path/to/portrait.png \
  attack.audio_path=/path/to/wav_dir \
  attack.output_path=/path/to/output \
  attack.min_timesteps=200
```

## Evaluation

MUTE is evaluated on the talking-head videos generated from the protected audio.
This repository does not include a video renderer. Use the official
[Hallo](https://github.com/fudan-generative-vision/hallo) inference script to
generate a video from a portrait and an audio file:

```bash
# in the Hallo repository
python scripts/inference.py \
  --source_image /path/to/portrait.png \
  --driving_audio /path/to/NAME_adv.wav \
  --output /path/to/NAME_adv_inference.mp4
```

Then run the evaluation scripts from the root of this repository:

```bash
# Audio-visual synchronization (SyncNet)
python eval/syncnet/evaluate_syncnet.py \
  --data_root /path/to/generated_videos \
  --mode method \
  --method METHOD_NAME \
  --output_csv syncnet_results.csv

# Mouth landmark distance (M-LMD)
python eval/evaluate_mlmd.py \
  --ref_dir /path/to/reference_videos \
  --gen_dir /path/to/generated_videos \
  --output_csv mlmd_results.csv

# Visual quality (FID, FVD, E-FID)
python eval/evaluate_fid.py \
  --ref_dir /path/to/reference_videos \
  --gen_dir /path/to/generated_videos \
  --metrics fid fvd efid \
  --output_csv visual_quality_results.csv
```

SyncNet evaluation needs two additional checkpoints, which can be obtained with
`download_model.sh` from [syncnet_python](https://github.com/joonson/syncnet_python):

```text
eval/syncnet/data/syncnet_v2.model
eval/syncnet/detectors/s3fd/weights/sfd_face.pth
```

See [`eval/README.md`](eval/README.md) for the expected directory layouts and
the remaining options. Scripts for the audio-quality metrics reported in the
paper (WER, STOI, and PESQ) are not included in this repository.

## Results

Main results of MUTE against Hallo (white-box), as reported in Table 1 of the
paper. Values are mean ± standard deviation over three random seeds. Full comparison
tables and generated videos are on the
[project page](https://jindl465.github.io/website/MUTE/).

| Dataset | WER ↓ | STOI ↑ / PESQ ↑ | SyncNet ↓ | M-LMD ↑ | FID ↑ / FVD ↑ | E-FID ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TalkingHead-1KH | 11.681 ± 0.330 | 0.936 ± 0.001 / 2.522 ± 0.002 | 1.332 ± 0.012 | 1.627 ± 0.012 | 3.183 ± 0.036 / 2.363 ± 0.049 | 5.489 ± 0.087 |
| CelebV-HQ | 10.467 ± 0.406 | 0.959 ± 0.000 / 2.940 ± 0.001 | 2.152 ± 0.040 | 1.693 ± 0.049 | 2.756 ± 0.048 / 1.790 ± 0.042 | 5.062 ± 0.183 |

The arrows give the direction that is better for protection. WER (measured with
Whisper), STOI, and PESQ
measure the protected audio, which should stay intelligible and close to the
original. The remaining metrics measure the videos generated from the protected
audio, where weaker synchronization and a larger distance from the reference
videos mean stronger protection.

## Acknowledgements

The model and preprocessing code under `hallo/` is adapted from
[Hallo](https://github.com/fudan-generative-vision/hallo), and the SyncNet
evaluation code under `eval/syncnet/` is adapted from
[syncnet_python](https://github.com/joonson/syncnet_python). The
TalkingHead-1KH samples used in our experiments come from the preprocessed data
released by [Silencer](https://github.com/yuangan/silencer). We thank the
authors for releasing their code, models, and data.

## License

This project is released under the MIT License; see [`LICENSE`](LICENSE).
Third-party attributions are listed in [`NOTICE`](NOTICE). Pretrained
checkpoints and datasets are governed by the terms of their respective sources.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{kim2026mute,
  title     = {MUTE: Multi-Level Alignment Uncoupling Against Talking-Head Exploitation for Voice Protection},
  author    = {Kim, Donghyun and Hong, Jin and Kim, Seungmin and Kim, Dain and Kwon, Junseok and Choi, Daeseon},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026},
  eprint    = {2611.11989},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD}
}
```
