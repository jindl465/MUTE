"""
Multi-speaker entry point for MUTE adversarial audio generation.

The wrapper loads Hallo once, preprocesses each speaker portrait, computes the
paper target attention distribution, and runs the two-phase PGD attack.
"""

import argparse
import gc
import glob
import os
import sys
import time

import soundfile as sf
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

_PROTECT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_PROTECT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _PROTECT_DIR)

from audio_loss import LipAttnWeightHook
from hallo.datasets.image_processor import ImageProcessor
from protect_audio import (
    _EPSILON,
    _HALLO_RUNTIME_DTYPE,
    _LOSS_WEIGHTS,
    _SAVE_GT_TARGET_ATTN_MAPS,
    _SCORE,
    _STEPS,
    _compute_kl_target,
    _load_hallo_model,
    _load_target_lip_mask_pyramid,
    protect_audio,
)
from utils import cprint


def _active_loss_suffix(min_timesteps: int) -> str:
    active_losses = "+".join(name for name, weight in _LOSS_WEIGHTS.items() if weight > 0)
    return f"audio_{active_losses}_eps{_EPSILON}_steps{_STEPS}_{min_timesteps}_{_SCORE}"


def _find_speaker_dirs(dataset_dir: str, speaker_ids: list[str] | None, num_speakers: int | None) -> list[str]:
    speaker_dirs = sorted(
        path for path in glob.glob(os.path.join(dataset_dir, "*"))
        if os.path.isdir(path)
    )
    if speaker_ids:
        speaker_map = {os.path.basename(path): path for path in speaker_dirs}
        missing = [speaker_id for speaker_id in speaker_ids if speaker_id not in speaker_map]
        if missing:
            cprint(f"Warning: requested speaker IDs were not found: {missing}", "r")
        speaker_dirs = [speaker_map[speaker_id] for speaker_id in speaker_ids if speaker_id in speaker_map]
    if num_speakers is not None and num_speakers > 0:
        speaker_dirs = speaker_dirs[:num_speakers]
    return speaker_dirs


def _find_first_image(speaker_dir: str) -> str | None:
    candidates = (
        glob.glob(os.path.join(speaker_dir, "*.png"))
        + glob.glob(os.path.join(speaker_dir, "*.jpg"))
        + glob.glob(os.path.join(speaker_dir, "*.jpeg"))
    )
    return sorted(candidates)[0] if candidates else None


def _find_audio_files(speaker_dir: str) -> list[str]:
    return sorted(
        glob.glob(os.path.join(speaker_dir, "*.wav"))
        + glob.glob(os.path.join(speaker_dir, "*.WAV"))
    )


def _build_runtime_config(orig_cwd: str, output_dir: str) -> tuple[str, OmegaConf]:
    hallo_cfg_path = os.path.join(orig_cwd, "hallo/configs/train/stage2.yaml")
    cfg = OmegaConf.load(hallo_cfg_path)
    cfg.solver.enable_xformers_memory_efficient_attention = False
    cfg.solver.gradient_checkpointing = False
    cfg.weight_dtype = _HALLO_RUNTIME_DTYPE
    os.makedirs(output_dir, exist_ok=True)
    tmp_cfg = os.path.join(output_dir, "_stage2_audio.yaml")
    OmegaConf.save(cfg, tmp_cfg)
    return tmp_cfg, cfg


def _preprocess_speaker_image(
    image_processor: ImageProcessor,
    source_image_path: str,
    cache_path: str,
    face_expand_ratio: float,
) -> tuple[dict, list[torch.Tensor]]:
    (
        source_image_pixels,
        source_image_face_region,
        source_image_face_emb,
        source_image_full_mask,
        source_image_face_mask,
        source_image_lip_mask,
    ) = image_processor.preprocess(source_image_path, cache_path, face_expand_ratio)

    lip_mask_path = os.path.join(
        cache_path,
        f"{os.path.splitext(os.path.basename(source_image_path))[0]}_lip_mask.png",
    )
    target_lip_mask = _load_target_lip_mask_pyramid(lip_mask_path)
    source_input = {
        "image": source_image_pixels,
        "face_region": source_image_face_region,
        "face_emb": torch.from_numpy(source_image_face_emb),
        "full_mask": source_image_full_mask,
        "face_mask": source_image_face_mask,
        "lip_mask": source_image_lip_mask,
    }
    return source_input, target_lip_mask


def main():
    parser = argparse.ArgumentParser(
        description="Run the MUTE audio attack for multiple speaker folders."
    )
    parser.add_argument("--dataset_dir", type=str, default="dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/mute_attack")
    parser.add_argument("--min_timesteps", type=int, default=200)
    parser.add_argument("--num_speakers", type=int, default=None)
    parser.add_argument("--speaker_ids", nargs="+", default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_dir):
        raise FileNotFoundError(f"Dataset directory does not exist: {args.dataset_dir}")

    orig_cwd = os.getcwd()
    suffix = _active_loss_suffix(args.min_timesteps)
    output_root = os.path.join(args.output_dir, suffix)
    tmp_cfg, runtime_cfg = _build_runtime_config(orig_cwd, output_root)

    speaker_ids = [speaker_id.strip() for speaker_id in args.speaker_ids] if args.speaker_ids else None
    speaker_dirs = _find_speaker_dirs(args.dataset_dir, speaker_ids, args.num_speakers)
    cprint(f"Processing {len(speaker_dirs)} speaker folder(s).", "y")
    if not speaker_dirs:
        return

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cprint(f"Loading HalloNet once ({_HALLO_RUNTIME_DTYPE}, xformers disabled)...", "y")
    model = _load_hallo_model(tmp_cfg, args.min_timesteps, runtime_dtype=_HALLO_RUNTIME_DTYPE)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

    face_analysis_model_path = os.path.abspath(runtime_cfg.face_analysis_model_path)
    models_cwd = os.path.dirname(os.path.dirname(face_analysis_model_path))
    cache_path = os.path.join(orig_cwd, ".cache")
    face_expand_ratio = 1.2
    img_size = (512, 512)

    os.chdir(models_cwd)
    image_processor = ImageProcessor(img_size, face_analysis_model_path)
    os.chdir(orig_cwd)

    use_kl = _LOSS_WEIGHTS.get("kl", 0.0) > 0
    start_time = time.time()
    processed_count = 0

    try:
        for speaker_dir in speaker_dirs:
            speaker_id = os.path.basename(speaker_dir)
            cprint("=" * 60, "y")
            cprint(f"Starting speaker {speaker_id}", "y")
            cprint("=" * 60, "y")

            source_image_path = _find_first_image(speaker_dir)
            if source_image_path is None:
                cprint(f"No portrait image found, skipping: {speaker_dir}", "r")
                continue

            audio_paths = _find_audio_files(speaker_dir)
            if not audio_paths:
                cprint(f"No wav files found, skipping: {speaker_dir}", "r")
                continue

            speaker_output = os.path.join(args.output_dir, speaker_id, suffix)
            os.makedirs(speaker_output, exist_ok=True)

            os.chdir(models_cwd)
            try:
                source_input, target_lip_mask = _preprocess_speaker_image(
                    image_processor,
                    source_image_path,
                    cache_path,
                    face_expand_ratio,
                )
            finally:
                os.chdir(orig_cwd)

            source_input["image"] = source_input["image"].to(args.device)
            cprint("Caching image features...", "y")
            model.cache_image_features(source_input)

            target_attn = None
            common_N = 0
            if use_kl:
                target_attn, common_N = _compute_kl_target(
                    model,
                    audio_paths[0],
                    target_lip_mask,
                    args.min_timesteps,
                    args.device,
                    source_image_path=source_image_path if _SAVE_GT_TARGET_ATTN_MAPS else None,
                    output_path=speaker_output if _SAVE_GT_TARGET_ATTN_MAPS else None,
                )
                model.cache_image_features(source_input)

            weight_hook = LipAttnWeightHook(model.denoising_unet) if use_kl else None
            try:
                for audio_path in tqdm(audio_paths, desc=f"speaker {speaker_id}"):
                    cprint(f"Processing {audio_path}", "y")
                    attack_start = time.perf_counter()
                    adv, stats = protect_audio(
                        audio_path=audio_path,
                        hallo_net=model,
                        loss_weights=_LOSS_WEIGHTS,
                        weight_hook=weight_hook,
                        target_attn=None if target_attn is None else target_attn.to(model.denoising_unet.device),
                        common_N=common_N,
                        source_image_path=source_image_path,
                        output_path=speaker_output,
                        return_stats=True,
                    )
                    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
                    output_path = os.path.join(speaker_output, f"{audio_name}_adv.wav")
                    sf.write(output_path, adv, samplerate=16000, subtype="FLOAT")
                    processed_count += 1
                    cprint(
                        f"Saved {output_path} | elapsed={time.perf_counter() - attack_start:.3f}s | "
                        f"attack_compute={stats.get('attack_compute_elapsed_sec', 0.0):.3f}s",
                        "g",
                    )
            finally:
                if weight_hook is not None:
                    weight_hook.remove()

            model._image_cache = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cprint(f"Finished speaker {speaker_id}", "g")
    finally:
        os.chdir(orig_cwd)

    cprint(
        f"All done. Processed {processed_count} audio file(s) in {time.time() - start_time:.1f}s.",
        "g",
    )


if __name__ == "__main__":
    main()
