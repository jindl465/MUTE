"""
MUTE adversarial audio generation.

This file keeps only the losses used in the paper:
  L_emb  : frame-wise embedding cosine loss.
  L_attn : target attention KL loss.
  L_stft : STFT-mask consistency loss.
"""

import gc
import glob
import math
import os
import random
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
_PROTECT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_PROTECT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _PROTECT_DIR)

import hydra
import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from PIL import Image as PILImage
from tqdm import tqdm

from audio_loss import (
    LipAttnWeightHook,
    compute_target_attn_map,
    forward_framewise_embedding_loss,
    forward_kl,
)
from hallo.datasets.image_processor import ImageProcessor
from protect_hallo import HalloNet, process_audio_emb
from utils import cprint, mp

matplotlib.use("Agg", force=True)


_EPSILON = 0.004
_ALPHA = 0.0002
_STEPS = 200
_PHASE_RATIO = 0.1
_SCORE = "score"
_TARGET_ATTN_SOFTMAX_ALPHA = 1.0
_SAVE_GT_TARGET_ATTN_MAPS = os.environ.get("SAVE_GT_TARGET_ATTN_MAPS", "0") != "0"

_STFT_MASK_SAMPLE_RATE = 16000
_STFT_MASK_N_FFT = 512
_STFT_MASK_HOP_LENGTH = 160
_STFT_MASK_WIN_LENGTH = 400
_STFT_MASK_MAG_PERCENTILE = 0.25
_STFT_MASK_MIN_FREQ_HZ = 80.0
_STFT_MASK_MAX_FREQ_HZ = 7600.0
_STFT_MASK_TIME_DILATION = 2
_STFT_MASK_FREQ_DILATION = 1
_STFT_MASK_SOFT_STRENGTH = 1.0

_HALLO_RUNTIME_DTYPE = "fp16"

_LOSS_WEIGHTS = {
    "emb_frame_l1": 1.0,
    "kl": 1.0,
    "stft_mask_cons": 5.0,
}


def _load_target_lip_mask_pyramid(mask_path: str) -> list[torch.Tensor]:
    mask_pil = PILImage.open(mask_path).convert("L")
    masks = []
    for size in (64, 32, 16, 8):
        resized = mask_pil.resize((size, size), PILImage.Resampling.BILINEAR)
        mask_np = np.asarray(resized, dtype=np.float32) / 255.0
        masks.append(torch.from_numpy(mask_np).reshape(1, -1))
    return masks


def _aggregate_clean_attn_map_from_capture(
    results: list[dict],
    common_N: int,
) -> torch.Tensor:
    common_H = int(math.isqrt(common_N))
    maps = []
    for result in results:
        attn = result["A"].float().mean(dim=0)
        N = attn.shape[0]
        H = int(math.isqrt(N))
        if N != common_N:
            attn = F.interpolate(
                attn.reshape(1, 1, H, H),
                size=(common_H, common_H),
                mode="bilinear",
                align_corners=False,
            ).reshape(common_N)
        maps.append(attn)
    clean_attn = torch.stack(maps).mean(dim=0)
    return clean_attn / clean_attn.sum().clamp_min(1e-12)


def _save_attn_vis(
    A_norm: torch.Tensor,
    label: str,
    source_image_path: str,
    output_path: str,
) -> None:
    common_H = int(math.isqrt(A_norm.shape[0]))
    attn_map = A_norm.float().detach().cpu().reshape(common_H, common_H).numpy()
    source = PILImage.open(source_image_path).convert("RGB").resize((512, 512))
    source_np = np.asarray(source, dtype=np.float32) / 255.0
    vmax = float(np.percentile(attn_map, 99))
    if vmax <= float(attn_map.min()):
        vmax = float(attn_map.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(source_np)
    axes[0].set_title("Original")
    axes[0].axis("off")
    im = axes[1].imshow(
        attn_map,
        cmap="hot",
        interpolation="nearest",
        vmin=float(attn_map.min()),
        vmax=vmax,
    )
    axes[1].set_title(label)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1])

    attn_norm = np.clip(
        (attn_map - float(attn_map.min())) / (vmax - float(attn_map.min()) + 1e-12),
        0.0,
        1.0,
    )
    attn_up = np.asarray(
        PILImage.fromarray((attn_norm * 255).astype(np.uint8)).resize(
            (512, 512),
            PILImage.Resampling.BILINEAR,
        ),
        dtype=np.float32,
    ) / 255.0
    overlay = source_np * 0.5 + plt.cm.hot(attn_up)[..., :3] * 0.5
    axes[2].imshow(np.clip(overlay, 0.0, 1.0))
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    plt.tight_layout()
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, f"attn_map_{label}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_gt_and_target_attn_vis(
    results: list[dict],
    targets: dict[str, torch.Tensor],
    common_N: int,
    selected_score: str,
    source_image_path: str | None,
    output_path: str | None,
) -> None:
    if not source_image_path or not output_path or selected_score not in targets:
        return
    os.makedirs(output_path, exist_ok=True)
    clean_attn = _aggregate_clean_attn_map_from_capture(results, common_N)
    _save_attn_vis(clean_attn, "clean_attn", source_image_path, output_path)
    _save_attn_vis(targets[selected_score], f"target_attn_{selected_score}", source_image_path, output_path)


def _restore_hallo_audio_to_waveform(
    audio_hallo: torch.Tensor,
    raw_mean: float,
    raw_std: float,
    raw_ref_waveform: np.ndarray,
    epsilon_raw: float,
) -> np.ndarray:
    audio_np = audio_hallo.detach().float().squeeze(0).cpu().numpy().astype(np.float64)
    restored = audio_np * float(raw_std) + float(raw_mean)
    ref = np.asarray(raw_ref_waveform, dtype=np.float64).squeeze()
    epsilon_safe = max(float(epsilon_raw) - 1e-6, 0.0)
    restored = np.minimum(np.maximum(restored, ref - epsilon_safe), ref + epsilon_safe)
    restored = np.clip(restored, -1.0, 1.0)
    return restored.astype(np.float32, copy=False)


def _load_hallo_model(
    cfg_path: str,
    min_ts: int,
    runtime_dtype: str = _HALLO_RUNTIME_DTYPE,
) -> HalloNet:
    model = HalloNet(cfg_path, min_timesteps=min_ts)
    if runtime_dtype == "fp32":
        model.net.float()
        model.vae.float()
        model.weight_dtype = torch.float32
    elif runtime_dtype == "bf16":
        model.net.bfloat16()
        model.vae.bfloat16()
        model.weight_dtype = torch.bfloat16
    elif runtime_dtype == "fp16":
        model.net.half()
        model.vae.half()
        model.weight_dtype = torch.float16
    else:
        raise ValueError(f"Unsupported runtime dtype: {runtime_dtype}")
    model.eval()
    return model


def _project_to_linf_box(
    audio_adv: torch.Tensor,
    audio_ref: torch.Tensor,
    epsilon_norm: float,
) -> torch.Tensor:
    return torch.minimum(
        torch.maximum(audio_adv, audio_ref - epsilon_norm),
        audio_ref + epsilon_norm,
    )


def _build_stft_mask(
    audio_ref: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    audio_float = audio_ref.detach().float()
    window = torch.hann_window(
        _STFT_MASK_WIN_LENGTH,
        device=audio_float.device,
        dtype=audio_float.dtype,
    )
    stft = torch.stft(
        audio_float,
        n_fft=_STFT_MASK_N_FFT,
        hop_length=_STFT_MASK_HOP_LENGTH,
        win_length=_STFT_MASK_WIN_LENGTH,
        window=window,
        center=True,
        return_complex=True,
    )
    magnitude = stft.abs()
    freqs = torch.fft.rfftfreq(
        _STFT_MASK_N_FFT,
        d=1.0 / _STFT_MASK_SAMPLE_RATE,
    ).to(device=audio_float.device, dtype=audio_float.dtype)
    band = (freqs >= _STFT_MASK_MIN_FREQ_HZ) & (freqs <= _STFT_MASK_MAX_FREQ_HZ)
    if not bool(band.any()):
        raise ValueError("The configured STFT frequency band keeps no bins.")

    threshold = torch.quantile(magnitude[:, band, :].reshape(-1), _STFT_MASK_MAG_PERCENTILE)
    keep_mask = torch.zeros_like(magnitude, dtype=audio_float.dtype)
    keep_mask[:, band, :] = (magnitude[:, band, :] >= threshold).to(audio_float.dtype)

    if _STFT_MASK_TIME_DILATION > 0 or _STFT_MASK_FREQ_DILATION > 0:
        keep_mask = F.max_pool2d(
            keep_mask.unsqueeze(1),
            kernel_size=(2 * _STFT_MASK_FREQ_DILATION + 1, 2 * _STFT_MASK_TIME_DILATION + 1),
            stride=1,
            padding=(_STFT_MASK_FREQ_DILATION, _STFT_MASK_TIME_DILATION),
        ).squeeze(1)
        keep_mask = (keep_mask > 0).to(audio_float.dtype)

    keep_mask = keep_mask * band.view(1, -1, 1).to(audio_float.dtype)
    return keep_mask, window, float(keep_mask.mean().item())


def _forward_stft_mask_consistency_loss(
    audio_adv: torch.Tensor,
    audio_ref: torch.Tensor,
    keep_mask: torch.Tensor,
    window: torch.Tensor,
) -> torch.Tensor:
    delta = (audio_adv - audio_ref).float()
    delta_stft = torch.stft(
        delta,
        n_fft=_STFT_MASK_N_FFT,
        hop_length=_STFT_MASK_HOP_LENGTH,
        win_length=_STFT_MASK_WIN_LENGTH,
        window=window,
        center=True,
        return_complex=True,
    )
    soft_mask = 1.0 - _STFT_MASK_SOFT_STRENGTH * (1.0 - keep_mask.to(delta_stft.real.dtype))
    projected_stft = delta_stft * soft_mask
    projected_delta = torch.istft(
        projected_stft,
        n_fft=_STFT_MASK_N_FFT,
        hop_length=_STFT_MASK_HOP_LENGTH,
        win_length=_STFT_MASK_WIN_LENGTH,
        window=window,
        center=True,
        length=delta.shape[-1],
    )
    if projected_delta.dim() == 1:
        projected_delta = projected_delta.unsqueeze(0)
    return F.mse_loss(delta, projected_delta.float())


def _prepare_fixed_unet_inputs(
    hallo_net: HalloNet,
    audio_raw: torch.Tensor,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        audio_norm = (audio_raw - norm_mean) / norm_std
        outputs = hallo_net.wav2vec_model(
            audio_norm,
            seq_len=hallo_net.seq_len,
            output_hidden_states=True,
        )
        if hallo_net.only_last_features:
            audio_emb = outputs.last_hidden_state.squeeze()
        else:
            audio_emb = torch.stack(outputs.hidden_states[1:], dim=1).squeeze(0)
            audio_emb = rearrange(audio_emb, "batch layers dim -> layers batch dim")
        audio_emb = process_audio_emb(audio_emb)

        max_start = max(audio_emb.shape[0] - hallo_net.clip_length, 0)
        min_start = min(int(hallo_net.cfg.data.n_motion_frames), max_start)
        start_idx = random.randint(min_start, max_start) if max_start > min_start else min_start

        cache = hallo_net._image_cache
        if cache is None:
            raise RuntimeError("Image features must be cached before preparing KL inputs.")
        latents = cache["latents"]
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            hallo_net.min_timesteps,
            hallo_net.max_timesteps,
            (latents.shape[0],),
            device=latents.device,
        ).long()
        noisy_latents = hallo_net.train_noise_scheduler.add_noise(latents, noise, timesteps)
    return start_idx, noisy_latents, timesteps


def _compute_kl_target(
    model: HalloNet,
    audio_path: str,
    target_image_lip_mask: list[torch.Tensor],
    min_ts: int,
    device: str,
    source_image_path: str | None = None,
    output_path: str | None = None,
) -> tuple[torch.Tensor, int]:
    from visualize_lip_attn import LipAttnGradCapture

    cprint("Computing target attention distribution...", "y")
    lip_capture = LipAttnGradCapture(model.denoising_unet)
    lip_capture.clear()

    cache = model._image_cache
    if cache is None:
        raise RuntimeError("Image features must be cached before target computation.")

    latents = cache["latents"]
    noise = torch.randn_like(latents)
    timesteps = torch.randint(min_ts, min_ts + 1, (1,), device=latents.device).long()
    noisy_latents = model.train_noise_scheduler.add_noise(latents, noise, timesteps)

    audio_raw = model.load_audio(audio_path)
    norm_mean = audio_raw.mean().detach()
    norm_std = torch.sqrt(audio_raw.var() + 1e-7).detach()
    with torch.no_grad():
        audio_norm = (audio_raw - norm_mean) / norm_std
        outputs = model.wav2vec_model(
            audio_norm,
            seq_len=model.seq_len,
            output_hidden_states=True,
        )
        if model.only_last_features:
            audio_emb = outputs.last_hidden_state.squeeze()
        else:
            audio_emb = torch.stack(outputs.hidden_states[1:], dim=1).squeeze(0)
            audio_emb = rearrange(audio_emb, "batch layers dim -> layers batch dim")
        audio_emb = process_audio_emb(audio_emb)

    start_idx = model.cfg.data.n_motion_frames
    audio_tensor = audio_emb[start_idx:start_idx + model.clip_length].unsqueeze(0).to(
        device=model.net.audioproj.device,
        dtype=model.net.audioproj.dtype,
    )

    with torch.enable_grad():
        _ = model.net(
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            ref_image_latents=cache["ref_image_latents"],
            face_emb=cache["image_prompt_embeds"],
            mask=cache["pixel_values_mask"],
            full_mask=cache["pixel_values_full_mask"],
            face_mask=cache["pixel_values_face_mask"],
            lip_mask=cache["pixel_values_lip_mask"],
            audio_emb=audio_tensor,
            uncond_img_fwd=False,
            uncond_audio_fwd=False,
        )

    results = lip_capture.compute_all()
    lip_capture.remove()
    model.net.reference_control_reader.clear()
    model.net.reference_control_writer.clear()
    if not results:
        raise RuntimeError("LipAttnGradCapture returned no attention sensitivity results.")

    targets, common_N = compute_target_attn_map(
        results,
        target_image_lip_mask,
        epsilon=_EPSILON,
        softmax_alpha=_TARGET_ATTN_SOFTMAX_ALPHA,
    )
    _save_gt_and_target_attn_vis(
        results,
        targets,
        common_N,
        _SCORE,
        source_image_path,
        output_path,
    )
    target_attn = targets[_SCORE].detach().cpu()

    del results, targets, latents, noise, timesteps, noisy_latents
    del audio_raw, norm_mean, norm_std, audio_emb, audio_tensor, lip_capture
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return target_attn, common_N


def _loss_history_summary(history: dict[str, list[float]]) -> dict[str, float]:
    summary = {}
    for name, values in history.items():
        if not values:
            continue
        summary[f"loss_{name}_first"] = float(values[0])
        summary[f"loss_{name}_last"] = float(values[-1])
        summary[f"loss_{name}_min"] = float(min(values))
        summary[f"loss_{name}_max"] = float(max(values))
    return summary


def protect_audio(
    audio_path: str,
    hallo_net: HalloNet,
    loss_weights: dict,
    weight_hook: LipAttnWeightHook | None = None,
    target_attn: torch.Tensor | None = None,
    common_N: int = 0,
    source_image_path: str = "",
    output_path: str = "",
    epsilon: float = _EPSILON,
    alpha: float = _ALPHA,
    steps: int = _STEPS,
    return_stats: bool = False,
    **_unused_kwargs,
) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
    use_emb = loss_weights.get("emb_frame_l1", 0.0) > 0
    use_kl = loss_weights.get("kl", 0.0) > 0
    stft_weight = float(loss_weights.get("stft_mask_cons", 0.0))
    use_stft = stft_weight > 0
    if not use_emb and not use_kl:
        raise ValueError("At least one paper task loss must be enabled: emb_frame_l1 or kl.")
    if use_kl and (weight_hook is None or target_attn is None or common_N <= 0):
        raise ValueError("KL loss requires weight_hook, target_attn, and common_N.")

    audio_raw = hallo_net.load_audio(audio_path)
    speech_raw, _ = librosa.load(audio_path, sr=16000)
    raw_mean = float(np.mean(speech_raw, dtype=np.float64))
    raw_std = float(np.sqrt(np.var(speech_raw, dtype=np.float64) + 1e-7))
    if raw_std <= 0:
        raise ValueError(f"Invalid waveform standard deviation for {audio_path}: {raw_std}")

    epsilon_norm = float(epsilon) / raw_std
    alpha_norm = float(alpha) / raw_std
    fixed_norm_mean = audio_raw.mean().detach()
    fixed_norm_std = torch.sqrt(audio_raw.var() + 1e-7).detach()

    stft_mask = None
    stft_window = None
    if use_stft:
        stft_mask, stft_window, keep_ratio = _build_stft_mask(audio_raw)
        cprint(
            f"  [stft] lambda={stft_weight:g} | keep_ratio={keep_ratio:.2%} | "
            f"freq=[{_STFT_MASK_MIN_FREQ_HZ:.0f},{_STFT_MASK_MAX_FREQ_HZ:.0f}]Hz",
            "y",
        )

    fixed_start_idx = None
    fixed_noisy_latents = None
    fixed_timesteps = None
    if use_kl:
        fixed_start_idx, fixed_noisy_latents, fixed_timesteps = _prepare_fixed_unet_inputs(
            hallo_net,
            audio_raw,
            fixed_norm_mean,
            fixed_norm_std,
        )

    cprint(
        f"  [pgd] epsilon={epsilon:.5f} alpha={alpha:.6f} steps={steps} "
        f"phase_ratio={_PHASE_RATIO:.2f}",
        "y",
    )
    with torch.no_grad():
        audio_adv = audio_raw + (torch.rand_like(audio_raw) * 2.0 - 1.0) * epsilon_norm
        audio_adv = _project_to_linf_box(audio_adv, audio_raw, epsilon_norm)

    phase_split = int(math.ceil(max(steps, 1) * _PHASE_RATIO))
    phase_split = max(1, min(phase_split, steps))
    loss_history: dict[str, list[float]] = {
        "emb_frame_l1": [],
        "kl": [],
        "stft_mask_cons": [],
        "total": [],
    }

    attack_start = time.perf_counter()
    last_A_norm = None
    pbar = tqdm(range(steps))
    for step_idx in pbar:
        audio_adv = audio_adv.detach().requires_grad_(True)
        total_loss = None
        log_values = {}

        # Phase 1 (k <= rho*K): alignment level (L_attn).
        # Phase 2 (k > rho*K): representation level (L_emb).
        run_kl_phase = use_kl and (not use_emb or step_idx < phase_split)
        run_emb_phase = use_emb and (not run_kl_phase)

        if run_emb_phase:
            emb_loss = forward_framewise_embedding_loss(
                hallo_net,
                audio_adv,
                audio_raw,
                fixed_norm_mean,
                fixed_norm_std,
            )
            total_loss = loss_weights["emb_frame_l1"] * emb_loss
            log_values["emb_frame_l1"] = float(emb_loss.detach().item())
            loss_history["emb_frame_l1"].append(log_values["emb_frame_l1"])

        if run_kl_phase:
            kl_loss, last_A_norm = forward_kl(
                hallo_net=hallo_net,
                audio_adv=audio_adv,
                weight_hook=weight_hook,
                target_attn=target_attn,
                common_N=common_N,
                start_idx=fixed_start_idx,
                noisy_latents=fixed_noisy_latents,
                timesteps=fixed_timesteps,
                norm_mean=fixed_norm_mean,
                norm_std=fixed_norm_std,
            )
            term = loss_weights["kl"] * kl_loss
            total_loss = term if total_loss is None else total_loss + term
            log_values["kl"] = float(kl_loss.detach().item())
            loss_history["kl"].append(log_values["kl"])

        if use_stft:
            stft_loss = _forward_stft_mask_consistency_loss(
                audio_adv,
                audio_raw,
                stft_mask,
                stft_window,
            )
            term = stft_weight * stft_loss
            total_loss = term if total_loss is None else total_loss + term
            log_values["stft_mask_cons"] = float(stft_loss.detach().item())
            loss_history["stft_mask_cons"].append(log_values["stft_mask_cons"])

        if total_loss is None:
            raise RuntimeError("No loss was computed for this PGD step.")

        grad, = torch.autograd.grad(total_loss, [audio_adv], retain_graph=False)
        if run_kl_phase:
            hallo_net.net.reference_control_reader.clear()
            hallo_net.net.reference_control_writer.clear()

        with torch.no_grad():
            audio_adv = audio_adv - grad.detach().sign() * alpha_norm
            audio_adv = _project_to_linf_box(audio_adv, audio_raw, epsilon_norm)

        total_value = float(total_loss.detach().item())
        loss_history["total"].append(total_value)
        pbar.set_description(f"[PGD] total={total_value:.6f}")
        if step_idx < 3 or (step_idx + 1) % 10 == 0 or step_idx == steps - 1:
            parts = [f"{key}={value:.6e}" for key, value in log_values.items()]
            delta = (audio_adv - audio_raw).detach().abs()
            print(
                f"[PGD] iter={step_idx + 1}/{steps} | total={total_value:.6e} | "
                f"{' | '.join(parts)} | delta_linf/eps="
                f"{delta.max().item() / max(epsilon_norm, 1e-12):.4f}"
            )

    attack_elapsed = time.perf_counter() - attack_start

    if use_kl and _SAVE_GT_TARGET_ATTN_MAPS and source_image_path and output_path:
        try:
            with torch.no_grad():
                _, last_A_norm = forward_kl(
                    hallo_net=hallo_net,
                    audio_adv=audio_adv.detach(),
                    weight_hook=weight_hook,
                    target_attn=target_attn,
                    common_N=common_N,
                    start_idx=fixed_start_idx,
                    noisy_latents=fixed_noisy_latents,
                    timesteps=fixed_timesteps,
                    norm_mean=fixed_norm_mean,
                    norm_std=fixed_norm_std,
                )
            _save_attn_vis(last_A_norm, "adv_final_attn", source_image_path, output_path)
        finally:
            hallo_net.net.reference_control_reader.clear()
            hallo_net.net.reference_control_writer.clear()

    finalize_start = time.perf_counter()
    result = _restore_hallo_audio_to_waveform(
        audio_adv,
        raw_mean=raw_mean,
        raw_std=raw_std,
        raw_ref_waveform=speech_raw,
        epsilon_raw=epsilon,
    )
    finalize_elapsed = time.perf_counter() - finalize_start
    stats = {
        "perturbation_update_elapsed_sec": float(attack_elapsed),
        "adv_waveform_finalize_elapsed_sec": float(finalize_elapsed),
        "attack_compute_elapsed_sec": float(attack_elapsed + finalize_elapsed),
        "num_update_steps": float(steps),
        "phase_ratio": float(_PHASE_RATIO),
        "phase_split_step": float(phase_split),
        "pgd_summary": _loss_history_summary(loss_history),
    }
    if return_stats:
        return result, stats
    return result


@hydra.main(version_base=None, config_path="./configs/attack", config_name="base")
def main(cfg: DictConfig):
    import hydra.utils as hu

    orig_cwd = hu.get_original_cwd()
    os.chdir(orig_cwd)

    args = cfg.attack
    img_path = args.img_path
    audio_dir = args.audio_path
    output_path = args.output_path
    device = args.device
    min_ts = args.min_timesteps

    active_losses = "+".join(k for k, w in _LOSS_WEIGHTS.items() if w > 0)
    output_path = os.path.join(
        output_path,
        f"audio_{active_losses}_eps{_EPSILON}_steps{_STEPS}_{min_ts}_{_SCORE}",
    )
    mp(output_path)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    hallo_cfg_path = os.path.join(orig_cwd, "hallo/configs/train/stage2.yaml")
    runtime_cfg = OmegaConf.load(hallo_cfg_path)
    runtime_cfg.solver.enable_xformers_memory_efficient_attention = False
    runtime_cfg.solver.gradient_checkpointing = False
    runtime_cfg.weight_dtype = _HALLO_RUNTIME_DTYPE
    tmp_cfg = os.path.join(output_path, "_stage2_audio.yaml")
    OmegaConf.save(runtime_cfg, tmp_cfg)

    cprint(f"Loading HalloNet ({_HALLO_RUNTIME_DTYPE}, xformers disabled)...", "y")
    model = _load_hallo_model(tmp_cfg, min_ts, runtime_dtype=_HALLO_RUNTIME_DTYPE)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

    image_candidates = []
    if os.path.isfile(img_path):
        image_candidates = [img_path]
    else:
        image_candidates = (
            glob.glob(os.path.join(img_path, "*.png"))
            + glob.glob(os.path.join(img_path, "*.jpg"))
            + glob.glob(os.path.join(img_path, "*.jpeg"))
        )
    if not image_candidates:
        raise FileNotFoundError(f"No portrait image found under {img_path}")
    source_image_path = sorted(image_candidates)[0]

    img_size = (512, 512)
    face_analysis_model_path = os.path.abspath(runtime_cfg.face_analysis_model_path)
    cache_path = os.path.join(orig_cwd, ".cache")
    face_expand_ratio = 1.2
    models_cwd = os.path.dirname(os.path.dirname(face_analysis_model_path))
    os.chdir(models_cwd)
    with ImageProcessor(img_size, face_analysis_model_path) as image_processor:
        (
            source_image_pixels,
            source_image_face_region,
            source_image_face_emb,
            source_image_full_mask,
            source_image_face_mask,
            source_image_lip_mask,
        ) = image_processor.preprocess(source_image_path, cache_path, face_expand_ratio)
        source_image_target_lip_mask = _load_target_lip_mask_pyramid(
            os.path.join(
                cache_path,
                f"{os.path.splitext(os.path.basename(source_image_path))[0]}_lip_mask.png",
            )
        )
    os.chdir(orig_cwd)

    source_input = {
        "image": source_image_pixels.to(device),
        "face_region": source_image_face_region,
        "face_emb": torch.from_numpy(source_image_face_emb),
        "full_mask": source_image_full_mask,
        "face_mask": source_image_face_mask,
        "lip_mask": source_image_lip_mask,
    }
    model.cache_image_features(source_input)

    audio_paths = sorted(glob.glob(os.path.join(audio_dir, "*.wav")) + glob.glob(os.path.join(audio_dir, "*.WAV")))
    if not audio_paths:
        raise FileNotFoundError(f"No wav files found under {audio_dir}")

    target_attn = None
    common_N = 0
    if _LOSS_WEIGHTS.get("kl", 0.0) > 0:
        target_attn, common_N = _compute_kl_target(
            model,
            audio_paths[0],
            source_image_target_lip_mask,
            min_ts,
            device,
            source_image_path=source_image_path if _SAVE_GT_TARGET_ATTN_MAPS else None,
            output_path=output_path if _SAVE_GT_TARGET_ATTN_MAPS else None,
        )
        model.cache_image_features(source_input)

    weight_hook = LipAttnWeightHook(model.denoising_unet) if _LOSS_WEIGHTS.get("kl", 0.0) > 0 else None
    try:
        for audio_path in tqdm(audio_paths, desc="protect"):
            adv = protect_audio(
                audio_path=audio_path,
                hallo_net=model,
                loss_weights=_LOSS_WEIGHTS,
                weight_hook=weight_hook,
                target_attn=None if target_attn is None else target_attn.to(model.denoising_unet.device),
                common_N=common_N,
                source_image_path=source_image_path,
                output_path=output_path,
            )
            audio_name = os.path.splitext(os.path.basename(audio_path))[0]
            out_path = os.path.join(output_path, f"{audio_name}_adv.wav")
            sf.write(out_path, adv, samplerate=16000, subtype="FLOAT")
            cprint(f"Saved {out_path}", "g")
    finally:
        if weight_hook is not None:
            weight_hook.remove()


if __name__ == "__main__":
    main()
