"""
Loss components used by MUTE.

The paper attack uses two task losses:
  L_emb  : frame-wise cosine loss on temporal wav2vec embeddings.
  L_attn : KL divergence between target and adversarial attention maps.

The STFT regularizer is implemented in protect_audio.py because it operates
directly on the waveform perturbation.
"""

import math
import random

import torch
import torch.nn.functional as F
from einops import rearrange

from hallo.models.attention import AudioTemporalBasicTransformerBlock
from protect_hallo import HalloNet, process_audio_emb


class _AttentionWeightCapture:
    """Attention processor that records raw attention weights for KL loss."""

    def __init__(self):
        self.attn_weights: torch.Tensor | None = None
        self.batch_size: int | None = None

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ):
        self.batch_size = hidden_states.shape[0]
        kv_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        query = attn.head_to_batch_dim(attn.to_q(hidden_states))
        key = attn.head_to_batch_dim(attn.to_k(kv_states))
        value = attn.head_to_batch_dim(attn.to_v(kv_states))

        scores = torch.bmm(query, key.transpose(-1, -2)) * (query.shape[-1] ** -0.5)
        if attention_mask is not None:
            scores = scores + attention_mask

        weights = torch.softmax(scores, dim=-1)
        self.attn_weights = weights

        hidden_states = torch.bmm(weights, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class LipAttnWeightHook:
    """
    Replace Hallo lip audio cross-attention processors and capture attention maps.

    The hook targets attn2_2, which is the lip branch used for the paper's
    alignment-level disruption objective.
    """

    def __init__(self, denoising_unet: torch.nn.Module):
        self._processors: list[tuple] = []

        for module in denoising_unet.modules():
            if not isinstance(module, AudioTemporalBasicTransformerBlock):
                continue
            attn_module = getattr(module, "attn2_2", None)
            if attn_module is None:
                continue
            capture = _AttentionWeightCapture()
            original_processor = attn_module.processor
            attn_module.set_processor(capture)
            self._processors.append((attn_module, original_processor, capture))

        if not self._processors:
            raise RuntimeError(
                "No attn2_2 layers were found. Disable xformers before registering "
                "LipAttnWeightHook."
            )
        print(f"[LipAttnWeightHook] Hooked {len(self._processors)} lip attention layer(s).")

    def clear(self) -> None:
        for _, _, capture in self._processors:
            capture.attn_weights = None
            capture.batch_size = None

    def remove(self) -> None:
        for attn_module, original_processor, _ in self._processors:
            attn_module.set_processor(original_processor)
        self._processors.clear()

    def get_A_per_layer(self) -> list[tuple[torch.Tensor, int]]:
        """
        Return spatial attention maps from captured layers.

        For each layer, raw weights have shape [B*H, N, M]. They are reduced to
        [B, N] by taking max over audio tokens and mean over heads.
        """
        maps = []
        for _, _, capture in self._processors:
            if capture.attn_weights is None:
                continue
            weights = capture.attn_weights
            batch = capture.batch_size or 1
            heads = weights.shape[0] // batch
            attn_map = (
                weights.reshape(batch, heads, weights.shape[1], weights.shape[2])
                .max(dim=3).values
                .mean(dim=1)
            )
            maps.append((attn_map, weights.shape[1]))
        return maps


def compute_target_attn_map(
    results: list[dict],
    _lip_mask_list: list[torch.Tensor] | None = None,
    epsilon: float = 0.01,
    softmax_alpha: float = 1.0,
) -> tuple[dict[str, torch.Tensor], int]:
    """
    Build the paper target attention distribution.

    The sensitivity score is:
        s(n) = (abs(A'(n)) + epsilon / 2 * abs(A''(n))) / A(n)

    Lip masks are accepted only for API compatibility with older wrappers; the
    paper target does not mask out lip pixels.
    """
    if not results:
        raise ValueError("No attention sensitivity results were provided.")

    eps = 1e-8
    spatial_sizes = [result["A"].shape[1] for result in results]
    common_N = max(spatial_sizes)
    common_H = int(math.isqrt(common_N))
    if common_H * common_H != common_N:
        raise ValueError(f"Attention size is not square: {common_N}")

    score_maps = []
    for result in results:
        A = result["A"].float().mean(dim=0).clamp_min(eps)
        Ap = result["Ap"].float().abs().mean(dim=0)
        App = result["App"].float().abs().mean(dim=0)
        N = A.shape[0]
        H = int(math.isqrt(N))
        if H * H != N:
            raise ValueError(f"Attention size is not square: {N}")

        score = (Ap + (float(epsilon) / 2.0) * App) / A
        if N != common_N:
            score = F.interpolate(
                score.reshape(1, 1, H, H),
                size=(common_H, common_H),
                mode="bilinear",
                align_corners=False,
            ).reshape(common_N)
        score_maps.append(score)

    score = torch.stack(score_maps).mean(dim=0)
    target = torch.softmax(float(softmax_alpha) * score, dim=0)
    return {"score": target}, common_N


def _get_temporal_audio_emb_seq(
    hallo_net: HalloNet,
    audio: torch.Tensor,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    apply_context_window: bool = False,
) -> torch.Tensor:
    audio_norm = (audio - norm_mean) / norm_std
    outputs = hallo_net.wav2vec_model(
        audio_norm,
        seq_len=hallo_net.seq_len,
        output_hidden_states=True,
    )
    if hallo_net.only_last_features:
        emb_seq = outputs.last_hidden_state.squeeze(0)
    else:
        emb_seq = torch.stack(outputs.hidden_states[1:], dim=1).squeeze(0)

    if apply_context_window:
        emb_seq = process_audio_emb(emb_seq)
    if emb_seq.ndim > 2:
        emb_seq = emb_seq.reshape(emb_seq.shape[0], -1)
    return emb_seq


def forward_framewise_embedding_loss(
    hallo_net: HalloNet,
    audio_adv: torch.Tensor,
    audio_raw: torch.Tensor,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    start_idx: int | None = None,
    clip_length: int | None = None,
    processed_context: bool = False,
) -> torch.Tensor:
    """
    Compute L_emb, the frame-wise embedding cosine loss.

    Minimizing this value pushes each adversarial temporal embedding away from
    the matching clean temporal embedding.
    """
    with torch.no_grad():
        clean = _get_temporal_audio_emb_seq(
            hallo_net,
            audio_raw,
            norm_mean,
            norm_std,
            apply_context_window=processed_context,
        )
    adv = _get_temporal_audio_emb_seq(
        hallo_net,
        audio_adv,
        norm_mean,
        norm_std,
        apply_context_window=processed_context,
    )

    length = min(adv.shape[0], clean.shape[0])
    if length <= 0:
        raise RuntimeError("Wav2Vec returned an empty temporal embedding sequence.")
    if clip_length is not None:
        if clip_length <= 0:
            raise ValueError(f"clip_length must be positive, got {clip_length}")
        clip_length = min(int(clip_length), length)
        max_start = max(length - clip_length, 0)
        if start_idx is None:
            start_idx = 0 if max_start == 0 else random.randint(0, max_start)
        start_idx = max(0, min(int(start_idx), max_start))
        end_idx = start_idx + clip_length
        adv = adv[start_idx:end_idx]
        clean = clean[start_idx:end_idx]
    else:
        adv = adv[:length]
        clean = clean[:length]

    return F.cosine_similarity(adv, clean, dim=-1).mean()


def _audio_condition_for_unet(
    hallo_net: HalloNet,
    audio_adv: torch.Tensor,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    start_idx: int | None,
) -> torch.Tensor:
    audio_norm = (audio_adv - norm_mean) / norm_std
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
    if start_idx is None:
        start_idx = random.randint(
            hallo_net.cfg.data.n_motion_frames,
            audio_emb.shape[0] - hallo_net.clip_length,
        )
    return audio_emb[start_idx:start_idx + hallo_net.clip_length].unsqueeze(0)


def forward_kl(
    hallo_net: HalloNet,
    audio_adv: torch.Tensor,
    weight_hook: LipAttnWeightHook,
    target_attn: torch.Tensor,
    common_N: int,
    start_idx: int | None = None,
    noisy_latents: torch.Tensor | None = None,
    timesteps: torch.Tensor | None = None,
    norm_mean: torch.Tensor | None = None,
    norm_std: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute L_attn = KL(A_target || A_adv)."""
    if hallo_net._image_cache is None:
        raise RuntimeError("Call hallo_net.cache_image_features(batch) before KL loss.")
    if hallo_net.seq_len is None:
        raise RuntimeError("Call hallo_net.load_audio(audio_path) before KL loss.")

    cache = hallo_net._image_cache
    weight_hook.clear()

    latents = cache["latents"]
    if noisy_latents is None or timesteps is None:
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            hallo_net.min_timesteps,
            hallo_net.max_timesteps,
            (latents.shape[0],),
            device=latents.device,
        ).long()
        noisy_latents = hallo_net.train_noise_scheduler.add_noise(latents, noise, timesteps)

    if norm_mean is None:
        norm_mean = audio_adv.mean().detach()
    if norm_std is None:
        norm_std = torch.sqrt(audio_adv.var() + 1e-7).detach()

    audio_tensor = _audio_condition_for_unet(
        hallo_net,
        audio_adv,
        norm_mean,
        norm_std,
        start_idx,
    )

    _ = hallo_net.net(
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

    common_H = int(math.isqrt(common_N))
    maps = []
    for attn_map, N in weight_hook.get_A_per_layer():
        H = int(math.isqrt(N))
        attn_map = attn_map.float().mean(dim=0)
        if N != common_N:
            attn_map = F.interpolate(
                attn_map.reshape(1, 1, H, H),
                size=(common_H, common_H),
                mode="bilinear",
                align_corners=False,
            ).reshape(common_N)
        maps.append(attn_map)

    if not maps:
        raise RuntimeError(
            "LipAttnWeightHook captured no maps. Ensure xformers is disabled."
        )

    A_adv = torch.stack(maps).mean(dim=0)
    A_adv = A_adv / A_adv.sum().clamp_min(1e-10)
    target = target_attn.to(device=A_adv.device, dtype=A_adv.dtype)
    target = target / target.sum().clamp_min(1e-10)
    loss = F.kl_div((A_adv + 1e-10).log(), target, reduction="sum")
    return loss, A_adv
