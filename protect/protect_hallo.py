import argparse
import math
import os
import random
import ssl
from typing import Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange, repeat
from omegaconf import OmegaConf
from transformers import Wav2Vec2FeatureExtractor

from hallo.models.audio_proj import AudioProjModel
from hallo.models.face_locator import FaceLocator
from hallo.models.image_proj import ImageProjModel
from hallo.models.mutual_self_attention import ReferenceAttentionControl
from hallo.models.unet_2d_condition import UNet2DConditionModel
from hallo.models.unet_3d import UNet3DConditionModel
from hallo.models.wav2vec import Wav2VecModel


ssl._create_default_https_context = ssl._create_unverified_context
os.environ["TORCH_HOME"] = os.getcwd()
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "hub")


def load_config(config_path: str):
    if config_path.endswith(".yaml"):
        return OmegaConf.load(config_path)
    raise ValueError("Unsupported config file format.")


def process_audio_emb(audio_emb):
    concatenated = []
    for i in range(audio_emb.shape[0]):
        window = [
            audio_emb[max(min(i + j, audio_emb.shape[0] - 1), 0)]
            for j in range(-2, 3)
        ]
        concatenated.append(torch.stack(window, dim=0))
    return torch.stack(concatenated, dim=0)


class Net(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        face_locator: FaceLocator,
        reference_control_writer,
        reference_control_reader,
        imageproj,
        audioproj,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.face_locator = face_locator
        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader
        self.imageproj = imageproj
        self.audioproj = audioproj

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        ref_image_latents: torch.Tensor,
        face_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        mask: torch.Tensor,
        full_mask: torch.Tensor,
        face_mask: torch.Tensor,
        lip_mask: torch.Tensor,
        uncond_img_fwd: bool = False,
        uncond_audio_fwd: bool = False,
    ):
        with torch.no_grad():
            face_emb = self.imageproj(
                face_emb.to(device=self.imageproj.device, dtype=self.imageproj.dtype)
            )
            mask = mask.to(device=self.face_locator.device, dtype=self.face_locator.dtype)
            mask_feature = self.face_locator(mask)

        audio_emb = audio_emb.to(
            device=self.audioproj.device,
            dtype=self.audioproj.dtype,
        )
        audio_emb = self.audioproj(audio_emb)

        if not uncond_img_fwd:
            with torch.no_grad():
                ref_timesteps = torch.zeros_like(timesteps)
                ref_timesteps = repeat(
                    ref_timesteps,
                    "b -> (repeat b)",
                    repeat=ref_image_latents.size(0) // ref_timesteps.size(0),
                )
                self.reference_unet(
                    ref_image_latents,
                    ref_timesteps,
                    encoder_hidden_states=face_emb,
                    return_dict=False,
                )
                self.reference_control_reader.update(self.reference_control_writer)

        if uncond_audio_fwd:
            audio_emb = torch.zeros_like(audio_emb).to(
                device=audio_emb.device,
                dtype=audio_emb.dtype,
            )

        return self.denoising_unet(
            noisy_latents,
            timesteps,
            mask_cond_fea=mask_feature,
            encoder_hidden_states=face_emb,
            audio_embedding=audio_emb,
            full_mask=full_mask,
            face_mask=face_mask,
            lip_mask=lip_mask,
        ).sample


def get_noise_scheduler(cfg: argparse.Namespace) -> Tuple[DDIMScheduler, DDIMScheduler]:
    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)
    return train_noise_scheduler, val_noise_scheduler


class HalloNet(nn.Module):
    def __init__(self, cfg, min_timesteps):
        super().__init__()
        self.cfg = load_config(cfg)
        self.weight_dtype = self.get_weight_dtype(self.cfg.weight_dtype)
        self.vae, self.imageproj, self.denoising_unet, self.net = (
            self.initialize_models()
        )
        self.train_noise_scheduler, self.val_noise_scheduler = get_noise_scheduler(
            self.cfg
        )

        audio_emb_path = "./th1kh/th1kh_tmp/audio_emb/0001.pt"
        self.audio_emb = (
            process_audio_emb(torch.load(audio_emb_path))
            if os.path.exists(audio_emb_path)
            else None
        )
        self.clip_length = 1
        self.min_timesteps = min_timesteps
        self.max_timesteps = self.min_timesteps + 100

        wav2vec_path = self.cfg.wav2vec_config.model_path
        self.wav2vec_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            wav2vec_path,
            local_files_only=True,
        )
        self.wav2vec_model = Wav2VecModel.from_pretrained(
            wav2vec_path,
            local_files_only=True,
        ).to("cuda")
        self.wav2vec_model.requires_grad_(False)
        self.wav2vec_model.eval()
        self.only_last_features = self.cfg.wav2vec_config.features == "last"
        self.seq_len = None
        self._image_cache = None

    def initialize_models(self):
        cfg = self.cfg
        load_dtype = torch.float16

        vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
            "cuda",
            dtype=load_dtype,
        )
        torch.cuda.empty_cache()

        reference_unet = UNet2DConditionModel.from_pretrained(
            cfg.base_model_path,
            subfolder="unet",
        ).to(device="cuda", dtype=load_dtype)
        torch.cuda.empty_cache()

        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            cfg.base_model_path,
            cfg.motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(
                cfg.unet_additional_kwargs
            ),
            use_landmark=False,
        ).to(device="cuda", dtype=load_dtype)
        torch.cuda.empty_cache()

        imageproj = ImageProjModel(
            cross_attention_dim=denoising_unet.config.cross_attention_dim,
            clip_embeddings_dim=512,
            clip_extra_context_tokens=4,
        ).to(device="cuda", dtype=load_dtype)
        face_locator = FaceLocator(conditioning_embedding_channels=320).to(
            device="cuda",
            dtype=load_dtype,
        )
        audioproj = AudioProjModel(
            seq_len=5,
            blocks=12,
            channels=768,
            intermediate_dim=512,
            output_dim=768,
            context_tokens=32,
        ).to(device="cuda", dtype=load_dtype)

        vae.requires_grad_(False)
        imageproj.requires_grad_(False)
        reference_unet.requires_grad_(False)
        denoising_unet.requires_grad_(False)
        face_locator.requires_grad_(False)
        audioproj.requires_grad_(False)

        reference_control_writer = ReferenceAttentionControl(
            reference_unet,
            do_classifier_free_guidance=False,
            mode="write",
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            denoising_unet,
            do_classifier_free_guidance=False,
            mode="read",
            fusion_blocks="full",
        )

        net = Net(
            reference_unet,
            denoising_unet,
            face_locator,
            reference_control_writer,
            reference_control_reader,
            imageproj,
            audioproj,
        )
        missing, unexpected = net.load_state_dict(
            torch.load(
                os.path.join(cfg.audio_ckpt_dir, "net.pth"),
                map_location="cpu",
            )
        )
        if missing or unexpected:
            raise RuntimeError(
                f"Failed to load Hallo checkpoint: missing={missing}, "
                f"unexpected={unexpected}"
            )
        print("loaded weight from", os.path.join(cfg.audio_ckpt_dir, "net.pth"))

        if cfg.solver.enable_xformers_memory_efficient_attention:
            if not is_xformers_available():
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly."
                )
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()

        if cfg.solver.gradient_checkpointing:
            reference_unet.enable_gradient_checkpointing()
            denoising_unet.enable_gradient_checkpointing()

        return vae, imageproj, denoising_unet, net

    def get_weight_dtype(self, weight_dtype):
        if weight_dtype == "fp16":
            return torch.float16
        if weight_dtype == "bf16":
            return torch.bfloat16
        if weight_dtype == "fp32":
            return torch.float32
        raise ValueError(f"Unsupported weight dtype: {weight_dtype}")

    def load_audio(self, audio_path: str) -> torch.Tensor:
        speech_array, _ = librosa.load(audio_path, sr=16000)
        audio_feature = np.squeeze(
            self.wav2vec_extractor(
                speech_array,
                sampling_rate=16000,
            ).input_values
        )
        self.seq_len = math.ceil(len(audio_feature) / 16000 * 25)
        return torch.from_numpy(audio_feature).float().unsqueeze(0).cuda()

    def cache_image_features(self, batch: dict):
        weight_dtype = self.weight_dtype
        pixel_values_vid = batch["image"].unsqueeze(0).unsqueeze(0).to(weight_dtype)
        pixel_values_vid = pixel_values_vid.repeat(1, self.clip_length, 1, 1, 1)

        pixel_values_face_mask = [
            m.repeat(self.clip_length, 1) for m in batch["face_mask"]
        ]
        pixel_values_lip_mask = [
            m.repeat(self.clip_length, 1) for m in batch["lip_mask"]
        ]
        pixel_values_full_mask = [
            m.repeat(self.clip_length, 1) for m in batch["full_mask"]
        ]

        pixel_values_face_mask = [
            m.to(device=self.denoising_unet.device, dtype=self.denoising_unet.dtype)
            for m in pixel_values_face_mask
        ]
        pixel_values_lip_mask = [
            m.to(device=self.denoising_unet.device, dtype=self.denoising_unet.dtype)
            for m in pixel_values_lip_mask
        ]
        pixel_values_full_mask = [
            m.to(device=self.denoising_unet.device, dtype=self.denoising_unet.dtype)
            for m in pixel_values_full_mask
        ]

        with torch.no_grad():
            video_length = pixel_values_vid.shape[1]
            pixel_values = rearrange(pixel_values_vid, "b f c h w -> (b f) c h w")
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = rearrange(
                latents,
                "(b f) c h w -> b c f h w",
                f=video_length,
            )
            latents = latents * 0.18215

            pixel_values_mask = batch["face_region"].unsqueeze(0).unsqueeze(1).to(
                dtype=weight_dtype
            )
            pixel_values_mask = repeat(
                pixel_values_mask,
                "b f c h w -> b (repeat f) c h w",
                repeat=video_length,
            )
            pixel_values_mask = pixel_values_mask.transpose(1, 2)

            pixel_values_ref_img = batch["image"].to(dtype=weight_dtype).unsqueeze(0)
            motion_zeros = pixel_values_ref_img.repeat(
                self.cfg.data.n_motion_frames,
                1,
                1,
                1,
            ).to(
                dtype=pixel_values_ref_img.dtype,
                device=pixel_values_ref_img.device,
            )
            pixel_values_ref_img = torch.cat(
                [pixel_values_ref_img, motion_zeros],
                dim=0,
            ).unsqueeze(0)
            pixel_values_ref_img[:, 1:] = 0.0
            ref_img_and_motion = rearrange(
                pixel_values_ref_img,
                "b f c h w -> (b f) c h w",
            )
            ref_image_latents = (
                self.vae.encode(ref_img_and_motion).latent_dist.sample() * 0.18215
            )

            image_prompt_embeds = batch["face_emb"].to(
                dtype=self.imageproj.dtype,
                device=self.imageproj.device,
            ).reshape(1, -1)

        self._image_cache = {
            "latents": latents,
            "ref_image_latents": ref_image_latents,
            "image_prompt_embeds": image_prompt_embeds,
            "pixel_values_mask": pixel_values_mask,
            "pixel_values_full_mask": pixel_values_full_mask,
            "pixel_values_face_mask": pixel_values_face_mask,
            "pixel_values_lip_mask": pixel_values_lip_mask,
        }

    def forward_audio_attack(self, audio_adv: torch.Tensor) -> torch.Tensor:
        if self._image_cache is None:
            raise RuntimeError("Call cache_image_features() before the attack loop.")
        if self.seq_len is None:
            raise RuntimeError("Call load_audio() before the attack loop.")

        cache = self._image_cache
        latents = cache["latents"]
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(
            self.min_timesteps,
            self.max_timesteps,
            (bsz,),
            device=latents.device,
        ).long()
        noisy_latents = self.train_noise_scheduler.add_noise(latents, noise, timesteps)

        mean = audio_adv.mean()
        var = audio_adv.var()
        audio_feature_norm = (audio_adv - mean) / torch.sqrt(var + 1e-7)
        embeddings = self.wav2vec_model(
            audio_feature_norm,
            seq_len=self.seq_len,
            output_hidden_states=True,
        )
        if self.only_last_features:
            audio_emb = embeddings.last_hidden_state.squeeze()
        else:
            audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
            audio_emb = rearrange(audio_emb, "b s d -> s b d")

        audio_emb = process_audio_emb(audio_emb)
        start_idx = random.randint(
            self.cfg.data.n_motion_frames,
            audio_emb.shape[0] - self.clip_length,
        )
        audio_tensor = audio_emb[start_idx:start_idx + self.clip_length].unsqueeze(0)

        model_pred = self.net(
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
        return F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
