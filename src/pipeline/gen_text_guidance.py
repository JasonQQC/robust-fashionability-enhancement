#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mid-U × CLIP(ViT) note (positive vs negative)
- note UNet mid [B,1280,Hm,Wm] note RGB
- note CLIP note image_features（note）
- note：note positive/negative prompts -> note text_features
- note：maximize mean( cos(img, pos) - cos(img, neg) )
- note；note _gen
"""

import os, argparse, glob
from typing import List
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPTokenizer, CLIPTextModel, CLIPImageProcessor, CLIPModel
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    UniPCMultistepScheduler,
    AutoencoderKL,
)

# ----------------- note adapter：mid[1280,H,W] -> noteRGB -----------------
class AdapterSmall(nn.Module):
    def __init__(self, c_in=1280, c_mid=256):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_mid, 3, padding=1)
        self.gn1   = nn.GroupNorm(32, c_mid)
        self.conv2 = nn.Conv2d(c_mid, 64, 3, padding=1)
        self.gn2   = nn.GroupNorm(32, 64)
        self.se1   = nn.Conv2d(64, 16, 1)
        self.se2   = nn.Conv2d(16, 64, 1)
        self.out   = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        x = F.gelu(self.gn1(self.conv1(x)))
        x = F.gelu(self.gn2(self.conv2(x)))
        # note SE note
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.gelu(self.se1(w))
        w = torch.sigmoid(self.se2(w))
        x = x * w
        x = self.out(x)  # (B,3,H,W)
        return x

# ----------------- CLIP note&note（note） -----------------
class CLIPGuidance(nn.Module):
    """
    - note CLIP note image_features（note adapter noteRGB）
    - note：note/note prompts -> note normalize
    """
    def __init__(self, clip_name: str = "openai/clip-vit-base-patch16", device: torch.device | None = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip = CLIPModel.from_pretrained(clip_name).to(self.device).eval()
        self.processor = CLIPImageProcessor.from_pretrained(clip_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_name)
        # note
        size = self.processor.size
        if isinstance(size, dict):
            self.out_h = size.get("height", size.get("shortest_edge", 224))
            self.out_w = size.get("width",  size.get("shortest_edge", 224))
        else:
            self.out_h = self.out_w = int(size)
        # note
        mean = torch.tensor(self.processor.image_mean, dtype=torch.float32).view(1,3,1,1)
        std  = torch.tensor(self.processor.image_std,  dtype=torch.float32).view(1,3,1,1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        # adapter
        self.adapter = AdapterSmall(1280, 256)

    @torch.no_grad()
    def encode_text_mean(self, prompts: List[str]) -> torch.Tensor:
        """note -> note text feature（note normalize）"""
        if len(prompts) == 0:
            raise ValueError("prompts note")
        tokens = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        text_feat = self.clip.get_text_features(**tokens)            # [N, D]
        text_feat = F.normalize(text_feat, dim=-1)                   # note
        return text_feat.mean(dim=0, keepdim=True)                   # [1, D]

    def image_features_from_mid(self, mid: torch.Tensor) -> torch.Tensor:
        """
        mid: [B,1280,Hm,Wm] -> adapter -> noteRGB -> note -> CLIP image_features（note）
        """
        x = self.adapter(mid.float()).to(self.device)                 # (B,3,Hm,Wm)
        # note，note“note/note”
        x = torch.tanh(x * 1.5).add(1).div(2).clamp(0, 1)  # [0,1]
        x = F.interpolate(x, size=(self.out_h, self.out_w), mode="bilinear", align_corners=False)
        x = (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)
        img_feat = self.clip.get_image_features(pixel_values=x.to(self.clip.device))  # [B, D]
        img_feat = F.normalize(img_feat, dim=-1)
        return img_feat


# ----------------- note SD+ControlNet pipeline -----------------
def build_pipeline(controlnet_repo: str, sd_repo: str, use_tiny_vae: bool, token: str | None):
    controlnet = ControlNetModel.from_pretrained(
        controlnet_repo, token=token, torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        sd_repo, controlnet=controlnet, safety_checker=None, low_cpu_mem_usage=True, torch_dtype=torch.float16
    )
    # note KL-VAE note
    if use_tiny_vae:
        from diffusers import AutoencoderTiny
        pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16)
    else:
        pipe.vae = AutoencoderKL.from_pretrained(sd_repo, subfolder="vae", torch_dtype=torch.float16)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # note
    pipe.enable_attention_slicing("max")
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_model_cpu_offload()
    pipe.to(pipe._execution_device)
    return pipe


# ----------------- note（note） -----------------
DEFAULT_POS = [
    "a high-fashion outfit, stylish, on-trend, editorial photo",
    "runway look, premium fabrics, well-coordinated styling",
    "fashion-forward streetwear, crisp details, sharp contrast",
    "lookbook style, high-end, elegant, refined silhouette",
]
DEFAULT_NEG = [
    "outdated outfit, poorly styled, unfashionable",
    "bad coordination, cheap-looking materials, off-trend",
    "low fashionability, unappealing styling, sloppy look",
    "washed out, dull, low contrast fashion photo",
]

# ----------------- note with mid-U × CLIP note -----------------
def _encode_prompt(pipe, prompt: str, negative_prompt: str):
    try:
        pe, _, _ = pipe.encode_prompt(prompt=prompt, device=pipe._execution_device,
                                      num_images_per_prompt=1, do_classifier_free_guidance=True,
                                      negative_prompt=negative_prompt)
    except Exception:
        pe = pipe._encode_prompt(prompt, pipe._execution_device, 1, True, negative_prompt)
    return pe

def sample_with_text_guidance(
    pipe, guider: CLIPGuidance,
    image_pil: Image.Image, control_pil: Image.Image, prompt: str,
    pos_text: List[str], neg_text: List[str],
    generator=None,
    controlnet_conditioning_scale=1.0,
    guidance_scale=8.5,
    negative_prompt="blurry, soft focus, low contrast, washed out",
    num_inference_steps=32,
    early_stop=15,
    cfg_norm=False,
    cfg_decay=True,
    guidance_loss_scale=0.05,  # note 0.02~0.08 note
    loss_every_n_steps=1,
):
    dev = pipe._execution_device

    with torch.no_grad():
        prompt_embeds = _encode_prompt(pipe, prompt, negative_prompt)

    image = pipe.image_processor.preprocess(
        image_pil, height=image_pil.height, width=image_pil.width
    ).to(dtype=torch.float16)
    control = pipe.prepare_control_image(
        image=control_pil, width=control_pil.width, height=control_pil.height,
        batch_size=1, num_images_per_prompt=1, device=dev, dtype=pipe.controlnet.dtype
    )

    # note（note）
    pos_feat = guider.encode_text_mean(pos_text).to(dev)  # [1,D]
    neg_feat = guider.encode_text_mean(neg_text).to(dev)  # [1,D]

    pipe.scheduler.set_timesteps(num_inference_steps, device=dev)
    timesteps, num_inference_steps = pipe.get_timesteps(num_inference_steps, 0.8, dev)
    latent_timestep = timesteps[:1].repeat(1)

    if generator is None:
        generator = torch.Generator(device=dev).manual_seed(0)
    latents = pipe.prepare_latents(image, latent_timestep, 1, 1, prompt_embeds.dtype, dev, generator)

    # note UNet mid features
    cache = {}
    def hook_mid(module, inp, out):
        cache["mid"] = out
    h = pipe.unet.mid_block.register_forward_hook(hook_mid)

    for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
        do_guidance = (i <= early_stop) and (guidance_loss_scale != 0) and (i % loss_every_n_steps == 0)
        sigma = pipe.scheduler.sigmas[i]

        if do_guidance:
            latents = latents.detach().requires_grad_()

        do_cfg = guidance_scale > 1.0
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        latent_model_input = latent_model_input.to(pipe.controlnet.dtype)

        down_res, mid_res = pipe.controlnet(
            latent_model_input, t, encoder_hidden_states=prompt_embeds,
            controlnet_cond=control, conditioning_scale=controlnet_conditioning_scale,
            guess_mode=False, return_dict=False
        )

        if do_guidance:
            noise_pred = pipe.unet(
                latent_model_input, t, encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=down_res,
                mid_block_additional_residual=mid_res,
            ).sample
        else:
            with torch.no_grad():
                noise_pred = pipe.unet(
                    latent_model_input, t, encoder_hidden_states=prompt_embeds,
                    down_block_additional_residuals=down_res,
                    mid_block_additional_residual=mid_res,
                ).sample

        if do_cfg:
            uncond, cond = noise_pred.chunk(2)
            cfg = 1 + guidance_scale * (1 - i / num_inference_steps) if cfg_decay else guidance_scale
            noise_pred = uncond + cfg * (cond - uncond)
            if cfg_norm:
                noise_pred = noise_pred * (torch.linalg.norm(uncond) / (torch.linalg.norm(noise_pred) + 1e-8))

        if do_guidance:
            # note mid
            mid = cache.get("mid", None)  # [2B,C,H,W] note [B,C,H,W]
            assert mid is not None, "mid not captured"
            mid_use = mid.chunk(2)[1] if do_cfg else mid  # [B,1280,Hm,Wm]

            # note image_features
            img_feat = guider.image_features_from_mid(mid_use)  # [B,D], note normalize
            # note
            pos_sim = (img_feat * pos_feat).sum(-1)  # [B]
            neg_sim = (img_feat * neg_feat).sum(-1)  # [B]
            score = (pos_sim - neg_sim).mean()
            loss  = - guidance_loss_scale * score
            grad  = torch.autograd.grad(loss, latents, retain_graph=False, create_graph=False)[0]
            latents = latents - grad * (sigma.to(latents.dtype) ** 2)

        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    h.remove()

    with torch.no_grad():
        latents_out = latents / pipe.vae.config.scaling_factor
        vae_dtype = next(pipe.vae.parameters()).dtype
        image = pipe.vae.decode(latents_out.to(vae_dtype)).sample  # [-1,1]
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    return image


# ----------------- CLI & note -----------------
def parse_args():
    ap = argparse.ArgumentParser()
    # note
    ap.add_argument("--clip_name", type=str, default="openai/clip-vit-base-patch16")
    ap.add_argument("--sd_repo", type=str, default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--controlnet_repo", type=str, default="checkpoints/controlnet_nobg/best")
    ap.add_argument("--use_tiny_vae", action="store_true")
    # note（note | note）
    ap.add_argument("--pos_text", type=str, default="|".join(DEFAULT_POS))
    ap.add_argument("--neg_text", type=str, default="|".join(DEFAULT_NEG))
    # note
    ap.add_argument("--prompt", type=str, default="a high-quality, detailed, editorial fashion photo, sharp focus, high-contrast, texture-rich")
    ap.add_argument("--negative_prompt", type=str, default="blurry, soft focus, low contrast, washed out")
    ap.add_argument("--cfg_scale", type=float, default=8.5)
    ap.add_argument("--guidance_loss", type=float, default=0.05)
    ap.add_argument("--ctrl_scale", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--early_stop", type=int, default=15)
    ap.add_argument("--cfg_norm", action="store_true")
    ap.add_argument("--no_cfg_decay", action="store_true")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)

    sub = ap.add_subparsers(dest="mode", required=True)
    sp1 = sub.add_parser("single")
    sp1.add_argument("--image", required=True)
    sp1.add_argument("--control", required=True)
    sp1.add_argument("--out", default="out_text_guidance.png")

    sp2 = sub.add_parser("folder")
    sp2.add_argument("--image_dir", required=True)
    sp2.add_argument("--control_dir", required=True)
    sp2.add_argument("--out_dir", required=True)

    return ap.parse_args()


def run_single(args, pipe, guider: CLIPGuidance):
    src  = Image.open(args.image).convert("RGB")
    ctrl = Image.open(args.control).convert("RGB")
    old_size = src.size
    src_r  = src.resize((args.resolution, args.resolution))
    ctrl_r = ctrl.resize((args.resolution, args.resolution))

    pos_text = [t.strip() for t in args.pos_text.split("|") if t.strip()]
    neg_text = [t.strip() for t in args.neg_text.split("|") if t.strip()]

    g = torch.Generator(device=pipe._execution_device).manual_seed(args.seed)
    out = sample_with_text_guidance(
        pipe, guider, src_r, ctrl_r, args.prompt,
        pos_text=pos_text, neg_text=neg_text,
        generator=g,
        controlnet_conditioning_scale=args.ctrl_scale,
        guidance_scale=args.cfg_scale,
        guidance_loss_scale=args.guidance_loss,
        num_inference_steps=args.steps,
        early_stop=args.early_stop,
        cfg_norm=args.cfg_norm,
        cfg_decay=not args.no_cfg_decay,
        negative_prompt=args.negative_prompt,
    )
    out = out.resize(old_size)
    out.save(args.out)
    print(f"✅ saved to {args.out}")


def run_folder(args, pipe, guider: CLIPGuidance):
    os.makedirs(args.out_dir, exist_ok=True)
    img_paths = sorted(glob.glob(os.path.join(args.image_dir, "*")))
    name2ctrl = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in glob.glob(os.path.join(args.control_dir, "*.png"))
    }

    pos_text = [t.strip() for t in args.pos_text.split("|") if t.strip()]
    neg_text = [t.strip() for t in args.neg_text.split("|") if t.strip()]

    for ip in tqdm(img_paths):
        basename = os.path.basename(ip)
        stem, ext = os.path.splitext(basename)
        if stem not in name2ctrl:
            print(f"[WARN] control not found for {stem}")
            continue
        cp = name2ctrl[stem]
        src  = Image.open(ip).convert("RGB")
        ctrl = Image.open(cp).convert("RGB")
        old_size = src.size
        src_r  = src.resize((args.resolution, args.resolution))
        ctrl_r = ctrl.resize((args.resolution, args.resolution))

        g = torch.Generator(device=pipe._execution_device).manual_seed(args.seed)
        out = sample_with_text_guidance(
            pipe, guider, src_r, ctrl_r, args.prompt,
            pos_text=pos_text, neg_text=neg_text,
            generator=g,
            controlnet_conditioning_scale=args.ctrl_scale,
            guidance_scale=args.cfg_scale,
            guidance_loss_scale=args.guidance_loss,
            num_inference_steps=args.steps,
            early_stop=args.early_stop,
            cfg_norm=args.cfg_norm,
            cfg_decay=not args.no_cfg_decay,
            negative_prompt=args.negative_prompt,
        )
        out = out.resize(old_size)
        out.save(os.path.join(args.out_dir, f"{stem}_gen{ext}"))

    print(f"✅ batch saved to {args.out_dir}")


if __name__ == "__main__":
    # note/note
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HF_TOKEN = os.environ.get("HF_TOKEN", None)

    args = parse_args()
    pipe = build_pipeline(
        controlnet_repo=args.controlnet_repo,
        sd_repo=args.sd_repo,
        use_tiny_vae=args.use_tiny_vae,
        token=HF_TOKEN,
    )
    guider = CLIPGuidance(clip_name=args.clip_name, device=device).eval()
    guider = guider.to(device)

    if args.mode == "single":
        run_single(args, pipe, guider)
    else:
        run_folder(args, pipe, guider)
