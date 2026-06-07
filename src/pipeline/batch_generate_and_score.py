#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, csv, argparse
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, SiglipModel

# =============== note SigLIP note ===============
class SiglipFashionRegressor(nn.Module):
    def __init__(self, name="google/siglip2-base-patch16-224"):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(name, use_fast=True)
        self.backbone  = SiglipModel.from_pretrained(name)
        h = self.backbone.config.vision_config.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(h, h//2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(h//2, 1)
        )
    def forward(self, pixel_values):  # [B,3,H,W]
        v = self.backbone.vision_model(pixel_values).last_hidden_state.mean(1)
        return self.mlp(v).squeeze(-1)

@torch.no_grad()
def score_images(reg, proc, pil_list, device):
    """note PIL note，note numpy note"""
    px = proc(images=pil_list, return_tensors="pt")["pixel_values"].to(device)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        pred = reg(px)
    return pred.float().cpu().numpy()

# =============== note sample(...) ===============
def import_existing_sample():
    """
    note sample(...) notebook/session note，
    note gen_with_midu_siglip.py note import。
    """
    try:
        from gen_with_midu_siglip import sample_with_midu as existing_sample
        return existing_sample
    except Exception:
        return None

# =============== note（note） ===============
def make_fallback_pipe(device, hf_token=None):
    from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, UniPCMultistepScheduler, AutoencoderTiny
    controlnet = ControlNetModel.from_pretrained(
        "Jason773/model_out",
        token=hf_token,
        revision="366312bcbc8c12efcfca5b7a3b3e275ce73ee875",
        torch_dtype=torch.float16,
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        safety_checker=None,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    # note AutoencoderKL（note，note）
    from diffusers import AutoencoderTiny
    pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    # note
    pipe.enable_attention_slicing("max")
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.enable_model_cpu_offload()
    return pipe

@torch.no_grad()
def simple_generate(pipe, src_rgb: Image.Image, ctrl_rgb: Image.Image, prompt: str,
                    steps=25, guidance_scale=9.0, control_scale=0.9, seed=0, size=(512,512)):
    # note resize（note）
    s = src_rgb.resize(size)
    c = ctrl_rgb.resize(size)
    g = torch.Generator(device=pipe.device).manual_seed(seed)
    out = pipe(
        prompt=prompt,
        image=s,
        control_image=c,
        negative_prompt="zoomed in, blurry, oversaturated, warped",
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=control_scale,
        generator=g,
        strength=0.8
    ).images[0]
    return out

# =============== note ===============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="")  # note
    ap.add_argument("--img_dirname", default="image")           # note
    ap.add_argument("--ctrl_dirname", default="labelcolor")     # Control note
    ap.add_argument("--out_dir", default="")  # note
    ap.add_argument("--prompt", default="a high-quality, detailed, and professional image, High fashionability")
    ap.add_argument("--seed", type=int, default=45)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance_scale", type=float, default=9.0)
    ap.add_argument("--control_scale", type=float, default=0.9)
    ap.add_argument("--size", type=int, nargs=2, default=[512,512])
    ap.add_argument("--hf_token", default=None)
    ap.add_argument("--reg_ckpt", default="ckpts/siglip_reg/regressor_best.pt")  # note
    ap.add_argument("--model_name", default="google/siglip2-base-patch16-224")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) note
    reg = SiglipFashionRegressor(args.model_name).to(device)
    state = torch.load(args.reg_ckpt, map_location=device)
    reg.load_state_dict(state, strict=True)
    reg.eval()
    proc = reg.processor

    # 2) note：note sample(...)；note fallback
    sample_fn = import_existing_sample()
    if sample_fn is None:
        print("⚠️ note sample(...)，note（note）。")
        pipe = make_fallback_pipe(device, args.hf_token)

    # 3) note 50 note（note image note）
    img_dir  = os.path.join(args.root, args.img_dirname)
    ctrl_dir = os.path.join(args.root, args.ctrl_dirname)
    names = [n for n in os.listdir(img_dir) if n.lower().endswith((".png",".jpg",".jpeg"))]
    names.sort()

    # 4) note：note + note + note
    rows = []
    improved = 0
    pbar = tqdm(names, ncols=100)
    for name in pbar:
        stem = os.path.splitext(name)[0]
        # note control
        ctrl_path = None
        for ext in (".png",".jpg",".jpeg"):
            p = os.path.join(ctrl_dir, stem + ext)
            if os.path.isfile(p): ctrl_path = p; break
        if ctrl_path is None:
            print(f"❌ control not found for {name}, skip.")
            continue

        src_path = os.path.join(img_dir, name)
        src = Image.open(src_path).convert("RGB")
        ctrl = Image.open(ctrl_path).convert("RGB")

        # note
        if sample_fn is not None:
            gen = sample_fn(
                image=src.resize(tuple(args.size)),
                control_image=ctrl.resize(tuple(args.size)),
                prompt=args.prompt,
                generator=torch.Generator(device=device).manual_seed(args.seed),
                controlnet_conditioning_scale=args.control_scale,
                guidance_scale=args.guidance_scale,
                guidance_loss_scale=0.0,  # note Mid-U note，note 0.05~0.2
                num_inference_steps=args.steps,
                early_stop=15,
                cfg_norm=True,
                cfg_decay=True,
                loss_every_n_steps=1
            )
        else:
            gen = simple_generate(pipe, src, ctrl, args.prompt, args.steps,
                                  args.guidance_scale, args.control_scale, args.seed, tuple(args.size))

        # note（note）
        gen_resized = gen.resize(src.size)
        out_path = os.path.join(args.out_dir, f"{stem}_gen.png")
        gen_resized.save(out_path)

        # note（note vs note）
        s_orig, s_gen = score_images(reg, proc, [src, gen_resized], device)
        delta = float(s_gen - s_orig)
        improved += int(delta > 0)

        rows.append([name, src_path, out_path, s_orig, s_gen, delta])
        pbar.set_postfix(delta=f"{delta:+.4f}")

    # 5) note CSV & note
    csv_path = os.path.join(args.out_dir, "scores_compare.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "src_path", "gen_path", "score_orig", "score_gen", "delta"])
        for r in rows: w.writerow(r)

    deltas = np.array([r[-1] for r in rows], dtype=np.float32)
    print("\n=== Summary ===")
    print(f"Samples: {len(rows)}")
    if len(rows) > 0:
        print(f"Mean Δ (gen - orig): {deltas.mean():.4f}")
        print(f"Median Δ: {np.median(deltas):.4f}")
        print(f"Improved count: {improved}/{len(rows)} ({improved/len(rows)*100:.1f}%)")
        print(f"CSV saved to: {csv_path}")
        print(f"Images saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
