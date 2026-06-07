#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate text-guidance sweep results using SigLIP regression model.

note：
sweep_root/
    cfg7p00_g0p010/
        123_gen.png
        456_gen.png
        ...
    cfg7p00_g0p020/
    cfg8p50_g0p020/
    ...
    summary.csv   <- note sweep，note

note：
sweep_root/scores_siglip_reg.csv
note summary & per-condition note
"""

import os
import csv
import argparse
import re
from collections import defaultdict

import numpy as np
import torch
from tqdm.auto import tqdm
from PIL import Image

from transformers import AutoImageProcessor, SiglipModel


# --------------------------------------------------------
# note（note）
# --------------------------------------------------------
class SiglipRegressor(torch.nn.Module):
    def __init__(self, model_name="google/siglip2-base-patch16-224"):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.backbone = SiglipModel.from_pretrained(model_name)
        hidden = self.backbone.config.vision_config.hidden_size
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden // 2, 1),
        )

    def forward(self, pixel_values):
        vision_out = self.backbone.vision_model(pixel_values=pixel_values)
        feats = vision_out.last_hidden_state.mean(1)
        return self.mlp(feats).squeeze(-1)


# --------------------------------------------------------
# note
# --------------------------------------------------------
def load_image(path):
    return Image.open(path).convert("RGB")


def score_images(model, processor, img_paths, device):
    """note，note python list"""
    imgs = [load_image(p) for p in img_paths]
    inputs = processor(images=imgs, return_tensors="pt")
    px = inputs["pixel_values"].to(device)
    with torch.no_grad():
        scores = model(px).cpu().tolist()
    return scores


def detect_sweep_folders(sweep_root):
    """note cfg_xxx_g_xxx note"""
    subdirs = []
    for name in os.listdir(sweep_root):
        p = os.path.join(sweep_root, name)
        if os.path.isdir(p) and ("cfg" in name and "g" in name):
            subdirs.append(p)
    return sorted(subdirs)


# --------------------------------------------------------
# note
# --------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--sweep_root", required=True,
                    help="note cfg_xxx_g_xxx/ note")

    ap.add_argument("--orig_dir", required=True,
                    help="note（note test50fromFashion144/image）")

    ap.add_argument("--reg_ckpt", required=True,
                    help="SigLIP regression best.pt")

    ap.add_argument("--siglip_name", default="google/siglip2-base-patch16-224",
                    help="description")

    ap.add_argument("--out_csv", default="scores_siglip_reg.csv")

    ap.add_argument("--topk", type=int, default=0,
                    help="note >0，note top-k note（note）")

    return ap.parse_args()


def main():
    args = parse_args()
    sweep_root = args.sweep_root
    orig_dir = args.orig_dir

    out_csv_path = os.path.join(sweep_root, args.out_csv)

    # --------------------------
    # 1) note
    # --------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(args.siglip_name)

    model = SiglipRegressor(args.siglip_name).to(device)
    state = torch.load(args.reg_ckpt, map_location=device)
    load_res = model.load_state_dict(state, strict=False)
    if load_res.missing_keys or load_res.unexpected_keys:
        raise RuntimeError(
            f"state_dict mismatch: missing={load_res.missing_keys}, "
            f"unexpected={load_res.unexpected_keys}"
        )
    model.eval()

    print(f"✅ Loaded SigLIP regressor from {args.reg_ckpt}")

    # --------------------------
    # 2) note sweep note
    # --------------------------
    subdirs = detect_sweep_folders(sweep_root)
    print(f"Found {len(subdirs)} sweep subfolders.")

    all_results = []   # note DataFrame

    # --------------------------
    # 3) note cfg/g note
    # --------------------------
    for sub in subdirs:
        name = os.path.basename(sub)

        # note cfg/g
        # note：cfg7p00_g0p020
        print(name)
        m = re.search(r"cfg_([0-9.]+)_g_([0-9.]+)", name)
        print(m)
        print(m.group(1))
        print(m.group(2))
        if not m:
            return None, None
        try:
            cfg = float(m.group(1))
            g   = float(m.group(2))
        except ValueError:
            return None, None

        print(f"\n=== Evaluating {name}  (cfg={cfg}, g={g}) ===")

        gen_files = sorted([
            f for f in os.listdir(sub)
            if f.lower().endswith((".png", ".jpg"))
        ])

        # note
        orig_paths = []
        gen_paths = []
        names = []

        for gf in gen_files:
            stem = os.path.splitext(gf)[0].replace("_gen", "")
            orig_path = os.path.join(orig_dir, stem + ".png")
            if not os.path.exists(orig_path):
                orig_path = os.path.join(orig_dir, stem + ".jpg")
            if not os.path.exists(orig_path):
                print(f"[Skip] No original image found for {gf}")
                continue

            orig_paths.append(orig_path)
            gen_paths.append(os.path.join(sub, gf))
            names.append(stem)

        if len(orig_paths) == 0:
            print("[Warn] No matched images, skip this folder.")
            continue

        # ---- note ----
        print("Scoring originals...")
        orig_scores = score_images(model, processor, orig_paths, device)
        print("Scoring generated...")
        gen_scores = score_images(model, processor, gen_paths, device)

        # ---- note ----
        for nm, so, sg, op, gp in zip(
            names, orig_scores, gen_scores, orig_paths, gen_paths
        ):
            all_results.append({
                "name": nm,
                "cfg": cfg,
                "g": g,
                "orig_score": so,
                "gen_score": sg,
                "delta": sg - so,
                "orig_path": op,
                "gen_path": gp,
            })

    # --------------------------
    # 4) note CSV
    # --------------------------
    import pandas as pd
    df = pd.DataFrame(all_results)
    df.to_csv(out_csv_path, index=False)
    print(f"\n💾 Saved all results to {out_csv_path}")

    # --------------------------
    # 5) note summary
    # --------------------------
    print("\n=== Global Summary ===")
    print(f"Samples: {len(df)}")
    print(f"Mean Δ: {df['delta'].mean():.4f}")
    print(f"Median Δ: {df['delta'].median():.4f}")
    improved = (df["delta"] > 0).sum()
    print(f"Improved: {improved}/{len(df)} ({improved/len(df)*100:.1f}%)")

    # --------------------------
    # 6) per (cfg,g)
    # --------------------------
    print("\n=== Per-condition Summary ===")
    for (cfg, g), group in df.groupby(["cfg", "g"]):
        mean = group["delta"].mean()
        med = group["delta"].median()
        imp = (group["delta"] > 0).sum()
        n = len(group)
        print(
            f"cfg={cfg:.2f}, g={g:.3f} | n={n}, "
            f"meanΔ={mean:.4f}, medΔ={med:.4f}, improved={imp}/{n} ({imp/n*100:.1f}%)"
        )

    # --------------------------
    # 7) note：note top-k
    # --------------------------
    if args.topk > 0:
        topk_dir = os.path.join(args.sweep_root, "topk_examples")
        os.makedirs(topk_dir, exist_ok=True)
        top = df.sort_values("delta", ascending=False).head(args.topk)

        print(f"\nSaving top-{args.topk} improved examples to {topk_dir} ...")

        for _, r in top.iterrows():
            # note
            import shutil
            shutil.copy(r["orig_path"], os.path.join(topk_dir, f"{r['name']}_orig.png"))
            shutil.copy(r["gen_path"],  os.path.join(topk_dir, f"{r['name']}_gen.png"))

        print(f"Top-{args.topk} saved.")

    print("\n✨ Done.")


if __name__ == "__main__":
    main()
