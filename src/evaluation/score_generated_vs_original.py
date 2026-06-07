#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, csv, argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch, torch.nn as nn
from transformers import AutoImageProcessor, SiglipModel ,SiglipImageProcessor

# note
class SiglipFashionRegressor(nn.Module):
    def __init__(self, name="google/siglip2-base-patch16-224"):
        super().__init__()
        self.processor = SiglipImageProcessor.from_pretrained(name)
        self.backbone  = SiglipModel.from_pretrained(name)
        h = self.backbone.config.vision_config.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(h, h//2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(h//2, 1)
        )
    def forward(self, pixel_values):
        v = self.backbone.vision_model(pixel_values).last_hidden_state.mean(1)
        return self.mlp(v).squeeze(-1)

@torch.no_grad()
def score_batch(model, processor, pil_list, device):
    px = processor(images=pil_list, return_tensors="pt")["pixel_values"].to(device)
    if device.type == "cuda":
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pred = model(px)
    else:
        pred = model(px)
    return pred.float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_dir", required=True)      # note（test50fromFashion144k/image）
    ap.add_argument("--gen_dir",  required=True)      # note（note out_dir）
    ap.add_argument("--ckpt",     default="ckpts/siglip_reg/regressor_best.pt")
    ap.add_argument("--model",    default="google/siglip2-base-patch16-224")
    ap.add_argument("--out_csv",  default="scores_compare.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # note
    reg = SiglipFashionRegressor(args.model).to(device)
    reg.load_state_dict(torch.load(args.ckpt, map_location=device), strict=True)
    reg.eval()
    proc = reg.processor

    # note：note，note *_gen.png
    names = [n for n in os.listdir(args.orig_dir) if n.lower().endswith((".png",".jpg",".jpeg"))]
    names.sort()

    rows, improved = [], 0
    for name in tqdm(names, ncols=100, desc="Scoring"):
        stem = os.path.splitext(name)[0]
        # note
        gen_path = None
        for ext in (".png",".jpg",".jpeg"):
            cand = os.path.join(args.gen_dir, f"{stem}_gen{ext}")
            if os.path.isfile(cand):
                gen_path = cand; break
        if gen_path is None:
            # note _gen note
            for ext in (".png",".jpg",".jpeg"):
                cand = os.path.join(args.gen_dir, f"{stem}{ext}")
                if os.path.isfile(cand):
                    gen_path = cand; break
        if gen_path is None:
            continue

        src = Image.open(os.path.join(args.orig_dir, name)).convert("RGB")
        gen = Image.open(gen_path).convert("RGB")

        s_orig, s_gen = score_batch(reg, proc, [src, gen], device)
        delta = float(s_gen - s_orig)
        improved += int(delta > 0)
        rows.append([name, os.path.join(args.orig_dir, name), gen_path, s_orig, s_gen, delta])

    # note CSV & note
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "src_path", "gen_path", "score_orig", "score_gen", "delta"])
        for r in rows: w.writerow(r)
        
    # note top-K note
    K = 20
    rows_sorted = sorted(rows, key=lambda r: r[-1], reverse=True)
    with open(os.path.splitext(args.out_csv)[0] + "_topK.txt", "w") as f:
        for r in rows_sorted[:K]:
            name, src_path, gen_path, s_orig, s_gen, delta = r
            f.write(f"{name}\t{src_path}\t{gen_path}\t{s_orig:.4f}\t{s_gen:.4f}\t{delta:.4f}\n")


    deltas = np.array([r[-1] for r in rows], dtype=np.float32)
    print("\n=== Summary ===")
    print(f"Samples: {len(rows)}")
    if len(rows) > 0:
        print(f"Mean Δ (gen - orig): {deltas.mean():.4f}")
        print(f"Median Δ: {np.median(deltas):.4f}")
        print(f"Improved count: {improved}/{len(rows)} ({improved/len(rows)*100:.1f}%)")
        print(f"CSV saved to: {args.out_csv}")

if __name__ == "__main__":
    main()
