#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, csv, argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch, torch.nn as nn
from transformers import AutoImageProcessor, AutoProcessor, SiglipModel

# note
class SiglipFashionRegressor(nn.Module):
    def __init__(self, name="google/siglip2-base-patch16-224"):
        super().__init__()

        # 🔧 note transformers：note AutoImageProcessor，
        # note，note Siglip note AutoProcessor
        try:
            self.processor = AutoImageProcessor.from_pretrained(name)
        except Exception:
            # note AutoProcessor
            proc = AutoProcessor.from_pretrained(name)
            # note AutoProcessor note image_processor，note
            if hasattr(proc, "image_processor"):
                self.processor = proc.image_processor
            else:
                self.processor = proc

        self.backbone = SiglipModel.from_pretrained(name)
        h = self.backbone.config.vision_config.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(h // 2, 1),
        )

    def forward(self, pixel_values):
        v = self.backbone.vision_model(pixel_values).last_hidden_state.mean(1)
        return self.mlp(v).squeeze(-1)


@torch.no_grad()
def score_batch(model, processor, pil_list, device):
    px = processor(images=pil_list, return_tensors="pt")["pixel_values"].to(device)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        pred = model(px)
    return pred.float().cpu().numpy()

def apply_bg_mask(img_pil, seg_pil, bg_color=(255, 255, 255), thr=5):
    """
    note seg note mask，note bg_color。
    note seg note 0 note/note，0 note。
    """
    img = img_pil.convert("RGB")
    seg = seg_pil.convert("L")

    img_np = np.array(img).astype(np.float32) / 255.0   # [H,W,3]
    seg_np = np.array(seg).astype(np.float32)           # [H,W]

    # note 0 note
    mask = (seg_np > thr).astype(np.float32)[..., None]  # [H,W,1]
    bg   = np.array(bg_color, dtype=np.float32)[None, None, :] / 255.0

    out = img_np * mask + bg * (1.0 - mask)
    out = (out * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_dir", required=True)
    ap.add_argument("--gen_dir",  required=True)
    ap.add_argument("--ckpt",     default="ckpts/siglip_reg/regressor_best.pt")
    ap.add_argument("--model",    default="google/siglip2-base-patch16-224")
    ap.add_argument("--out_csv",  default="scores_compare.csv")

    # seg + note
    ap.add_argument("--seg_dir",  default=None,
                    help="seg / labelcolor note（note，note）")
    ap.add_argument("--bg_color", type=int, nargs=3, default=[255, 255, 255])
    ap.add_argument("--mask_thr", type=int, default=5)

    # ⭐ note：note
    ap.add_argument("--save_masked_dir", default=None,
                    help="note，note/note")
    ap.add_argument("--save_limit", type=int, default=0,
                    help="note；0 note")

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reg = SiglipFashionRegressor(args.model).to(device)
    reg.load_state_dict(torch.load(args.ckpt, map_location=device), strict=True)
    reg.eval()
    proc = reg.processor

    use_mask = args.seg_dir is not None
    save_masked = args.save_masked_dir is not None
    if save_masked:
        os.makedirs(args.save_masked_dir, exist_ok=True)

    names = [n for n in os.listdir(args.orig_dir)
             if n.lower().endswith((".png",".jpg",".jpeg"))]
    names.sort()

    rows, improved = [], 0
    saved_count = 0

    for name in tqdm(names, ncols=100, desc="Scoring"):
        stem, ext = os.path.splitext(name)

        # note
        gen_path = None
        for e in (".png",".jpg",".jpeg"):
            cand = os.path.join(args.gen_dir, f"{stem}_gen{e}") 
            if os.path.isfile(cand):
                gen_path = cand; break
        if gen_path is None:
            for e in (".png",".jpg",".jpeg"):
                cand = os.path.join(args.gen_dir, f"{stem}_comp{e}")
                if os.path.isfile(cand):
                    gen_path = cand; break
        if gen_path is None:
            for e in (".png",".jpg",".jpeg"):
                cand = os.path.join(args.gen_dir, f"{stem}_best{e}")
                if os.path.isfile(cand):
                    gen_path = cand; break
        if gen_path is None:
            for e in (".png",".jpg",".jpeg"):
                cand = os.path.join(args.gen_dir, f"{stem}{e}")
                if os.path.isfile(cand):
                    gen_path = cand; break
        if gen_path is None:
            continue

        src_path = os.path.join(args.orig_dir, name)
        src = Image.open(src_path).convert("RGB")
        gen = Image.open(gen_path).convert("RGB")

        if use_mask:
            seg_path = None
            for e in (".png",".jpg",".jpeg"):
                cand = os.path.join(args.seg_dir, stem + e)
                if os.path.isfile(cand):
                    seg_path = cand; break
            if seg_path is None:
                continue

            seg = Image.open(seg_path)
            src = apply_bg_mask(src, seg,
                                bg_color=tuple(args.bg_color),
                                thr=args.mask_thr)
            gen = apply_bg_mask(gen, seg,
                                bg_color=tuple(args.bg_color),
                                thr=args.mask_thr)

            # ⭐ note（note）
            if save_masked and (args.save_limit == 0 or saved_count < args.save_limit):
                src_out = os.path.join(args.save_masked_dir, f"{stem}_src_masked{ext}")
                gen_out = os.path.join(args.save_masked_dir, f"{stem}_gen_masked{ext}")
                src.save(src_out)
                gen.save(gen_out)
                saved_count += 1

        s_orig, s_gen = score_batch(reg, proc, [src, gen], device)
        delta = float(s_gen - s_orig)
        improved += int(delta > 0)
        rows.append([name, src_path, gen_path, s_orig, s_gen, delta])

    # note CSV & note
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "src_path", "gen_path", "score_orig", "score_gen", "delta"])
        for r in rows:
            w.writerow(r)

    if len(rows) == 0:
        print("No valid pairs found.")
        return

    deltas = np.array([r[-1] for r in rows], dtype=np.float32)
    print("\n=== Summary ===")
    print(f"Samples: {len(rows)}")
    print(f"Mean Δ (gen - orig): {deltas.mean():.4f}")
    print(f"Median Δ: {np.median(deltas):.4f}")
    print(f"Improved count: {improved}/{len(rows)} ({improved/len(rows)*100:.1f}%)")
    print(f"CSV saved to: {args.out_csv}")

if __name__ == "__main__":
    main()
