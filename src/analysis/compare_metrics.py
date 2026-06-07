#!/usr/bin/env python
import os, csv, numpy as np
from PIL import Image
from glob import glob

def img_to_np(img):
    return np.asarray(img).astype(np.float32) / 255.0

def metrics(image: Image.Image):
    x = img_to_np(image.convert("RGB"))
    r,g,b = x[...,0], x[...,1], x[...,2]
    # note (Rec.601)
    y = 0.299*r + 0.587*g + 0.114*b
    brightness = float(y.mean())
    contrast   = float(y.std())

    # note (note HSV S)
    maxc = x.max(axis=2)
    minc = x.min(axis=2)
    sat  = np.where(maxc==0, 0.0, (maxc - minc) / (maxc + 1e-8)).mean().item()

    # note（RGB note）
    gray = y
    color_cast = (np.abs(r-gray) + np.abs(g-gray) + np.abs(b-gray)).mean().item()/3.0
    return brightness, contrast, sat, color_cast

# note compare_metrics.py note main() note
def main(orig_dir, gen_dir, out_csv):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    orig_files = sorted(glob(os.path.join(orig_dir, "*")))
    names = [os.path.basename(p) for p in orig_files]
    # note：note“note + _gen”
    gen_map = {}
    for p in glob(os.path.join(gen_dir, "*")):
        stem = os.path.basename(p)
        # note "_gen" note
        base = stem.replace("_gen", "")
        gen_map[base] = p

    rows = []
    improved = 0
    for name in names:
        if name not in gen_map:
            continue
        o = Image.open(os.path.join(orig_dir, name)).convert("RGB")
        g = Image.open(gen_map[name]).convert("RGB")
        ob, oc, osat, occ = metrics(o)
        gb, gc, gsat, gcc = metrics(g)
        rows.append([name, ob, oc, osat, occ, gb, gc, gsat, gcc,
                     gb-ob, gc-oc, gsat-osat, gcc-occ])

    if not rows:
        print("No matching files.")
        return

    # === note ===
    mean_delta = np.mean([r[-4] for r in rows])
    mean_contr = np.mean([r[-3] for r in rows])
    mean_sat   = np.mean([r[-2] for r in rows])
    mean_cast  = np.mean([r[-1] for r in rows])

    improved = sum(1 for r in rows if (r[-3] > 0 and r[-2] > 0))
    print("=== Summary ===")
    print(f"Samples: {len(rows)}")
    print(f"Mean Δ Brightness: {mean_delta:+.4f}")
    print(f"Mean Δ Contrast  : {mean_contr:+.4f}")
    print(f"Mean Δ Saturation: {mean_sat:+.4f}")
    print(f"Mean Δ ColorCast : {mean_cast:+.4f}")
    print(f"Improved (contrast&sat up): {improved}/{len(rows)} ({improved/len(rows)*100:.1f}%)")

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename",
                    "orig_brightness","orig_contrast","orig_saturation","orig_colorcast",
                    "gen_brightness","gen_contrast","gen_saturation","gen_colorcast",
                    "delta_brightness","delta_contrast","delta_saturation","delta_colorcast"])
        w.writerows(rows)
    print(f"CSV saved to: {out_csv}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_dir", required=True)
    ap.add_argument("--gen_dir", required=True)
    ap.add_argument("--out_csv", default="metrics_compare.csv")
    args = ap.parse_args()
    main(args.orig_dir, args.gen_dir, args.out_csv)
