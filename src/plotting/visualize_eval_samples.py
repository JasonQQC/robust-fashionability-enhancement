#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import List

import pandas as pd
from PIL import Image, ImageDraw

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def find_original(input_dir: Path, image_id: str) -> Path:
    for ext in IMAGE_EXTS:
        cand = input_dir / f"{image_id}{ext}"
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"Original image not found for image_id={image_id} in {input_dir}")


def resize_fit(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    s = min(size / max(w, 1), size / max(h, 1))
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    x = (size - nw) // 2
    y = (size - nh) // 2
    canvas = Image.new("RGB", (size, size), (250, 250, 250))
    if hasattr(Image, "Resampling"):
        resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    else:
        resized = img.resize((nw, nh), Image.LANCZOS)
    canvas.paste(resized, (x, y))
    return canvas


def make_pair(original_path: Path, edited_path: Path, tile_size: int, label: str) -> Image.Image:
    orig = resize_fit(Image.open(original_path).convert("RGB"), tile_size)
    edit = resize_fit(Image.open(edited_path).convert("RGB"), tile_size)
    out = Image.new("RGB", (tile_size * 2, tile_size + 34), (255, 255, 255))
    out.paste(orig, (0, 0))
    out.paste(edit, (tile_size, 0))
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, tile_size, tile_size * 2, tile_size + 34), fill=(245, 245, 245))
    draw.text((8, tile_size + 8), label, fill=(20, 20, 20))
    draw.text((8, 8), "original", fill=(30, 30, 30))
    draw.text((tile_size + 8, 8), "edited", fill=(30, 30, 30))
    return out


def chunked(items: List[Image.Image], n: int) -> List[List[Image.Image]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build side-by-side visualization from evaluation results")
    ap.add_argument("--scores_csv", default="results/scores.csv")
    ap.add_argument("--input_dir", default="eval50/images")
    ap.add_argument("--outputs_dir", default="outputs")
    ap.add_argument("--output_path", default="results/vis/side_by_side_grid.png")
    ap.add_argument("--method", default="nano_banana2")
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--tile_size", type=int, default=320)
    ap.add_argument("--cols", type=int, default=2)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    scores_csv = Path(args.scores_csv)
    input_dir = Path(args.input_dir)
    outputs_dir = Path(args.outputs_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not scores_csv.is_file():
        raise SystemExit(f"scores_csv not found: {scores_csv}")

    df = pd.read_csv(scores_csv)
    df = df[df["method"] == args.method].copy()
    if df.empty:
        raise SystemExit(f"No rows for method={args.method} in {scores_csv}")

    # Pick stronger positive deltas first for quick qualitative check.
    df.sort_values("delta", ascending=False, inplace=True)
    if args.num_samples > 0:
        df = df.head(args.num_samples)

    pairs: List[Image.Image] = []
    for _, row in df.iterrows():
        image_id = str(row["image_id"])
        orig = find_original(input_dir, image_id)
        edit = outputs_dir / args.method / f"{image_id}.png"
        if not edit.is_file():
            continue
        label = f"{image_id} | delta={float(row['delta']):+.4f}"
        pairs.append(make_pair(orig, edit, args.tile_size, label))

    if not pairs:
        raise SystemExit("No valid sample pairs found to visualize")

    cols = max(1, args.cols)
    rows = (len(pairs) + cols - 1) // cols
    cell_w, cell_h = pairs[0].size
    canvas = Image.new("RGB", (cell_w * cols, cell_h * rows), (255, 255, 255))

    for i, p in enumerate(pairs):
        r = i // cols
        c = i % cols
        canvas.paste(p, (c * cell_w, r * cell_h))

    canvas.save(output_path)
    print(f"Saved visualization to: {output_path}")


if __name__ == "__main__":
    main()
