#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize GPT strength-sweep eval50 results.

Per-image: 4-panel card  [Original | Subtle | Moderate | Aggressive]
Contact sheets: all, top-delta, bottom-delta

CLI
---
python visualize_gpt_strength_eval50.py
python visualize_gpt_strength_eval50.py --top-k 10
python visualize_gpt_strength_eval50.py --sort delta_desc
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# ─── Pillow compat helpers ────────────────────────────────────────────────────
def _lanczos():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS  # type: ignore


def fit_with_pad(
    img: Image.Image, target_w: int, target_h: int, bg: Tuple[int, int, int] = (240, 240, 240)
) -> Image.Image:
    """Resize img to fit in (target_w × target_h) preserving aspect ratio, then pad."""
    iw, ih = img.size
    scale = min(target_w / iw, target_h / ih)
    new_w, new_h = int(iw * scale), int(ih * scale)
    resized = img.resize((new_w, new_h), _lanczos())
    canvas = Image.new("RGB", (target_w, target_h), color=bg)
    ox = (target_w - new_w) // 2
    oy = (target_h - new_h) // 2
    canvas.paste(resized, (ox, oy))
    return canvas


def _font(size: int) -> ImageFont.ImageFont:
    """Return a truetype font or fallback bitmap font."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", size)
        except Exception:
            return ImageFont.load_default()


def _text_block_height(n_lines: int, font_size: int) -> int:
    return n_lines * (font_size + 4) + 10


def draw_text_block(draw: ImageDraw.Draw, x: int, y: int, lines: List[str],
                    font_size: int = 14, color: str = "#222222") -> int:
    """Draws lines of text, returns total height."""
    font = _font(font_size)
    dh = font_size + 4
    for line in lines:
        draw.text((x, y), line, fill=color, font=font)
        y += dh
    return y


# ─── card builders ────────────────────────────────────────────────────────────

TEXT_H = 60      # height reserved below each panel for score annotations
LABEL_H = 22     # header row per-panel showing strength label

STRENGTHS = ["subtle", "moderate", "aggressive"]
STRENGTH_COLORS = {"subtle": "#1a75c8", "moderate": "#d97706", "aggressive": "#b91c1c"}


def make_comparison_card(
    orig_img: Image.Image,
    tiles: Dict[str, Optional[Image.Image]],
    scores_row: Dict[str, float],   # orig_score, {strength}_score, {strength}_delta
    image_id: str,
    tile_w: int,
    tile_h: int,
) -> Image.Image:
    """4-panel card: Original + 3 strength edits, each with label + score text."""
    cols = 1 + len(STRENGTHS)
    card_w = cols * tile_w
    card_h = LABEL_H + tile_h + TEXT_H
    card = Image.new("RGB", (card_w, card_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(card)

    def draw_panel(idx: int, img: Image.Image, label: str,
                   label_color: str, text_lines: List[str]) -> None:
        x = idx * tile_w
        # Label bar
        draw.rectangle([x, 0, x + tile_w - 1, LABEL_H - 1], fill=label_color)
        font_lbl = _font(13)
        draw.text((x + 6, 3), label, fill="white", font=font_lbl)
        # Image tile
        fit = fit_with_pad(img, tile_w, tile_h)
        card.paste(fit, (x, LABEL_H))
        # Text block
        ty = LABEL_H + tile_h + 4
        draw_text_block(draw, x + 6, ty, text_lines, font_size=12)

    orig_score = scores_row.get("original_score", float("nan"))
    draw_panel(0, orig_img, f"Original  ({image_id})", "#333333", [
        f"score: {orig_score:.4f}",
    ])

    for i, s in enumerate(STRENGTHS):
        img_s = tiles.get(s)
        if img_s is None:
            img_s = Image.new("RGB", (tile_w, tile_h), (200, 200, 200))
        e_score = scores_row.get(f"{s}_score", float("nan"))
        delta = scores_row.get(f"{s}_delta", float("nan"))
        delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "N/A"
        clr = STRENGTH_COLORS.get(s, "#555555")
        draw_panel(
            i + 1, img_s, s.capitalize(), clr,
            [f"score: {e_score:.4f}" if not np.isnan(e_score) else "N/A",
             f"Δ={delta_str}"],
        )

    return card


# ─── contact sheet helpers ────────────────────────────────────────────────────

def make_sheet_cell(
    orig_img: Image.Image,
    tiles: Dict[str, Optional[Image.Image]],
    image_id: str,
    scores_row: Dict[str, float],
    cell_w: int,
    cell_h: int,
) -> Image.Image:
    """4-column mini-row for contact sheet."""
    return make_comparison_card(
        orig_img, tiles, scores_row, image_id,
        tile_w=cell_w // (1 + len(STRENGTHS)),
        tile_h=cell_h - LABEL_H - TEXT_H,
    )


def build_contact_sheet(
    rows: List[Image.Image],
    sheet_w: int,
    cols: int,
) -> Image.Image:
    if not rows:
        return Image.new("RGB", (sheet_w, 100), (255, 255, 255))
    cell_w = rows[0].width
    cell_h = rows[0].height
    n_rows = (len(rows) + cols - 1) // cols
    # Each "row" here is one full comparison card – arrange in a vertical strip
    # (cols=1 by default since each card is already 4 panels wide)
    sheet = Image.new("RGB", (cell_w * cols, cell_h * n_rows), (250, 250, 250))
    for idx, row_img in enumerate(rows):
        c = idx % cols
        r = idx // cols
        sheet.paste(row_img, (c * cell_w, r * cell_h))
    return sheet


# ─── main ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Visualize GPT strength-sweep eval50")
    ap.add_argument("--scores-csv", default="results/gpt_image_strength_scores.csv")
    ap.add_argument("--input-dir", default=None)
    ap.add_argument("--output-dir", default="outputs/gpt_image_strength")
    ap.add_argument("--out-comparisons", default="results/comparisons/gpt_image_strength")
    ap.add_argument("--out-sheet", default="results/gpt_image_strength_contact_sheet.png")
    ap.add_argument("--out-sheet-top", default="results/gpt_image_strength_contact_sheet_top_delta.png")
    ap.add_argument("--out-sheet-bottom", default="results/gpt_image_strength_contact_sheet_bottom_delta.png")
    ap.add_argument("--tile-w", type=int, default=256,
                    help="Width per panel in individual comparison cards")
    ap.add_argument("--tile-h", type=int, default=320,
                    help="Image height per panel (excl. text/label rows)")
    ap.add_argument("--top-k", type=int, default=20,
                    help="Number of images in top/bottom contact sheets (0 = all)")
    ap.add_argument("--cols", type=int, default=1,
                    help="Number of card columns in contact sheet")
    ap.add_argument("--sort", default="image_id",
                    choices=["image_id", "delta_desc", "delta_asc"],
                    help="Sort order for main contact sheet")
    return ap.parse_args()


def _find_input_dir() -> Path:
    for c in [Path("data/eval50/images"), Path("eval50/images"), Path("eval50/image")]:
        if c.is_dir():
            return c
    raise FileNotFoundError("Cannot find input image dir. Pass --input-dir explicitly.")


def main() -> None:
    args = parse_args()
    scores_csv = Path(args.scores_csv)
    if not scores_csv.is_file():
        raise SystemExit(f"Scores CSV not found: {scores_csv}")

    input_dir = Path(args.input_dir) if args.input_dir else _find_input_dir()
    output_root = Path(args.output_dir)
    out_comp = Path(args.out_comparisons)
    out_comp.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(scores_csv)
    df["delta"] = df["delta"].astype(float)
    df["original_score"] = df["original_score"].astype(float)
    df["edited_score"] = df["edited_score"].astype(float)

    image_ids = df["image_id"].unique().tolist()
    image_ids_sorted = sorted(image_ids)

    # Build per-image scores_row lookup.
    def build_scores(image_id: str) -> Dict[str, float]:
        sub = df[df["image_id"] == image_id]
        row: Dict[str, float] = {}
        if not sub.empty:
            row["original_score"] = float(sub.iloc[0]["original_score"])
        for s in STRENGTHS:
            sr = sub[sub["strength"] == s]
            if not sr.empty:
                row[f"{s}_score"] = float(sr.iloc[0]["edited_score"])
                row[f"{s}_delta"] = float(sr.iloc[0]["delta"])
        return row

    # Per-image best delta (across strengths) for contact sheet sorting.
    # "best_delta" = max delta over all available strengths for that image.
    def best_delta(image_id: str) -> float:
        sub = df[df["image_id"] == image_id]["delta"]
        return float(sub.max()) if not sub.empty else float("-inf")

    IMAGE_EXTS = (".png", ".jpg", ".jpeg")

    def load_orig(image_id: str) -> Optional[Image.Image]:
        for ext in IMAGE_EXTS:
            p = input_dir / f"{image_id}{ext}"
            if p.is_file():
                return Image.open(p).convert("RGB")
        return None

    def load_edit(image_id: str, strength: str) -> Optional[Image.Image]:
        p = output_root / strength / f"{image_id}.png"
        if p.is_file():
            return Image.open(p).convert("RGB")
        return None

    print(f"Building cards for {len(image_ids_sorted)} images…")
    cards: Dict[str, Image.Image] = {}

    for image_id in image_ids_sorted:
        orig = load_orig(image_id)
        if orig is None:
            print(f"  [skip] original not found: {image_id}")
            continue
        tiles = {s: load_edit(image_id, s) for s in STRENGTHS}
        scores_row = build_scores(image_id)
        card = make_comparison_card(
            orig, tiles, scores_row, image_id,
            tile_w=args.tile_w,
            tile_h=args.tile_h,
        )
        out_path = out_comp / f"{image_id}_comparison.png"
        card.save(out_path)
        cards[image_id] = card

    print(f"Saved {len(cards)} comparison cards to {out_comp}")

    if not cards:
        print("No cards produced – nothing to sheet.")
        return

    # Sort for main sheet.
    if args.sort == "delta_desc":
        ordered = sorted(cards.keys(), key=best_delta, reverse=True)
    elif args.sort == "delta_asc":
        ordered = sorted(cards.keys(), key=best_delta)
    else:
        ordered = sorted(cards.keys())

    def save_sheet(ids: List[str], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [cards[i] for i in ids if i in cards]
        sheet = build_contact_sheet(rows, sheet_w=rows[0].width if rows else 1024, cols=args.cols)
        sheet.save(path)
        print(f"Saved sheet ({len(rows)} cards): {path}")

    save_sheet(ordered, Path(args.out_sheet))

    k = args.top_k if args.top_k > 0 else len(ordered)
    by_delta = sorted(cards.keys(), key=best_delta, reverse=True)
    save_sheet(by_delta[:k], Path(args.out_sheet_top))
    save_sheet(by_delta[-k:], Path(args.out_sheet_bottom))


if __name__ == "__main__":
    main()
