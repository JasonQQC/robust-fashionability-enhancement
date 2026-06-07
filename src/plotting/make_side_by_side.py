#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from PIL import Image
from tqdm import tqdm
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_dir", required=True, help="description")
    ap.add_argument("--gen_dir",  required=True, help="note（*_gen.png note）")
    ap.add_argument("--out_dir",  required=True, help="description")
    ap.add_argument("--max_width", type=int, default=1024,
                    help="note，note (note)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    names = [n for n in os.listdir(args.orig_dir)
             if n.lower().endswith((".png", ".jpg", ".jpeg"))]
    names.sort()

    for name in tqdm(names, ncols=100, desc="Make side-by-side"):
        stem, ext = os.path.splitext(name)
        orig_path = os.path.join(args.orig_dir, name)

        # note
        gen_path = None
        for e in (".png", ".jpg", ".jpeg"):
            p1 = os.path.join(args.gen_dir, f"{stem}_gen{e}")
            p2 = os.path.join(args.gen_dir, f"{stem}{e}")
            if os.path.isfile(p1):
                gen_path = p1
                break
            if os.path.isfile(p2):
                gen_path = p2
                break
        if gen_path is None:
            # note
            continue

        orig = Image.open(orig_path).convert("RGB")
        gen  = Image.open(gen_path).convert("RGB")

        # note：note，note
        if orig.width > args.max_width:
            scale = args.max_width / orig.width
            new_size = (args.max_width, int(orig.height * scale))
            orig = orig.resize(new_size, Image.BICUBIC)
        # note（note，note）
        gen = gen.resize(orig.size, Image.BICUBIC)

        w, h = orig.size
        canvas = Image.new("RGB", (w * 2, h), (255, 255, 255))
        canvas.paste(orig, (0, 0))
        canvas.paste(gen, (w, 0))

        out_name = f"{stem}_side_by_side{ext}"
        canvas.save(os.path.join(args.out_dir, out_name))

if __name__ == "__main__":
    main()
