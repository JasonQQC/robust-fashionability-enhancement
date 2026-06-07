#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, csv, argparse, random, shutil
from tqdm import tqdm

def find_with_ext(root, stem):
    """
    stem: note id，note '-63'
    note snap-images/-63.jpg, snap-images/-63.png ...
    """
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(root, stem + ext)
        if os.path.isfile(p):
            return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv",       required=True, help="test.csv，note image_id note")
    ap.add_argument("--image_root",     required=True, help="snap-images note，note 20250319_.../snap-images")
    ap.add_argument("--out_image_dir",  required=True, help="note eval/image note")

    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--seed",        type=int, default=2)
    args = ap.parse_args()

    random.seed(args.seed)

    os.makedirs(args.out_image_dir, exist_ok=True)

    # === note test.csv ===
    image_ids = []
    with open(args.test_csv, newline="") as f:
        reader = csv.DictReader(f)
        if "image_id" not in reader.fieldnames:
            raise ValueError("CSV note image_id note")

        for row in reader:
            image_ids.append(str(row["image_id"]))   # note "-63"

    # note
    image_ids = sorted(list(set(image_ids)))
    print(f"Test CSV note {len(image_ids)} note image_id")

    # === note ===
    if args.num_samples > len(image_ids):
        print("num_samples > note，note。")
        chosen = image_ids
    else:
        chosen = random.sample(image_ids, args.num_samples)

    print(f"note {len(chosen)} note → note {args.out_image_dir}")

    # === note ===
    miss_img = 0
    for img_id in tqdm(chosen, ncols=100):
        stem = str(img_id)   # note '-63'
        src_img = find_with_ext(args.image_root, stem)

        if src_img is None:
            miss_img += 1
            continue

        dst_img = os.path.join(args.out_image_dir, os.path.basename(src_img))
        shutil.copy2(src_img, dst_img)

    print(f"note！note：{miss_img}")

if __name__ == "__main__":
    main()

