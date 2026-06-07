#!/usr/bin/env python3
"""
note mid note .pt note .npz（float16）
note：
 - note
 - note/note
 - note
note：
  python convert_pt_to_npz_verbose.py --src ./mid_feats --dst ./mid_feats_npz --delete_pt
"""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Convert .pt mid features to .npz (compressed)")
    parser.add_argument("--src", required=True, help="note（note .pt）")
    parser.add_argument("--dst", required=True, help="note（note .npz）")
    parser.add_argument("--delete_pt", action="store_true", help="note .pt note")
    parser.add_argument("--ext", default=".pt", help="note（note .pt）")
    parser.add_argument("--dry_run", action="store_true", help="note，note")
    args = parser.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    files = [f for f in os.listdir(args.src) if f.endswith(args.ext)]
    total = len(files)

    if total == 0:
        print(f"❌ note {args.src} note {args.ext} note")
        return

    print(f"✅ note {total} note {args.ext} note，note...")
    total_bytes_pt, total_bytes_npz = 0, 0
    success, fail = 0, 0

    for name in tqdm(files, desc="Converting", ncols=100):
        srcp = os.path.join(args.src, name)
        try:
            if args.dry_run:
                continue

            obj = torch.load(srcp, map_location="cpu")

            # note
            mid = obj["mid"]
            if isinstance(mid, torch.Tensor):
                mid_np = mid.cpu().numpy().astype(np.float16)
            else:
                raise ValueError("mid note Tensor")
            label = int(obj.get("label", 0))
            img_id = obj.get("id", os.path.splitext(name)[0])

            # note（note id_ note）
            dst_name = f"id_{img_id}.npz"
            dstp = os.path.join(args.dst, dst_name)

            # note npz（note）
            np.savez_compressed(dstp, mid=mid_np, label=label, id=img_id)
            success += 1

            # note
            total_bytes_pt += os.path.getsize(srcp)
            total_bytes_npz += os.path.getsize(dstp)

            if args.delete_pt:
                os.remove(srcp)

        except Exception as e:
            fail += 1
            tqdm.write(f"⚠️ note: {name} -> {e}")

    # note
    ratio = (total_bytes_npz / total_bytes_pt) if total_bytes_pt > 0 else 1
    ratio_str = f"{ratio:.3f}x ({ratio*100:.1f}%)"

    print("\n=== note ===")
    print(f"note {success} note, note {fail} note")
    if not args.dry_run:
        print(f"note .pt note: {total_bytes_pt/1024/1024:.2f} MB")
        print(f"note .npz note: {total_bytes_npz/1024/1024:.2f} MB")
        print(f"note: {ratio_str}")
    if args.delete_pt:
        print("🗑️ note .pt note")
    print(f"note: {args.dst}")

if __name__ == "__main__":
    main()
