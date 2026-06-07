#!/usr/bin/env python3
import sys, os
from pathlib import Path
import pandas as pd
import numpy as np
import argparse

# thresholds 
TH_LARGE_UP = 0.05    #  > +0.05
TH_SMALL_UP = 0.01    #  between +0.01 and +0.05
TH_SMALL_DOWN = -0.01
TH_LARGE_DOWN = -0.05

def stats_from_df(df):
    d = df["delta"].to_numpy()
    n = len(d)
    mean = float(np.mean(d))
    median = float(np.median(d))
    trimmed = float(np.mean(np.sort(d)[int(0.1*n):int(0.9*n)])) if n>10 else mean
    # winsorized 5%
    lo = int(0.05*n); hi = int((1-0.05)*n)
    dl = np.sort(d)
    if n>20:
        dl[:lo] = dl[lo]
        dl[hi:] = dl[hi-1]
    wins = float(np.mean(dl))
    # counts
    cnt_large_up = int((d > TH_LARGE_UP).sum())
    cnt_small_up = int(((d > TH_SMALL_UP) & (d <= TH_LARGE_UP)).sum())
    cnt_neutral = int(((d >= TH_SMALL_DOWN) & (d <= TH_SMALL_UP)).sum())
    cnt_small_down = int(((d < TH_SMALL_DOWN) & (d >= TH_LARGE_DOWN)).sum())
    cnt_large_down = int((d < TH_LARGE_DOWN).sum())
    improved_pct = float((d>0).mean()*100)
    return {
        "n": n,
        "mean_delta": mean,
        "median_delta": median,
        "trimmed_mean": trimmed,
        "winsorized_mean": wins,
        "improved_pct": improved_pct,
        "cnt_large_up": cnt_large_up,
        "cnt_small_up": cnt_small_up,
        "cnt_neutral": cnt_neutral,
        "cnt_small_down": cnt_small_down,
        "cnt_large_down": cnt_large_down,
    }

def analyze(root_dir: Path, out_root: Path):
    exps = sorted([p for p in root_dir.glob("g*") if p.is_dir()])
    if not exps:
        print("No experiments found under", root_dir)
        return
    rows = []
    # collect per-image effects across exps
    image_effects = {}  # name -> {exp: delta}
    for exp in exps:
        scores = exp / "scores.csv"
        if not scores.exists():
            print("Skipping (no scores.csv):", exp)
            continue
        df = pd.read_csv(scores)
        if "delta" not in df.columns:
            print("scores.csv has no 'delta' column in", exp)
            continue
        s = stats_from_df(df)
        s["exp"] = exp.name
        rows.append(s)

        # write top lists
        df_sorted_up = df.sort_values("delta", ascending=False).reset_index(drop=True)
        df_sorted_up.head(20).to_csv(exp / "top_improved.csv", index=False)
        df_sorted_up.tail(20).to_csv(exp / "top_worsened.csv", index=False)

        for _, r in df.iterrows():
            name = r.get("name", r.get("filename", r.get("path", None)))
            if name is None:
                # try to find first column
                name = r.iloc[0]
            image_effects.setdefault(name, {})[exp.name] = float(r["delta"])

    out_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("mean_delta", ascending=False).to_csv(out_root / "ANALYSIS_SUMMARY.csv", index=False)

    # build image consistency table
    img_rows = []
    exps_names = [e.name for e in exps]
    for img, dmap in image_effects.items():
        row = {"image": img}
        deltas = []
        for e in exps_names:
            val = dmap.get(e, None)
            row[e] = val
            deltas.append(val)
        # count how many exps improved this image
        improved_count = sum(1 for v in deltas if v is not None and v>0)
        worsened_count = sum(1 for v in deltas if v is not None and v<0)
        row["improved_count"] = improved_count
        row["worsened_count"] = worsened_count
        img_rows.append(row)
    pd.DataFrame(img_rows).to_csv(out_root / "image_consistency.csv", index=False)

    print("Wrote analysis to:", out_root)
    print("Summary:")
    print(pd.DataFrame(rows).sort_values("mean_delta", ascending=False).to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="runs/eval50/sweep_midu_cls", help="sweep root dir")
    parser.add_argument("--out", default="runs/eval50/sweep_analysis", help="analysis out dir")
    args = parser.parse_args()
    analyze(Path(args.root), Path(args.out))
