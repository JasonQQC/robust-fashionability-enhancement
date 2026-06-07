#!/usr/bin/env python3
import argparse
import pandas as pd

def find_col(df, candidates):
    for c in df.columns:
        if c.lower() in candidates:
            return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="final evaluation csv (must include orig_score & gen_score or delta)")
    ap.add_argument("--out_csv", default=None, help="optional: save per-image with group label")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    orig_col = find_col(df, {"orig_score"})
    gen_col  = find_col(df, {"gen_score"})
    delta_col = find_col(df, {"delta"})

    if orig_col is None:
        raise ValueError("orig_score column not found in csv.")
    if delta_col is None:
        if gen_col is None:
            raise ValueError("need either delta or gen_score column.")
        df["delta"] = df[gen_col] - df[orig_col]
        delta_col = "delta"

    # 3-bin split by tertiles of orig_score
    q1 = df[orig_col].quantile(1/3)
    q2 = df[orig_col].quantile(2/3)

    def bucket(x):
        if x <= q1: return "bad"      # note
        if x <= q2: return "mid"      # note
        return "good"                 # note

    df["group"] = df[orig_col].apply(bucket)
    df["improved"] = df[delta_col] > 0

    # group stats
    grp = df.groupby("group").agg(
        n=("group","size"),
        improved_n=("improved","sum"),
        improved_pct=("improved", lambda s: 100.0 * s.mean()),
        mean_orig=(orig_col, "mean"),
        mean_delta=(delta_col, "mean"),
        median_delta=(delta_col, "median"),
    ).reset_index()

    # order groups
    order = {"bad":0, "mid":1, "good":2}
    grp["order"] = grp["group"].map(order)
    grp = grp.sort_values("order").drop(columns=["order"])

    print("\n=== 3-way split by orig_score tertiles ===")
    print(f"q1(33%)={q1:.4f}  q2(67%)={q2:.4f}\n")
    print(grp.to_string(index=False))

    # overall
    overall = {
        "n": len(df),
        "improved_n": int(df["improved"].sum()),
        "improved_pct": 100.0 * df["improved"].mean(),
        "mean_delta": df[delta_col].mean(),
        "median_delta": df[delta_col].median(),
    }
    print("\n=== Overall ===")
    print(overall)

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print("\nSaved per-image table with group labels ->", args.out_csv)

if __name__ == "__main__":
    main()
