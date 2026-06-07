#!/usr/bin/env python3
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="final_scores csv WITHOUT header")
    ap.add_argument("--out_csv", default=None, help="optional output with group labels")
    args = ap.parse_args()

    
    df = pd.read_csv(args.csv, header=None)
    df.columns = ["name", "orig_score", "gen_score", "delta"]

    
    q1 = df["orig_score"].quantile(1/3)
    q2 = df["orig_score"].quantile(2/3)

    def bucket(x):
        if x <= q1:
            return "bad"    # note
        elif x <= q2:
            return "mid"    # note
        else:
            return "good"   # note

    df["group"] = df["orig_score"].apply(bucket)
    df["improved"] = df["delta"] > 0

    
    summary = (
        df.groupby("group")
        .agg(
            n=("name", "count"),
            improved_n=("improved", "sum"),
            improved_pct=("improved", lambda x: 100.0 * x.mean()),
            mean_orig=("orig_score", "mean"),
            mean_delta=("delta", "mean"),
            median_delta=("delta", "median"),
        )
        .reset_index()
    )

    # （bad -> mid -> good）
    order = {"bad": 0, "mid": 1, "good": 2}
    summary["order"] = summary["group"].map(order)
    summary = summary.sort_values("order").drop(columns=["order"])

    print("\n=== Split by original score (tertiles) ===")
    print(f"Thresholds: q1={q1:.4f}, q2={q2:.4f}\n")
    print(summary.to_string(index=False))

    # Overall
    print("\n=== Overall ===")
    print({
        "n": len(df),
        "improved_n": int((df["delta"] > 0).sum()),
        "improved_pct": 100.0 * (df["delta"] > 0).mean(),
        "mean_delta": float(df["delta"].mean()),
        "median_delta": float(df["delta"].median()),
    })

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print("\nSaved detailed csv with group labels to:", args.out_csv)

if __name__ == "__main__":
    main()
