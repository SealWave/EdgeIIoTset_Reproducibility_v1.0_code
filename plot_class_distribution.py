#!/usr/bin/env python3
"""Generate vector and PNG class-distribution figures from audit CSV output."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="class_distribution.csv from audit_dataset.py")
    parser.add_argument("--out", required=True, help="output file stem without extension")
    args = parser.parse_args()
    data = pd.read_csv(args.input).sort_values("count", ascending=True)
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    bars = ax.barh(data["class_label"], data["count"], color="#2a6f9e")
    ax.bar_label(bars, labels=[f"{int(n):,}" for n in data["count"]], padding=3, fontsize=8)
    ax.set(xlabel="Records in original ML-selected CSV", ylabel="Attack type", title="Edge-IIoTset ML-selected class distribution")
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout()
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(target.with_suffix(".png"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
