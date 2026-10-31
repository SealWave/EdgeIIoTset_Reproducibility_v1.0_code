#!/usr/bin/env python3
"""Chunked audit for one Edge-IIoTset CSV without altering the source data."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_CANDIDATES = ("Attack_type", "Attack_label", "label", "Label", "class", "Class")


def find_label(columns: list[str]) -> str:
    for candidate in LABEL_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(f"No recognised label column. Columns: {columns}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--chunksize", type=int, default=25_000)
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    columns = list(pd.read_csv(csv_path, nrows=0).columns)
    label = find_label(columns)
    total_rows = 0
    missing = Counter()
    infinity = Counter()
    classes = Counter()
    dtypes: dict[str, Counter] = {name: Counter() for name in columns}
    row_hashes: set[bytes] = set()
    duplicate_rows = 0

    for chunk in pd.read_csv(csv_path, chunksize=args.chunksize, low_memory=False):
        total_rows += len(chunk)
        missing.update(chunk.isna().sum().to_dict())
        for col, dtype in chunk.dtypes.items():
            dtypes[col][str(dtype)] += len(chunk)
        numeric = chunk.select_dtypes(include=[np.number])
        if not numeric.empty:
            infinity.update(np.isinf(numeric).sum().to_dict())
        labels = chunk[label].fillna("<MISSING>").astype(str).str.strip()
        classes.update(labels.tolist())
        # Stable row hashes allow exact duplicate detection across chunk boundaries.
        for value in pd.util.hash_pandas_object(chunk, index=False).to_numpy():
            digest = int(value).to_bytes(8, "little", signed=False)
            if digest in row_hashes:
                duplicate_rows += 1
            else:
                row_hashes.add(digest)

    class_rows = [{"class_label": k, "count": int(v), "proportion": v / total_rows}
                  for k, v in classes.most_common()]
    pd.DataFrame(class_rows).to_csv(out / "class_distribution.csv", index=False)
    manifest = {
        "input_csv": str(csv_path),
        "file_bytes": csv_path.stat().st_size,
        "sha256": hashlib.file_digest(csv_path.open("rb"), "sha256").hexdigest(),
        "label_column_selected": label,
        "columns": columns,
        "column_count": len(columns),
        "rows_scanned": total_rows,
        "exact_duplicate_rows_detected": duplicate_rows,
        "missing_values_by_column": {k: int(v) for k, v in missing.items() if v},
        "infinite_values_by_numeric_column": {k: int(v) for k, v in infinity.items() if v},
        "observed_dtypes": {k: dict(v) for k, v in dtypes.items()},
        "class_counts": {k: int(v) for k, v in classes.items()},
        "notes": [
            "Duplicates are exact whole-row hash matches; hash collisions are extremely unlikely but not impossible.",
            "This audit does not alter the CSV or infer a legitimate pre-existing train/test split.",
        ],
    }
    (out / "dataset_audit.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("label_column_selected", "rows_scanned", "exact_duplicate_rows_detected", "column_count")}, indent=2))


if __name__ == "__main__":
    main()
