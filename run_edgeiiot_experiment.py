#!/usr/bin/env python3
"""Reproducible, bounded-memory Edge-IIoTset multiclass experiment.

This script intentionally reports laptop software timing only.  It does not
measure board-level power or establish energy efficiency on an IoT device.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_SEED = 42
IDENTIFIER_HINTS = ("frame.time", "timestamp", "src_host", "dst_host", "flow_id",
                    "src_ip", "dst_ip", "source_ip", "destination_ip")
LABEL_HINTS = ("attack_label", "attack_type", "label", "target", "class")


class UInt8Tree:
    """A trained sklearn tree represented by uint8 features and thresholds.

    Feature quantisation uses per-feature affine calibration on the training
    set only.  The stored predictor uses uint8 comparison operations and leaf
    class IDs.  It is deliberately simple and portable rather than optimised.
    """

    def __init__(self, source_tree: DecisionTreeClassifier, x_train: np.ndarray):
        self.feature_min = np.min(x_train, axis=0).astype(np.float64)
        self.feature_max = np.max(x_train, axis=0).astype(np.float64)
        self.scale = (self.feature_max - self.feature_min) / 255.0
        self.scale[self.scale == 0] = 1.0
        tree = source_tree.tree_
        self.feature = tree.feature.astype(np.int32)
        self.left = tree.children_left.astype(np.int32)
        self.right = tree.children_right.astype(np.int32)
        self.threshold = np.zeros(tree.node_count, dtype=np.uint8)
        self.leaf_class = np.argmax(tree.value[:, 0, :], axis=1).astype(np.int32)
        for node, feature in enumerate(self.feature):
            if feature >= 0:
                raw = (tree.threshold[node] - self.feature_min[feature]) / self.scale[feature]
                self.threshold[node] = np.uint8(np.clip(np.rint(raw), 0, 255))

    def quantize(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.rint((x - self.feature_min) / self.scale), 0, 255).astype(np.uint8)

    def predict(self, x: np.ndarray) -> np.ndarray:
        q = self.quantize(x)
        out = np.empty(len(q), dtype=np.int32)
        for row, values in enumerate(q):
            node = 0
            while self.feature[node] >= 0:
                node = self.left[node] if values[self.feature[node]] <= self.threshold[node] else self.right[node]
            out[row] = self.leaf_class[node]
        return out

    def export(self, path: Path, feature_names: list[str]) -> None:
        payload = {
            "format": "edgeiiot_uint8_tree_v1",
            "input_precision": "uint8, per-feature affine calibrated from training data",
            "supported_operations": ["uint8 affine input quantisation", "uint8 <= threshold", "integer child traversal", "integer leaf class ID"],
            "feature_names": feature_names,
            "feature_min": self.feature_min.tolist(),
            "feature_max": self.feature_max.tolist(),
            "feature_scale": self.scale.tolist(),
            "nodes": [{"feature": int(self.feature[i]), "threshold_uint8": int(self.threshold[i]),
                       "left": int(self.left[i]), "right": int(self.right[i]),
                       "leaf_class": int(self.leaf_class[i])} for i in range(len(self.feature))],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sampled_rows(csv_path: Path, label: str, cap: int, chunksize: int, seed: int) -> pd.DataFrame:
    """Scan CSV in chunks and retain at most cap labelled records per class."""
    reservoir: dict[str, pd.DataFrame] = {}
    rng = np.random.default_rng(seed)
    for chunk_number, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize, low_memory=False), start=1):
        chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna(subset=[label])
        chunk[label] = chunk[label].astype(str).str.strip()
        for class_name, group in chunk.groupby(label, sort=False):
            existing = reservoir.get(class_name)
            pool = group if existing is None else pd.concat([existing, group], ignore_index=True)
            if len(pool) > cap:
                indices = rng.choice(len(pool), size=cap, replace=False)
                pool = pool.iloc[indices].reset_index(drop=True)
            reservoir[class_name] = pool
        if chunk_number % 10 == 0:
            print(f"scanned {chunk_number * chunksize:,} source rows; retained {sum(map(len, reservoir.values())):,}")
    return pd.concat(reservoir.values(), ignore_index=True)


def prepare_features(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    y = frame[target].astype(str).str.strip()
    excluded = [target]
    for col in frame.columns:
        lowered = col.lower()
        if col != target and (any(hint in lowered for hint in IDENTIFIER_HINTS) or any(hint == lowered for hint in LABEL_HINTS)):
            excluded.append(col)
    numeric = frame.drop(columns=sorted(set(excluded)), errors="ignore").select_dtypes(include=[np.number]).copy()
    numeric = numeric.dropna(axis=1, how="all")
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    if numeric.empty:
        raise RuntimeError("No nonconstant numeric features remained after leakage controls.")
    return numeric, y, numeric.columns.tolist(), sorted(set(excluded))


def time_predict(predictor, x: np.ndarray, repetitions: int = 15) -> tuple[np.ndarray, list[float]]:
    predictor(x[: min(128, len(x))])  # warm-up, excluded from measurements
    elapsed: list[float] = []
    pred = predictor(x)
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        pred = predictor(x)
        elapsed.append((time.perf_counter_ns() - start) / 1_000_000_000)
    return pred, elapsed


def metrics_for(model: str, y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray, timing: list[float],
                size: int, n_features: int, source: str, quantization: str) -> tuple[dict, list[dict]]:
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    row = {
        "model": model, "accuracy": accuracy_score(y_true, y_pred), "precision_macro": p_macro,
        "recall_macro": r_macro, "f1_macro": f_macro, "test_rows": int(len(y_true)), "n_features": n_features,
        "serialized_model_bytes": int(size), "inference_batch_size": int(len(y_true)), "timing_repetitions": len(timing),
        "inference_seconds_batch_median": float(np.median(timing)), "inference_seconds_batch_p05": float(np.quantile(timing, .05)),
        "inference_seconds_batch_p95": float(np.quantile(timing, .95)),
        "latency_ms_per_sample_median": float(np.median(timing) * 1000 / len(y_true)),
        "measurement_scope": "Laptop software prediction after fitted preprocessing; not IoT-device latency or energy.",
        "model_source": source, "quantization": quantization,
    }
    per_class = [{"model": model, "class_id": int(cls), "precision": float(p), "recall": float(r),
                  "f1": float(f), "support": int(s)}
                 for cls, p, r, f, s in zip(labels, precision, recall, f1, support)]
    return row, per_class


def save_confusion(path_base: Path, model: str, y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    fig, ax = plt.subplots(figsize=(8.4, 7.3))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)), xticklabels=classes,
           yticklabels=classes, xlabel="Predicted class", ylabel="True class", title=f"Confusion matrix: {model}")
    plt.setp(ax.get_xticklabels(), rotation=55, ha="right", rotation_mode="anchor", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04, label="Records")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(path_base.with_suffix("." + extension), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plots(metrics: pd.DataFrame, per_class: pd.DataFrame, classes: np.ndarray, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    score_columns = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    x = np.arange(len(metrics)); width = .19
    for j, col in enumerate(score_columns):
        ax.bar(x + (j - 1.5) * width, metrics[col], width, label=col.replace("_", " "))
    ax.set(xticks=x, xticklabels=metrics["model"], ylim=(0, 1.05), ylabel="Score", title="Held-out multiclass detection performance")
    ax.tick_params(axis="x", rotation=16); ax.legend(ncols=2); ax.grid(axis="y", alpha=.25); fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(figures / f"performance_comparison.{ext}", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for model, group in per_class.groupby("model", sort=False):
        ordered = group.set_index("class_id").reindex(range(len(classes)))
        ax.plot(classes, ordered["f1"], marker="o", linewidth=1.3, markersize=3, label=model)
    ax.set(ylim=(0, 1.05), ylabel="F1 score", title="Per-class F1 on the same held-out records")
    ax.tick_params(axis="x", rotation=55); ax.legend(fontsize=7); ax.grid(axis="y", alpha=.25); fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(figures / f"per_class_f1.{ext}", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    sizes_kb = metrics["serialized_model_bytes"] / 1024
    ax.scatter(metrics["latency_ms_per_sample_median"], sizes_kb, s=72, color="#1f77b4")
    for _, row in metrics.iterrows():
        ax.annotate(row["model"], (row["latency_ms_per_sample_median"], row["serialized_model_bytes"] / 1024),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Median laptop inference time per sample (ms)", ylabel="Serialized model size (KiB)",
           title="Model footprint versus laptop inference timing")
    ax.grid(alpha=.25); fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(figures / f"size_latency.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--label", default="Attack_type")
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-class", type=int, default=4_000)
    parser.add_argument("--chunksize", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--mlp-max-iter", type=int, default=100)
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve(); out = Path(args.out)
    models_dir = out / "models"; figures_dir = out / "figures"; predictions_dir = out / "predictions"
    for directory in (models_dir, figures_dir, predictions_dir): directory.mkdir(parents=True, exist_ok=True)
    if args.label not in pd.read_csv(csv_path, nrows=0).columns:
        raise ValueError(f"{args.label!r} is not present in {csv_path}")

    sampled = sampled_rows(csv_path, args.label, args.per_class, args.chunksize, args.seed)
    before_dedup = len(sampled); sampled = sampled.drop_duplicates().reset_index(drop=True)
    x_frame, y_raw, features, excluded = prepare_features(sampled, args.label)
    encoder = LabelEncoder(); y = encoder.fit_transform(y_raw)
    x_train, x_test, y_train, y_test = train_test_split(x_frame, y, test_size=.20, random_state=args.seed, stratify=y)
    imputer = SimpleImputer(strategy="median")
    x_train_i = imputer.fit_transform(x_train); x_test_i = imputer.transform(x_test)
    joblib.dump(imputer, models_dir / "numeric_median_imputer.joblib")

    conventional = DecisionTreeClassifier(max_depth=12, min_samples_leaf=2, class_weight="balanced", random_state=args.seed)
    compact = DecisionTreeClassifier(max_depth=8, min_samples_leaf=3, class_weight="balanced", random_state=args.seed)
    conventional.fit(x_train_i, y_train); compact.fit(x_train_i, y_train)
    compact_int8 = UInt8Tree(compact, x_train_i)
    mlp = Pipeline([("scale", StandardScaler()), ("mlp", MLPClassifier(hidden_layer_sizes=(32,), activation="relu",
                   alpha=.0001, early_stopping=True, validation_fraction=.1, max_iter=args.mlp_max_iter,
                   random_state=args.seed, learning_rate_init=.001))])
    mlp.fit(x_train_i, y_train)

    joblib.dump(conventional, models_dir / "decision_tree_float.joblib")
    joblib.dump(compact, models_dir / "compact_tree_float.joblib")
    joblib.dump(mlp, models_dir / "tiny_mlp_float.joblib")
    compact_int8.export(models_dir / "compact_tree_uint8.json", features)
    specs = [
        ("Decision Tree (float)", conventional.predict, models_dir / "decision_tree_float.joblib", "sklearn CART float64", "none"),
        ("Compact Tree (float)", compact.predict, models_dir / "compact_tree_float.joblib", "sklearn CART float64", "none"),
        ("Compact Tree (uint8)", compact_int8.predict, models_dir / "compact_tree_uint8.json", "custom integer tree", "per-feature uint8 affine input and threshold quantisation"),
        ("Tiny MLP (float)", mlp.predict, models_dir / "tiny_mlp_float.joblib", "sklearn MLP float64", "none; compact neural baseline, not a TFLite deployment"),
    ]

    metric_rows: list[dict] = []; class_rows: list[dict] = []
    prediction_output = pd.DataFrame({"row_in_held_out_set": np.arange(len(y_test)), "true_class": encoder.inverse_transform(y_test)})
    for name, predictor, model_path, source, quantization in specs:
        predicted, timings = time_predict(predictor, x_test_i)
        row, per_class = metrics_for(name, y_test, predicted, np.arange(len(encoder.classes_)), timings,
                                     model_path.stat().st_size, len(features), source, quantization)
        metric_rows.append(row); class_rows.extend(per_class)
        prediction_output[name] = encoder.inverse_transform(predicted)
        safe = "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")
        save_confusion(figures_dir / f"confusion_{safe}", name, y_test, predicted, encoder.classes_)
        print(f"{name}: accuracy={row['accuracy']:.4f}; macro-F1={row['f1_macro']:.4f}; laptop median={row['latency_ms_per_sample_median']:.5f} ms/sample")

    metrics = pd.DataFrame(metric_rows); per_class = pd.DataFrame(class_rows)
    metrics.to_csv(out / "metrics.csv", index=False); per_class.to_csv(out / "per_class_metrics.csv", index=False)
    prediction_output.to_csv(predictions_dir / "held_out_predictions.csv", index=False)
    pd.DataFrame({"class_id": np.arange(len(encoder.classes_)), "class_label": encoder.classes_}).to_csv(out / "label_mapping.csv", index=False)
    pd.DataFrame({"feature": features}).to_csv(out / "selected_features.csv", index=False)
    plots(metrics, per_class, encoder.classes_, figures_dir)

    metadata = {
        "input_csv": str(csv_path), "label_column": args.label, "seed": args.seed, "sampling": {
            "strategy": "one-pass bounded per-class random reservoir", "per_class_cap": args.per_class,
            "rows_retained_before_deduplication": before_dedup, "rows_after_exact_deduplication": len(sampled)},
        "split": {"strategy": "stratified random row split", "test_fraction": .20, "train_rows": len(y_train), "test_rows": len(y_test),
                  "leakage_assessment": "Exact duplicate rows were removed after sampling. No group/session ID was available in the selected numeric schema; therefore random row split may still be optimistic for related flows."},
        "preprocessing": {"numeric_features_only": True, "imputation": "median, fit only on training data", "features": features,
                          "excluded_target_or_identifier_or_leakage_columns": excluded},
        "quantized_tree": {"format": "edgeiiot_uint8_tree_v1", "verification": "Predictions were generated by the uint8 traversal and evaluated on the shared held-out set; disagreement with its float compact-tree source is directly visible in predictions/held_out_predictions.csv.",
                           "limitation": "Quantisation has not been compiled or measured on an IoT device."},
        "measurement": {"warm_up_batch": min(128, len(y_test)), "repetitions": 15, "scope": "prediction after fitted preprocessing on this laptop", "energy_or_power": "not measured", "peak_memory": "not measured reliably"},
        "software": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "hardware": {"platform": platform.platform(), "cpu": platform.processor(), "logical_cpus": psutil.cpu_count(logical=True), "memory_total_bytes": psutil.virtual_memory().total},
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved reproducible experiment outputs to {out.resolve()}")


if __name__ == "__main__":
    main()
