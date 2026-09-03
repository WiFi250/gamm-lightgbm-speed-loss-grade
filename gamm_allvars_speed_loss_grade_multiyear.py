#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多年份全变量航速损失五等级 GAMM 风格控制实验脚本。

说明：
1. 复用主实验的数据清洗、时间切分与五等级分桶口径。
2. 使用 Python 中可全量扩展的 GAMM 风格实现：
   - 连续变量 -> 样条平滑项
   - 空间网格 cell -> 组效应项
   - 最终以多类对数损失 SGD 分类器拟合
3. 输出与主实验尽量一致的指标、混淆矩阵、空间分布图和解释性结果。
"""

from __future__ import annotations

import argparse
import inspect
import json
import pickle
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lightgbm_allvars_speed_loss_grade_multiyear import (
    GRADE_LABELS,
    NUM_CLASSES,
    build_thresholds,
    build_usecols_from_header,
    classification_metrics,
    classify_direction,
    chronological_split,
    compute_grade_columns,
    ensure_output_dir,
    infer_feature_columns,
    load_marine_regions,
    load_multi_year_data,
    make_time_summary,
    read_header_columns,
    save_arctic_region_basemap,
    save_arctic_spatial_map,
    save_classification_bar,
    save_classification_report_table,
    save_confusion_matrix,
    train_valid_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a scalable GAMM-style grade classifier for LNG ship speed loss."
    )
    parser.add_argument("--input-csvs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("gamm_speed_loss_grade_outputs"))
    parser.add_argument("--sample-cap-per-file", type=int, default=0)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--time-col", type=str, default="acq_time")
    parser.add_argument("--target-col", type=str, default="speed_loss_rate")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--positive-only", action="store_true")
    parser.add_argument("--spline-fit-sample-size", type=int, default=400_000)
    parser.add_argument("--monitor-sample-size", type=int, default=300_000)
    parser.add_argument("--n-knots", type=int, default=6)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--lon-bin-width", type=float, default=10.0)
    parser.add_argument("--lat-bin-width", type=float, default=2.5)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=500_000)
    parser.add_argument("--alpha", type=float, default=1e-5)
    parser.add_argument("--learning-rate", type=str, default="optimal")
    parser.add_argument("--eta0", type=float, default=0.0)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--marine-regions-dir", type=Path, default=Path("IHO_marine_regions_58N"))
    return parser.parse_args()


def numeric_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df[feature_cols].copy()
    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    return out


def make_spatial_cell_id(
    df: pd.DataFrame,
    lon_bin_width: float,
    lat_bin_width: float,
) -> pd.Series:
    lon_idx = np.floor((pd.to_numeric(df["longitude"], errors="coerce") + 180.0) / lon_bin_width).astype("Int64")
    lat_idx = np.floor((pd.to_numeric(df["latitude"], errors="coerce") + 90.0) / lat_bin_width).astype("Int64")
    return lon_idx.astype(str) + "_" + lat_idx.astype(str)


def make_one_hot_encoder() -> OneHotEncoder:
    kwargs = {"handle_unknown": "ignore"}
    sig = inspect.signature(OneHotEncoder)
    if "sparse_output" in sig.parameters:
        kwargs["sparse_output"] = True
    else:
        kwargs["sparse"] = True
    return OneHotEncoder(**kwargs)


def make_spline_transformer(n_knots: int, degree: int) -> SplineTransformer:
    kwargs = {
        "n_knots": n_knots,
        "degree": degree,
        "knots": "quantile",
        "include_bias": False,
    }
    sig = inspect.signature(SplineTransformer)
    if "sparse_output" in sig.parameters:
        kwargs["sparse_output"] = True
    return SplineTransformer(**kwargs)


def sample_frame(df: pd.DataFrame, size: int, random_state: int) -> pd.DataFrame:
    if size <= 0 or size >= len(df):
        return df.copy()
    return df.sample(n=size, random_state=random_state).reset_index(drop=True)


def batched_indices(n: int, batch_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + batch_size, n)) for start in range(0, n, batch_size)]


def transform_frame(
    df: pd.DataFrame,
    numeric_cols: list[str],
    cell_col: str,
    spline: SplineTransformer,
    encoder: OneHotEncoder,
) -> sparse.csr_matrix:
    X_num = spline.transform(numeric_frame(df, numeric_cols))
    X_cell = encoder.transform(df[[cell_col]])
    return sparse.hstack([X_num, X_cell], format="csr")


def expected_grade_from_proba(proba: np.ndarray) -> np.ndarray:
    return np.sum(proba * np.arange(NUM_CLASSES, dtype=float), axis=1)


def class_weight_map(y: pd.Series) -> dict[int, float]:
    counts = y.value_counts().sort_index()
    total = float(counts.sum())
    num_classes = float(len(counts))
    return {int(cls): total / (num_classes * float(cnt)) for cls, cnt in counts.items() if cnt > 0}


def sample_weights(y: pd.Series, weights_map: dict[int, float]) -> np.ndarray:
    return y.map(weights_map).fillna(1.0).astype("float32").to_numpy()


def fit_epochwise_classifier(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    numeric_cols: list[str],
    cell_col: str,
    spline: SplineTransformer,
    encoder: OneHotEncoder,
    random_state: int,
    epochs: int,
    batch_size: int,
    alpha: float,
    learning_rate: str,
    eta0: float,
    early_stop_patience: int,
    monitor_sample_size: int,
) -> tuple[SGDClassifier, dict[str, list[float]], pd.DataFrame, pd.DataFrame]:
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        learning_rate=learning_rate,
        eta0=eta0,
        fit_intercept=True,
        random_state=random_state,
        average=True,
    )

    classes = np.arange(NUM_CLASSES, dtype=int)
    weights_map = class_weight_map(train_df["loss_grade"])
    train_monitor = sample_frame(train_df, monitor_sample_size, random_state)
    valid_monitor = sample_frame(valid_df, monitor_sample_size, random_state + 1)

    history = {
        "epoch": [],
        "train_logloss": [],
        "valid_logloss": [],
        "train_accuracy": [],
        "valid_accuracy": [],
    }
    best_state: bytes | None = None
    best_valid_loss = float("inf")
    wait = 0
    for epoch in range(1, epochs + 1):
        for start, end in batched_indices(len(train_df), batch_size):
            batch_df = train_df.iloc[start:end]
            X_batch = transform_frame(batch_df, numeric_cols, cell_col, spline, encoder)
            y_batch = batch_df["loss_grade"].to_numpy(dtype=int, copy=True)
            w_batch = sample_weights(batch_df["loss_grade"], weights_map)
            clf.partial_fit(X_batch, y_batch, classes=classes, sample_weight=w_batch)

        train_proba = clf.predict_proba(transform_frame(train_monitor, numeric_cols, cell_col, spline, encoder))
        valid_proba = clf.predict_proba(transform_frame(valid_monitor, numeric_cols, cell_col, spline, encoder))
        train_pred = np.argmax(train_proba, axis=1)
        valid_pred = np.argmax(valid_proba, axis=1)

        train_logloss = float(log_loss(train_monitor["loss_grade"], train_proba, labels=classes))
        valid_logloss = float(log_loss(valid_monitor["loss_grade"], valid_proba, labels=classes))
        train_acc = float(np.mean(train_pred == train_monitor["loss_grade"].to_numpy()))
        valid_acc = float(np.mean(valid_pred == valid_monitor["loss_grade"].to_numpy()))

        history["epoch"].append(epoch)
        history["train_logloss"].append(train_logloss)
        history["valid_logloss"].append(valid_logloss)
        history["train_accuracy"].append(train_acc)
        history["valid_accuracy"].append(valid_acc)
        print(
            f"Epoch {epoch}: train_logloss={train_logloss:.6f}, valid_logloss={valid_logloss:.6f}, "
            f"train_acc={train_acc:.4f}, valid_acc={valid_acc:.4f}"
            ,
            flush=True,
        )

        if valid_logloss < best_valid_loss - 1e-5:
            best_valid_loss = valid_logloss
            best_state = pickle.dumps(clf, protocol=pickle.HIGHEST_PROTOCOL)
            wait = 0
        else:
            wait += 1
            if wait >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        clf = pickle.loads(best_state)
    return clf, history, train_monitor, valid_monitor


def predict_proba_in_batches(
    clf: SGDClassifier,
    df: pd.DataFrame,
    numeric_cols: list[str],
    cell_col: str,
    spline: SplineTransformer,
    encoder: OneHotEncoder,
    batch_size: int,
) -> np.ndarray:
    proba_parts: list[np.ndarray] = []
    for start, end in batched_indices(len(df), batch_size):
        batch_df = df.iloc[start:end]
        X_batch = transform_frame(batch_df, numeric_cols, cell_col, spline, encoder)
        proba_parts.append(clf.predict_proba(X_batch))
    return np.vstack(proba_parts)


def save_fit_curve(history: dict[str, list[float]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(history["epoch"], history["train_logloss"], label="Train", color="#2563eb", lw=1.7)
    axes[0].plot(history["epoch"], history["valid_logloss"], label="Valid", color="#dc2626", lw=1.7)
    axes[0].set_title("Monitor Log Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Log loss")
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_accuracy"], label="Train", color="#2563eb", lw=1.7)
    axes[1].plot(history["epoch"], history["valid_accuracy"], label="Valid", color="#dc2626", lw=1.7)
    axes[1].set_title("Monitor Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(alpha=0.2)
    axes[1].legend()

    fig.suptitle("GAMM-style Grade Classification Fit Curves")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def feature_spline_slices(spline: SplineTransformer, feature_names: list[str]) -> dict[str, slice]:
    per_feature = spline.n_features_out_ // len(feature_names)
    return {
        feature: slice(i * per_feature, (i + 1) * per_feature)
        for i, feature in enumerate(feature_names)
    }


def save_feature_importance(
    clf: SGDClassifier,
    spline: SplineTransformer,
    numeric_cols: list[str],
    output_csv: Path,
    output_png: Path,
) -> pd.DataFrame:
    coef = np.asarray(clf.coef_, dtype=float)
    slices = feature_spline_slices(spline, numeric_cols)
    rows = []
    for feature in numeric_cols:
        sl = slices[feature]
        rows.append(
            {
                "feature": feature,
                "mean_abs_coef": float(np.mean(np.abs(coef[:, sl]))),
                "max_abs_coef": float(np.max(np.abs(coef[:, sl]))),
            }
        )
    table = pd.DataFrame(rows).sort_values("mean_abs_coef", ascending=False)
    table.to_csv(output_csv, index=False, encoding="utf-8-sig")

    top = table.head(min(20, len(table))).sort_values("mean_abs_coef", ascending=True)
    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    ax.barh(top["feature"], top["mean_abs_coef"], color="#2563eb")
    ax.set_title("GAMM-style Feature Importance")
    ax.set_xlabel("Mean |coefficient| over spline bases")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return table


def save_smooth_effect_panels(
    spline: SplineTransformer,
    clf: SGDClassifier,
    reference_df: pd.DataFrame,
    numeric_cols: list[str],
    feature_table: pd.DataFrame,
    output_path: Path,
    class_index: int = 4,
    top_n: int = 9,
) -> None:
    coef = np.asarray(clf.coef_, dtype=float)
    intercept = np.asarray(clf.intercept_, dtype=float)
    slices = feature_spline_slices(spline, numeric_cols)
    top_features = feature_table.head(top_n)["feature"].tolist()

    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    axes = axes.ravel()

    for ax, feature in zip(axes, top_features, strict=False):
        values = pd.to_numeric(reference_df[feature], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) == 0:
            ax.set_visible(False)
            continue
        x_low, x_high = np.nanpercentile(values, [1.0, 99.0])
        if not np.isfinite(x_low) or not np.isfinite(x_high) or x_high <= x_low:
            ax.set_visible(False)
            continue
        grid = np.linspace(x_low, x_high, 240)
        temp = pd.DataFrame(
            {
                col: np.full_like(grid, np.nanmedian(pd.to_numeric(reference_df[col], errors="coerce")), dtype=float)
                for col in numeric_cols
            }
        )
        temp[feature] = grid
        basis = spline.transform(temp[numeric_cols])
        sl = slices[feature]
        effect = basis[:, sl] @ coef[class_index, sl]
        effect = np.asarray(effect).ravel() + intercept[class_index]
        effect = effect - np.mean(effect)

        ax.plot(grid, effect, color="#dc2626", lw=2)
        ax.axhline(0, color="#94a3b8", lw=1, ls="--")
        ax.set_title(feature)
        ax.set_xlabel(feature)
        ax.set_ylabel(f"Class {GRADE_LABELS[class_index]} effect")
        ax.grid(alpha=0.18)

    for ax in axes[len(top_features) :]:
        ax.set_visible(False)

    fig.suptitle("Top Smooth Effects for High-loss Class")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output_dir: Path,
    summary: dict,
    class_report: pd.DataFrame,
    feature_table: pd.DataFrame,
) -> None:
    test_metrics = summary["test_metrics"]
    valid_metrics = summary["valid_metrics"]
    class_rows = [
        "| Grade | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in class_report.iterrows():
        class_rows.append(
            f"| {row['grade']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {int(row['support'])} |"
        )
    top_rows = [
        "| 排名 | 变量 | mean(|coef|) |",
        "|---|---|---:|",
    ]
    for i, (_, row) in enumerate(feature_table.head(8).iterrows(), start=1):
        top_rows.append(f"| {i} | `{row['feature']}` | {row['mean_abs_coef']:.4f} |")

    lines = [
        "# 全量正损失五等级航速损失 GAMM 控制实验报告",
        "",
        "本实验与主实验保持相同的数据、时间切分与五等级评估口径，采用 Python 中可全量运行的 GAMM 风格实现：连续变量使用样条平滑项，空间网格 `cell_id` 使用组效应项，再用多类对数损失分类器拟合五档标签。",
        "",
        "## 1. 模型配置",
        f"- 平滑项：`SplineTransformer(n_knots={summary['n_knots']}, degree={summary['degree']})`",
        f"- 组效应：空间网格 `cell_id`，经纬度分箱宽度分别为 `lon={summary['lon_bin_width']}`、`lat={summary['lat_bin_width']}`",
        f"- 分类器：`SGDClassifier(loss='log_loss', average=True)`",
        f"- 实际训练轮数：`{summary['best_epoch']}` / 最大 `epochs={summary['epochs']}`",
        "",
        "## 2. 总体指标",
        "| 数据集 | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | QWK | Within-one Acc | Top-2 Acc | MAE(class) | Corr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| Train | {summary['train_metrics']['accuracy']:.4f} | {summary['train_metrics']['balanced_accuracy']:.4f} | {summary['train_metrics']['macro_f1']:.4f} | {summary['train_metrics']['weighted_f1']:.4f} | {summary['train_metrics']['quadratic_kappa']:.4f} | {summary['train_metrics']['within_one_accuracy']:.4f} | {summary['train_metrics']['top2_accuracy']:.4f} | {summary['train_metrics']['mae_class']:.4f} | {summary['train_metrics']['corr']:.4f} |",
        f"| Valid | {valid_metrics['accuracy']:.4f} | {valid_metrics['balanced_accuracy']:.4f} | {valid_metrics['macro_f1']:.4f} | {valid_metrics['weighted_f1']:.4f} | {valid_metrics['quadratic_kappa']:.4f} | {valid_metrics['within_one_accuracy']:.4f} | {valid_metrics['top2_accuracy']:.4f} | {valid_metrics['mae_class']:.4f} | {valid_metrics['corr']:.4f} |",
        f"| Test | {test_metrics['accuracy']:.4f} | {test_metrics['balanced_accuracy']:.4f} | {test_metrics['macro_f1']:.4f} | {test_metrics['weighted_f1']:.4f} | {test_metrics['quadratic_kappa']:.4f} | {test_metrics['within_one_accuracy']:.4f} | {test_metrics['top2_accuracy']:.4f} | {test_metrics['mae_class']:.4f} | {test_metrics['corr']:.4f} |",
        "",
        "## 3. 测试集分类报告",
        *class_rows,
        "",
        "## 4. 变量重要性",
        *top_rows,
        "",
        "## 5. 结论",
        "- 该 GAMM 风格控制模型能够学习到稳定的有序等级信号。",
        "- 如果主模型在 Macro F1、Balanced Accuracy、QWK 等指标上持续高于该基线，就可以更有力地说明主模型确实利用了更强的非线性与交互结构。",
    ]
    (output_dir / "GAMM控制实验报告.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)

    header_cols = read_header_columns(args.input_csvs[0])
    usecols = build_usecols_from_header(header_cols, args.target_col, args.time_col)

    df, manifests = load_multi_year_data(
        csv_paths=args.input_csvs,
        sample_cap_per_file=args.sample_cap_per_file,
        chunksize=args.chunksize,
        random_state=args.random_state,
        usecols=usecols,
    )
    (args.output_dir / "sampling_manifest.json").write_text(
        json.dumps(manifests, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    df[args.time_col] = pd.to_datetime(df[args.time_col], errors="coerce")
    df = df.dropna(subset=[args.time_col, args.target_col, "longitude", "latitude"]).copy()
    rows_before_positive_filter = int(len(df))
    if args.positive_only:
        df = df[pd.to_numeric(df[args.target_col], errors="coerce") > 0].copy()
        if len(df) == 0:
            raise ValueError("No rows left after applying positive-only filter.")

    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce").astype("float32")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce").astype("float32")
    df = df.sort_values(args.time_col).reset_index(drop=True)
    df["year"] = df[args.time_col].dt.year

    train_df, test_df = chronological_split(df, args.time_col, args.test_ratio)
    train_part, valid_part = train_valid_split(train_df, args.time_col, args.valid_ratio)

    thresholds = build_thresholds(train_part[args.target_col])
    train_part = compute_grade_columns(train_part, args.target_col, thresholds, "quantile")
    valid_part = compute_grade_columns(valid_part, args.target_col, thresholds, "quantile")
    test_df = compute_grade_columns(test_df, args.target_col, thresholds, "quantile")

    feature_cols, categorical_cols, dropped_constant_cols = infer_feature_columns(train_part)
    if categorical_cols:
        raise ValueError(f"GAMM control expects numeric features only, got categorical columns: {categorical_cols}")

    train_part = train_part.dropna(subset=feature_cols + ["loss_grade"]).copy()
    valid_part = valid_part.dropna(subset=feature_cols + ["loss_grade"]).copy()
    test_df = test_df.dropna(subset=feature_cols + ["loss_grade"]).copy()

    train_part["cell_id"] = make_spatial_cell_id(train_part, args.lon_bin_width, args.lat_bin_width)
    valid_part["cell_id"] = make_spatial_cell_id(valid_part, args.lon_bin_width, args.lat_bin_width)
    test_df["cell_id"] = make_spatial_cell_id(test_df, args.lon_bin_width, args.lat_bin_width)

    spline_fit_df = sample_frame(train_part, args.spline_fit_sample_size, args.random_state)
    numeric_cols = feature_cols.copy()
    spline = make_spline_transformer(args.n_knots, args.degree)
    spline.fit(numeric_frame(spline_fit_df, numeric_cols))

    encoder = make_one_hot_encoder()
    encoder.fit(train_part[["cell_id"]])

    clf, history, train_monitor, valid_monitor = fit_epochwise_classifier(
        train_df=train_part,
        valid_df=valid_part,
        numeric_cols=numeric_cols,
        cell_col="cell_id",
        spline=spline,
        encoder=encoder,
        random_state=args.random_state,
        epochs=args.epochs,
        batch_size=args.batch_size,
        alpha=args.alpha,
        learning_rate=args.learning_rate,
        eta0=args.eta0,
        early_stop_patience=args.early_stop_patience,
        monitor_sample_size=args.monitor_sample_size,
    )

    train_proba = predict_proba_in_batches(clf, train_part, numeric_cols, "cell_id", spline, encoder, args.batch_size)
    valid_proba = predict_proba_in_batches(clf, valid_part, numeric_cols, "cell_id", spline, encoder, args.batch_size)
    test_proba = predict_proba_in_batches(clf, test_df, numeric_cols, "cell_id", spline, encoder, args.batch_size)

    train_pred = np.argmax(train_proba, axis=1).astype(int)
    valid_pred = np.argmax(valid_proba, axis=1).astype(int)
    test_pred = np.argmax(test_proba, axis=1).astype(int)

    train_metrics = classification_metrics(train_part["loss_grade"].to_numpy(), train_pred, train_proba)
    valid_metrics = classification_metrics(valid_part["loss_grade"].to_numpy(), valid_pred, valid_proba)
    test_metrics = classification_metrics(test_df["loss_grade"].to_numpy(), test_pred, test_proba)

    keep_cols = ["longitude", "latitude", args.target_col, "loss_grade", "cell_id"]
    pred_df = test_df[keep_cols].copy()
    pred_df["pred_grade"] = test_pred
    pred_df["actual_grade_label"] = pred_df["loss_grade"].map(lambda x: GRADE_LABELS[int(x)])
    pred_df["pred_grade_label"] = pred_df["pred_grade"].map(lambda x: GRADE_LABELS[int(x)])
    pred_df["pred_expected_grade"] = expected_grade_from_proba(test_proba)
    pred_df["correct"] = pred_df["pred_grade"] == pred_df["loss_grade"]
    pred_df.to_csv(args.output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    save_fit_curve(history, args.output_dir / "fit_curves.png")
    save_confusion_matrix(test_df["loss_grade"].to_numpy(), test_pred, args.output_dir / "confusion_matrix.png")
    save_confusion_matrix(
        test_df["loss_grade"].to_numpy(),
        test_pred,
        args.output_dir / "confusion_matrix_normalized.png",
        normalize=True,
    )
    class_report = save_classification_report_table(
        test_df["loss_grade"].to_numpy(),
        test_pred,
        args.output_dir / "classification_report.csv",
    )
    save_classification_bar(class_report, args.output_dir / "class_metrics_bar.png")

    feature_table = save_feature_importance(
        clf,
        spline,
        numeric_cols,
        args.output_dir / "feature_importance_gamm.csv",
        args.output_dir / "feature_importance_gamm.png",
    )
    save_smooth_effect_panels(
        spline=spline,
        clf=clf,
        reference_df=train_monitor,
        numeric_cols=numeric_cols,
        feature_table=feature_table,
        output_path=args.output_dir / "smooth_effects_top9_high_class.png",
        class_index=4,
    )

    marine_region_shapes = load_marine_regions(args.marine_regions_dir)
    save_arctic_region_basemap(marine_region_shapes, args.output_dir / "marine_regions_basemap.png")
    save_arctic_spatial_map(
        pred_df,
        value_col="loss_grade",
        output_path=args.output_dir / "test_spatial_actual_arctic.png",
        title="Test Set Spatial Distribution - Actual Grade (Arctic Projection)",
        region_shapes=marine_region_shapes,
        cbar_ticks=list(range(NUM_CLASSES)),
        cbar_ticklabels=GRADE_LABELS,
        cbar_label="Actual Grade",
    )
    save_arctic_spatial_map(
        pred_df,
        value_col="pred_grade",
        output_path=args.output_dir / "test_spatial_predicted_arctic.png",
        title="Test Set Spatial Distribution - Predicted Grade (Arctic Projection)",
        region_shapes=marine_region_shapes,
        cbar_ticks=list(range(NUM_CLASSES)),
        cbar_ticklabels=GRADE_LABELS,
        cbar_label="Predicted Grade",
    )
    save_arctic_spatial_map(
        pred_df.assign(error_class=(pred_df["pred_grade"] != pred_df["loss_grade"]).astype(int)),
        value_col="error_class",
        output_path=args.output_dir / "test_spatial_error_arctic.png",
        title="Test Set Spatial Distribution - Misclassified Points (Arctic Projection)",
        region_shapes=marine_region_shapes,
        cbar_ticks=[0, 1],
        cbar_ticklabels=["Correct", "Misclassified"],
        cbar_label="Point Status",
        vmin=0,
        vmax=1,
        cmap="coolwarm",
    )

    direction_base = test_df[numeric_cols].copy()
    direction_base["pred_expected_grade"] = pred_df["pred_expected_grade"].to_numpy()
    direction_sample = sample_frame(direction_base, min(args.monitor_sample_size, len(direction_base)), args.random_state)
    direction_rows = []
    for feature in numeric_cols:
        feat_values = pd.to_numeric(direction_sample[feature], errors="coerce").to_numpy(dtype=float)
        expected_grade = direction_sample["pred_expected_grade"].to_numpy(dtype=float)
        mask = np.isfinite(feat_values) & np.isfinite(expected_grade)
        corr = float(pd.Series(feat_values[mask]).corr(pd.Series(expected_grade[mask]), method="spearman")) if mask.sum() >= 3 else float("nan")
        direction_rows.append(
            {
                "feature": feature,
                "mean_abs_coef": float(feature_table.loc[feature_table["feature"] == feature, "mean_abs_coef"].iloc[0]),
                "expected_grade_spearman": corr,
                "direction_label": classify_direction(corr, int(mask.sum())),
                "support_n": int(mask.sum()),
            }
        )
    direction_df = pd.DataFrame(direction_rows).sort_values("mean_abs_coef", ascending=False)
    direction_df.to_csv(args.output_dir / "directional_analysis.csv", index=False, encoding="utf-8-sig")

    with open(args.output_dir / "gamm_style_grade_model.pkl", "wb") as f:
        pickle.dump(
            {
                "spline": spline,
                "encoder": encoder,
                "classifier": clf,
                "numeric_cols": numeric_cols,
                "grade_labels": GRADE_LABELS,
                "cell_id_categories": getattr(encoder, "categories_", None),
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    summary = {
        "input_csvs": [str(p) for p in args.input_csvs],
        "sample_cap_per_file": args.sample_cap_per_file,
        "chunksize": args.chunksize,
        "target_col": args.target_col,
        "positive_only": bool(args.positive_only),
        "rows_before_positive_filter": rows_before_positive_filter,
        "rows_after_positive_filter": int(len(df)),
        "rows_dropped_by_positive_filter": int(rows_before_positive_filter - len(df)) if args.positive_only else 0,
        "all_features": numeric_cols,
        "categorical_features": categorical_cols,
        "dropped_constant_features": dropped_constant_cols,
        "grade_labels": GRADE_LABELS,
        "grade_thresholds": thresholds,
        "num_rows_total": int(len(df)),
        "num_rows_train": int(len(train_part)),
        "num_rows_valid": int(len(valid_part)),
        "num_rows_test": int(len(test_df)),
        "train_years": make_time_summary(train_part, args.time_col),
        "valid_years": make_time_summary(valid_part, args.time_col),
        "test_years": make_time_summary(test_df, args.time_col),
        "train_grade_counts": train_part["loss_grade"].value_counts().sort_index().to_dict(),
        "valid_grade_counts": valid_part["loss_grade"].value_counts().sort_index().to_dict(),
        "test_grade_counts": test_df["loss_grade"].value_counts().sort_index().to_dict(),
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "epochs": args.epochs,
        "best_epoch": int(history["epoch"][int(np.argmin(history["valid_logloss"]))]),
        "n_knots": args.n_knots,
        "degree": args.degree,
        "lon_bin_width": args.lon_bin_width,
        "lat_bin_width": args.lat_bin_width,
        "num_spatial_cells_train": int(train_part["cell_id"].nunique()),
        "monitor_train_size": int(len(train_monitor)),
        "monitor_valid_size": int(len(valid_monitor)),
        "feature_importance_top": feature_table.head(8)["feature"].tolist(),
        "direction_top": direction_df[["feature", "direction_label"]].head(8).to_dict("records"),
        "history": history,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "grade_thresholds.json").write_text(
        json.dumps(
            {
                "grade_binning": "quantile",
                "thresholds": thresholds,
                "grade_labels": GRADE_LABELS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_report(args.output_dir, summary, class_report, feature_table)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
