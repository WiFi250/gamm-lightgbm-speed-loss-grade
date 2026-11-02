#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""船速损失等级模型 —— 对照实验版（2026 挑战杯赛前改进）

与原脚本 lightgbm_allvars_speed_loss_grade_multiyear.py 的差异：
  1. 时间切分改为严格按年份：train <= 2020 / valid = 2021 / test >= 2022
     （原脚本 chronological_split 按行数比例切，只是碰巧近似年份边界）
  2. 新增季节特征 month_sin / month_cos / doy_sin / doy_cos（原脚本 datetime 全被排除，
     模型完全不知道月份）
  3. 早停指标可选 QWK（原脚本用 multi_logloss 早停，但汇报 accuracy/QWK，口径不一致）
  4. 新增 regression 目标（预测连续 speed_loss_rate，再用训练集分位阈值切级），
     利用 G1-G5 的有序性
  5. 新增 noenv 消融（去掉 8 个环境变量，回答"环境到底贡献多少"）
  6. 新增 binary 高风险口径（G4+G5 vs 其余）

所有实验共用同一套训练集分位阈值、同一测试集，指标可直接对比。
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

DATA = Path("/Users/wangyifan/Documents/seaice/博四/挑战者杯比赛/model_v2/data")
OUT = Path("/Users/wangyifan/Documents/seaice/博四/挑战者杯比赛/model_v2/runs")
NUM_CLASSES = 5

# 原始 17 项特征（与报告 2.1 节一致）
SHIP_FEATURES = ["DWT", "Length", "Width", "Maxspeed", "MCR", "reference_speed"]
NAV_FEATURES = ["longitude", "latitude", "nearest_port_nm"]
ENV_FEATURES = [
    "VHM0_m", "VTM10_s", "WindSpeed_ms", "Temperature_C",
    "SST_C", "era5_cloudwater_sw99_visibility_km", "sic", "sit",
]
SEASON_FEATURES = ["month_sin", "month_cos", "doy_sin", "doy_cos"]


# ----------------------------------------------------------------------------
# 数据
# ----------------------------------------------------------------------------
def load_years(years: list[int], use_season: bool) -> pd.DataFrame:
    frame = []
    for y in years:
        p = DATA / f"cruising_{y}.parquet"
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_parquet(p)
        # 只保留正损失样本（与报告口径一致：2868 万条）
        df = df[df["speed_loss_rate"] > 0]
        # 季节特征：月份/年积日在圆周上的正余弦编码
        if use_season:
            df["month_sin"] = np.sin(2 * np.pi * df["month"].astype("float32") / 12.0)
            df["month_cos"] = np.cos(2 * np.pi * df["month"].astype("float32") / 12.0)
            df["doy_sin"] = np.sin(2 * np.pi * df["doy"].astype("float32") / 365.25)
            df["doy_cos"] = np.cos(2 * np.pi * df["doy"].astype("float32") / 365.25)
        # 释放不再需要的列，压低 16 GB 机器上的峰值内存
        # 保留 year / doy 作为排序键，切分完成后再删
        drop = [c for c in ("acq_time", "month") if c in df.columns]
        df = df.drop(columns=drop)
        frame.append(df)
    df = pd.concat(frame, ignore_index=True)
    del frame
    return df


def build_thresholds(train_target: pd.Series) -> np.ndarray:
    q = np.linspace(0.0, 1.0, NUM_CLASSES + 1)
    th = train_target.quantile(q).to_numpy(dtype=float)
    # 防重复阈值
    th = np.maximum.accumulate(th)
    for i in range(1, len(th)):
        if th[i] <= th[i - 1]:
            th[i] = np.nextafter(th[i - 1], np.inf)
    return th


def cut_grades(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    g = np.searchsorted(thresholds[1:-1], values, side="right").astype(int)
    return np.clip(g, 0, NUM_CLASSES - 1)


# ----------------------------------------------------------------------------
# 指标
# ----------------------------------------------------------------------------
def qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def feval_qwk_factory(num_class: int):
    """LightGBM 自定义评估函数：用 QWK 做早停，和汇报口径对齐。"""

    def feval(preds: np.ndarray, eval_data: lgb.Dataset) -> tuple[str, float, bool]:
        y = eval_data.get_label().astype(int)
        p = np.asarray(preds)
        if p.ndim == 1:
            p = p.reshape(-1, num_class)
        return "qwk", qwk(y, p.argmax(axis=1)), True

    return feval


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None = None,
             ice_flag: np.ndarray | None = None) -> dict:
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "qwk": qwk(y_true, y_pred),
        "within_one": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        "mae_class": float(np.mean(np.abs(y_true - y_pred))),
    }
    if proba is not None:
        top2 = np.argpartition(-proba, kth=1, axis=1)[:, :2]
        m["top2"] = float(np.mean([yt in row for yt, row in zip(y_true, top2, strict=False)]))
        exp = (proba * np.arange(proba.shape[1])).sum(axis=1)
        m["expected_mae"] = float(np.mean(np.abs(y_true - exp)))
    # 高风险口径：G4+G5 合并
    hi_true = (y_true >= 3).astype(int)
    hi_pred = (y_pred >= 3).astype(int)
    m["hi_recall"] = float(recall_score(hi_true, hi_pred, zero_division=0))
    m["hi_precision"] = float(precision_score(hi_true, hi_pred, zero_division=0))
    m["hi_f1"] = float(f1_score(hi_true, hi_pred, zero_division=0))
    # 分域评估：把"是否有冰"拆开看，回答"模型在冰区到底准不准"
    if ice_flag is not None:
        for tag, sel in (("ice", ice_flag), ("noice", ~ice_flag)):
            if sel.sum() < 1000:
                continue
            m[f"{tag}_n"] = int(sel.sum())
            m[f"{tag}_acc"] = float(accuracy_score(y_true[sel], y_pred[sel]))
            m[f"{tag}_qwk"] = qwk(y_true[sel], y_pred[sel])
            m[f"{tag}_within1"] = float(np.mean(np.abs(y_true[sel] - y_pred[sel]) <= 1))
            m[f"{tag}_hi_recall"] = float(
                recall_score((y_true[sel] >= 3).astype(int), (y_pred[sel] >= 3).astype(int),
                             zero_division=0))
    return m


# ----------------------------------------------------------------------------
# 训练
# ----------------------------------------------------------------------------
@dataclass
class Config:
    name: str
    mode: str = "multiclass"        # multiclass | regression | binary
    use_season: bool = False
    use_env: bool = True
    stop_metric: str = "logloss"    # logloss | qwk
    num_boost_round: int = 1200
    learning_rate: float = 0.04
    num_leaves: int = 127
    extra: dict = field(default_factory=dict)


def feature_cols(cfg: Config) -> list[str]:
    cols = list(SHIP_FEATURES) + list(NAV_FEATURES)
    if cfg.use_env:
        cols += list(ENV_FEATURES)
    if cfg.use_season:
        cols += list(SEASON_FEATURES)
    return cols


def train_one(cfg: Config, train: pd.DataFrame, valid: pd.DataFrame,
              test: pd.DataFrame, thresholds: np.ndarray) -> dict:
    cols = feature_cols(cfg)
    X_tr = train[cols].astype("float32")
    X_va = valid[cols].astype("float32")
    X_te = test[cols].astype("float32")

    y_grade_tr = cut_grades(train["speed_loss_rate"].to_numpy(), thresholds)
    y_grade_va = cut_grades(valid["speed_loss_rate"].to_numpy(), thresholds)
    y_grade_te = cut_grades(test["speed_loss_rate"].to_numpy(), thresholds)

    if cfg.mode == "regression":
        y_tr, y_va = train["speed_loss_rate"].to_numpy("float32"), valid["speed_loss_rate"].to_numpy("float32")
        params = {
            "objective": "regression", "metric": "l2",
            "learning_rate": cfg.learning_rate, "num_leaves": cfg.num_leaves,
            "min_data_in_leaf": 60, "feature_fraction": 0.9,
            "bagging_fraction": 0.9, "bagging_freq": 1,
            "lambda_l2": 0.15, "seed": 42, "num_threads": 0,
            "verbosity": -1, "feature_pre_filter": False,
        }
        feval = None
    elif cfg.mode == "binary":
        y_tr = (y_grade_tr >= 3).astype(int)
        y_va = (y_grade_va >= 3).astype(int)
        params = {
            "objective": "binary", "metric": "binary_logloss",
            "learning_rate": cfg.learning_rate, "num_leaves": cfg.num_leaves,
            "min_data_in_leaf": 60, "feature_fraction": 0.9,
            "bagging_fraction": 0.9, "bagging_freq": 1,
            "lambda_l2": 0.15, "seed": 42, "num_threads": 0,
            "verbosity": -1, "feature_pre_filter": False,
            "scale_pos_weight": float((y_tr == 0).sum() / max(1, (y_tr == 1).sum())),
        }
        feval = None
    else:
        y_tr, y_va = y_grade_tr, y_grade_va
        params = {
            "objective": "multiclass", "num_class": NUM_CLASSES,
            "metric": ["multi_logloss", "multi_error"],
            "learning_rate": cfg.learning_rate, "num_leaves": cfg.num_leaves,
            "min_data_in_leaf": 60, "feature_fraction": 0.9,
            "bagging_fraction": 0.9, "bagging_freq": 1,
            "lambda_l2": 0.15, "seed": 42, "num_threads": 0,
            "verbosity": -1, "feature_pre_filter": False,
        }
        feval = feval_qwk_factory(NUM_CLASSES) if cfg.stop_metric == "qwk" else None
        if feval is not None:
            # 让 QWK 成为唯一评估指标，否则 early_stopping 会先被 multi_logloss 触发，
            # 用 QWK 早停就形同虚设
            params["metric"] = "None"

    dtr = lgb.Dataset(X_tr, label=y_tr, free_raw_data=True)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr, free_raw_data=True)

    t0 = time.time()
    callbacks = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(50)]
    booster = lgb.train(
        params, dtr, num_boost_round=cfg.num_boost_round,
        valid_sets=[dva], valid_names=["valid"],
        feval=feval, callbacks=callbacks,
    )
    train_sec = time.time() - t0
    best_iter = booster.best_iteration or booster.current_iteration()

    # ---- 预测 ----
    if cfg.mode == "regression":
        pred_te = booster.predict(X_te, num_iteration=best_iter)
        y_pred = cut_grades(pred_te, thresholds)
        proba = None
    elif cfg.mode == "binary":
        p_te = booster.predict(X_te, num_iteration=best_iter)
        y_pred = np.where(p_te >= 0.5, 4, 0)  # 仅供二分类指标，五级指标不计
        proba = None
    else:
        proba = booster.predict(X_te, num_iteration=best_iter)
        y_pred = proba.argmax(axis=1)

    res = {"name": cfg.name, "mode": cfg.mode, "best_iter": int(best_iter),
           "train_sec": round(train_sec, 1), "n_features": len(cols)}
    if cfg.mode == "binary":
        hi_true = (y_grade_te >= 3).astype(int)
        res.update({
            "hi_recall": float(recall_score(hi_true, p_te >= 0.5, zero_division=0)),
            "hi_precision": float(precision_score(hi_true, p_te >= 0.5, zero_division=0)),
            "hi_f1": float(f1_score(hi_true, p_te >= 0.5, zero_division=0)),
            "hi_auc": None,
        })
    else:
        res.update(evaluate(y_grade_te, y_pred, proba,
                            (test["sic"].to_numpy() >= 80)))

    res["confusion"] = confusion_matrix(y_grade_te, y_pred, labels=list(range(NUM_CLASSES))).tolist() \
        if cfg.mode == "multiclass" else None
    return res


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="+", default=["all"])
    ap.add_argument("--max-round", type=int, default=1200)
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="对训练集随机下采样比例，仅用于快速验证管线")
    ap.add_argument("--split", choices=["ratio", "year", "random", "group"],
                    default="ratio",
                    help="ratio=按时间顺序切（与报告一致）；year=严格按年份；"
                         "random=纯随机切（有同航段/同船泄漏）；group=按 IMO 分组随机（无同船泄漏）")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    print("加载数据 ...", flush=True)
    alldf = load_years(list(range(2013, 2024)), use_season=True)
    print(f"total={len(alldf):,}", flush=True)

    if args.split == "ratio":
        # 与已提交报告一致：按时间排序后按行数比例切分
        # test 20%，valid 占训练部分的 15% —— 复现报告里的 1928万/340万/568万
        alldf = alldf.sort_values(["year", "doy"], kind="stable").reset_index(drop=True)
        n = len(alldf)
        n_test = int(n * 0.2)
        n_valid = int((n - n_test) * 0.15)
        train = alldf.iloc[: n - n_test - n_valid].copy()
        valid = alldf.iloc[n - n_test - n_valid: n - n_test].copy()
        test = alldf.iloc[n - n_test:].copy()
        del alldf
        test_span = (int(test["year"].min()), int(test["year"].max()))
    elif args.split == "year":
        yr = alldf["year"]
        train = alldf[yr <= 2020].copy()
        valid = alldf[yr == 2021].copy()
        test = alldf[yr >= 2022].copy()
        test_span = (int(test["year"].min()), int(test["year"].max()))
        del alldf, yr
    elif args.split == "random":
        import numpy as _np
        n = len(alldf)
        n_test = int(n * 0.2)
        n_valid = int((n - n_test) * 0.15)
        perm = _np.random.RandomState(42).permutation(n)
        test = alldf.iloc[perm[:n_test]].copy()
        valid = alldf.iloc[perm[n_test:n_test + n_valid]].copy()
        train = alldf.iloc[perm[n_test + n_valid:]].copy()
        test_span = (int(test["year"].min()), int(test["year"].max()))
        del alldf
    else:  # group：按 IMO 分组随机，同一艘船只出现在一侧
        import numpy as _np
        imos = alldf["imo"].fillna(-1).astype("int64").to_numpy()
        uniq = _np.unique(imos)
        _np.random.RandomState(42).shuffle(uniq)
        n_g = len(uniq)
        n_test_g = max(1, int(round(n_g * 0.35)))
        g_test = uniq[:n_test_g]          # 35% 的船留出
        m_test = _np.isin(imos, g_test)
        train = alldf[~m_test].copy()
        held = alldf[m_test].copy()
        # 验证集从留出船里随机抽 20%：保证 valid 与 test 同分布，
        # 否则按船分组后 valid 的冰区样本几乎为零（实测 0.90%），早停会失效。
        # valid 仅用于挑迭代轮数，最终指标全部在剩余 test 上算。
        _p = _np.random.RandomState(7).permutation(len(held))
        n_v = int(len(held) * 0.2)
        valid = held.iloc[_p[:n_v]].copy()
        test = held.iloc[_p[n_v:]].copy()
        print(f"group 切分：船 {n_g} 艘，留出 {n_test_g} 艘 -> "
              f"train {len(train):,} / valid {len(valid):,} / test {len(test):,}",
              flush=True)
        del held
        test_span = (int(test["year"].min()), int(test["year"].max()))
        del alldf, imos

    train = train.drop(columns=[c for c in ("year", "doy", "imo") if c in train.columns])
    valid = valid.drop(columns=[c for c in ("year", "doy", "imo") if c in valid.columns])
    test = test.drop(columns=[c for c in ("year", "doy", "imo") if c in test.columns])
    print(f"split={args.split}  test 年份跨度={test_span}", flush=True)
    print(f"train={len(train):,} valid={len(valid):,} test={len(test):,}", flush=True)
    print("冰区样本占比  train=%.2f%%  valid=%.2f%%  test=%.2f%%" % (
        (train["sic"] >= 80).mean() * 100, (valid["sic"] >= 80).mean() * 100,
        (test["sic"] >= 80).mean() * 100), flush=True)

    if args.train_frac < 1.0:
        train = train.sample(frac=args.train_frac, random_state=42).reset_index(drop=True)
        print(f"下采样后 train={len(train):,}", flush=True)

    # 阈值只由训练集决定，再原样用于验证与测试
    thresholds = build_thresholds(train["speed_loss_rate"])
    print("thresholds", np.round(thresholds, 6).tolist(), flush=True)
    np.save(OUT / "thresholds.npy", thresholds)

    configs = [
        Config("A_baseline", mode="multiclass", use_season=False, use_env=True,
               stop_metric="logloss", num_boost_round=args.max_round),
        Config("B_season", mode="multiclass", use_season=True, use_env=True,
               stop_metric="logloss", num_boost_round=args.max_round),
        Config("C_season_qwkstop", mode="multiclass", use_season=True, use_env=True,
               stop_metric="qwk", num_boost_round=args.max_round),
        Config("D_regression", mode="regression", use_season=True, use_env=True,
               num_boost_round=args.max_round),
        Config("E_noenv_ablation", mode="multiclass", use_season=True, use_env=False,
               stop_metric="logloss", num_boost_round=args.max_round),
        Config("F_binary_highrisk", mode="binary", use_season=True, use_env=True,
               num_boost_round=args.max_round),
    ]
    if args.experiments != ["all"]:
        configs = [c for c in configs if c.name in args.experiments]

    results = []
    for cfg in configs:
        print(f"\n=== {cfg.name} ({cfg.mode}, season={cfg.use_season}, env={cfg.use_env}) ===", flush=True)
        try:
            r = train_one(cfg, train, valid, test, thresholds)
        except Exception as exc:  # noqa: BLE001
            print("FAILED", cfg.name, repr(exc), flush=True)
            results.append({"name": cfg.name, "error": repr(exc)})
            continue
        print(json.dumps({k: v for k, v in r.items() if k != "confusion"},
                         ensure_ascii=False), flush=True)
        results.append(r)
        (OUT / f"{cfg.name}.json").write_text(json.dumps(r, ensure_ascii=False, indent=2))

    (OUT / "all_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))

    # 汇总表
    rows = []
    for r in results:
        if "error" in r:
            continue
        rows.append({
            "实验": r["name"],
            "Accuracy": round(r.get("accuracy", np.nan), 4),
            "MacroF1": round(r.get("macro_f1", np.nan), 4),
            "QWK": round(r.get("qwk", np.nan), 4),
            "WithinOne": round(r.get("within_one", np.nan), 4),
            "高风险召回": round(r.get("hi_recall", np.nan), 4),
            "高风险精确": round(r.get("hi_precision", np.nan), 4),
            "最佳轮数": r["best_iter"],
            "特征数": r["n_features"],
            "耗时s": r["train_sec"],
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    print("\n===== 汇总 =====")
    print(table.to_string(index=False))
    print(f"\n总耗时 {round(time.time() - t_all, 1)} s")


if __name__ == "__main__":
    main()
