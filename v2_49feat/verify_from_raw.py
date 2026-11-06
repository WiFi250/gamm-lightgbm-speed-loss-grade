#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 data_v2 + data_lead + data_depth 重建航段数据集，并重算 17 特征 SHAP，
与缓存 seg_1h_lead3.parquet 及昨晚的 shap17_mean_abs.json 逐项对比。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

BASE = Path("/Users/wangyifan/Documents/seaice/博四/挑战者杯比赛/model_v2")
sys.path.insert(0, str(BASE))

import train_v2 as T                      # noqa: E402
from train_lead import build_with_lead    # noqa: E402
from train_ensemble import PARAMS         # noqa: E402

CACHE = BASE / "data_seg" / "seg_1h_lead3.parquet"
VERIFY = BASE / "data_seg" / "seg_verify.parquet"
ENV8 = ["sic", "sit", "VHM0_m", "VTM10_s", "WindSpeed_ms", "Temperature_C", "SST_C",
        "era5_cloudwater_sw99_visibility_km"]
SHIP = T.SHIP_FEATURES
NAV = ["longitude", "latitude", "nearest_port_nm"]
FEATS = list(SHIP + NAV + ENV8)
KEY = ["speed_loss_rate", "sic", "sit", "VHM0_m", "WindSpeed_ms", "SST_C",
       "Temperature_C", "nearest_port_nm", "peer_loss_mean"]


def main() -> None:
    t0 = time.time()
    old = pd.read_parquet(CACHE)
    print(f"缓存 seg_1h_lead3: {len(old):,} 行 {len(old.columns)} 列", flush=True)

    print("\n重建中（读 data_v2 + data_lead + data_depth）…", flush=True)
    new = build_with_lead(1)
    print(f"\n重建完成 {len(new):,} 行 {len(new.columns)} 列，用时 {time.time()-t0:.0f}s", flush=True)
    new.to_parquet(VERIFY, index=False, compression="zstd")

    rep = {"rows_cached": int(len(old)), "rows_rebuilt": int(len(new)),
           "cols_identical": list(old.columns) == list(new.columns),
           "cols_only_cached": sorted(set(old.columns) - set(new.columns)),
           "cols_only_rebuilt": sorted(set(new.columns) - set(old.columns))}

    print("\n=== 关键列统计对比（缓存 / 重建）===")
    rows = []
    for c in KEY:
        a, b = old[c], new[c]
        rows.append(dict(
            col=c, mean_old=float(a.mean()), mean_new=float(b.mean()),
            med_old=float(a.median()), med_new=float(b.median()),
            nan_old=float(a.isna().mean()), nan_new=float(b.isna().mean()),
            d_mean=abs(float(a.mean()) - float(b.mean())),
        ))
        print(f"{c:24s} mean {a.mean():>12.5f} / {b.mean():>12.5f}   "
              f"NaN {a.isna().mean()*100:5.2f}% / {b.isna().mean()*100:5.2f}%")
    rep["key_cols"] = rows

    by_year_old = old.groupby("year").size().to_dict()
    by_year_new = new.groupby("year").size().to_dict()
    rep["rows_by_year"] = {int(k): [int(by_year_old.get(k, 0)), int(by_year_new.get(k, 0))]
                           for k in sorted(set(by_year_old) | set(by_year_new))}
    print("\n=== 分年航段数（缓存 / 重建）===")
    for k, v in rep["rows_by_year"].items():
        print(f"{k}  {v[0]:>9,} / {v[1]:>9,}")

    # ---------- 用重建数据训练并重算 SHAP ----------
    print("\n训练 17 特征模型并计算 SHAP …", flush=True)
    seg = new.sort_values(["year", "doy"], kind="stable").reset_index(drop=True)
    n = len(seg)
    n_test, n_valid = int(n * 0.2), int((n - int(n * 0.2)) * 0.15)
    i1, i2 = n - n_test - n_valid, n - n_test
    tr, va, te = seg.iloc[:i1], seg.iloc[i1:i2], seg.iloc[i2:]
    th = T.build_thresholds(tr["speed_loss_rate"])
    y_tr = T.cut_grades(tr["speed_loss_rate"].to_numpy(), th)
    y_va = T.cut_grades(va["speed_loss_rate"].to_numpy(), th)
    p = dict(PARAMS, objective="multiclass", num_class=5, metric="multi_logloss",
             seed=42, bagging_seed=43, feature_fraction_seed=44)
    bst = lgb.train(p, lgb.Dataset(tr[FEATS].astype("float32"), label=y_tr, free_raw_data=True),
                    num_boost_round=600,
                    valid_sets=[lgb.Dataset(va[FEATS].astype("float32"), label=y_va)],
                    callbacks=[lgb.early_stopping(100, verbose=False)])
    best = bst.best_iteration or bst.current_iteration()
    rng = np.random.RandomState(0)
    idx = np.sort(rng.choice(len(te), size=30_000, replace=False))
    X = te[FEATS].astype("float32").iloc[idx].reset_index(drop=True)
    contrib = np.asarray(bst.predict(X, num_iteration=best, pred_contrib=True))
    k = len(FEATS)
    sv = contrib.reshape(len(X), 5, k + 1)[:, :, :k]
    absmean = np.abs(sv).mean(axis=0).mean(axis=0)
    res = {f: float(v) for f, v in sorted(zip(FEATS, absmean), key=lambda x: -x[1])}
    rep["best_iter"] = int(best)
    rep["shap17_rebuilt"] = res

    prev = json.load(open(BASE / "figs" / "shap17_mean_abs.json", encoding="utf-8"))
    print(f"\n=== SHAP 对比（昨晚 / 本次重算），best_iter {best} ===")
    diff = []
    for f in sorted(res, key=lambda x: -res[x]):
        d = res[f] - prev.get(f, float("nan"))
        diff.append(abs(d))
        print(f"{f:40s} {prev.get(f, float('nan')):.4f}  ->  {res[f]:.4f}   Δ{d:+.5f}")
    rep["max_abs_diff"] = float(max(diff))
    print(f"\n最大绝对偏差 {max(diff):.2e}")
    (BASE / "figs" / "shap17_verify.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print("报告写入 figs/shap17_verify.json")


if __name__ == "__main__":
    main()
