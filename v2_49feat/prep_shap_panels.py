#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为文档图 4-5 / 4-6 生成绘图数据（在托管 venv 里跑，因为需要 lightgbm）。

输出 figs/shap_panels.npz：
  mean_abs        : 49 项平均绝对 SHAP
  feats           : 特征名顺序
  <feat>__x / __y : 8 个环境变量的抽样散点（x 为取值，y 为对期望等级的 SHAP 贡献）

与 shap49.py 完全同源（同切分、同种子、同早停），保证图 4-5 与图 4-6 一致。
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import train_v2 as T
from train_ensemble import PARAMS
from train_final import ENV, DYNAMICS, SHIP, NAV, PEER
from train_lead import DEPTH_FEATS, LEAD_FEATS

BASE = Path("/Users/wangyifan/Documents/seaice/博四/挑战者杯比赛/model_v2")
OUT = BASE / "figs"
SEG = BASE / "data_seg" / "seg_1h_lead3.parquet"

FEATS = list(SHIP + NAV + ENV
             + ["hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
                "speed_ratio", "power_per_dwt", "dwt_per_l3", "beam_ratio", "mcr_per_len"]
             + DYNAMICS + PEER + LEAD_FEATS + DEPTH_FEATS)

# 图 4-6 的八个面板，顺序与原报告一致
PANELS = ["sic", "sit", "VHM0_m", "VTM10_s", "SST_C", "WindSpeed_ms",
          "Temperature_C", "era5_cloudwater_sw99_visibility_km"]


def main() -> None:
    seg = pd.read_parquet(SEG).sort_values(["year", "doy"], kind="stable").reset_index(drop=True)
    n = len(seg)
    n_test = int(n * 0.2)
    n_valid = int((n - n_test) * 0.15)
    i1, i2 = n - n_test - n_valid, n - n_test
    tr, va, te = seg.iloc[:i1], seg.iloc[i1:i2], seg.iloc[i2:]

    th = T.build_thresholds(tr["speed_loss_rate"])
    y_tr = T.cut_grades(tr["speed_loss_rate"].to_numpy(), th)
    y_va = T.cut_grades(va["speed_loss_rate"].to_numpy(), th)

    p = dict(PARAMS, objective="multiclass", num_class=5, metric="multi_logloss",
             seed=42, bagging_seed=43, feature_fraction_seed=44)
    bst = lgb.train(p, lgb.Dataset(tr[FEATS].astype("float32"), label=y_tr),
                    num_boost_round=600,
                    valid_sets=[lgb.Dataset(va[FEATS].astype("float32"), label=y_va)],
                    callbacks=[lgb.early_stopping(100, verbose=False)])
    best = bst.best_iteration or bst.current_iteration()
    print("best_iter", best, flush=True)

    rng = np.random.RandomState(0)
    idx = np.sort(rng.choice(len(te), size=30_000, replace=False))
    X = te[FEATS].astype("float32").iloc[idx].reset_index(drop=True)
    contrib = np.asarray(bst.predict(X, num_iteration=best, pred_contrib=True))
    k = len(FEATS)
    contrib = contrib.reshape(len(X), 5, k + 1)
    sv = contrib[:, :, :k]

    absmean = np.abs(sv).mean(axis=0).mean(axis=0)
    rank = np.arange(5)[None, :, None]
    sv_rank = (sv * rank).sum(axis=1)          # (n, k) 对期望等级的贡献

    payload = {"mean_abs": absmean.astype("float32"),
               "feats": np.array(FEATS, dtype=object),
               "n_sample": np.array([len(X)])}
    for f in PANELS:
        j = FEATS.index(f)
        x = X[f].to_numpy(dtype="float32")
        y = sv_rank[:, j].astype("float32")
        m = np.isfinite(x) & np.isfinite(y)
        payload[f + "__x"] = x[m]
        payload[f + "__y"] = y[m]
        print(f"  {f}: n={m.sum()}  x[{np.nanmin(x[m]):.2f},{np.nanmax(x[m]):.2f}]", flush=True)

    np.savez_compressed(OUT / "shap_panels.npz", **payload)
    tab = {FEATS[i]: float(absmean[i]) for i in np.argsort(-absmean)}
    json.dump(tab, open(OUT / "shap_panels_mean_abs.json", "w"),
              ensure_ascii=False, indent=2)
    print("已写出 shap_panels.npz；前 10：", list(tab.items())[:10], flush=True)


if __name__ == "__main__":
    main()
