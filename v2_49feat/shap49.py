#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重算 49 特征模型的 SHAP（用 LightGBM 自带 TreeSHAP，不依赖 shap 包）。"""
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
ENV8 = ["sic", "sit", "VHM0_m", "VTM10_s", "WindSpeed_ms", "Temperature_C", "SST_C",
        "era5_cloudwater_sw99_visibility_km"]


def main() -> None:
    print("特征数", len(FEATS), flush=True)
    seg = pd.read_parquet(SEG).sort_values(["year", "doy"], kind="stable").reset_index(drop=True)
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
    print("num_feature", bst.num_feature(), "best_iter", best, flush=True)

    rng = np.random.RandomState(0)
    idx = np.sort(rng.choice(len(te), size=40_000, replace=False))
    X = te[FEATS].astype("float32").iloc[idx].reset_index(drop=True)
    contrib = np.asarray(bst.predict(X, num_iteration=best, pred_contrib=True))
    k = len(FEATS)
    print("contrib", contrib.shape, "期望", (len(X), 5 * (k + 1)), flush=True)
    contrib = contrib.reshape(len(X), 5, k + 1)
    sv = contrib[:, :, :k]                     # (n, 5类, k特征)
    absmean = np.abs(sv).mean(axis=0).mean(axis=0)   # (k,)
    print("absmean", absmean.shape, flush=True)

    order = np.argsort(-absmean)
    tab = {FEATS[i]: round(float(absmean[i]), 4) for i in order}
    json.dump(tab, open(OUT / "shap49_mean_abs.json", "w"), ensure_ascii=False, indent=2)
    print("前 12：", list(tab.items())[:12], flush=True)
    print("环境8：", {k2: tab.get(k2) for k2 in ENV8}, flush=True)

    rank = np.arange(5)[None, :, None]
    sv_rank = (sv * rank).sum(axis=1)          # (n, k)
    curves = {}
    for f in ENV8 + ["peer_loss_mean", "water_depth"]:
        j = FEATS.index(f)
        x = X[f].to_numpy()
        m = np.isfinite(x)
        qs = np.unique(np.quantile(x[m], np.linspace(0, 1, 21)))
        bx, by = [], []
        for a, b2 in zip(qs[:-1], qs[1:]):
            sel = m & (x >= a) & (x < b2)
            if sel.sum() < 150:
                continue
            bx.append(float(np.median(x[sel])))
            by.append(float(np.mean(sv_rank[sel, j])))
        curves[f] = {"x": bx, "y": by}
    json.dump(curves, open(OUT / "shap49_curves.json", "w"), ensure_ascii=False, indent=2)
    print("曲线变量：", list(curves), flush=True)


if __name__ == "__main__":
    main()
