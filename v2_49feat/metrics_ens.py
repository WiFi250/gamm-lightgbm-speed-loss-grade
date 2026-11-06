#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定稿模型的 5 种子集成指标 + IMO 分组 bootstrap 置信区间。

与 metrics_boot.py 的区别：LightGBM 那部分是**5 种子概率平均**（定稿口径），
而不是单种子。GAMM 沿用 metrics_boot.json 里同一次运行的结果（GAMM 无随机性）。

输出 figs/metrics_ens.json，结构与 metrics_boot.json 一致，供 make_doc_figs_tpl.py 直接读取。
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import train_v2 as T
from train_ensemble import PARAMS, qwk
from train_final import ENV, DYNAMICS, SHIP, NAV, PEER
from train_lead import DEPTH_FEATS, LEAD_FEATS

BASE = Path("/Users/wangyifan/Documents/seaice/博四/挑战者杯比赛/model_v2")
OUT = BASE / "figs"
SEG = BASE / "data_seg" / "seg_1h_lead3.parquet"
FEATS = (SHIP + NAV + ENV
         + ["hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
            "speed_ratio", "power_per_dwt", "dwt_per_l3", "beam_ratio", "mcr_per_len"]
         + DYNAMICS + PEER + LEAD_FEATS + DEPTH_FEATS)
SEEDS = [42, 7, 2024, 913, 555]
NBOOT = 600


def metrics(y, pred, proba):
    cm = np.bincount(y * 5 + pred, minlength=25).reshape(5, 5).astype(float)
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    f1s = np.where((2 * tp + fp + fn) > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0)
    top2 = (np.argsort(-proba, axis=1)[:, :2] == y[:, None]).any(axis=1).mean()
    return dict(acc=accuracy_score(y, pred), f1=float(f1s.mean()), qwk=qwk(y, pred),
                w1=float((np.abs(y - pred) <= 1).mean()), top2=float(top2))


def boot(y, pred, groups, nboot=NBOOT, seed=42):
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.RandomState(seed)
    out = {k: [] for k in ("acc", "f1", "qwk", "w1")}
    for _ in range(nboot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        yy, pp = y[idx], pred[idx]
        cm = np.bincount(yy * 5 + pp, minlength=25).reshape(5, 5).astype(float)
        tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
        f1s = np.where((2 * tp + fp + fn) > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0)
        out["acc"].append(tp.sum() / len(yy))
        out["f1"].append(float(f1s.mean()))
        out["qwk"].append(qwk(yy, pp))
        out["w1"].append(float((np.abs(yy - pp) <= 1).mean()))
    return {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
            for k, v in out.items()}


def main() -> None:
    seg = pd.read_parquet(SEG).sort_values(["year", "doy"], kind="stable").reset_index(drop=True)
    n = len(seg)
    n_test, n_valid = int(n * 0.2), int((n - int(n * 0.2)) * 0.15)
    i1, i2 = n - n_test - n_valid, n - n_test
    splits = {"Train": seg.iloc[:i1], "Valid": seg.iloc[i1:i2], "Test": seg.iloc[i2:]}
    tr, va = splits["Train"], splits["Valid"]
    th = T.build_thresholds(tr["speed_loss_rate"])
    y = {k: T.cut_grades(d["speed_loss_rate"].to_numpy(), th) for k, d in splits.items()}
    print(f"train={len(tr):,} valid={len(va):,} test={len(splits['Test']):,}", flush=True)

    probas = {k: [] for k in splits}
    for s in SEEDS:
        p = dict(PARAMS, objective="multiclass", num_class=5, metric="multi_logloss",
                 seed=s, bagging_seed=s + 1, feature_fraction_seed=s + 2)
        dtr = lgb.Dataset(tr[FEATS].astype("float32"), label=y["Train"], free_raw_data=True)
        dva = lgb.Dataset(va[FEATS].astype("float32"), label=y["Valid"], reference=dtr,
                          free_raw_data=True)
        b = lgb.train(p, dtr, num_boost_round=600, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        best = b.best_iteration or b.current_iteration()
        for k, d in splits.items():
            probas[k].append(b.predict(d[FEATS].astype("float32"), num_iteration=best))
        print(f"  seed={s} iter={best} QWK={qwk(y['Test'], probas['Test'][-1].argmax(1)):.4f}",
              flush=True)

    res = {"LightGBM": {}, "GAMM": json.load(open(OUT / "metrics_boot.json"))["GAMM"]}
    for k, d in splits.items():
        pr = np.mean(probas[k], axis=0)
        pred = pr.argmax(1)
        m = metrics(y[k], pred, pr)
        m["ci"] = boot(y[k], pred, d["imo"].fillna(-1).to_numpy("int64"))
        res["LightGBM"][k] = m
        print("ENS", k, {kk: round(vv, 4) for kk, vv in m.items() if kk != "ci"}, flush=True)

    json.dump(res, open(OUT / "metrics_ens.json", "w"), ensure_ascii=False, indent=2)
    print("写出 figs/metrics_ens.json")


if __name__ == "__main__":
    main()
