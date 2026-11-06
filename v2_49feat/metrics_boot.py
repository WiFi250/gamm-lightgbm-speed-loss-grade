#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补齐表12/17/18 需要的 Top-2 准确率与 IMO 分组 bootstrap 置信区间。

bootstrap 口径与原记录一致：以 IMO（船舶）为聚类单元、600 次重抽、报告 2.5/97.5 分位。
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

import train_v2 as T
from train_ensemble import PARAMS, qwk
from train_final import ENV, DYNAMICS, SHIP, NAV, PEER
from train_lead import DEPTH_FEATS, LEAD_FEATS
from gamm49 import design, grid_id, F49

BASE = Path("/Users/wangyifan/Documents/seaice/博四/挑战者杯比赛/model_v2")
OUT = BASE / "figs"
SEG = BASE / "data_seg" / "seg_1h_lead3.parquet"
F49L = (SHIP + NAV + ENV
        + ["hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
           "speed_ratio", "power_per_dwt", "dwt_per_l3", "beam_ratio", "mcr_per_len"]
        + DYNAMICS + PEER + LEAD_FEATS + DEPTH_FEATS)
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
    tr, va, te = seg.iloc[:i1], seg.iloc[i1:i2], seg.iloc[i2:]
    th = T.build_thresholds(tr["speed_loss_rate"])
    splits = {"Train": tr, "Valid": va, "Test": te}
    y = {k: T.cut_grades(d["speed_loss_rate"].to_numpy(), th) for k, d in splits.items()}

    # ---- LightGBM ----
    p = dict(PARAMS, objective="multiclass", num_class=5, metric="multi_logloss",
             seed=42, bagging_seed=43, feature_fraction_seed=44)
    bst = lgb.train(p, lgb.Dataset(tr[F49L].astype("float32"), label=y["Train"], free_raw_data=True),
                    num_boost_round=600,
                    valid_sets=[lgb.Dataset(va[F49L].astype("float32"), label=y["Valid"])],
                    callbacks=[lgb.early_stopping(100, verbose=False)])
    best = bst.best_iteration or bst.current_iteration()
    res = {"LightGBM": {}}
    for k, d in splits.items():
        pr = bst.predict(d[F49L].astype("float32"), num_iteration=best)
        m = metrics(y[k], pr.argmax(1), pr)
        m["ci"] = boot(y[k], pr.argmax(1), d["imo"].fillna(-1).to_numpy("int64"))
        res["LightGBM"][k] = m
        print("LGB", k, {kk: round(vv, 4) for kk, vv in m.items() if kk != "ci"}, flush=True)

    # ---- GAMM（12 轮固定，与 49 特征版一致）----
    imp = SimpleImputer(strategy="median"); imp.fit(tr[F49].astype("float64"))
    spl = SplineTransformer(n_knots=6, degree=3, knots="quantile", include_bias=False,
                            extrapolation="constant")
    spl.fit(imp.transform(tr[F49].astype("float64")))
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True); ohe.fit(grid_id(tr))
    Z = {k: design(d, F49, imp, spl, ohe) for k, d in splits.items()}
    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-5, learning_rate="optimal",
                        random_state=42, n_jobs=-1)
    classes = np.arange(5); rng = np.random.RandomState(42)
    for ep in range(12):
        o = rng.permutation(Z["Train"].shape[0])
        for b0 in range(0, len(o), 200_000):
            i = o[b0:b0 + 200_000]
            clf.partial_fit(Z["Train"][i], y["Train"][i], classes=classes)
    res["GAMM"] = {}
    for k in ("Train", "Valid", "Test"):
        pr = clf.predict_proba(Z[k])
        m = metrics(y[k], pr.argmax(1), pr)
        m["ci"] = boot(y[k], pr.argmax(1), splits[k]["imo"].fillna(-1).to_numpy("int64"))
        res["GAMM"][k] = m
        print("GAMM", k, {kk: round(vv, 4) for kk, vv in m.items() if kk != "ci"}, flush=True)

    json.dump(res, open(OUT / "metrics_boot.json", "w"), ensure_ascii=False, indent=2)
    print("写出 figs/metrics_boot.json")


if __name__ == "__main__":
    main()
