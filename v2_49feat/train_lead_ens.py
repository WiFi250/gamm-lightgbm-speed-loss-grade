#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L1（S2 + 前方环境特征）的 5 种子集成，取定稿数字。

复用 data_seg/seg_1h_lead.parquet，只训练、不重建。
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import train_v2 as T
from train_ensemble import PARAMS, FEATS as FEATS_S2, qwk
from train_lead import LEAD_FEATS, DEPTH_FEATS

BASE = Path("/Users/wangyifan/Documents/seaice/博四/挑战者杯比赛/model_v2")
OUT = BASE / "runs_lead"
OUT.mkdir(parents=True, exist_ok=True)
SEG = BASE / "data_seg" / "seg_1h_lead3.parquet"
FEATS = FEATS_S2 + LEAD_FEATS + DEPTH_FEATS
SEEDS = [42, 7, 2024, 913, 555]


def main() -> None:
    seg = pd.read_parquet(SEG).sort_values(["year", "doy"], kind="stable").reset_index(drop=True)
    n = len(seg)
    n_test, n_valid = int(n * 0.2), int((n - int(n * 0.2)) * 0.15)
    i1, i2 = n - n_test - n_valid, n - n_test
    tr, va, te = seg.iloc[:i1], seg.iloc[i1:i2], seg.iloc[i2:]
    th = T.build_thresholds(tr["speed_loss_rate"])
    y_tr = T.cut_grades(tr["speed_loss_rate"].to_numpy(), th)
    y_va = T.cut_grades(va["speed_loss_rate"].to_numpy(), th)
    y_te = T.cut_grades(te["speed_loss_rate"].to_numpy(), th)
    print(f"train={len(tr):,} valid={len(va):,} test={len(te):,}", flush=True)

    pts = []
    for s in SEEDS:
        p = dict(PARAMS, objective="multiclass", num_class=5, metric="multi_logloss",
                 seed=s, bagging_seed=s + 1, feature_fraction_seed=s + 2)
        dtr = lgb.Dataset(tr[FEATS].astype("float32"), label=y_tr, free_raw_data=True)
        dva = lgb.Dataset(va[FEATS].astype("float32"), label=y_va, reference=dtr, free_raw_data=True)
        b = lgb.train(p, dtr, num_boost_round=600, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        best = b.best_iteration or b.current_iteration()
        pr = b.predict(te[FEATS].astype("float32"), num_iteration=best)
        pts.append(pr)
        print(f"  seed={s} iter={best} QWK={qwk(y_te, pr.argmax(1)):.4f}  "
              f"Acc={accuracy_score(y_te, pr.argmax(1))*100:.2f}", flush=True)

    pt = np.mean(pts, axis=0)
    pred = pt.argmax(1)
    ice = te["sic"].to_numpy() >= 80
    res = {"方案": "L5 定稿：S2+lead+depth 5种子", "特征数": len(FEATS),
           "Accuracy": round(accuracy_score(y_te, pred) * 100, 2),
           "MacroF1": round(f1_score(y_te, pred, average="macro", zero_division=0) * 100, 2),
           "QWK": round(qwk(y_te, pred), 4),
           "误差≤1级": round(float(np.mean(np.abs(y_te - pred) <= 1)) * 100, 2),
           "冰区Acc": round(accuracy_score(y_te[ice], pred[ice]) * 100, 2),
           "冰区QWK": round(qwk(y_te[ice], pred[ice]), 4),
           "高风险召回": round(float((pred[ice | ~ice] >= 3)[y_te >= 3].mean()) * 100, 2),
           "高风险精确": round(float((y_te >= 3)[pred >= 3].mean()) * 100, 2)}
    print(json.dumps(res, ensure_ascii=False), flush=True)
    (OUT / "L5_ensemble.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    pd.DataFrame([res]).to_csv(OUT / "L2_ensemble.csv", index=False, encoding="utf-8-sig")
    np.save(OUT / "proba_test_L5.npy", pt)


if __name__ == "__main__":
    main()
