#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""边界诊断 + 代价敏感评估（文献驱动，不改主模型）。

两个产出
--------
1) 逐边界特征重要性。
   Wang, Yu, Kuang & Li (2026, Ocean Engineering, OTCSF) 的核心论点是：
   **有序等级相邻两个边界的判别证据不同**——"开始升险"与"进入高风险"
   由不同特征驱动，单一全局重要性排序会掩盖这一点。
   做法：把五级拆成 4 个累积二分类（G≥2 / G≥3 / G≥4 / G=5），
   各训一个模型，比较每个边界上排名靠前的特征是否不同。

2) 代价敏感指标。
   文献共识：把高风险判成低风险的代价远高于反向（OTCSF 明确提出
   "severe underestimation" 与 regulatory cost-sensitive loss）。
   这里给出：
     - 严重低估率（真实等级 − 预测等级 ≥ 2 的样本占比）
     - 二次代价（Σ(Δ等级)² / n，等价于 1−QWK 的分子部分）
     - 按真实等级的召回表，便于定位短板在哪一级

不训练新主模型，直接复用定稿集成的概率。
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score

import train_v2 as T
from train_ensemble import PARAMS
from train_final import ENV, DYNAMICS, SHIP, NAV, PEER
from train_lead import DEPTH_FEATS, LEAD_FEATS

BASE = Path("/Users/wangyifan/Documents/seaice/博四/挑战者杯比赛/model_v2")
OUT = BASE / "runs_diag"
try:
    OUT.mkdir(exist_ok=True)
except FileExistsError:
    pass
SEG = BASE / "data_seg" / "seg_1h_lead3.parquet"
PROBA = BASE / "runs_lead" / "proba_test_L5.npy"

FEATS = (SHIP + NAV + ENV
         + ["hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
            "speed_ratio", "power_per_dwt", "dwt_per_l3", "beam_ratio", "mcr_per_len"]
         + DYNAMICS + PEER + LEAD_FEATS + DEPTH_FEATS)


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

    # ---------- 1) 逐边界二分类 ----------
    print("=== 逐边界累积二分类 ===", flush=True)
    rows = []
    for k in (1, 2, 3, 4):        # 判 y >= k
        b_tr = (y_tr >= k).astype(int)
        b_va = (y_va >= k).astype(int)
        b_te = (y_te >= k).astype(int)
        pos = b_tr.mean()
        p = dict(PARAMS, objective="binary", metric="binary_logloss",
                 seed=42, bagging_seed=43, feature_fraction_seed=44)
        dtr = lgb.Dataset(tr[FEATS].astype("float32"), label=b_tr, free_raw_data=True)
        dva = lgb.Dataset(va[FEATS].astype("float32"), label=b_va, reference=dtr,
                          free_raw_data=True)
        bst = lgb.train(p, dtr, num_boost_round=400, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(80, verbose=False)])
        best = bst.best_iteration or bst.current_iteration()
        pr = bst.predict(te[FEATS].astype("float32"), num_iteration=best)
        auc = roc_auc_score(b_te, pr)
        gain = bst.feature_importance("gain")
        imp = dict(zip(FEATS, (gain / max(1, gain.sum()) * 100).round(2)))
        top = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:6])
        rows.append({"边界": f"G>={k}", "正类占比": round(float(pos) * 100, 1),
                     "AUC": round(float(auc), 4), "轮数": int(best),
                     "前提特征": ", ".join(f"{k2}({v2:.1f})" for k2, v2 in top.items())})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    pd.DataFrame(rows).to_csv(OUT / "boundary_auc.csv", index=False, encoding="utf-8-sig")

    # ---------- 2) 代价敏感评估（用定稿集成的预测） ----------
    pred = np.load(PROBA).argmax(1)
    d = y_te - pred                      # >0 = 低估
    severe = float(np.mean(d >= 2))
    under = float(np.mean(d > 0))
    over = float(np.mean(d < 0))
    sq = float(np.mean(d.astype(float) ** 2))
    rec = [float(recall_score((y_te == k).astype(int), (pred == k).astype(int), zero_division=0))
           for k in range(5)]
    res = {"n": int(len(y_te)),
           "严重低估率(≥2级)": round(severe * 100, 2),
           "低估率(≥1级)": round(under * 100, 2),
           "高估率(≥1级)": round(over * 100, 2),
           "二次代价(ΣΔ²/n)": round(sq, 4),
           "各级召回": [round(r * 100, 2) for r in rec]}
    print("\n=== 代价敏感评估（定稿集成） ===")
    print(json.dumps(res, ensure_ascii=False))
    (OUT / "cost_metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
