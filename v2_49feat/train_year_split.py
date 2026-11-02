#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按年份严格切分的对照实验（49 项特征，与定稿同特征同参数）。

报告 4.2 写的是"2013—2020 训练 / 2021 验证 / 2022—2023 测试"，
但原脚本 chronological_split 实际是按行数比例切的，三侧的真实年份覆盖是
   train 2013—2021 全年 + 2022 前 35% / valid 2022 中段 / test 2022 末 + 2023 全年。
这里把"报告里写的切分"真正实现一遍，看结果差多少。

输出 runs_year/，供 4.6 节引用。
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import train_v2 as T
from train_ensemble import PARAMS, qwk
from train_final import ENV, DYNAMICS, SHIP, NAV, PEER
from train_lead import DEPTH_FEATS, LEAD_FEATS

BASE = Path("/Users/wangyifan/Documents/seaice/博四/挑战者杯比赛/model_v2")
OUT = BASE / "runs_year"
try:
    OUT.mkdir(exist_ok=True)
except FileExistsError:
    pass
SEG = BASE / "data_seg" / "seg_1h_lead3.parquet"
SEEDS = [42, 7, 2024, 913, 555]

FEATS = (SHIP + NAV + ENV
         + ["hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
            "speed_ratio", "power_per_dwt", "dwt_per_l3", "beam_ratio", "mcr_per_len"]
         + DYNAMICS + PEER + LEAD_FEATS + DEPTH_FEATS)


def run(tag, tr, va, te, n_seeds=5):
    th = T.build_thresholds(tr["speed_loss_rate"])
    y_tr = T.cut_grades(tr["speed_loss_rate"].to_numpy(), th)
    y_va = T.cut_grades(va["speed_loss_rate"].to_numpy(), th)
    y_te = T.cut_grades(te["speed_loss_rate"].to_numpy(), th)
    probas = []
    for s in SEEDS[:n_seeds]:
        p = dict(PARAMS, objective="multiclass", num_class=5, metric="multi_logloss",
                 seed=s, bagging_seed=s + 1, feature_fraction_seed=s + 2)
        dtr = lgb.Dataset(tr[FEATS].astype("float32"), label=y_tr, free_raw_data=True)
        dva = lgb.Dataset(va[FEATS].astype("float32"), label=y_va, reference=dtr,
                          free_raw_data=True)
        b = lgb.train(p, dtr, num_boost_round=600, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        best = b.best_iteration or b.current_iteration()
        probas.append(b.predict(te[FEATS].astype("float32"), num_iteration=best))
    pr = np.mean(probas, axis=0)
    pred = pr.argmax(1)
    ice = te["sic"].to_numpy() >= 80
    res = {"方案": tag,
           "n_train": int(len(tr)), "n_valid": int(len(va)), "n_test": int(len(te)),
           "train年份": f"{int(tr.year.min())}-{int(tr.year.max())}",
           "test年份": f"{int(te.year.min())}-{int(te.year.max())}",
           "阈值": [round(float(x), 4) for x in th],
           "Accuracy": round(accuracy_score(y_te, pred) * 100, 2),
           "MacroF1": round(f1_score(y_te, pred, average="macro", zero_division=0) * 100, 2),
           "QWK": round(qwk(y_te, pred), 4),
           "误差≤1级": round(float(np.mean(np.abs(y_te - pred) <= 1)) * 100, 2),
           "冰区Acc": round(accuracy_score(y_te[ice], pred[ice]) * 100, 2),
           "冰区QWK": round(qwk(y_te[ice], pred[ice]), 4),
           "冰区样本占比": round(float(ice.mean()) * 100, 2),
           "高风险召回": round(float((pred[y_te >= 3] >= 3).mean()) * 100, 2),
           "高风险精确": round(float((y_te[pred >= 3] >= 3).mean()) * 100, 2)}
    print(json.dumps(res, ensure_ascii=False), flush=True)
    np.save(OUT / f"{tag}_proba.npy", pr)
    np.save(OUT / f"{tag}_y.npy", y_te)
    return res


def main() -> None:
    seg = pd.read_parquet(SEG)
    rows = []

    # A) 按行数比例（现有定稿口径）
    a = seg.sort_values(["year", "doy"], kind="stable").reset_index(drop=True)
    n = len(a)
    n_test, n_valid = int(n * 0.2), int((n - int(n * 0.2)) * 0.15)
    i1, i2 = n - n_test - n_valid, n - n_test
    rows.append(run("A_ratio", a.iloc[:i1].copy(), a.iloc[i1:i2].copy(), a.iloc[i2:].copy()))

    # C) 纯随机切分（同船同时段的航段会跨两侧，用于量化泄漏）
    import numpy as _np
    perm = _np.random.RandomState(42).permutation(len(seg))
    rows.append(run("C_random", seg.iloc[perm[:i1]].copy(),
                    seg.iloc[perm[i1:i2]].copy(), seg.iloc[perm[i2:]].copy()))

    # B) 严格按年份：≤2020 训练 / 2021 验证 / ≥2022 测试
    tr = seg[seg.year <= 2020].copy()
    va = seg[seg.year == 2021].copy()
    te = seg[seg.year >= 2022].copy()
    rows.append(run("B_year", tr, va, te))

    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    print("\n===== 切分口径对照（49 项特征，5 种子集成） =====")
    cols = ["方案", "n_train", "n_test", "train年份", "test年份", "Accuracy", "QWK",
            "误差≤1级", "冰区Acc", "冰区QWK", "冰区样本占比", "高风险召回", "高风险精确"]
    print(pd.DataFrame(rows)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
