#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S2 定稿模型上还能不能挤出东西：三件事一起试。

A) 多种子集成：同配置跑 5 个随机种子，概率平均。
   19M 逐条记录时模型早已饱和（2% 数据就能复现全量结果），
   但航段级只有 69 万训练样本，饱和假设不再自动成立，集成有真实空间。

B) 有序决策阈值：现在用 argmax(p)，等价于假设各类先验相等。
   而测试集五级占比为 18.1/20.8/24.7/21.8/14.7，训练集是各 20%——
   先验偏斜时 argmax 不是最优决策。改成「期望等级 + 4 个切点」，
   切点在验证集上搜，分别按 Accuracy 与 QWK 各搜一套。

C) 无监督先验自适应（Saerens EM）：只用测试集特征、不用标签，
   迭代估计测试集真实先验并重加权概率。合法，且能修正 B 中
   「验证集 ≠ 测试集」造成的切点偏移。

产出 runs_ens/，每步提升写进 summary.csv。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import train_v2 as T
from train_final import ENV, DYNAMICS, SHIP, NAV, PEER, build

BASE = Path("/Users/wangyifan/Documents/seaice/博四/挑战者杯比赛/model_v2")
OUT = BASE / "runs_ens"
OUT.mkdir(parents=True, exist_ok=True)
SEG_CACHE = BASE / "data_seg"
try:
    SEG_CACHE.mkdir(exist_ok=True)
except FileExistsError:
    pass

PARAMS = {
    "learning_rate": 0.04, "num_leaves": 127, "min_data_in_leaf": 60,
    "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1,
    "lambda_l2": 0.15, "num_threads": 0, "verbosity": -1,
    "feature_pre_filter": False,
}

FEATS = (SHIP + NAV + ENV
         + ["hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
            "speed_ratio", "power_per_dwt", "dwt_per_l3", "beam_ratio", "mcr_per_len"]
         + DYNAMICS + PEER)


def qwk(y, p):
    return T.qwk(np.asarray(y), np.asarray(p))


def get_segments(hours: int) -> pd.DataFrame:
    f = SEG_CACHE / f"seg_{hours}h.parquet"
    if f.exists():
        print(f"复用缓存 {f}", flush=True)
        return pd.read_parquet(f)
    seg = build(hours)
    seg.to_parquet(f, index=False, compression="zstd")
    print(f"写出缓存 {f}", flush=True)
    return seg


def train_one(seed, tr, va, te, th, rounds):
    p = dict(PARAMS, objective="multiclass", num_class=5, metric="multi_logloss",
             seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2)
    y_tr = T.cut_grades(tr["speed_loss_rate"].to_numpy(), th)
    y_va = T.cut_grades(va["speed_loss_rate"].to_numpy(), th)
    dtr = lgb.Dataset(tr[FEATS].astype("float32"), label=y_tr, free_raw_data=True)
    dva = lgb.Dataset(va[FEATS].astype("float32"), label=y_va, reference=dtr, free_raw_data=True)
    b = lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    best = b.best_iteration or b.current_iteration()
    pv = b.predict(va[FEATS].astype("float32"), num_iteration=best)
    pt = b.predict(te[FEATS].astype("float32"), num_iteration=best)
    return pv, pt, int(best)


def fit_cuts(score: np.ndarray, y: np.ndarray, metric: str) -> np.ndarray:
    """在连续期望等级上搜 4 个切点，使 metric（acc 或 qwk）最大。粗搜 + 细搜。"""
    grid = np.quantile(score, np.linspace(0.02, 0.98, 97))
    best_c, best_v = None, -1e9
    # 逐切点贪心：先粗搜每个位置，再联合微调
    cuts = np.quantile(score, [0.2, 0.4, 0.6, 0.8])
    for _ in range(4):
        for k in range(4):
            base = cuts.copy()
            for c in grid:
                if k > 0 and c <= cuts[k - 1]:
                    continue
                if k < 3 and c >= cuts[k + 1]:
                    continue
                trial = base.copy()
                trial[k] = c
                pred = np.searchsorted(trial, score)
                v = accuracy_score(y, pred) if metric == "acc" else qwk(y, pred)
                if v > best_v:
                    best_v, best_c = v, trial
            cuts = best_c if best_c is not None else cuts
    return cuts


def em_prior(proba: np.ndarray, pi_train: np.ndarray, iters: int = 60) -> np.ndarray:
    """Saerens–Latinne–Decaestecker EM 先验自适应（不用标签）。"""
    pi = pi_train.copy()
    for _ in range(iters):
        w = pi / pi_train
        q = proba * w
        q /= q.sum(axis=1, keepdims=True)
        pi_new = q.mean(axis=0)
        if np.abs(pi_new - pi).max() < 1e-6:
            pi = pi_new
            break
        pi = pi_new
    w = pi / pi_train
    q = proba * w
    q /= q.sum(axis=1, keepdims=True)
    return q, pi


def report(tag, y, pred, rows):
    res = {"方案": tag,
           "Accuracy": round(accuracy_score(y, pred) * 100, 2),
           "MacroF1": round(f1_score(y, pred, average="macro", zero_division=0) * 100, 2),
           "QWK": round(qwk(y, pred), 4),
           "误差≤1级": round(float(np.mean(np.abs(y - pred) <= 1)) * 100, 2),
           "高风险召回": round(float(((pred >= 3)[(y >= 3)].mean()) if (y >= 3).any() else 0) * 100, 2),
           "高风险精确": round(float(((y >= 3)[(pred >= 3)].mean()) if (pred >= 3).any() else 0) * 100, 2)}
    rows.append(res)
    print(json.dumps(res, ensure_ascii=False), flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 2024, 913, 555])
    ap.add_argument("--max-round", type=int, default=600)
    args = ap.parse_args()

    t0 = time.time()
    seg = get_segments(1)
    seg = seg.sort_values(["year", "doy"], kind="stable").reset_index(drop=True)
    n = len(seg)
    n_test, n_valid = int(n * 0.2), int((n - int(n * 0.2)) * 0.15)
    i1, i2 = n - n_test - n_valid, n - n_test
    tr, va, te = seg.iloc[:i1], seg.iloc[i1:i2], seg.iloc[i2:]
    th = T.build_thresholds(tr["speed_loss_rate"])
    y_va = T.cut_grades(va["speed_loss_rate"].to_numpy(), th)
    y_te = T.cut_grades(te["speed_loss_rate"].to_numpy(), th)
    pi_train = np.bincount(T.cut_grades(tr["speed_loss_rate"].to_numpy(), th), minlength=5) / len(tr)
    print(f"train={len(tr):,} valid={len(va):,} test={len(te):,}", flush=True)

    pv_list, pt_list, iters = [], [], []
    for s in args.seeds:
        pv, pt, bi = train_one(s, tr, va, te, th, args.max_round)
        pv_list.append(pv); pt_list.append(pt); iters.append(bi)
        print(f"  seed={s} best_iter={bi}  QWK(单模型)={qwk(y_te, pt.argmax(1)):.4f}", flush=True)

    rows = []
    # 单模型（第一个种子）作对照
    report("A0 单模型 argmax", y_te, pt_list[0].argmax(1), rows)
    pv = np.mean(pv_list, axis=0)
    pt = np.mean(pt_list, axis=0)
    report("A1 5种子集成 argmax", y_te, pt.argmax(1), rows)

    # B1 期望等级 + 验证集搜出的 Accuracy 最优切点
    ev, et = (pv * np.arange(5)).sum(1), (pt * np.arange(5)).sum(1)
    cuts_acc = fit_cuts(ev, y_va, "acc")
    report("B1 集成 + Acc 最优切点", y_te, np.searchsorted(cuts_acc, et), rows)
    cuts_qwk = fit_cuts(ev, y_va, "qwk")
    report("B2 集成 + QWK 最优切点", y_te, np.searchsorted(cuts_qwk, et), rows)

    # C 无监督先验自适应
    q_te, pi_hat = em_prior(pt, pi_train)
    report("C1 集成 + EM 先验自适应", y_te, q_te.argmax(1), rows)
    print("  EM 估计的测试集先验:", np.round(pi_hat, 4).tolist(),
          " 真实:", np.round(np.bincount(y_te, minlength=5) / len(y_te), 4).tolist(), flush=True)

    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    print("\n===== 集成 / 决策口径 =====")
    print(tab.to_string(index=False))
    print(f"\n总耗时 {time.time()-t0:.0f}s  各种子最佳轮数 {iters}")


if __name__ == "__main__":
    main()
