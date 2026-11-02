#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GAMM（等效拆解）在 49 项特征上的对照。

依据《算法模型及运行记录》中的口径：GAMM 并非直接调用 mgcv::gamm，
而是用「连续变量样条平滑项 + 经纬度网格独热（空间组效应）+ 线性分类器」等效实现。
这里沿用该结构，把特征集从 17 项扩到 49 项，与 LightGBM 用完全相同的切分、阈值和样本。

实现：
  连续项 → SplineTransformer(n_knots=6, degree=3, knots="quantile")
  空间项 → 2.5°×10° 网格编号 OneHot
  线性层 → SGDClassifier(log_loss, alpha=1e-5, learning_rate="optimal")，五类 OvA
超参与原仓库 gamm_allvars_*.py 的默认值一致，以保证与报告中的 GAMM 口径可比。
缺失值用训练集中位数填补（样条不接受 NaN，这是它与 LightGBM 的固有差别，需在报告中说明）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

import train_v2 as T
from train_ensemble import qwk
from train_final import ENV, DYNAMICS, SHIP, NAV, PEER
from train_lead import DEPTH_FEATS, LEAD_FEATS

BASE = Path("/Users/wangyifan/Documents/seaice/博四/挑战者杯比赛/model_v2")
OUT = BASE / "runs_gamm" if "--year" not in __import__("sys").argv else BASE / "runs_gamm_year"
try:
    OUT.mkdir(exist_ok=True)
except FileExistsError:
    pass
SEG = BASE / "data_seg" / "seg_1h_lead3.parquet"

F17 = SHIP + NAV + ENV
F49 = (SHIP + NAV + ENV
       + ["hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
          "speed_ratio", "power_per_dwt", "dwt_per_l3", "beam_ratio", "mcr_per_len"]
       + DYNAMICS + PEER + LEAD_FEATS + DEPTH_FEATS)
CHUNK = 120_000


def grid_id(df: pd.DataFrame):
    latb = np.floor(df["latitude"].to_numpy() / 2.5).astype(int) + 45
    lonb = np.floor(df["longitude"].to_numpy() / 10.0).astype(int) + 36
    return (latb * 100 + lonb).reshape(-1, 1)


def design(df, feats, imp, spl, ohe):
    """分块生成稀疏设计矩阵，控制内存峰值。"""
    blocks = []
    n = len(df)
    for s in range(0, n, CHUNK):
        sub = df.iloc[s:s + CHUNK]
        X = imp.transform(sub[feats].astype("float64"))
        Z = spl.transform(X)
        G = ohe.transform(grid_id(sub))
        blocks.append(sparse.hstack([sparse.csr_matrix(Z), G], format="csr"))
    return sparse.vstack(blocks, format="csr")


def evaluate(tag, y, pred, ice):
    res = {"方案": tag,
           "Accuracy": round(accuracy_score(y, pred) * 100, 2),
           "MacroF1": round(f1_score(y, pred, average="macro", zero_division=0) * 100, 2),
           "QWK": round(qwk(y, pred), 4),
           "误差≤1级": round(float(np.mean(np.abs(y - pred) <= 1)) * 100, 2),
           "冰区Acc": round(accuracy_score(y[ice], pred[ice]) * 100, 2),
           "冰区QWK": round(qwk(y[ice], pred[ice]), 4)}
    print(json.dumps(res, ensure_ascii=False), flush=True)
    return res


def run(tag, feats, tr, va, te, th, y_tr, y_va, y_te, ice):
    t0 = time.time()
    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(tr[feats].astype("float64"))
    spl = SplineTransformer(n_knots=6, degree=3, knots="quantile", include_bias=False,
                            extrapolation="constant")
    spl.fit(Xtr)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    ohe.fit(grid_id(tr))
    del Xtr

    Ztr = design(tr, feats, imp, spl, ohe)
    Zva = design(va, feats, imp, spl, ohe)
    Zte = design(te, feats, imp, spl, ohe)
    print(f"  设计矩阵 {Ztr.shape}  非零占比 {Ztr.nnz/Ztr.shape[0]/Ztr.shape[1]*100:.1f}%  "
          f"构建 {time.time()-t0:.0f}s", flush=True)

    # alpha 与原脚本一致固定 1e-5；轮数改为按验证集对数损失早停，
    # 因为硬性 6 轮会欠拟合（实测 GAMM 在 6 轮时 QWK 仅 0.365，明显未收敛）。
    alpha, max_epochs, patience = 1e-5, 30, 3
    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=alpha,
                        learning_rate="optimal", random_state=42, n_jobs=-1)
    classes = np.arange(5)
    best_ll, best_ep, wait = np.inf, 0, 0
    rng = np.random.RandomState(42)
    for ep in range(1, max_epochs + 1):
        order = rng.permutation(Ztr.shape[0])
        for b0 in range(0, len(order), 200_000):
            idx = order[b0:b0 + 200_000]
            clf.partial_fit(Ztr[idx], y_tr[idx], classes=classes)
        pv = clf.predict_proba(Zva)
        ll = -np.mean(np.log(np.clip(pv[np.arange(len(y_va)), y_va], 1e-9, 1)))
        if ll < best_ll - 1e-5:
            best_ll, best_ep, wait = ll, ep, 0
        else:
            wait += 1
        if ep % 3 == 0 or wait == patience:
            print(f"    epoch {ep:2d}  验证对数损失 {ll:.4f}  最好 {best_ll:.4f} @ {best_ep}",
                  flush=True)
        if wait >= patience:
            break
    print(f"  早停于第 {ep} 轮（最好第 {best_ep} 轮）  验证 QWK={qwk(y_va, clf.predict(Zva)):.4f}",
          flush=True)
    pred = clf.predict(Zte)
    res = evaluate(tag, y_te, pred, ice)
    res.update({"alpha": alpha, "n_design": int(Ztr.shape[1]),
                "epochs": int(best_ep), "sec": round(time.time() - t0, 1)})
    np.save(OUT / f"{tag}_pred.npy", pred)
    return res


def main() -> None:
    import sys
    year_split = "--year" in sys.argv
    seg = pd.read_parquet(SEG).sort_values(["year", "doy"], kind="stable").reset_index(drop=True)
    if year_split:
        tr = seg[seg.year <= 2020].copy()
        va = seg[seg.year == 2021].copy()
        te = seg[seg.year >= 2022].copy()
        print("[按年份切分]", flush=True)
    else:
        n = len(seg)
        n_test, n_valid = int(n * 0.2), int((n - int(n * 0.2)) * 0.15)
        i1, i2 = n - n_test - n_valid, n - n_test
        tr, va, te = seg.iloc[:i1], seg.iloc[i1:i2], seg.iloc[i2:]
    th = T.build_thresholds(tr["speed_loss_rate"])
    y_tr = T.cut_grades(tr["speed_loss_rate"].to_numpy(), th)
    y_va = T.cut_grades(va["speed_loss_rate"].to_numpy(), th)
    y_te = T.cut_grades(te["speed_loss_rate"].to_numpy(), th)
    ice = te["sic"].to_numpy() >= 80
    print(f"train={len(tr):,} valid={len(va):,} test={len(te):,}", flush=True)

    rows = [run("GAMM17", F17, tr, va, te, th, y_tr, y_va, y_te, ice),
            run("GAMM49", F49, tr, va, te, th, y_tr, y_va, y_te, ice)]
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    print("\n===== GAMM 等效实现：17 项 vs 49 项 =====")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
