#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""49 特征完整模型的多种子重复：给 PPT 图 (a) 的深色条配误差范围。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

BASE = Path("/Users/wangyifan/Documents/seaice/博四/挑战者杯比赛/model_v2")
sys.path.insert(0, str(BASE))

import train_v2 as T                # noqa: E402
from train_ensemble import PARAMS   # noqa: E402
from train_final import ENV, DYNAMICS, SHIP, NAV, PEER   # noqa: E402
from train_lead import DEPTH_FEATS, LEAD_FEATS           # noqa: E402

SEG = BASE / "data_seg" / "seg_1h_lead3.parquet"
FEATS = list(SHIP + NAV + ENV
             + ["hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
                "speed_ratio", "power_per_dwt", "dwt_per_l3", "beam_ratio", "mcr_per_len"]
             + DYNAMICS + PEER + LEAD_FEATS + DEPTH_FEATS)
SEEDS = [42, 43, 44]


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

    Xtr = tr[FEATS].astype("float32")
    Xva = va[FEATS].astype("float32")
    rng = np.random.RandomState(0)
    idx = np.sort(rng.choice(len(te), size=30_000, replace=False))
    Xte = te[FEATS].astype("float32").iloc[idx].reset_index(drop=True)
    del seg, tr, va
    k = len(FEATS)

    out = {}
    for s in SEEDS:
        p = dict(PARAMS, objective="multiclass", num_class=5, metric="multi_logloss",
                 seed=s, bagging_seed=s + 1, feature_fraction_seed=s + 2)
        bst = lgb.train(p, lgb.Dataset(Xtr, label=y_tr, free_raw_data=True),
                        num_boost_round=600,
                        valid_sets=[lgb.Dataset(Xva, label=y_va)],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
        best = bst.best_iteration or bst.current_iteration()
        c = np.asarray(bst.predict(Xte, num_iteration=best, pred_contrib=True))
        sv = c.reshape(len(Xte), 5, k + 1)[:, :, :k]
        am = np.abs(sv).mean(axis=0).mean(axis=0)
        out[s] = {"best_iter": int(best), "shap": {f: float(v) for f, v in zip(FEATS, am)}}
        print(f"seed {s}: best_iter {best}", flush=True)
        del bst, c, sv

    mat = np.array([[out[s]["shap"][f] for f in FEATS] for s in SEEDS])
    mean, std = mat.mean(0), mat.std(0, ddof=1)
    tot = mean.sum()
    ENV8 = ["sic", "sit", "VHM0_m", "VTM10_s", "WindSpeed_ms", "Temperature_C", "SST_C",
            "era5_cloudwater_sw99_visibility_km"]
    order = np.argsort(-mean)
    rows = []
    print("\n=== 3 次重复（均值 ± 标准差）===")
    for i in order[:20]:
        rows.append({"feature": FEATS[i], "mean": float(mean[i]), "std": float(std[i]),
                     "pct": float(mean[i] / tot * 100)})
        print(f"{FEATS[i]:40s} {mean[i]:.4f} ± {std[i]:.4f}   {mean[i]/tot*100:5.2f}%")

    env_sum = sum(mean[FEATS.index(f)] for f in ENV8)
    env_share = [float(sum(out[s]['shap'][f] for f in ENV8) / sum(out[s]['shap'].values()) * 100)
                 for s in SEEDS]
    print(f"\n环境 8 项合计 {env_sum:.4f}，份额 {env_share[0]:.2f}% / "
          f"{env_share[1]:.2f}% / {env_share[2]:.2f}%")

    (BASE / "figs" / "shap49_multiseed.json").write_text(
        json.dumps({"seeds": SEEDS, "per_seed": out, "summary": rows,
                    "env_share_pct": env_share}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("\n写入 figs/shap49_multiseed.json")


if __name__ == "__main__":
    main()
