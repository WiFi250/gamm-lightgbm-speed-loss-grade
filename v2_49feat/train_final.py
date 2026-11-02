#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最终候选模型：1 小时航段聚合 + F4 全套特征 + 同期群体特征。

把三条各自有效的改动合到一起：
  1) 航段聚合（唯一在 QWK 上有实质提升的改动：0.421 → 0.458）
  2) F4 全套（船舶比值 + 环境变化率 + 时刻编码：Acc 34.22 → 34.83）
  3) 群体特征 peer_loss_mean（本轮最好单模型 35.03，gain 重要性第 1）

对照三组，同切分同参数：
  S0  航段 + 基础 17 项（复现上一轮的 1 小时基线）
  S1  航段 + F4 全套
  S2  航段 + F4 + 群体特征
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import train_v2 as T
from train_peer import add_peer_features

BASE = Path("/Users/wangyifan/Documents/seaice/博四/挑战者杯比赛/model_v2")
DATA = BASE / "data_v2"
OUT = BASE / "runs_final"
OUT.mkdir(parents=True, exist_ok=True)

ENV = ["VHM0_m", "VTM10_s", "WindSpeed_ms", "Temperature_C", "SST_C",
       "era5_cloudwater_sw99_visibility_km", "sic", "sit"]
DYNAMICS = ["sic_d1", "sic_d12", "sic_d48", "sit_d1", "sit_d12",
            "WindSpeed_ms_d1", "WindSpeed_ms_d12", "VHM0_m_d12", "Temperature_C_d12"]
SHIP = T.SHIP_FEATURES
NAV = ["longitude", "latitude", "nearest_port_nm"]
PEER = ["peer_loss_mean"]

PARAMS = {
    "learning_rate": 0.04, "num_leaves": 127, "min_data_in_leaf": 60,
    "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1,
    "lambda_l2": 0.15, "seed": 42, "num_threads": 0, "verbosity": -1,
    "feature_pre_filter": False,
}


def build(hours: int) -> pd.DataFrame:
    parts = []
    for y in range(2013, 2024):
        d = pd.read_parquet(DATA / f"cruising_{y}.parquet")
        d = d[d["speed_loss_rate"] > 0].copy()
        if len(d) == 0:
            continue
        d["acq_time"] = pd.to_datetime(d["acq_time"], errors="coerce")
        d = add_peer_features(d)
        d["bucket"] = d["acq_time"].dt.floor(f"{hours}h")
        agg = {c: "mean" for c in ENV + NAV + DYNAMICS + PEER + ["speed_loss_rate"]}
        agg.update({c: "first" for c in SHIP})
        g = d.groupby(["imo", "bucket"], observed=True).agg(agg).reset_index()
        g["year"] = g["bucket"].dt.year.astype("int16")
        g["doy"] = g["bucket"].dt.dayofyear.astype("int16")
        g["hour_f"] = (g["bucket"].dt.hour + g["bucket"].dt.minute / 60.0).astype("float32")
        parts.append(g)
        print(f"  {y}: {len(g):,} 航段  peer 覆盖 {g.peer_loss_mean.notna().mean()*100:.1f}%",
              flush=True)
        del d, g
    seg = pd.concat(parts, ignore_index=True)
    del parts
    seg["hour_sin"] = np.sin(2 * np.pi * seg["hour_f"] / 24.0).astype("float32")
    seg["hour_cos"] = np.cos(2 * np.pi * seg["hour_f"] / 24.0).astype("float32")
    _m = ((seg["doy"] - 1) // 30.4 + 1).clip(1, 12)
    seg["month_sin"] = np.sin(2 * np.pi * _m / 12.0).astype("float32")
    seg["month_cos"] = np.cos(2 * np.pi * _m / 12.0).astype("float32")
    seg["doy_sin"] = np.sin(2 * np.pi * seg["doy"] / 365.25).astype("float32")
    seg["doy_cos"] = np.cos(2 * np.pi * seg["doy"] / 365.25).astype("float32")
    seg["speed_ratio"] = (seg["reference_speed"] / seg["Maxspeed"].replace(0, np.nan)).astype("float32")
    seg["power_per_dwt"] = (seg["MCR"] / seg["DWT"].replace(0, np.nan)).astype("float32")
    seg["dwt_per_l3"] = (seg["DWT"] / seg["Length"].replace(0, np.nan) ** 3).astype("float32")
    seg["beam_ratio"] = (seg["Width"] / seg["Length"].replace(0, np.nan)).astype("float32")
    seg["mcr_per_len"] = (seg["MCR"] / seg["Length"].replace(0, np.nan)).astype("float32")
    return seg.drop(columns=["bucket", "hour_f"])


def run(name, feats, tr, va, te, th, rounds):
    y_tr = T.cut_grades(tr["speed_loss_rate"].to_numpy(), th)
    y_va = T.cut_grades(va["speed_loss_rate"].to_numpy(), th)
    y_te = T.cut_grades(te["speed_loss_rate"].to_numpy(), th)
    p = dict(PARAMS, objective="multiclass", num_class=5, metric="multi_logloss")
    dtr = lgb.Dataset(tr[feats].astype("float32"), label=y_tr, free_raw_data=True)
    dva = lgb.Dataset(va[feats].astype("float32"), label=y_va, reference=dtr, free_raw_data=True)
    t0 = time.time()
    b = lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    best = b.best_iteration or b.current_iteration()
    proba = b.predict(te[feats].astype("float32"), num_iteration=best)
    res = {"name": name, "n_features": len(feats), "best_iter": int(best),
           "sec": round(time.time() - t0, 1)}
    res.update(T.evaluate(y_te, proba.argmax(1), proba, (te["sic"].to_numpy() >= 80)))
    gain = b.feature_importance("gain")
    imp = dict(zip(feats, (gain / max(1, gain.sum()) * 100).round(2)))
    res["top_gain"] = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:8])
    print(json.dumps(res, ensure_ascii=False), flush=True)
    (OUT / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=1)
    ap.add_argument("--max-round", type=int, default=600)
    args = ap.parse_args()

    print(f"聚合 {args.hours} 小时窗口 + 群体特征 ...", flush=True)
    seg = build(args.hours)
    print(f"航段样本 {len(seg):,}", flush=True)
    seg = seg.sort_values(["year", "doy"], kind="stable").reset_index(drop=True)
    n = len(seg)
    n_test = int(n * 0.2)
    n_valid = int((n - n_test) * 0.15)
    tr = seg.iloc[: n - n_test - n_valid].copy()
    va = seg.iloc[n - n_test - n_valid: n - n_test].copy()
    te = seg.iloc[n - n_test:].copy()
    th = T.build_thresholds(tr["speed_loss_rate"])
    print(f"train={len(tr):,} valid={len(va):,} test={len(te):,}", flush=True)

    base = SHIP + NAV + ENV
    f4 = base + ["hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
                 "speed_ratio", "power_per_dwt", "dwt_per_l3", "beam_ratio", "mcr_per_len"] + DYNAMICS

    rows = [run("S0_seg_base", base, tr, va, te, th, args.max_round),
            run("S1_seg_F4", f4, tr, va, te, th, args.max_round),
            run("S2_seg_F4_peer", f4 + PEER, tr, va, te, th, args.max_round)]

    tab = pd.DataFrame([{
        "方案": r["name"], "特征数": r["n_features"],
        "Accuracy": round(r["accuracy"] * 100, 2), "MacroF1": round(r["macro_f1"] * 100, 2),
        "QWK": round(r["qwk"], 4), "WithinOne": round(r["within_one"] * 100, 2),
        "高风险召回": round(r["hi_recall"] * 100, 2), "高风险精确": round(r["hi_precision"] * 100, 2),
        "冰区Acc": round(r["ice_acc"] * 100, 2), "冰区QWK": round(r["ice_qwk"], 4),
        "最佳轮数": r["best_iter"],
    } for r in rows])
    tab.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    print("\n===== 航段 + F4 + 群体 =====")
    print(tab.to_string(index=False))
    print("\nS2 重要性前 8：", json.dumps(rows[2]["top_gain"], ensure_ascii=False))


if __name__ == "__main__":
    main()
