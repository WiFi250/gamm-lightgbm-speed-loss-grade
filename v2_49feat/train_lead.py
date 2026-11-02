#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S2 + 前方环境特征（t+3 h / t+6 h）对照。

S2 已是定稿候选（1 h 航段 + F4 + 群体特征，Acc 36.6 / QWK 0.502）。
这一步只加一组特征，别的不动：
  sic_lead3 / sic_lead6 及变化量、风速 / 波高 / 气温的 3 h 变化量，
  共 8 项（取自 make_lead.py）。

两臂用完全相同的种子配置与切分，只差这 8 列。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import train_v2 as T
from train_peer import add_peer_features
from train_ensemble import PARAMS, FEATS as FEATS_S2, qwk
from train_final import ENV, DYNAMICS, SHIP, NAV, PEER, build

BASE = Path("/Users/wangyifan/Documents/seaice/博四/挑战者杯比赛/model_v2")
DATA = BASE / "data_v2"
LEAD = BASE / "data_lead"
OUT = BASE / "runs_lead"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = BASE / "data_seg" / "seg_1h_lead3.parquet"

DEPTH_FEATS = ["water_depth", "shallow50", "shelf200"]
LEAD_FEATS = ["sic_lead3", "sic_lead6", "sic_lead3_d", "sic_lead6_d",
              "WindSpeed_ms_lead3_d", "VHM0_m_lead3_d",
              "Temperature_C_lead3_d", "sit_lead3_d"]


def build_with_lead(hours: int) -> pd.DataFrame:
    parts = []
    for y in range(2013, 2024):
        d = pd.read_parquet(DATA / f"cruising_{y}.parquet")
        d = d[d["speed_loss_rate"] > 0].copy()
        if len(d) == 0:
            continue
        d["acq_time"] = pd.to_datetime(d["acq_time"], errors="coerce")
        d = add_peer_features(d)
        lf = LEAD / f"lead_{y}.parquet"
        if lf.exists():
            ld = pd.read_parquet(lf)
            ld["acq_time"] = pd.to_datetime(ld["acq_time"])
            # 原始 AIS 有重复报文，先按 (imo, acq_time) 去重再合并
            ld = ld.drop_duplicates(subset=["imo", "acq_time"], keep="first")
            d = d.drop_duplicates(subset=["imo", "acq_time"], keep="first")
            d = d.merge(ld, on=["imo", "acq_time"], how="left")
        dp = BASE / "data_depth" / f"depth_{y}.parquet"
        if dp.exists():
            dd = pd.read_parquet(dp)
            dd["acq_time"] = pd.to_datetime(dd["acq_time"])
            dd = dd.drop_duplicates(subset=["imo", "acq_time"], keep="first")
            d = d.merge(dd, on=["imo", "acq_time"], how="left")
        for c in LEAD_FEATS + DEPTH_FEATS:
            if c not in d.columns:
                d[c] = np.nan
        d["bucket"] = d["acq_time"].dt.floor(f"{hours}h")
        # 太阳高度角：比小时编码更能代表"有没有光"。
        # 小时编码不含纬度与季节，喀拉海 12 月与 6 月的同一钟点光照完全不同。
        # 文献依据：Rao et al. (2021) 明确指出日照/可见光信息能提升航速预测。
        _dec = np.deg2rad(23.44) * np.sin(2 * np.pi * (284 + d["acq_time"].dt.dayofyear) / 365.0)
        _ha = np.deg2rad(15.0 * (d["acq_time"].dt.hour + d["acq_time"].dt.minute / 60.0 - 12.0)
                         + d["longitude"])
        _lat = np.deg2rad(d["latitude"])
        d["solar_sin"] = (np.sin(_lat) * np.sin(_dec)
                          + np.cos(_lat) * np.cos(_dec) * np.cos(_ha)).astype("float32")
        d["daylight"] = (d["solar_sin"] > 0).astype("float32")
        agg = {c: "mean" for c in ENV + NAV + DYNAMICS + PEER + LEAD_FEATS + DEPTH_FEATS
               + ["solar_sin", "daylight", "speed_loss_rate"]}
        agg.update({c: "first" for c in SHIP})
        g = d.groupby(["imo", "bucket"], observed=True).agg(agg).reset_index()
        g["year"] = g["bucket"].dt.year.astype("int16")
        g["doy"] = g["bucket"].dt.dayofyear.astype("int16")
        g["hour_f"] = (g["bucket"].dt.hour + g["bucket"].dt.minute / 60.0).astype("float32")
        parts.append(g)
        print(f"  {y}: {len(g):,} 航段  lead3 覆盖 {g.sic_lead3.notna().mean()*100:.1f}%",
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


def run(name, feats, tr, va, te, th, rounds, seed=42):
    import lightgbm as lgb
    y_tr = T.cut_grades(tr["speed_loss_rate"].to_numpy(), th)
    y_va = T.cut_grades(va["speed_loss_rate"].to_numpy(), th)
    y_te = T.cut_grades(te["speed_loss_rate"].to_numpy(), th)
    p = dict(PARAMS, objective="multiclass", num_class=5, metric="multi_logloss",
             seed=seed, bagging_seed=seed + 1, feature_fraction_seed=seed + 2)
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
    g = b.feature_importance("gain")
    imp = dict(zip(feats, (g / max(1, g.sum()) * 100).round(2)))
    res["top_gain"] = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:10])
    print(json.dumps(res, ensure_ascii=False), flush=True)
    (OUT / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-round", type=int, default=600)
    args = ap.parse_args()

    if CACHE.exists():
        print("复用缓存", flush=True)
        seg = pd.read_parquet(CACHE)
    else:
        seg = build_with_lead(1)
        seg.to_parquet(CACHE, index=False, compression="zstd")
        print("写出缓存", CACHE, flush=True)

    seg = seg.sort_values(["year", "doy"], kind="stable").reset_index(drop=True)
    n = len(seg)
    n_test, n_valid = int(n * 0.2), int((n - int(n * 0.2)) * 0.15)
    i1, i2 = n - n_test - n_valid, n - n_test
    tr, va, te = seg.iloc[:i1], seg.iloc[i1:i2], seg.iloc[i2:]
    th = T.build_thresholds(tr["speed_loss_rate"])
    print(f"train={len(tr):,} valid={len(va):,} test={len(te):,}  "
          f"测试集 lead3 覆盖 {te.sic_lead3.notna().mean()*100:.1f}%", flush=True)

    rows = [run("L0_S2", FEATS_S2, tr, va, te, th, args.max_round),
            run("L1_S2_lead", FEATS_S2 + LEAD_FEATS, tr, va, te, th, args.max_round),
            run("L4_lead_depth", FEATS_S2 + LEAD_FEATS + DEPTH_FEATS,
                tr, va, te, th, args.max_round)]

    tab = pd.DataFrame([{
        "方案": r["name"], "特征数": r["n_features"],
        "Accuracy": round(r["accuracy"] * 100, 2), "MacroF1": round(r["macro_f1"] * 100, 2),
        "QWK": round(r["qwk"], 4), "WithinOne": round(r["within_one"] * 100, 2),
        "高风险召回": round(r["hi_recall"] * 100, 2), "高风险精确": round(r["hi_precision"] * 100, 2),
        "冰区Acc": round(r["ice_acc"] * 100, 2), "冰区QWK": round(r["ice_qwk"], 4),
        "最佳轮数": r["best_iter"],
    } for r in rows])
    tab.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    print("\n===== 前方环境特征增量 =====")
    print(tab.to_string(index=False))
    print("\nL1 重要性前 10：", json.dumps(rows[1]["top_gain"], ensure_ascii=False))


if __name__ == "__main__":
    main()
