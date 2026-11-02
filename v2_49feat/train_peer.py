#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同海域同期「船舶群体」特征实验。

动机
----
前面 20 次实验都撞在同一个数上，说明剩下的误差来自**没被观测的东西**：
洋流残余、载货吃水、破冰船护航编队、船东调度。这些量单船看不到，
但**同一海域、同一时段的其他船也在受影响**——群体行为本身就是这些
未观测因素的观测器。

于是构造「留一法群体特征」：
  peer_n          同格点同期（0.25° × 3 h）内其他船的记录数
  peer_loss_mean  其他船的平均速度损失（留自己这艘船，避免自证）
  peer_speed_mean 其他船的平均对地航速
  peer_loss_std   其他船损失的标准差（编队内一致性）

为什么不是泄漏：这些量在预测时刻全部可得（AIS 实时流 + 各船参考速度），
不依赖任何未来信息，也不依赖被预测样本自身的标签。
它把「护航编队」「局地流涡」「等泊拥堵」这些原来只能当噪声的东西变成特征。

对照组与 F4 完全一致（33 项特征、同切分、同参数、同 20% 采样）。
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

BASE = Path("/Users/wangyifan/Documents/seaice/博四/挑战者杯比赛/model_v2")
OUT = BASE / "runs_peer"
OUT.mkdir(parents=True, exist_ok=True)

F4_EXTRA = ["speed_ratio", "power_per_dwt", "dwt_per_l3", "beam_ratio", "mcr_per_len",
            "sic_d1", "sic_d12", "sic_d48", "sit_d1", "sit_d12",
            "WindSpeed_ms_d1", "WindSpeed_ms_d12", "VHM0_m_d12", "Temperature_C_d12",
            "hour_sin", "hour_cos"]
BASE_FEATS = T.SHIP_FEATURES + T.NAV_FEATURES + T.ENV_FEATURES + F4_EXTRA
PEER_FEATS = ["peer_ship_n", "peer_n", "peer_loss_mean", "peer_speed_mean", "peer_loss_std"]

PARAMS = {
    "learning_rate": 0.04, "num_leaves": 127, "min_data_in_leaf": 60,
    "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1,
    "lambda_l2": 0.15, "seed": 42, "num_threads": 0, "verbosity": -1,
    "feature_pre_filter": False,
}

CELL = 4          # 0.25° 网格
TBIN = 3 * 3600   # 3 小时时间桶


def add_peer_features(d: pd.DataFrame) -> pd.DataFrame:
    """留一法群体特征：同格点同期、排除本船。"""
    t = pd.to_datetime(d["acq_time"])
    tb = (t.astype("int64") // 10**9 // TBIN).astype("int64")
    la = np.rint(d["latitude"].to_numpy() * CELL).astype("int64")
    lo = np.rint(d["longitude"].to_numpy() * CELL).astype("int64")
    # (格点, 时间桶) 的联合键。用 hash 而不是拼接整数，避免桶号溢出与冲突
    key = pd.util.hash_pandas_object(
        pd.DataFrame({"a": la, "b": lo, "c": tb}), index=False).to_numpy()

    loss = d["speed_loss_rate"].to_numpy("float64")
    spd = (d["reference_speed"].to_numpy("float64") * (1.0 - loss))
    imo = d["imo"].fillna(-1).to_numpy("int64")

    ks = pd.Series(key)
    grp = pd.DataFrame({"key": key, "imo": imo, "loss": loss, "spd": spd, "loss2": loss * loss})

    tot = grp.groupby("key").agg(n=("loss", "size"), s=("loss", "sum"),
                                 v=("spd", "sum"), q=("loss2", "sum"),
                                 ns=("imo", "nunique"))
    own = grp.groupby(["key", "imo"]).agg(n=("loss", "size"), s=("loss", "sum"),
                                          v=("spd", "sum"), q=("loss2", "sum"))

    tot_r = tot.reindex(key)
    own_r = own.reindex(pd.MultiIndex.from_arrays([key, imo]))

    n_peer = tot_r["n"].to_numpy() - own_r["n"].to_numpy()
    s_peer = tot_r["s"].to_numpy() - own_r["s"].to_numpy()
    v_peer = tot_r["v"].to_numpy() - own_r["v"].to_numpy()
    q_peer = tot_r["q"].to_numpy() - own_r["q"].to_numpy()

    ok = n_peer > 0
    mean_loss = np.where(ok, s_peer / np.maximum(n_peer, 1), np.nan)
    mean_spd = np.where(ok, v_peer / np.maximum(n_peer, 1), np.nan)
    var = np.where(ok, np.maximum(q_peer / np.maximum(n_peer, 1) - mean_loss ** 2, 0), np.nan)

    # 记录数会被「同一条船 3 小时内约 180 条报文」放大，
    # 真正有意义的是**不同船**的数量
    n_ship = tot_r["ns"].to_numpy() - 1
    d["peer_ship_n"] = np.where(ok, n_ship, 0).astype("float32")
    d["peer_n"] = n_peer.astype("float32")
    d["peer_loss_mean"] = mean_loss.astype("float32")
    d["peer_speed_mean"] = mean_spd.astype("float32")
    d["peer_loss_std"] = np.sqrt(var).astype("float32")
    return d


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts = []
    for y in range(2013, 2024):
        d = pd.read_parquet(BASE / "data_v2" / f"cruising_{y}.parquet")
        d = d[d["speed_loss_rate"] > 0].copy()
        d = add_peer_features(d)
        _m = ((d["doy"] - 1) // 30.4 + 1).clip(1, 12)
        d["month_sin"] = np.sin(2 * np.pi * _m / 12.0).astype("float32")
        d["month_cos"] = np.cos(2 * np.pi * _m / 12.0).astype("float32")
        d["doy_sin"] = np.sin(2 * np.pi * d["doy"] / 365.25).astype("float32")
        d["doy_cos"] = np.cos(2 * np.pi * d["doy"] / 365.25).astype("float32")
        parts.append(d)
        print(f"  {y}: {len(d):,} 条  peer 覆盖 {d.peer_loss_mean.notna().mean()*100:.1f}%",
              flush=True)
    df = pd.concat(parts, ignore_index=True)
    del parts
    df = df.sort_values(["year", "doy"], kind="stable").reset_index(drop=True)
    n = len(df)
    n_test = int(n * 0.2)
    n_valid = int((n - n_test) * 0.15)
    i1, i2 = n - n_test - n_valid, n - n_test
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()


def run(name, feats, tr, va, te, frac, rounds):
    th = T.build_thresholds(tr["speed_loss_rate"])
    if frac < 1.0:
        idx = np.sort(np.random.RandomState(42).permutation(len(tr))[: int(len(tr) * frac)])
        tr = tr.iloc[idx]
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
    imp = dict(zip(feats, (b.feature_importance("gain") / max(1, b.feature_importance("gain").sum()) * 100).round(2)))
    res["top_gain"] = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:8])
    print(json.dumps(res, ensure_ascii=False), flush=True)
    (OUT / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--max-round", type=int, default=400)
    args = ap.parse_args()

    print("加载 data_v2 并构造群体特征 ...", flush=True)
    tr, va, te = load_all()
    cov = te["peer_loss_mean"].notna().mean() * 100
    print(f"train={len(tr):,} valid={len(va):,} test={len(te):,}  测试集 peer 覆盖 {cov:.1f}%",
          flush=True)

    rows = [run("P0_base", BASE_FEATS, tr, va, te, args.frac, args.max_round),
            run("P1_peer", BASE_FEATS + PEER_FEATS, tr, va, te, args.frac, args.max_round),
            run("P2_peer_meanonly", BASE_FEATS + ["peer_loss_mean"], tr, va, te,
                args.frac, args.max_round)]

    tab = pd.DataFrame([{
        "方案": r["name"], "特征数": r["n_features"],
        "Accuracy": round(r["accuracy"] * 100, 2), "MacroF1": round(r["macro_f1"] * 100, 2),
        "QWK": round(r["qwk"], 4), "WithinOne": round(r["within_one"] * 100, 2),
        "高风险召回": round(r["hi_recall"] * 100, 2), "高风险精确": round(r["hi_precision"] * 100, 2),
        "冰区Acc": round(r["ice_acc"] * 100, 2), "冰区QWK": round(r["ice_qwk"], 4),
        "最佳轮数": r["best_iter"],
    } for r in rows])
    tab.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    print("\n===== 群体特征增量 =====")
    print(tab.to_string(index=False))
    print("\nP1 重要性前 8：", json.dumps(rows[1]["top_gain"], ensure_ascii=False))


if __name__ == "__main__":
    main()
