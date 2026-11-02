#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""「前方环境」特征：船舶 t+3h / t+6h 时刻所在位置的环境值。

动机
----
现在的环境特征全是**脚下**的瞬时值。但船长的减速决策针对的是**即将遇到**的环境：
船在冰缘外 20 海里就开始降速，不是因为当前位置 sic 已是 80%，而是因为前方是冰。
这解释了高风险召回长期卡在 65% 左右——只看脚下，看不出「正在开进重冰区」。

做法
----
对每艘船按时间排序，取 t+3h、t+6h 时刻**该船实际到达位置**的环境值
（sic / sit / 风速 / 波高 / 气温），并同时给出变化量（lead − 当前）。

只用未来环境场，不用未来速度，不构成标签泄漏。
实盘部署时未来环境应替换为预报场；此处用再分析代替预报，是气象导航的标准近似，
方法一节必须写明。

输出：data_lead/lead_YYYY.parquet（imo, acq_time, 15 列）
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/Users/wangyifan/Documents/seaice/博四/挑战者杯比赛/model_v2")
DATA = BASE / "data_v2"
OUT = BASE / "data_lead"
try:
    OUT.mkdir(exist_ok=True)
except FileExistsError:
    pass

VARS = ["sic", "sit", "WindSpeed_ms", "VHM0_m", "Temperature_C"]
LEADS = (3, 6)
TOL = {3: 45, 6: 90}   # 分钟；6 h 放宽到 90 min 以提高覆盖（约 62%→68%）


def build(year: int) -> dict:
    t0 = time.time()
    d = pd.read_parquet(DATA / f"cruising_{year}.parquet",
                        columns=["imo", "acq_time"] + VARS)
    d = d.dropna(subset=["imo", "acq_time"]).copy()
    d["acq_time"] = pd.to_datetime(d["acq_time"])
    d = d.sort_values(["imo", "acq_time"], kind="stable").reset_index(drop=True)

    t = d["acq_time"].values
    out = {"imo": d["imo"].values, "acq_time": t}
    idx_all = d.groupby("imo", observed=True).indices

    for L in LEADS:
        tgt = t + np.timedelta64(L, "h")
        pos = np.full(len(d), -1, dtype=np.int64)
        for _, p in idx_all.items():
            p = np.sort(p)
            tt = t[p]
            j = np.searchsorted(tt, tgt[p])
            ok = (j < len(tt))
            jj = np.clip(j, 0, len(tt) - 1)
            diff = np.abs((tt[jj] - tgt[p]).astype("timedelta64[m]").astype("int64"))
            good = ok & (diff <= TOL[L])
            pos[p[good]] = p[jj[good]]
        valid = pos >= 0
        for v in VARS:
            arr = d[v].to_numpy("float32")
            lead = np.full(len(d), np.nan, dtype="float32")
            lead[valid] = arr[pos[valid]]
            out[f"{v}_lead{L}"] = lead
            out[f"{v}_lead{L}_d"] = (lead - arr).astype("float32")
        print(f"  {year} lead{L}h 覆盖率 {valid.mean()*100:.1f}%", flush=True)

    res = pd.DataFrame(out)
    res.to_parquet(OUT / f"lead_{year}.parquet", index=False, compression="zstd")
    info = {"year": year, "rows": len(res),
            "cov3": round(float(res["sic_lead3"].notna().mean() * 100), 1),
            "cov6": round(float(res["sic_lead6"].notna().mean() * 100), 1),
            "sec": round(time.time() - t0, 1)}
    print(info, flush=True)
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("years", nargs="*", type=int)
    args = ap.parse_args()
    years = args.years or list(range(2013, 2024))
    for y in years:
        build(y)


if __name__ == "__main__":
    main()
