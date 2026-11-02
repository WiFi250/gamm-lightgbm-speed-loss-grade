#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""水深特征：ETOPO1（1 弧分）插值到每个 AIS 船位。

文献依据
--------
Pant, Fernandez & Sahoo (2026, Ocean Engineering 359:125763) 在美国北极 AIS 上
做两阶段模型时发现：**离岸距离与水深是决定航速的主导因素，风和海冰影响反而有限**
（"Distance to coast and bathymetric depth were dominant determinants... Wind and
sea ice effects were modest."）。我们现有 46 项特征里没有任何地形量，
`nearest_port_nm` 只是"到最近港口"，不等于"到海岸"或"水深"。

数据
----
ETOPO1 60 弧秒，经 NOAA ERDDAP 取北极子集（58—90°N 全经度，1921×21601，79 MB）。
经度原始为 0—360，需转换。

输出：data_depth/depth_YYYY.parquet（imo, acq_time, water_depth, shallow50, shelf200）
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

BASE = Path("/Users/wangyifan/Documents/seaice/博四/挑战者杯比赛/model_v2")
DATA = BASE / "data_v2"
OUT = BASE / "data_depth"
try:
    OUT.mkdir(exist_ok=True)
except FileExistsError:
    pass
TOPO = Path("/tmp/etopo_arctic.nc")


def load_topo():
    ds = xr.open_dataset(TOPO)
    lat = ds["latitude"].values.astype("float64")
    lon = ds["longitude"].values.astype("float64")
    z = ds["altitude"].values.astype("float32")
    z = np.where(z > 0, 0.0, -z).astype("float32")      # 陆地为 0 m 水深
    f = RegularGridInterpolator((lat, lon), z, bounds_error=False, fill_value=np.nan)
    return f


def build(year: int, f) -> dict:
    t0 = time.time()
    d = pd.read_parquet(DATA / f"cruising_{year}.parquet", columns=["imo", "acq_time",
                                                                   "latitude", "longitude"])
    d = d.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    lon360 = np.mod(d["longitude"].to_numpy("float64"), 360.0)
    pts = np.column_stack([d["latitude"].to_numpy("float64"), lon360])
    depth = f(pts).astype("float32")
    out = pd.DataFrame({"imo": d["imo"].values,
                        "acq_time": pd.to_datetime(d["acq_time"]).values,
                        "water_depth": depth})
    out["shallow50"] = (out.water_depth < 50).astype("float32")
    out["shelf200"] = (out.water_depth < 200).astype("float32")
    out.to_parquet(OUT / f"depth_{year}.parquet", index=False, compression="zstd")
    info = {"year": year, "rows": len(out),
            "valid_pct": round(float(np.isfinite(depth).mean() * 100), 1),
            "med_depth": (round(float(np.nanmedian(depth)), 1) if np.isfinite(depth).any() else None),
            "shallow_pct": round(float(np.nanmean(out.shallow50) * 100), 1),
            "sec": round(time.time() - t0, 1)}
    print(info, flush=True)
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("years", nargs="*", type=int)
    args = ap.parse_args()
    years = args.years or list(range(2013, 2024))
    f = load_topo()
    for y in years:
        build(y, f)


if __name__ == "__main__":
    main()
