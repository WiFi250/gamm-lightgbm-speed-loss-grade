#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多年份全变量的 LNG 船速损失等级 LightGBM 建模脚本。

目标：
    将 speed_loss_rate 先按训练集区间做 0-1 归一化，再划分为 5 个有序等级：
        0-20%, 20-40%, 40-60%, 60-80%, 80-100%
    模型训练、验证和预测阶段都只使用“等级”作为目标，而不是连续数值。

输出：
    - 等级分类拟合曲线
    - 测试集混淆矩阵与分类指标
    - 测试集空间分布图（真实等级 / 预测等级）
    - SHAP 重要性与 dependence 图
    - 变量方向性分析表
    - 结果汇总 JSON

说明：
    本版纳入除 distance、time_gap_hour、avg_speed 之外的其他自变量，并保留 reference_speed。
    其中 acq_time 仅用于时间切分，不作为特征输入。
    可选地只保留 speed_loss_rate > 0 的正损失样本，用于“只学习正损失部分”的等级建模。
"""

from __future__ import annotations

import argparse
import json
import os
from warnings import warn
from pathlib import Path

import lightgbm as lgb
import matplotlib
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import matplotlib.path as mpath
from matplotlib.patches import Circle
from matplotlib.patches import Patch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
except ImportError:  # pragma: no cover - cartopy is optional at runtime
    ccrs = None
    cfeature = None
    LatitudeFormatter = None
    LongitudeFormatter = None

try:
    import shapefile
except ImportError:  # pragma: no cover - handled at runtime on the server
    shapefile = None


ENV_FEATURES = [
    "VHM0_m",
    "VTM10_s",
    "WindSpeed_ms",
    "Temperature_C",
    "SST_C",
    "era5_cloudwater_sw99_visibility_km",
    "sic",
    "sit",
]

FEATURE_TYPE_COLORS = {
    "气象": "#D98C6A",
    "海况": "#5E9E99",
    "海冰": "#86A66D",
    "能见度": "#B08CCB",
    "其他": "#8B919A",
}

FEATURE_TYPE_ORDER = ["气象", "海况", "海冰", "能见度", "其他"]

FEATURE_TYPE_DISPLAY = {
    "气象": "Meteorology",
    "海况": "Sea State",
    "海冰": "Sea Ice",
    "能见度": "Visibility",
    "其他": "Other",
}

FEATURE_TYPE_GROUPS = {
    "气象": {"WindSpeed_ms", "Temperature_C", "SST_C"},
    "海况": {"VHM0_m", "VTM10_s"},
    "海冰": {"sic", "sit"},
    "能见度": {"era5_cloudwater_sw99_visibility_km"},
}


def feature_type_for_bar(feature: str) -> str:
    for label, members in FEATURE_TYPE_GROUPS.items():
        if feature in members:
            return label
    return "其他"


def feature_color(feature: str) -> str:
    return FEATURE_TYPE_COLORS[feature_type_for_bar(feature)]


def feature_display_name(feature: str) -> str:
    display_names = {
        "era5_cloudwater_sw99_visibility_km": "Visibility (km)",
    }
    return display_names.get(feature, feature)

GRADE_LABELS = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
NUM_CLASSES = len(GRADE_LABELS)
GRADE_CMAP = ListedColormap(
    ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
    name="grade5",
)
GRADE_NORM = BoundaryNorm(np.arange(-0.5, NUM_CLASSES + 0.5, 1), GRADE_CMAP.N)
ERROR_CMAP = ListedColormap(["#e5e7eb", "#dc2626"], name="binary_error")
ERROR_NORM = BoundaryNorm(np.array([-0.5, 0.5, 1.5]), ERROR_CMAP.N)
LAMBERT_LON_MIN = -50.17982864379883
LAMBERT_LON_MAX = 192.0
LAMBERT_LAT_MIN = 57.500797271728516
LAMBERT_CENTRAL_LON = 70.91008567810059
ARCTIC_MIN_LAT = LAMBERT_LAT_MIN
BASE_EXCLUDED_COLS = {
    "acq_time",
    "avg_speed",
    "distance",
    "time_gap_hour",
    "risk_bin_start_30min",
    "imo",
    "speed_loss_rate",
}


def _prepare_colorarray(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape[-1] != 3:
        raise ValueError("Input image has less than 3 channels.")
    return arr


_XYZ_FROM_RGB = np.array(
    [
        [0.412453, 0.35758, 0.180423],
        [0.212671, 0.71516, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ]
)
_RGB_FROM_XYZ = np.linalg.inv(_XYZ_FROM_RGB)


def _convert_color(matrix: np.ndarray, arr: np.ndarray) -> np.ndarray:
    arr = _prepare_colorarray(arr)
    return arr @ matrix.T.astype(arr.dtype)


def _xyz2rgb(xyz: np.ndarray) -> np.ndarray:
    arr = _convert_color(_RGB_FROM_XYZ, xyz)
    mask = arr > 0.0031308
    arr[mask] = 1.055 * np.power(arr[mask], 1 / 2.4) - 0.055
    arr[~mask] *= 12.92
    np.clip(arr, 0, 1, out=arr)
    return arr


def _lab2xyz(lab: np.ndarray) -> np.ndarray:
    arr = _prepare_colorarray(lab).copy()
    L, a, b = arr[..., 0], arr[..., 1], arr[..., 2]
    y = (L + 16.0) / 116.0
    x = (a / 500.0) + y
    z = y - (b / 200.0)
    invalid = np.atleast_1d(z < 0).nonzero()
    n_invalid = invalid[0].size
    if n_invalid != 0:
        if z.ndim > 0:
            z[invalid] = 0
        else:
            z = 0
    out = np.stack([x, y, z], axis=-1)
    mask = out > 0.2068966
    out[mask] = np.power(out[mask], 3.0)
    out[~mask] = (out[~mask] - 16.0 / 116.0) / 7.787
    out *= np.array([0.95047, 1.0, 1.08883])
    return out, n_invalid


def _lab2rgb(lab: np.ndarray) -> np.ndarray:
    xyz, n_invalid = _lab2xyz(lab)
    if n_invalid != 0:
        warn(
            "Conversion from CIE-LAB, via XYZ to sRGB color space resulted in "
            f"{n_invalid} negative Z values that have been clipped to zero",
            stacklevel=3,
        )
    return _xyz2rgb(xyz)


def _lch2lab(lch: np.ndarray) -> np.ndarray:
    arr = _prepare_colorarray(lch).copy()
    c, h = arr[..., 1], arr[..., 2]
    arr[..., 1], arr[..., 2] = c * np.cos(h), c * np.sin(h)
    return arr


def _lch2rgb(lch: np.ndarray) -> np.ndarray:
    return _lab2rgb(_lch2lab([[lch]]))[0][0]


def _make_shap_red_blue_cmap() -> matplotlib.colors.LinearSegmentedColormap:
    blue_lch = [54.0, 70.0, 4.6588]
    l_mid = 40.0
    red_lch = [54.0, 90.0, 0.35470565 + 2 * np.pi]
    gray_lch = [55.0, 0.0, 0.0]

    blue_rgb = _lch2rgb(blue_lch)
    red_rgb = _lch2rgb(red_lch)
    gray_rgb = _lch2rgb(gray_lch)

    reds: list[tuple[float, float, float]] = []
    greens: list[tuple[float, float, float]] = []
    blues: list[tuple[float, float, float]] = []
    alphas: list[tuple[float, float, float]] = []
    nsteps = 100
    l_vals = list(np.linspace(blue_lch[0], l_mid, nsteps // 2)) + list(np.linspace(l_mid, red_lch[0], nsteps // 2))
    c_vals = np.linspace(blue_lch[1], red_lch[1], nsteps)
    h_vals = np.linspace(blue_lch[2], red_lch[2], nsteps)
    for pos, l, c, h in zip(np.linspace(0, 1, nsteps), l_vals, c_vals, h_vals):
        rgb = _lch2rgb([l, c, h])
        reds.append((pos, rgb[0], rgb[0]))
        greens.append((pos, rgb[1], rgb[1]))
        blues.append((pos, rgb[2], rgb[2]))
        alphas.append((pos, 1.0, 1.0))

    cmap = matplotlib.colors.LinearSegmentedColormap(
        "red_blue",
        {"red": reds, "green": greens, "blue": blues, "alpha": alphas},
    )
    cmap.set_bad(gray_rgb.tolist(), 1.0)
    cmap.set_over(gray_rgb.tolist(), 1.0)
    cmap.set_under(gray_rgb.tolist(), 1.0)
    return cmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a multi-year environment-only LightGBM grade classifier for LNG ship speed loss."
    )
    parser.add_argument(
        "--input-csvs",
        nargs="+",
        type=Path,
        required=True,
        help="Input yearly CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("speed_loss_grade_multiyear_outputs"),
        help="Output directory.",
    )
    parser.add_argument(
        "--sample-cap-per-file",
        type=int,
        default=250_000,
        help="Maximum rows kept per yearly file. Set 0 to disable sampling.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="Chunk size for sampling large files.",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default="acq_time",
        help="Timestamp column used for chronological split.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="speed_loss_rate",
        help="Continuous target column used to derive grade labels.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Chronological test split ratio.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.15,
        help="Chronological validation split ratio inside the training portion.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--shap-sample-size",
        type=int,
        default=20_000,
        help="Sample size for SHAP analysis from the test set.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help="LightGBM training threads. 0 means use os.cpu_count().",
    )
    parser.add_argument(
        "--top-class-index",
        type=int,
        default=4,
        help="Class index used for SHAP direction analysis. Default 4 = 80-100% grade.",
    )
    parser.add_argument(
        "--grade-binning",
        choices=["quantile", "absolute01"],
        default="quantile",
        help="How to convert the continuous target into 5 grades. 'quantile' matches the current setup; 'absolute01' uses equal-width bins on train min-max normalized values.",
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Keep only rows with target > 0 before chronological splitting and training.",
    )
    parser.add_argument(
        "--marine-regions-dir",
        type=Path,
        default=Path("IHO_marine_regions_58N"),
        help="Directory containing IHO marine region shapefiles.",
    )
    return parser.parse_args()


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def count_rows(csv_path: Path) -> int:
    import subprocess

    out = subprocess.check_output(["wc", "-l", str(csv_path)], text=True).strip()
    return int(out.split()[0]) - 1


def sampled_rows_for_chunk(
    seen_before: int,
    seen_after: int,
    total_rows: int,
    target_rows: int,
    already_kept: int,
    chunk_len: int,
) -> int:
    if total_rows <= 0:
        return 0
    if target_rows <= 0 or target_rows >= total_rows:
        return chunk_len

    target_cumulative = target_rows * seen_after / total_rows
    keep_now = int(round(target_cumulative - already_kept))
    return max(0, min(chunk_len, keep_now))


def load_one_csv(
    csv_path: Path,
    sample_cap_per_file: int,
    chunksize: int,
    random_state: int,
    usecols: list[str],
) -> tuple[pd.DataFrame, dict]:
    total_rows = count_rows(csv_path)

    if sample_cap_per_file <= 0 or total_rows <= sample_cap_per_file:
        df = pd.read_csv(
            csv_path,
            usecols=usecols,
            encoding="utf-8-sig",
            parse_dates=["acq_time"],
        )
        manifest = {
            "file": str(csv_path),
            "rows_total": total_rows,
            "rows_kept": int(len(df)),
            "sampled": False,
            "sample_mode": "all",
        }
        return df, manifest

    # Fast deterministic sampling: keep the first N rows from each yearly file.
    # This is much quicker than chunk-wise sampling and is sufficient for a
    # server-side pilot run when a full multi-year read is too expensive.
    df = pd.read_csv(
        csv_path,
        usecols=usecols,
        encoding="utf-8-sig",
        parse_dates=["acq_time"],
        nrows=sample_cap_per_file,
    )
    manifest = {
        "file": str(csv_path),
        "rows_total": total_rows,
        "rows_kept": int(len(df)),
        "sampled": True,
        "sample_mode": "head",
    }
    return df, manifest


def load_multi_year_data(
    csv_paths: list[Path],
    sample_cap_per_file: int,
    chunksize: int,
    random_state: int,
    usecols: list[str],
) -> tuple[pd.DataFrame, list[dict]]:
    parts: list[pd.DataFrame] = []
    manifests: list[dict] = []
    for csv_path in csv_paths:
        df_part, manifest = load_one_csv(
            csv_path=csv_path,
            sample_cap_per_file=sample_cap_per_file,
            chunksize=chunksize,
            random_state=random_state,
            usecols=usecols,
        )
        parts.append(df_part)
        manifests.append(manifest)

    combined = pd.concat(parts, ignore_index=True)
    return combined, manifests


def chronological_split(df: pd.DataFrame, time_col: str, test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values(time_col).reset_index(drop=True)
    split_idx = int(len(ordered) * (1 - test_ratio))
    split_idx = max(1, min(split_idx, len(ordered) - 1))
    return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()


def train_valid_split(df: pd.DataFrame, time_col: str, valid_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values(time_col).reset_index(drop=True)
    split_idx = int(len(ordered) * (1 - valid_ratio))
    split_idx = max(1, min(split_idx, len(ordered) - 1))
    return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()


def make_time_summary(df: pd.DataFrame, time_col: str) -> dict:
    years = pd.to_datetime(df[time_col], errors="coerce").dt.year
    counts = years.value_counts().sort_index()
    return {str(int(k)): int(v) for k, v in counts.items() if pd.notna(k)}


def read_header_columns(csv_path: Path) -> list[str]:
    return pd.read_csv(csv_path, nrows=0, encoding="utf-8-sig").columns.tolist()


def build_usecols_from_header(header_cols: list[str], target_col: str, time_col: str) -> list[str]:
    usecols = [col for col in header_cols if col not in BASE_EXCLUDED_COLS]
    for required in (time_col, target_col):
        if required in header_cols and required not in usecols:
            usecols.insert(0, required)
    return usecols


def infer_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    feature_cols: list[str] = []
    categorical_cols: list[str] = []
    dropped_constant_cols: list[str] = []

    for col in df.columns:
        if col in BASE_EXCLUDED_COLS:
            continue
        if col in {"loss_grade", "loss_norm", "year"}:
            continue

        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            continue

        nunique = series.dropna().nunique()
        if nunique <= 1:
            dropped_constant_cols.append(col)
            continue

        feature_cols.append(col)
        if series.dtype == object or pd.api.types.is_categorical_dtype(series):
            categorical_cols.append(col)

    return feature_cols, categorical_cols, dropped_constant_cols


def polar_stereo_north(
    lon: np.ndarray | pd.Series | float,
    lat: np.ndarray | pd.Series | float,
    lon0: float = 0.0,
    radius_m: float = 6_371_000.0,
) -> tuple[np.ndarray, np.ndarray]:
    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    lon_rad = np.deg2rad(lon_arr)
    lat_rad = np.deg2rad(lat_arr)
    lon0_rad = np.deg2rad(lon0)
    rho = 2.0 * radius_m * np.tan(np.pi / 4.0 - lat_rad / 2.0)
    x = rho * np.sin(lon_rad - lon0_rad)
    y = -rho * np.cos(lon_rad - lon0_rad)
    return x / 1000.0, y / 1000.0


def polar_stereo_radius_km(lat: float, radius_m: float = 6_371_000.0) -> float:
    lat_rad = np.deg2rad(float(lat))
    return float(2.0 * radius_m * np.tan(np.pi / 4.0 - lat_rad / 2.0) / 1000.0)


def format_lon_label(lon: float) -> str:
    lon_int = int(round(float(lon)))
    if lon_int == 0:
        return "0°"
    if abs(lon_int) == 180:
        return "180°"
    suffix = "E" if lon_int > 0 else "W"
    return f"{abs(lon_int)}°{suffix}"


def draw_polar_reference_frame(
    ax: plt.Axes,
    central_lon: float,
    boundary_lat: float = 58.0,
    *,
    lon_step: int = 30,
    lat_ticks: tuple[int, ...] = (60, 70, 80),
) -> None:
    boundary_r = polar_stereo_radius_km(boundary_lat)
    theta = np.linspace(0, 2 * np.pi, 361)
    meridian_lons = np.arange(-180, 181, lon_step)
    circle_color = "#d1d5db"
    label_color = "#4b5563"
    boundary_color = "#6b7280"

    # Draw the boundary and lat/lon graticules in projected coordinates.
    boundary = Circle((0.0, 0.0), boundary_r, fill=False, edgecolor=boundary_color, linewidth=0.9, linestyle="--")
    ax.add_patch(boundary)

    for lat in lat_ticks:
        r = polar_stereo_radius_km(lat)
        ax.plot(r * np.cos(theta), r * np.sin(theta), color=circle_color, lw=0.45, alpha=0.95, zorder=0)

        # Place latitude labels near the upper-left arc, similar to the reference figure.
        theta_deg = 110.0 - (lat - 60.0)
        theta_rad = np.deg2rad(theta_deg)
        ax.text(
            (r + 55.0) * np.cos(theta_rad),
            (r + 55.0) * np.sin(theta_rad),
            f"{lat}°N",
            fontsize=6.5,
            color=label_color,
            ha="center",
            va="center",
        )

    rho = np.linspace(0.0, boundary_r, 240)
    for lon in meridian_lons:
        delta = np.deg2rad(float(lon) - float(central_lon))
        x = rho * np.sin(delta)
        y = -rho * np.cos(delta)
        ax.plot(x, y, color=circle_color, lw=0.35, alpha=0.95, zorder=0)

        r_label = boundary_r + 85.0
        lx = r_label * np.sin(delta)
        ly = -r_label * np.cos(delta)
        ax.text(lx, ly, format_lon_label(lon), fontsize=6.1, color=label_color, ha="center", va="center")

    pad = 0.08 * boundary_r
    ax.set_xlim(-boundary_r - pad, boundary_r + pad)
    ax.set_ylim(-boundary_r - pad, boundary_r + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def load_marine_regions(region_dir: Path) -> list[dict]:
    if shapefile is None:
        raise RuntimeError(
            "The 'shapefile' package is required for marine region plots. "
            "Install it with `python3 -m pip install --user pyshp`."
        )
    if not region_dir.exists():
        raise FileNotFoundError(f"Marine region directory not found: {region_dir}")

    shapes: list[dict] = []
    for shp_path in sorted(region_dir.glob("*/region.shp")):
        reader = shapefile.Reader(str(shp_path))
        field_names = [f[0] for f in reader.fields[1:]]
        for shape_rec, shape in zip(reader.records(), reader.shapes()):
            record = dict(zip(field_names, shape_rec))
            shapes.append(
                {
                    "name": str(record.get("FIG_NAME") or record.get("IHO_NAME") or shp_path.parent.name),
                    "area_km2": float(record.get("AREA_KM2") or 0.0),
                    "points": np.asarray(shape.points, dtype=float),
                    "parts": list(shape.parts) + [len(shape.points)],
                }
            )
    return shapes


def project_shape_points(points: np.ndarray, lon0: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    return polar_stereo_north(points[:, 0], points[:, 1], lon0=lon0)


def arctic_projection_extent(
    region_shapes: list[dict],
    point_lons: np.ndarray,
    point_lats: np.ndarray,
    lon0: float = 0.0,
) -> float:
    x_pts, y_pts = polar_stereo_north(point_lons, point_lats, lon0=lon0)
    max_abs = float(np.nanmax(np.abs(np.concatenate([x_pts, y_pts]))))
    for region in region_shapes:
        x_reg, y_reg = project_shape_points(region["points"], lon0=lon0)
        max_abs = max(max_abs, float(np.nanmax(np.abs(np.concatenate([x_reg, y_reg])))))
    return max_abs * 1.08 if np.isfinite(max_abs) and max_abs > 0 else 1.0


def create_arctic_axis(fig: plt.Figure, lon0: float = 0.0) -> tuple[plt.Axes, bool]:
    if ccrs is None or cfeature is None:
        ax = fig.add_subplot(111)
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("white")
        return ax, False

    data_crs = ccrs.PlateCarree()
    projection = ccrs.LambertAzimuthalEqualArea(
        central_longitude=LAMBERT_CENTRAL_LON if lon0 == 0.0 else lon0,
        central_latitude=90.0,
    )

    lon_bottom = np.linspace(LAMBERT_LON_MIN, LAMBERT_LON_MAX, 720)
    lat_bottom = np.full_like(lon_bottom, LAMBERT_LAT_MIN)
    lat_side = np.linspace(LAMBERT_LAT_MIN, 89.999, 180)
    lon = np.concatenate(
        [
            lon_bottom,
            np.full_like(lat_side, LAMBERT_LON_MAX),
            np.full_like(lat_side, LAMBERT_LON_MIN),
        ]
    )
    lat = np.concatenate([lat_bottom, lat_side, lat_side[::-1]])
    xy = projection.transform_points(data_crs, lon, lat)
    boundary = mpath.Path(xy[:, :2])

    ax = fig.add_subplot(111, projection=projection)
    ax.set_boundary(boundary, transform=projection)
    xmin, ymin = boundary.vertices.min(axis=0)
    xmax, ymax = boundary.vertices.max(axis=0)
    pad = 0.012 * max(xmax - xmin, ymax - ymin)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.add_feature(cfeature.OCEAN, facecolor="#F7F9FB")
    ax.add_feature(cfeature.LAND, facecolor="#C7D0D6", edgecolor="#33383D", linewidth=0.25)
    ax.coastlines(resolution="50m", color="#33383D", linewidth=0.35)
    gl = ax.gridlines(
        crs=data_crs,
        xlocs=np.arange(-30, 193, 30),
        ylocs=[60, 70, 80],
        color="#66737D",
        linewidth=0.3,
        alpha=0.25,
        draw_labels=False,
    )
    return ax, True


def compute_grade_columns(
    df: pd.DataFrame,
    target_col: str,
    thresholds: list[float],
    grade_binning: str,
    train_min: float | None = None,
    train_max: float | None = None,
) -> pd.DataFrame:
    out = df.copy()
    threshold_arr = np.asarray(thresholds, dtype=float)
    if len(threshold_arr) != NUM_CLASSES + 1 or not np.all(np.isfinite(threshold_arr)):
        raise ValueError("Grade thresholds are invalid.")

    values = out[target_col].to_numpy(dtype=float)
    if grade_binning == "quantile":
        # Convert the raw target into an empirical percentile score on the training distribution.
        norm = np.interp(values, threshold_arr, np.linspace(0.0, 1.0, len(threshold_arr)))
        norm = np.clip(norm, 0.0, 1.0)
        # Map percentile ranks to 0, 1, 2, 3, 4.
        grade = np.searchsorted(threshold_arr[1:-1], values, side="right").astype(int)
    elif grade_binning == "absolute01":
        if train_min is None or train_max is None or not np.isfinite(train_min) or not np.isfinite(train_max):
            raise ValueError("Absolute 0-1 binning requires valid train_min and train_max.")
        scale = float(train_max - train_min)
        if scale <= 0:
            raise ValueError("Absolute 0-1 binning requires train_max > train_min.")
        norm = (values - train_min) / scale
        norm = np.clip(norm, 0.0, 1.0)
        # Equal-width bins on the normalized 0-1 interval.
        grade = np.searchsorted(threshold_arr[1:-1], norm, side="right").astype(int)
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"Unsupported grade binning mode: {grade_binning}")
    grade = np.clip(grade, 0, NUM_CLASSES - 1)

    out["loss_norm"] = norm
    out["loss_grade"] = grade
    return out


def build_thresholds(train_series: pd.Series) -> list[float]:
    quantiles = np.linspace(0.0, 1.0, NUM_CLASSES + 1)
    values = train_series.quantile(quantiles).to_numpy(dtype=float)
    values = np.asarray(values, dtype=float)
    if len(np.unique(values)) < len(values):
        # Rare, but can happen if the target is heavily discretized. Nudge duplicates forward slightly.
        values = values.copy()
        for i in range(1, len(values)):
            if values[i] <= values[i - 1]:
                values[i] = np.nextafter(values[i - 1], float("inf"))
    return values.tolist()


def build_absolute01_thresholds() -> list[float]:
    return np.linspace(0.0, 1.0, NUM_CLASSES + 1).tolist()


def class_weight_map(y: pd.Series) -> dict[int, float]:
    counts = y.value_counts().sort_index()
    total = float(counts.sum())
    num_classes = float(len(counts))
    return {int(cls): total / (num_classes * float(cnt)) for cls, cnt in counts.items() if cnt > 0}


def regression_on_classes_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "corr": corr}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None = None) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "quadratic_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "mae_class": float(np.mean(np.abs(y_true - y_pred))),
        "within_one_accuracy": float(np.mean(np.abs(y_true - y_pred) <= 1)),
    }
    if proba is not None and len(proba) == len(y_true):
        top2 = np.argpartition(-proba, kth=1, axis=1)[:, :2]
        metrics["top2_accuracy"] = float(np.mean([yt in row for yt, row in zip(y_true, top2, strict=False)]))
        expected_grade = np.sum(proba * np.arange(proba.shape[1], dtype=float), axis=1)
        metrics.update(
            regression_on_classes_metrics(y_true.astype(float), expected_grade),
        )
    return metrics


def train_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    label_col: str,
    random_state: int,
    num_threads: int,
) -> tuple[lgb.Booster, dict]:
    X_train = train_df[feature_cols].copy()
    y_train = train_df[label_col].to_numpy(copy=True)
    X_valid = valid_df[feature_cols].copy()
    y_valid = valid_df[label_col].to_numpy(copy=True)

    for col in feature_cols:
        if col in categorical_cols:
            X_train[col] = X_train[col].astype("category")
            X_valid[col] = X_valid[col].astype("category")
        else:
            X_train[col] = pd.to_numeric(X_train[col], errors="coerce").astype("float32")
            X_valid[col] = pd.to_numeric(X_valid[col], errors="coerce").astype("float32")

    weights_map = class_weight_map(train_df[label_col])
    train_weights = train_df[label_col].map(weights_map).fillna(1.0).astype("float32").to_numpy()
    valid_weights = valid_df[label_col].map(weights_map).fillna(1.0).astype("float32").to_numpy()

    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        weight=train_weights,
        categorical_feature=categorical_cols,
        free_raw_data=False,
    )
    valid_set = lgb.Dataset(
        X_valid,
        label=y_valid,
        weight=valid_weights,
        reference=train_set,
        categorical_feature=categorical_cols,
        free_raw_data=False,
    )

    params = {
        "objective": "multiclass",
        "num_class": NUM_CLASSES,
        "metric": ["multi_logloss", "multi_error"],
        "learning_rate": 0.04,
        "num_leaves": 127,
        "min_data_in_leaf": 60,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 0.15,
        "seed": random_state,
        "num_threads": num_threads,
        "verbosity": -1,
        "feature_pre_filter": False,
    }

    eval_result: dict[str, dict[str, list[float]]] = {}
    model = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=1200,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=20),
            lgb.record_evaluation(eval_result),
        ],
    )
    return model, eval_result


def predict_frame(
    model: lgb.Booster,
    df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    X = df[feature_cols].copy()
    for col in feature_cols:
        if col in categorical_cols:
            X[col] = X[col].astype("category")
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype("float32")
    best_iter = model.best_iteration or model.current_iteration()
    proba = model.predict(X, num_iteration=best_iter)
    pred = np.argmax(proba, axis=1).astype(int)
    return pred, np.asarray(proba)


def save_fit_curve(eval_result: dict, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for metric_name, ax in zip(["multi_logloss", "multi_error"], axes, strict=False):
        train_vals = eval_result.get("train", {}).get(metric_name, [])
        valid_vals = eval_result.get("valid", {}).get(metric_name, [])
        ax.plot(train_vals, label="Train", color="#2563eb", lw=1.6)
        ax.plot(valid_vals, label="Valid", color="#dc2626", lw=1.6)
        ax.set_title(metric_name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(metric_name)
        ax.grid(alpha=0.2)
        ax.legend()

    fig.suptitle("LightGBM Grade Classification Fit Curves")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    normalize: bool = False,
) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm = cm / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(cm, cmap="Blues", vmin=0 if normalize else None)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(GRADE_LABELS, rotation=30, ha="right")
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_yticklabels(GRADE_LABELS)
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("True grade")
    ax.set_title("Confusion Matrix (normalized)" if normalize else "Confusion Matrix")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = cm[i, j]
            text = f"{val:.2f}" if normalize else f"{int(val):,}"
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Rate" if normalize else "Count", rotation=270, labelpad=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    cm_df = pd.DataFrame(cm, index=GRADE_LABELS, columns=GRADE_LABELS)
    return cm_df


def save_classification_report_table(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )
    table = pd.DataFrame(
        {
            "grade": GRADE_LABELS,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
        }
    )
    table.to_csv(output_path, index=False, encoding="utf-8-sig")
    return table


def save_classification_bar(table: pd.DataFrame, output_path: Path) -> None:
    x = np.arange(len(table))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.bar(x - width, table["precision"], width=width, label="Precision", color="#2563eb")
    ax.bar(x, table["recall"], width=width, label="Recall", color="#059669")
    ax.bar(x + width, table["f1"], width=width, label="F1", color="#dc2626")
    ax.set_xticks(x)
    ax.set_xticklabels(table["grade"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-class Precision / Recall / F1")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_spatial_map(
    df: pd.DataFrame,
    value_col: str,
    output_path: Path,
    title: str,
    sample_size: int = 120_000,
    *,
    vmin: float | None = 0,
    vmax: float | None = NUM_CLASSES - 1,
    cmap: str = "viridis",
    cbar_ticks: list[float] | None = None,
    cbar_ticklabels: list[str] | None = None,
    cbar_label: str = "Grade",
) -> None:
    plot_df = df[["longitude", "latitude", value_col]].dropna().copy()
    if len(plot_df) > sample_size:
        plot_df = plot_df.sample(n=sample_size, random_state=42)

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(
        plot_df["longitude"],
        plot_df["latitude"],
        c=plot_df[value_col],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=4,
        alpha=0.45,
        linewidths=0,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
    if cbar_ticklabels is not None:
        cbar.set_ticklabels(cbar_ticklabels)
    cbar.ax.set_ylabel(cbar_label, rotation=270, labelpad=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def draw_arctic_region_outlines(
    ax: plt.Axes,
    region_shapes: list[dict],
    lon0: float = 0.0,
    color: str = "#475569",
    linewidth: float = 0.6,
    alpha: float = 0.85,
) -> None:
    for region in region_shapes:
        points = region["points"]
        parts = region["parts"]
        for start, end in zip(parts[:-1], parts[1:]):
            seg = points[start:end]
            if len(seg) < 2:
                continue
            if ccrs is not None and hasattr(ax, "projection"):
                ax.plot(
                    seg[:, 0],
                    seg[:, 1],
                    transform=ccrs.PlateCarree(),
                    color=color,
                    lw=linewidth,
                    alpha=alpha,
                    zorder=2,
                )
            else:
                x, y = polar_stereo_north(seg[:, 0], seg[:, 1], lon0=lon0)
                ax.plot(x, y, color=color, lw=linewidth, alpha=alpha)


def save_arctic_region_basemap(
    region_shapes: list[dict],
    output_path: Path,
    lon0: float = 0.0,
) -> None:
    fig = plt.figure(figsize=(8.6, 6.3))
    ax, use_cartopy = create_arctic_axis(fig, lon0=lon0)
    draw_arctic_region_outlines(ax, region_shapes, lon0=lon0, color="#475569", linewidth=0.7, alpha=0.9)
    for region in region_shapes:
        pts = region["points"]
        if len(pts) == 0:
            continue
        if use_cartopy:
            cx = float(np.nanmean(pts[:, 0]))
            cy = float(np.nanmean(pts[:, 1]))
            if np.isfinite(cx) and np.isfinite(cy):
                ax.text(
                    cx,
                    cy,
                    region["name"],
                    transform=ccrs.PlateCarree(),
                    fontsize=6.5,
                    color="#111827",
                    ha="center",
                    va="center",
                    zorder=3,
                )
        else:
            x, y = polar_stereo_north(pts[:, 0], pts[:, 1], lon0=lon0)
            if not np.isfinite(x).any() or not np.isfinite(y).any():
                continue
            cx = float(np.nanmean(x))
            cy = float(np.nanmean(y))
            if np.isfinite(cx) and np.isfinite(cy):
                ax.text(cx, cy, region["name"], fontsize=6.5, color="#111827", ha="center", va="center")

    boundary_lon = np.linspace(-180, 180, 721)
    boundary_lat = np.full_like(boundary_lon, 58.0, dtype=float)
    if use_cartopy:
        ax.plot(
            boundary_lon,
            boundary_lat,
            transform=ccrs.PlateCarree(),
            color="#111827",
            lw=1.0,
            ls="--",
            alpha=0.75,
            zorder=2,
        )
    else:
        draw_polar_reference_frame(ax, central_lon=LAMBERT_CENTRAL_LON if lon0 == 0.0 else lon0, boundary_lat=58.0)
    ax.set_title("IHO Marine Regions North of 58N")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_arctic_spatial_map(
    df: pd.DataFrame,
    value_col: str,
    output_path: Path,
    title: str,
    region_shapes: list[dict],
    sample_size: int = 120_000,
    *,
    vmin: float | None = 0,
    vmax: float | None = NUM_CLASSES - 1,
    cmap: str = "viridis",
    cbar_ticks: list[float] | None = None,
    cbar_ticklabels: list[str] | None = None,
    cbar_label: str = "Grade",
    lon0: float = 0.0,
) -> None:
    plot_df = df[["longitude", "latitude", value_col]].dropna().copy()
    if len(plot_df) > sample_size:
        plot_df = plot_df.sample(n=sample_size, random_state=42)

    fig = plt.figure(figsize=(8.6, 6.3))
    ax, use_cartopy = create_arctic_axis(fig, lon0=lon0)
    draw_arctic_region_outlines(ax, region_shapes, lon0=lon0, color="#64748b", linewidth=0.45, alpha=0.8)

    if value_col in {"loss_grade", "pred_grade"}:
        cmap_obj = GRADE_CMAP
        norm_obj = GRADE_NORM
        default_ticks = list(range(NUM_CLASSES))
        default_ticklabels = GRADE_LABELS
    elif value_col == "error_class":
        cmap_obj = ERROR_CMAP
        norm_obj = ERROR_NORM
        default_ticks = [0, 1]
        default_ticklabels = ["Correct", "Misclassified"]
    else:
        cmap_obj = cmap
        norm_obj = None
        default_ticks = cbar_ticks
        default_ticklabels = cbar_ticklabels

    scatter_kwargs = {
        "c": plot_df[value_col].to_numpy(),
        "cmap": cmap_obj,
        "norm": norm_obj,
        "s": 4,
        "alpha": 0.55,
        "linewidths": 0,
    }
    if use_cartopy:
        scatter_kwargs["transform"] = ccrs.PlateCarree()
        sc = ax.scatter(plot_df["longitude"], plot_df["latitude"], **scatter_kwargs)
    else:
        x, y = polar_stereo_north(plot_df["longitude"].to_numpy(), plot_df["latitude"].to_numpy(), lon0=lon0)
        sc = ax.scatter(x, y, **scatter_kwargs)

    if not use_cartopy:
        draw_polar_reference_frame(ax, central_lon=LAMBERT_CENTRAL_LON if lon0 == 0.0 else lon0, boundary_lat=58.0)
    ax.set_title(title)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    ticks = cbar_ticks if cbar_ticks is not None else default_ticks
    ticklabels = cbar_ticklabels if cbar_ticklabels is not None else default_ticklabels
    if ticks is not None:
        cbar.set_ticks(ticks)
    if ticklabels is not None:
        cbar.set_ticklabels(ticklabels)
    cbar.ax.set_ylabel(cbar_label, rotation=270, labelpad=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def shap_contrib_multiclass(model: lgb.Booster, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    best_iter = model.best_iteration or model.current_iteration()
    contrib = model.predict(X, num_iteration=best_iter, pred_contrib=True)
    contrib = np.asarray(contrib)

    n_samples, n_features = X.shape
    block = n_features + 1
    if contrib.ndim != 2 or contrib.shape[1] % block != 0:
        raise ValueError(f"Unexpected pred_contrib shape: {contrib.shape}")
    num_class = contrib.shape[1] // block
    contrib = contrib.reshape(n_samples, num_class, block)
    shap_values = contrib[:, :, :n_features]
    base_values = contrib[:, :, -1]
    return shap_values, base_values


def save_shap_bar(feature_table: pd.DataFrame, output_path: Path, title: str) -> None:
    top = feature_table.head(20).sort_values("mean_abs_shap", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    bar_colors = [feature_color(feat) for feat in top["feature"]]
    ax.barh(top["feature"], top["mean_abs_shap"], color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_ylabel("Feature")
    handles = [
        Patch(facecolor=FEATURE_TYPE_COLORS[label], edgecolor="none", label=FEATURE_TYPE_DISPLAY[label])
        for label in FEATURE_TYPE_ORDER
        if label in set(feature_type_for_bar(feat) for feat in top["feature"])
    ]
    if handles:
        ax.legend(handles=handles, title="Feature type", loc="lower right", frameon=False, fontsize=9, title_fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _gaussian_smooth(values: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size < 3 or not np.isfinite(arr).any():
        return arr
    sigma = float(max(sigma, 1e-6))
    radius = max(1, int(np.ceil(3.0 * sigma)))
    kernel_x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (kernel_x / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(arr, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _binned_feature_trend(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 20:
        return None

    x_low, x_high = np.nanpercentile(x, [1.0, 99.0])
    if not np.isfinite(x_low) or not np.isfinite(x_high) or x_high <= x_low:
        x_low = float(np.nanmin(x))
        x_high = float(np.nanmax(x))
    if not np.isfinite(x_low) or not np.isfinite(x_high) or x_high <= x_low:
        return None

    x_clipped = np.clip(x, x_low, x_high)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(x_clipped, quantiles))
    if edges.size < 5:
        edges = np.linspace(x_low, x_high, max(6, min(n_bins + 1, max(6, x.size // 30))))
    if edges.size < 3:
        return None

    centers: list[float] = []
    medians: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []

    for i in range(edges.size - 1):
        left = float(edges[i])
        right = float(edges[i + 1])
        if i == edges.size - 2:
            idx = (x_clipped >= left) & (x_clipped <= right)
        else:
            idx = (x_clipped >= left) & (x_clipped < right)
        if idx.sum() < 3:
            continue
        values_x = x_clipped[idx]
        values_y = y[idx]
        centers.append(float(np.median(values_x)))
        medians.append(float(np.median(values_y)))
        lowers.append(float(np.nanpercentile(values_y, 10)))
        uppers.append(float(np.nanpercentile(values_y, 90)))

    if len(centers) < 3:
        return None

    centers_arr = np.asarray(centers, dtype=float)
    order = np.argsort(centers_arr)
    centers_arr = centers_arr[order]
    med_arr = np.asarray(medians, dtype=float)[order]
    low_arr = np.asarray(lowers, dtype=float)[order]
    high_arr = np.asarray(uppers, dtype=float)[order]

    grid = np.linspace(float(centers_arr.min()), float(centers_arr.max()), 240)
    med_grid = np.interp(grid, centers_arr, med_arr)
    low_grid = np.interp(grid, centers_arr, low_arr)
    high_grid = np.interp(grid, centers_arr, high_arr)

    med_grid = _gaussian_smooth(med_grid, sigma=3.0)
    low_grid = _gaussian_smooth(low_grid, sigma=3.0)
    high_grid = _gaussian_smooth(high_grid, sigma=3.0)

    low_grid = np.minimum(low_grid, med_grid)
    high_grid = np.maximum(high_grid, med_grid)
    return grid, med_grid, low_grid, high_grid


def _binned_density(x: np.ndarray, n_bins: int = 40) -> tuple[np.ndarray, np.ndarray] | None:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 20:
        return None

    x_low, x_high = np.nanpercentile(x, [1.0, 99.0])
    if not np.isfinite(x_low) or not np.isfinite(x_high) or x_high <= x_low:
        x_low = float(np.nanmin(x))
        x_high = float(np.nanmax(x))
    if not np.isfinite(x_low) or not np.isfinite(x_high) or x_high <= x_low:
        return None

    clipped = np.clip(x, x_low, x_high)
    counts, edges = np.histogram(clipped, bins=n_bins, range=(x_low, x_high))
    if counts.sum() <= 0:
        return None
    centers = 0.5 * (edges[:-1] + edges[1:])
    density = counts.astype(float)
    density = _gaussian_smooth(density, sigma=1.6)
    if np.nanmax(density) <= 0:
        return None
    density /= np.nanmax(density)
    return centers, density


def save_shap_summary_combo(
    X: pd.DataFrame,
    shap_values: np.ndarray,
    feature_table: pd.DataFrame,
    output_path: Path,
    title: str,
    bar_features: int | None = None,
    panel_features: int = 9,
) -> None:
    ranked = feature_table.sort_values("mean_abs_shap", ascending=False, kind="mergesort")
    bar_top = ranked["feature"].tolist() if bar_features is None else ranked.head(bar_features)["feature"].tolist()
    panel_top = ranked.head(panel_features)["feature"].tolist()
    feature_set = set(bar_top) | set(panel_top)
    panel_letters = list("bcdefghij")

    fig = plt.figure(figsize=(18.2, 10.4), facecolor="white")
    gs = fig.add_gridspec(3, 4, width_ratios=[1.18, 1.0, 1.0, 1.0], wspace=0.34, hspace=0.42)
    ax_bar = fig.add_subplot(gs[:, 0])

    bar_vals = ranked.set_index("feature").loc[bar_top, "mean_abs_shap"].to_numpy(dtype=float)
    bar_colors = [feature_color(feat) for feat in bar_top]
    y = np.arange(len(bar_top))
    ax_bar.barh(y, bar_vals, color=bar_colors, edgecolor="white", linewidth=0.6)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([feature_display_name(feat) for feat in bar_top], fontsize=7.5)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Mean |SHAP value|", fontsize=8.5)
    ax_bar.set_ylabel("")
    ax_bar.grid(axis="x", linestyle="--", alpha=0.18)
    ax_bar.text(-0.16, 1.02, "a", transform=ax_bar.transAxes, fontsize=14, fontweight="bold", va="bottom")
    for spine in ("top", "right"):
        ax_bar.spines[spine].set_visible(False)
    ax_bar.spines["left"].set_alpha(0.35)
    ax_bar.spines["bottom"].set_alpha(0.35)

    legend_handles = [
        Patch(facecolor=FEATURE_TYPE_COLORS[label], edgecolor="none", label=FEATURE_TYPE_DISPLAY[label])
        for label in FEATURE_TYPE_ORDER
        if label in {feature_type_for_bar(feat) for feat in feature_set}
    ]

    for i, feat in enumerate(panel_top):
        row = i // 3
        col = i % 3 + 1
        ax = fig.add_subplot(gs[row, col])
        feat_idx = X.columns.get_loc(feat)
        x = pd.to_numeric(X[feat], errors="coerce").to_numpy(dtype=float)
        yv = shap_values[:, feat_idx].astype(float)
        mask = np.isfinite(x) & np.isfinite(yv)
        color = feature_color(feat)

        if mask.sum() >= 10:
            x_valid = x[mask]
            y_valid = yv[mask]
            ax.scatter(
                x_valid,
                y_valid,
                s=6,
                color=color,
                alpha=0.07,
                linewidths=0,
                rasterized=True,
                zorder=1,
            )

            trend = _binned_feature_trend(x_valid, y_valid)
            if trend is not None:
                grid, med_grid, low_grid, high_grid = trend
                ax.fill_between(grid, low_grid, high_grid, color=color, alpha=0.14, linewidth=0, zorder=2)
                ax.plot(grid, med_grid, color=color, lw=2.0, zorder=3)

            density = _binned_density(x_valid)
            if density is not None:
                density_x, density_y = density
                y_low = float(np.nanpercentile(y_valid, 1))
                y_high = float(np.nanpercentile(y_valid, 99))
                if not np.isfinite(y_low) or not np.isfinite(y_high) or y_high <= y_low:
                    y_low = float(np.nanmin(y_valid))
                    y_high = float(np.nanmax(y_valid))
                y_span = max(y_high - y_low, float(np.nanstd(y_valid)) * 2.0, 1e-3)
                baseline = y_low - 0.18 * y_span
                height = 0.10 * y_span
                ax.fill_between(
                    density_x,
                    baseline,
                    baseline + density_y * height,
                    color=color,
                    alpha=0.20,
                    linewidth=0,
                    zorder=0,
                )
                ax.plot(
                    density_x,
                    baseline + density_y * height,
                    color=color,
                    alpha=0.25,
                    lw=0.8,
                    zorder=0,
                )
                ax.set_ylim(baseline - 0.06 * y_span, y_high + 0.12 * y_span)

        ax.axhline(0, color="#94A3B8", lw=1.0, ls=(0, (2.2, 2.2)), alpha=0.9, zorder=0)
        ax.set_xlabel(feature_display_name(feat), fontsize=8.5)
        ax.set_ylabel("SHAP value", fontsize=8.5)
        ax.tick_params(axis="both", labelsize=7.5, length=3.0, direction="out")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_alpha(0.35)
        ax.spines["bottom"].set_alpha(0.35)
        ax.text(0.02, 0.96, panel_letters[i], transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
        if mask.sum() >= 10 and np.isfinite(x_valid).any():
            x_low, x_high = np.nanpercentile(x_valid, [1.0, 99.0])
            if np.isfinite(x_low) and np.isfinite(x_high) and x_high > x_low:
                ax.set_xlim(x_low, x_high)

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=len(legend_handles),
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
            fontsize=7.0,
            handlelength=1.0,
            columnspacing=1.8,
            handletextpad=0.45,
        )

    fig.subplots_adjust(left=0.12, right=0.995, bottom=0.13, top=0.98)
    fig.savefig(output_path, dpi=600, facecolor="white")
    plt.close(fig)


def _stack_beeswarm_offsets(shap_vals: np.ndarray, max_spread: float = 0.38) -> np.ndarray:
    """Create cone-like vertical offsets with width driven by local SHAP density."""
    x = np.asarray(shap_vals, dtype=float)
    offsets = np.zeros_like(x, dtype=float)
    finite_mask = np.isfinite(x)
    finite_x = x[finite_mask]
    if finite_x.size <= 1:
        return offsets

    lo, hi = np.nanpercentile(finite_x, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return offsets

    n_bins = int(np.clip(np.sqrt(finite_x.size) * 1.8, 28, 96))
    bin_edges = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(np.clip(finite_x, lo, hi), bins=bin_edges)
    counts = counts.astype(float)

    # Smooth the histogram so the envelope follows the local density rather than
    # raw per-bin counts. This removes the long "tails" from sparse bins.
    sigma = 1.25
    radius = int(np.ceil(3.0 * sigma))
    kernel_x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (kernel_x / sigma) ** 2)
    kernel /= kernel.sum()
    smooth = np.convolve(counts, kernel, mode="same")
    if np.max(smooth) <= 0:
        return offsets

    density = smooth / np.max(smooth)
    clipped_x = np.clip(x, lo, hi)
    bin_ids = np.digitize(clipped_x, bin_edges[1:-1], right=False)

    for b in range(n_bins):
        idx = np.where(bin_ids == b)[0]
        if idx.size == 0:
            continue

        # Wider center, narrower tails.
        bin_width = max_spread * (0.14 + 0.86 * float(density[b]) ** 0.72)
        bin_width = float(np.clip(bin_width, 0.02, max_spread))

        idx = idx[np.argsort(x[idx])]
        n = idx.size
        if n == 1:
            offsets[idx[0]] = 0.0
            continue

        step = bin_width / max(1, (n - 1) / 2.0)
        positions = np.zeros(n, dtype=float)
        for j in range(1, n):
            ring = (j + 1) // 2
            sign = 1.0 if j % 2 == 1 else -1.0
            positions[j] = sign * ring * step
        max_abs = float(np.max(np.abs(positions)))
        if max_abs > bin_width and max_abs > 0:
            positions *= bin_width / max_abs
        offsets[idx] = positions

    return offsets


def save_shap_beeswarm(
    X: pd.DataFrame,
    shap_values: np.ndarray,
    feature_table: pd.DataFrame,
    output_path: Path,
    title: str,
    max_display: int = 20,
) -> None:
    ranked = feature_table.sort_values("mean_abs_shap", ascending=False, kind="mergesort")
    top_features = ranked.head(max_display)["feature"].tolist()
    # Keep the visual order explicit: the most important feature must be at the top.
    y_positions = {feat: len(top_features) - 1 - i for i, feat in enumerate(top_features)}
    fig, ax = plt.subplots(figsize=(11, 8))
    all_values: list[np.ndarray] = []
    shap_red_blue = _make_shap_red_blue_cmap()

    for feat in top_features:
        idx = X.columns.get_loc(feat)
        x = pd.to_numeric(X[feat], errors="coerce")
        y = shap_values[:, idx]
        mask = x.notna() & np.isfinite(y)
        if not mask.any():
            continue
        feat_values = x[mask].to_numpy(dtype=float)
        shap_vals = y[mask]
        all_values.append(feat_values)
        offsets = _stack_beeswarm_offsets(shap_vals)
        ax.scatter(
            shap_vals,
            np.full(mask.sum(), y_positions[feat]) + offsets,
            c=feat_values,
            cmap=shap_red_blue,
            s=9,
            alpha=0.82,
            linewidths=0,
            rasterized=True,
        )

    ax.axvline(0, color="gray", lw=1, alpha=0.6)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(list(reversed(top_features)))
    ax.set_xlabel("SHAP value")
    ax.set_ylabel("feature")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.15)
    ax.set_ylim(-0.5, len(top_features) - 0.5)
    if all_values:
        values = np.concatenate(all_values)
        vmin = float(np.nanpercentile(values, 1))
        vmax = float(np.nanpercentile(values, 99))
        if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
            sm = plt.cm.ScalarMappable(cmap=shap_red_blue)
            sm.set_clim(vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel("Feature value", rotation=270, labelpad=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_dependence_plots(
    X: pd.DataFrame,
    shap_values: np.ndarray,
    feature_table: pd.DataFrame,
    output_dir: Path,
    max_features: int,
) -> list[str]:
    top_features = feature_table.head(max_features)["feature"].tolist()
    saved: list[str] = []
    for idx, feat in enumerate(top_features, start=1):
        fig, ax = plt.subplots(figsize=(9, 6))
        feature_idx = X.columns.get_loc(feat)
        x = pd.to_numeric(X[feat], errors="coerce")
        y = shap_values[:, feature_idx]
        mask = x.notna() & np.isfinite(y)
        ax.scatter(x[mask], y[mask], s=12, alpha=0.45, color="#2b6cb0")
        ax.axhline(0, color="gray", lw=1, alpha=0.6)
        ax.set_xlabel(feat)
        ax.set_ylabel("SHAP value")
        ax.set_title(f"SHAP dependence: {feat}")
        fig.tight_layout()
        out_path = output_dir / f"shap_dependence_top_{idx}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(out_path))
    return saved


def classify_direction(spearman_corr: float, support: int) -> str:
    if support < 200:
        return "样本不足"
    if np.isnan(spearman_corr):
        return "不稳定"
    if spearman_corr >= 0.15:
        return "总体正相关"
    if spearman_corr <= -0.15:
        return "总体负相关"
    return "弱相关/非单调"


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)

    header_cols = read_header_columns(args.input_csvs[0])
    usecols = build_usecols_from_header(header_cols, args.target_col, args.time_col)
    if args.target_col not in header_cols:
        raise ValueError(f"Target column {args.target_col!r} not found in input CSV headers.")

    df, manifests = load_multi_year_data(
        csv_paths=args.input_csvs,
        sample_cap_per_file=args.sample_cap_per_file,
        chunksize=args.chunksize,
        random_state=args.random_state,
        usecols=usecols,
    )

    (args.output_dir / "sampling_manifest.json").write_text(
        json.dumps(manifests, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    df["acq_time"] = pd.to_datetime(df["acq_time"], errors="coerce")
    df = df.dropna(subset=["acq_time", args.target_col, "longitude", "latitude"]).copy()
    rows_before_positive_filter = int(len(df))
    if args.positive_only:
        df = df[pd.to_numeric(df[args.target_col], errors="coerce") > 0].copy()
        if len(df) == 0:
            raise ValueError("No rows left after applying positive-only filter on the target column.")
        print(f"Filtered positive-only rows: {rows_before_positive_filter:,} -> {len(df):,}")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce").astype("float32")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce").astype("float32")
    df = df.sort_values("acq_time").reset_index(drop=True)

    df["year"] = df["acq_time"].dt.year

    train_df, test_df = chronological_split(df, args.time_col, args.test_ratio)
    train_part, valid_part = train_valid_split(train_df, args.time_col, args.valid_ratio)

    if args.grade_binning == "quantile":
        thresholds = build_thresholds(train_part[args.target_col])
        train_min = float("nan")
        train_max = float("nan")
    else:
        thresholds = build_absolute01_thresholds()
        train_min = float(train_part[args.target_col].min())
        train_max = float(train_part[args.target_col].max())

    train_part = compute_grade_columns(
        train_part,
        args.target_col,
        thresholds,
        args.grade_binning,
        train_min=train_min,
        train_max=train_max,
    )
    valid_part = compute_grade_columns(
        valid_part,
        args.target_col,
        thresholds,
        args.grade_binning,
        train_min=train_min,
        train_max=train_max,
    )
    test_df = compute_grade_columns(
        test_df,
        args.target_col,
        thresholds,
        args.grade_binning,
        train_min=train_min,
        train_max=train_max,
    )

    feature_cols, categorical_cols, dropped_constant_cols = infer_feature_columns(train_part)
    if len(feature_cols) == 0:
        raise ValueError("No usable feature columns were found after excluding leakage and constant columns.")

    numeric_cols = [col for col in feature_cols if col not in categorical_cols]
    train_part = train_part.dropna(subset=feature_cols + ["loss_grade"]).copy()
    valid_part = valid_part.dropna(subset=feature_cols + ["loss_grade"]).copy()
    test_df = test_df.dropna(subset=feature_cols + ["loss_grade"]).copy()
    for col in numeric_cols:
        train_part[col] = pd.to_numeric(train_part[col], errors="coerce")
        valid_part[col] = pd.to_numeric(valid_part[col], errors="coerce")
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")
    train_part = train_part.dropna(subset=numeric_cols).copy()
    valid_part = valid_part.dropna(subset=numeric_cols).copy()
    test_df = test_df.dropna(subset=numeric_cols).copy()
    for col in categorical_cols:
        category_values = (
            pd.concat([train_part[col], valid_part[col], test_df[col]], ignore_index=True)
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        cat_dtype = pd.CategoricalDtype(categories=category_values)
        train_part[col] = train_part[col].astype(str).astype(cat_dtype)
        valid_part[col] = valid_part[col].astype(str).astype(cat_dtype)
        test_df[col] = test_df[col].astype(str).astype(cat_dtype)

    (args.output_dir / "grade_thresholds.json").write_text(
        json.dumps(
            {
                "grade_binning": args.grade_binning,
                "train_min": float(np.min(train_part[args.target_col])),
                "train_max": float(np.max(train_part[args.target_col])),
                "thresholds": thresholds,
                "grade_labels": GRADE_LABELS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    model, eval_result = train_model(
        train_df=train_part,
        valid_df=valid_part,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        label_col="loss_grade",
        random_state=args.random_state,
        num_threads=args.num_threads if args.num_threads > 0 else (os.cpu_count() or 1),
    )

    train_pred, train_proba = predict_frame(model, train_part, feature_cols, categorical_cols)
    valid_pred, valid_proba = predict_frame(model, valid_part, feature_cols, categorical_cols)
    test_pred, test_proba = predict_frame(model, test_df, feature_cols, categorical_cols)

    train_metrics = classification_metrics(train_part["loss_grade"].to_numpy(), train_pred, train_proba)
    valid_metrics = classification_metrics(valid_part["loss_grade"].to_numpy(), valid_pred, valid_proba)
    test_metrics = classification_metrics(test_df["loss_grade"].to_numpy(), test_pred, test_proba)

    pred_df = test_df.copy()
    pred_df["pred_grade"] = test_pred
    pred_df["pred_grade_label"] = pred_df["pred_grade"].map(lambda x: GRADE_LABELS[int(x)])
    pred_df["actual_grade_label"] = pred_df["loss_grade"].map(lambda x: GRADE_LABELS[int(x)])
    pred_df["pred_expected_grade"] = np.sum(test_proba * np.arange(NUM_CLASSES, dtype=float), axis=1)
    pred_df["correct"] = pred_df["pred_grade"] == pred_df["loss_grade"]
    pred_df.to_csv(args.output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    save_fit_curve(eval_result, args.output_dir / "fit_curves.png")
    conf_df = save_confusion_matrix(test_df["loss_grade"].to_numpy(), test_pred, args.output_dir / "confusion_matrix.png")
    conf_norm_df = save_confusion_matrix(
        test_df["loss_grade"].to_numpy(),
        test_pred,
        args.output_dir / "confusion_matrix_normalized.png",
        normalize=True,
    )
    class_report = save_classification_report_table(
        test_df["loss_grade"].to_numpy(),
        test_pred,
        args.output_dir / "classification_report.csv",
    )
    save_classification_bar(class_report, args.output_dir / "class_metrics_bar.png")

    marine_region_shapes = load_marine_regions(args.marine_regions_dir)
    save_arctic_region_basemap(marine_region_shapes, args.output_dir / "marine_regions_basemap.png")

    save_arctic_spatial_map(
        pred_df,
        value_col="loss_grade",
        output_path=args.output_dir / "test_spatial_actual_arctic.png",
        title="Test Set Spatial Distribution - Actual Grade (Arctic Projection)",
        region_shapes=marine_region_shapes,
        cbar_ticks=list(range(NUM_CLASSES)),
        cbar_ticklabels=GRADE_LABELS,
        cbar_label="Actual Grade",
    )
    save_arctic_spatial_map(
        pred_df,
        value_col="pred_grade",
        output_path=args.output_dir / "test_spatial_predicted_arctic.png",
        title="Test Set Spatial Distribution - Predicted Grade (Arctic Projection)",
        region_shapes=marine_region_shapes,
        cbar_ticks=list(range(NUM_CLASSES)),
        cbar_ticklabels=GRADE_LABELS,
        cbar_label="Predicted Grade",
    )
    save_arctic_spatial_map(
        pred_df.assign(error_class=(pred_df["pred_grade"] != pred_df["loss_grade"]).astype(int)),
        value_col="error_class",
        output_path=args.output_dir / "test_spatial_error_arctic.png",
        title="Test Set Spatial Distribution - Misclassified Points (Arctic Projection)",
        region_shapes=marine_region_shapes,
        sample_size=120_000,
        vmin=0,
        vmax=1,
        cmap="coolwarm",
        cbar_ticks=[0, 1],
        cbar_ticklabels=["Correct", "Misclassified"],
        cbar_label="Point Status",
    )

    shap_sample = pred_df.sample(n=min(args.shap_sample_size, len(pred_df)), random_state=42).copy()
    X_shap = shap_sample[feature_cols].copy()
    shap_all_classes, shap_base_values = shap_contrib_multiclass(model, X_shap)

    high_class_index = int(np.clip(args.top_class_index, 0, NUM_CLASSES - 1))
    shap_high = shap_all_classes[:, high_class_index, :]

    global_importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "mean_abs_shap": np.abs(shap_all_classes).mean(axis=(0, 1)),
            "mean_shap": shap_all_classes.mean(axis=(0, 1)),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    class_importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "mean_abs_shap": np.abs(shap_high).mean(axis=0),
            "mean_shap": shap_high.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    global_importance.to_csv(args.output_dir / "shap_feature_importance_global.csv", index=False, encoding="utf-8-sig")
    class_importance.to_csv(args.output_dir / "shap_feature_importance_high_class.csv", index=False, encoding="utf-8-sig")

    save_shap_bar(
        global_importance,
        args.output_dir / "shap_summary_bar_global.png",
        title="SHAP Global Importance (All Classes)",
    )
    save_shap_bar(
        class_importance,
        args.output_dir / "shap_summary_bar_high_class.png",
        title=f"SHAP Importance - {GRADE_LABELS[high_class_index]} Class",
    )
    save_shap_summary_combo(
        X=X_shap,
        shap_values=shap_high,
        feature_table=global_importance,
        output_path=args.output_dir / "shap_summary_combo_high_class.png",
        title="SHAP Summary - All Classes",
    )
    save_shap_beeswarm(
        X=X_shap,
        shap_values=shap_high,
        feature_table=class_importance,
        output_path=args.output_dir / "shap_summary_beeswarm_high_class.png",
        title=f"SHAP Beeswarm - {GRADE_LABELS[high_class_index]} Class",
    )
    dependence_files = save_dependence_plots(
        X=X_shap,
        shap_values=shap_high,
        feature_table=class_importance,
        output_dir=args.output_dir,
        max_features=len(feature_cols),
    )

    expected_grade = shap_sample["pred_expected_grade"].to_numpy(dtype=float)
    direction_rows = []
    for i, feat in enumerate(feature_cols):
        feat_values = pd.to_numeric(X_shap[feat], errors="coerce").to_numpy(dtype=float)
        shap_values_feat = shap_high[:, i]
        mask = np.isfinite(feat_values) & np.isfinite(shap_values_feat) & np.isfinite(expected_grade)
        if mask.sum() >= 3:
            feature_expected_corr = float(
                pd.Series(feat_values[mask]).corr(pd.Series(expected_grade[mask]), method="spearman")
            )
            shap_spearman = float(
                pd.Series(feat_values[mask]).corr(pd.Series(shap_values_feat[mask]), method="spearman")
            )
        else:
            feature_expected_corr = float("nan")
            shap_spearman = float("nan")

        direction_rows.append(
            {
                "feature": feat,
                "global_mean_abs_shap": float(
                    global_importance.loc[global_importance["feature"] == feat, "mean_abs_shap"].iloc[0]
                ),
                "high_class_mean_abs_shap": float(
                    class_importance.loc[class_importance["feature"] == feat, "mean_abs_shap"].iloc[0]
                ),
                "shap_class_spearman": shap_spearman,
                "expected_grade_spearman": feature_expected_corr,
                "direction_label": classify_direction(feature_expected_corr, int(mask.sum())),
                "support_n": int(mask.sum()),
            }
        )

    direction_df = pd.DataFrame(direction_rows).sort_values("high_class_mean_abs_shap", ascending=False)
    direction_df.to_csv(args.output_dir / "directional_analysis.csv", index=False, encoding="utf-8-sig")
    shap_sample_out = X_shap.copy()
    shap_sample_out["actual_grade"] = shap_sample["loss_grade"].to_numpy()
    shap_sample_out["pred_grade"] = shap_sample["pred_grade"].to_numpy()
    shap_sample_out["pred_expected_grade"] = shap_sample["pred_expected_grade"].to_numpy()
    shap_sample_out.to_csv(args.output_dir / "shap_values_sample.csv", index=False, encoding="utf-8-sig")

    summary = {
        "input_csvs": [str(p) for p in args.input_csvs],
        "sample_cap_per_file": args.sample_cap_per_file,
        "chunksize": args.chunksize,
        "target_col": args.target_col,
        "positive_only": bool(args.positive_only),
        "rows_before_positive_filter": rows_before_positive_filter,
        "rows_after_positive_filter": int(len(df)),
        "rows_dropped_by_positive_filter": int(rows_before_positive_filter - len(df)) if args.positive_only else 0,
        "all_features": feature_cols,
        "categorical_features": categorical_cols,
        "dropped_constant_features": dropped_constant_cols,
        "grade_labels": GRADE_LABELS,
        "grade_thresholds": thresholds,
        "num_rows_total": int(len(df)),
        "num_rows_train": int(len(train_part)),
        "num_rows_valid": int(len(valid_part)),
        "num_rows_test": int(len(test_df)),
        "train_years": make_time_summary(train_part, args.time_col),
        "valid_years": make_time_summary(valid_part, args.time_col),
        "test_years": make_time_summary(test_df, args.time_col),
        "train_grade_counts": train_part["loss_grade"].value_counts().sort_index().to_dict(),
        "valid_grade_counts": valid_part["loss_grade"].value_counts().sort_index().to_dict(),
        "test_grade_counts": test_df["loss_grade"].value_counts().sort_index().to_dict(),
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "best_iteration": int(model.best_iteration or model.current_iteration()),
        "feature_importance_top_global": global_importance.head(8)["feature"].tolist(),
        "feature_importance_top_high_class": class_importance.head(8)["feature"].tolist(),
        "combo_figure_high_class": str(args.output_dir / "shap_summary_combo_high_class.png"),
        "direction_top": direction_df[["feature", "direction_label"]].head(8).to_dict("records"),
        "dependence_plots": dependence_files,
    }

    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    model.save_model(str(args.output_dir / "lightgbm_env_speed_loss_grade_model.txt"))

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nModel saved to: {args.output_dir / 'lightgbm_env_speed_loss_grade_model.txt'}")


if __name__ == "__main__":
    main()
