# GAMM & LightGBM 航速损失等级建模

本仓库包含 LNG 船舶航速损失五等级分类的两个模型脚本：

- `lightgbm_allvars_speed_loss_grade_multiyear.py`：基于 LightGBM 的多分类模型，支持 SHAP 解释、空间分布图和变量方向性分析。
- `gamm_allvars_speed_loss_grade_multiyear.py`：GAMM 风格控制模型，将连续变量转换为样条项，并加入空间网格组效应，再使用 `SGDClassifier` 进行多分类训练。

仓库只发布代码，不包含原始 AIS 数据、模型输出或 IHO 海洋区域边界数据。请确保你有权使用输入数据，并根据数据来源要求进行引用和合规处理。

## 模型结构

![GAMM 与 LightGBM 模型结构](figures/fig4_1_gamm_lightgbm_structure_editable_20260901_01.png)

图示展示了 GAMM 风格控制模型与 LightGBM 模型的整体结构和结果对比流程。

## 方法概览

脚本默认使用 `acq_time` 做时间排序切分，使用 `speed_loss_rate` 生成 5 个等级：

`0-20%`、`20-40%`、`40-60%`、`60-80%`、`80-100%`

默认会排除 `distance`、`time_gap_hour`、`avg_speed`、`imo` 等字段，自动将其余可用字段识别为特征。`acq_time` 只用于时间切分，不进入模型特征；`longitude` 和 `latitude` 需要存在，因为脚本会用它们生成空间图或空间网格。

## 环境安装

建议使用 Python 3.10 或更高版本：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

空间图需要 IHO 海洋区域目录，目录结构应类似：

```text
IHO_marine_regions_58N/
└── <region-name>/region.shp
```

同一目录下的 `.dbf`、`.shx` 等配套文件也需要保留。若不需要 Cartopy 底图，可不安装可选依赖 `cartopy`；但 IHO 区域边界仍需要 `pyshp` 和对应数据目录。

## 运行 LightGBM

最小示例：

```bash
python3 lightgbm_allvars_speed_loss_grade_multiyear.py \
  --input-csvs /path/to/data_2021.csv /path/to/data_2022.csv \
  --output-dir outputs/lightgbm \
  --marine-regions-dir /path/to/IHO_marine_regions_58N
```

常用选项：

- `--positive-only`：只保留 `speed_loss_rate > 0` 的样本。
- `--grade-binning quantile`：按训练集分位点分级，默认设置。
- `--grade-binning absolute01`：按训练集最小值和最大值归一化后等宽分级。
- `--sample-cap-per-file 0`：每个输入文件不限制读取行数；默认每个文件最多读取 250,000 行。
- `--shap-sample-size 20000`：控制 SHAP 分析样本量。

主要输出包括 `run_summary.json`、`classification_report.csv`、`confusion_matrix.png`、`test_predictions.csv`、SHAP 重要性文件和空间分布图。

## 运行 GAMM 风格控制模型

```bash
python3 gamm_allvars_speed_loss_grade_multiyear.py \
  --input-csvs /path/to/data_2021.csv /path/to/data_2022.csv \
  --output-dir outputs/gamm \
  --marine-regions-dir /path/to/IHO_marine_regions_58N
```

该脚本与 LightGBM 脚本共用数据读取、时间切分、等级划分、评价指标和空间可视化函数。常用调节项包括 `--n-knots`、`--degree`、`--epochs`、`--batch-size`、`--lon-bin-width` 和 `--lat-bin-width`。

## 输入数据要求

每个 CSV 至少应包含以下字段：

- `acq_time`：可解析为时间戳的采集时间；
- `speed_loss_rate`：航速损失率目标变量；
- `longitude`、`latitude`：经纬度；
- 用于建模的数值或类别特征，例如风、浪、温度、海温、能见度和海冰变量。

如果字段名不同，可通过 `--time-col` 和 `--target-col` 指定时间列及目标列；空间坐标字段目前仍需要使用 `longitude` 和 `latitude`。

## 可复现性说明

两个脚本默认随机种子为 `42`，但运行结果仍可能受到 Python、LightGBM、scikit-learn 版本、硬件线程数和输入数据排序的影响。大文件运行前建议先使用较小的 `--sample-cap-per-file` 做试运行。

## 项目结构

```text
.
├── gamm_allvars_speed_loss_grade_multiyear.py
├── lightgbm_allvars_speed_loss_grade_multiyear.py
├── figures/
│   └── fig4_1_gamm_lightgbm_structure_editable_20260901_01.png
├── requirements.txt
└── README.md
```

## 49-feature version (2026-11)

A segment-level (1 h) extension with 49 features and a matched GAMM baseline is available in
[`v2_49feat/`](v2_49feat/). It reports test-set Accuracy 0.3727 / Macro F1 0.3829 / QWK 0.5160
for LightGBM, versus 0.3418 / 0.3319 / 0.3978 for the 17-feature record-level model above.
