# 49 项特征版本（航段级）复现说明

本目录是 `lightgbm_allvars_*` / `gamm_allvars_*`（17 项特征、逐条 AIS 记录）的扩展版本：
**1 小时航段聚合 + 49 项特征 + 五分类**，LightGBM 与 GAMM 使用完全相同的特征、切分与标签。

## 一、目录内文件

| 文件 | 作用 |
|---|---|
| `make_depth.py` | 用 ETOPO1（1 弧分）插值出每个船位的水深 |
| `make_lead.py` | 构造 t+3 h / t+6 h 的「前方环境」特征 |
| `train_peer.py` | 构造「同海域同期其他船」特征（留一法） |
| `train_final.py` | 1 小时航段聚合 + 特征装配 + LightGBM 训练 |
| `train_lead.py` | 在 49 项特征上做逐组增量对照（含太阳高度角、水深） |
| `train_ensemble.py` | 5 种子概率平均、有序切点、EM 先验自适应对照 |
| `gamm49.py` | GAMM 等效实现（样条 + 空间独热 + SGD），17 项与 49 项对照 |
| `diag_boundary.py` | 逐边界累积二分类（AUC）与代价敏感指标 |
| `train_year_split.py` | 三种切分（随机 / 时间顺序 / 严格按年份）的对照 |

脚本内路径为原作者本机绝对路径，复跑时请只改各文件顶部的 `BASE` / `DATA` 常量。

## 二、运行顺序

```bash
python make_depth.py          # 生成 data_depth/
python make_lead.py           # 生成 data_lead/
python train_final.py         # 首次运行会构建 data_seg/seg_1h_lead3.parquet
python train_lead.py          # 增量对照
python train_ensemble.py      # 5 种子集成
python gamm49.py              # GAMM 对照
python diag_boundary.py       # 边界 AUC 与代价指标
```

## 三、主要结果（测试集 204,322 个航段）

| 模型 | 特征数 | Accuracy | Macro F1 | QWK | 误差≤1级 | 冰区 Accuracy | 冰区 QWK |
|---|---|---|---|---|---|---|---|
| LightGBM | 49 | **0.3727** | **0.3829** | **0.5160** | **0.7591** | **0.5145** | **0.3987** |
| GAMM | 49 | 0.3306 | 0.3147 | 0.4685 | 0.6831 | 0.4681 | 0.2515 |
| GAMM | 17 | 0.3077 | 0.2864 | 0.4300 | 0.6617 | 0.3942 | 0.1828 |

作为对照，原 17 项特征、逐条记录版本为 Accuracy 0.3418、QWK 0.3978。

## 四、切分口径

- **时间顺序切分（正文采用）**：按 (年份, 年积日) 排序后按行数比例切，训练集覆盖 2013—2022 年上半年，
  测试集为 2022 年下半年至 2023 年。
- **严格按年份切分**：≤2020 训练 / 2021 验证 / ≥2022 测试，Accuracy 0.3584、QWK 0.4820。
- **纯随机切分**：Accuracy 0.5331、QWK 0.6814 —— 同一艘船相邻航段跨两侧造成的泄漏，不可用于评估。

## 五、特征清单（49 项）

| 组 | 数量 | 说明 |
|---|---|---|
| 船舶属性 | 6 | DWT、Length、Width、Maxspeed、MCR、reference_speed |
| 航行背景 | 3 | longitude、latitude、nearest_port_nm |
| 环境即时值 | 8 | VHM0、VTM10、风速、气温、SST、能见度、SIC、SIT |
| 时刻与季节 | 6 | 时/月/年积日的正余弦编码 |
| 船舶比值 | 5 | 基准/设计航速、功率/载重吨、载重吨/长度立方、宽长比、功率/船长 |
| 环境变化率 | 9 | 冰情、风、浪、气温的 1—12 h 差分 |
| 同期船群行为 | 1 | peer_loss_mean（留一法，排除本船） |
| 航段前方环境 | 8 | 3/6 h 后船位处的环境及变化量 |
| 水深 | 3 | 水深、浅水指示、陆架指示 |
