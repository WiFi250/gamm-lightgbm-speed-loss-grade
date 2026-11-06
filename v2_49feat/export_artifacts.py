#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出可核验的模型与预测文件，使算法记录中的“模型文件/预测文件/分类报告”真实存在。

输出目录 model_v2/output_49feat/：
  lightgbm_49feat_segment.txt   单种子（seed=42）原生模型
  run_summary.json              样本规模、切分、参数与三集指标（集成口径）
  test_predictions.csv          测试集真实等级、预测等级、五类概率（集成）
  classification_report.csv     分等级 Precision/Recall/F1/Support（集成）
  shap_feature_importance_global.csv  平均绝对 SHAP（49 项）
  metrics_bootstrap.csv         IMO 分组 bootstrap 点估计与区间
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

import train_v2 as T
from train_ensemble import PARAMS
from train_final import ENV, DYNAMICS, SHIP, NAV, PEER
from train_lead import DEPTH_FEATS, LEAD_FEATS

BASE = Path("/Users/wangyifan/Documents/seaice/博四/挑战者杯比赛/model_v2")
OUT = BASE / "output_49feat"
SEG = BASE / "data_seg" / "seg_1h_lead3.parquet"
FEATS = (SHIP + NAV + ENV
         + ["hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos",
            "speed_ratio", "power_per_dwt", "dwt_per_l3", "beam_ratio", "mcr_per_len"]
         + DYNAMICS + PEER + LEAD_FEATS + DEPTH_FEATS)


def main() -> None:
    try:
        OUT.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass
    seg = pd.read_parquet(SEG).sort_values(["year", "doy"], kind="stable").reset_index(drop=True)
    n = len(seg)
    n_test, n_valid = int(n * 0.2), int((n - int(n * 0.2)) * 0.15)
    i1, i2 = n - n_test - n_valid, n - n_test
    tr, va, te = seg.iloc[:i1], seg.iloc[i1:i2], seg.iloc[i2:]
    th = T.build_thresholds(tr["speed_loss_rate"])
    y_tr = T.cut_grades(tr["speed_loss_rate"].to_numpy(), th)
    y_va = T.cut_grades(va["speed_loss_rate"].to_numpy(), th)
    y_te = T.cut_grades(te["speed_loss_rate"].to_numpy(), th)

    p = dict(PARAMS, objective="multiclass", num_class=5, metric="multi_logloss",
             seed=42, bagging_seed=43, feature_fraction_seed=44)
    bst = lgb.train(p, lgb.Dataset(tr[FEATS].astype("float32"), label=y_tr),
                    num_boost_round=600,
                    valid_sets=[lgb.Dataset(va[FEATS].astype("float32"), label=y_va)],
                    callbacks=[lgb.early_stopping(100, verbose=False)])
    best = bst.best_iteration or bst.current_iteration()
    bst.save_model(str(OUT / "lightgbm_49feat_segment.txt"))
    print("模型已保存，best_iter", best, flush=True)

    proba = np.load(BASE / "runs_lead" / "proba_test_L5.npy")
    pred = proba.argmax(1)

    sub = te[["longitude", "latitude", "speed_loss_rate"]].reset_index(drop=True)
    for j in range(5):
        sub[f"p{j + 1}"] = proba[:, j]
    sub["loss_grade"] = y_te
    sub["pred_grade"] = pred
    sub["correct"] = (sub["loss_grade"] == sub["pred_grade"]).astype(int)
    sub.to_csv(OUT / "test_predictions.csv", index=False, float_format="%.5f")

    P, R, F, S = precision_recall_fscore_support(y_te, pred, labels=range(5), zero_division=0)
    rep = pd.DataFrame({"grade": [f"G{i + 1}" for i in range(5)],
                        "precision": P.round(4), "recall": R.round(4),
                        "f1": F.round(4), "support": S})
    rep.to_csv(OUT / "classification_report.csv", index=False, encoding="utf-8-sig")

    tab = json.load(open(BASE / "figs" / "shap_panels_mean_abs.json"))
    pd.DataFrame({"feature": list(tab), "mean_abs_shap": list(tab.values())}).to_csv(
        OUT / "shap_feature_importance_global.csv", index=False, encoding="utf-8-sig")

    mb = json.load(open(BASE / "figs" / "metrics_ens.json"))
    rows = []
    for m in ("LightGBM", "GAMM"):
        for s in ("Train", "Valid", "Test"):
            d = mb[m][s]
            rows.append({"model": m, "split": s,
                         "accuracy": d["acc"], "accuracy_lo": d["ci"]["acc"][0],
                         "accuracy_hi": d["ci"]["acc"][1],
                         "macro_f1": d["f1"], "macro_f1_lo": d["ci"]["f1"][0],
                         "macro_f1_hi": d["ci"]["f1"][1],
                         "qwk": d["qwk"], "qwk_lo": d["ci"]["qwk"][0], "qwk_hi": d["ci"]["qwk"][1],
                         "within_one": d["w1"], "within_one_lo": d["ci"]["w1"][0],
                         "within_one_hi": d["ci"]["w1"][1], "top2": d["top2"]})
    pd.DataFrame(rows).to_csv(OUT / "metrics_bootstrap.csv", index=False, encoding="utf-8-sig")

    summary = {"input_files": "LNG_ML_cruising_2013.csv — LNG_ML_cruising_2023.csv",
               "records_total": 32954743, "positive_records": 28684792,
               "segments_1h": int(n), "train": int(len(tr)), "valid": int(len(va)),
               "test": int(len(te)), "features": len(FEATS), "classes": 5,
               "params": {"learning_rate": p["learning_rate"], "num_leaves": p["num_leaves"],
                          "min_data_in_leaf": p["min_data_in_leaf"],
                          "feature_fraction": p["feature_fraction"],
                          "bagging_fraction": p["bagging_fraction"],
                          "bagging_freq": p["bagging_freq"],
                          "lambda_l1": 0.0, "lambda_l2": p["lambda_l2"]},
               "num_boost_round": 600, "early_stopping": 100, "best_iteration": int(best),
               "thresholds": [round(float(v), 6) for v in th],
               "ensemble_seeds": [42, 7, 2024, 913, 555],
               "metrics": mb["LightGBM"]}
    json.dump(summary, open(OUT / "run_summary.json", "w"), ensure_ascii=False, indent=2)
    print("写出", OUT)


if __name__ == "__main__":
    main()
