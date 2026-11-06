import json, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score
import train_v2 as T
from train_ensemble import PARAMS, qwk
from train_final import ENV, DYNAMICS, SHIP, NAV, PEER
from train_lead import DEPTH_FEATS, LEAD_FEATS
FEATS=(SHIP+NAV+ENV+["hour_sin","hour_cos","month_sin","month_cos","doy_sin","doy_cos",
 "speed_ratio","power_per_dwt","dwt_per_l3","beam_ratio","mcr_per_len"]+DYNAMICS+PEER+LEAD_FEATS+DEPTH_FEATS)
seg=pd.read_parquet("data_seg/seg_1h_lead3.parquet").sort_values(["year","doy"],kind="stable").reset_index(drop=True)
n=len(seg); nt=int(n*0.2); nv=int((n-nt)*0.15)
tr=seg.iloc[:n-nt-nv]; va=seg.iloc[n-nt-nv:n-nt]; te=seg.iloc[n-nt:]
th=T.build_thresholds(tr["speed_loss_rate"])
y={k:T.cut_grades(d["speed_loss_rate"].to_numpy(),th) for k,d in [("Train",tr),("Valid",va),("Test",te)]}
p=dict(PARAMS,objective="multiclass",num_class=5,metric="multi_logloss",seed=42,bagging_seed=43,feature_fraction_seed=44)
b=lgb.train(p,lgb.Dataset(tr[FEATS].astype("float32"),label=y["Train"]),
  num_boost_round=600,valid_sets=[lgb.Dataset(va[FEATS].astype("float32"),label=y["Valid"])],
  callbacks=[lgb.early_stopping(100,verbose=False)])
best=b.best_iteration
out={}
for k,d in [("Train",tr),("Valid",va),("Test",te)]:
    pr=b.predict(d[FEATS].astype("float32"),num_iteration=best)
    pred=pr.argmax(1); yt=y[k]
    out[k]=dict(acc=round(accuracy_score(yt,pred),4), f1=round(f1_score(yt,pred,average="macro"),4),
                qwk=round(qwk(yt,pred),4), w1=round(float((np.abs(yt-pred)<=1).mean()),4))
print("best_iter",best); print(json.dumps(out))
json.dump(out,open("figs/lgb_3sets.json","w"),indent=2)
