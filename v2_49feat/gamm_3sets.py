import json, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import train_v2 as T
from gamm49 import design, grid_id, F49
from train_ensemble import qwk
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, SplineTransformer
from sklearn.linear_model import SGDClassifier
seg=pd.read_parquet("data_seg/seg_1h_lead3.parquet").sort_values(["year","doy"],kind="stable").reset_index(drop=True)
n=len(seg); nt=int(n*0.2); nv=int((n-nt)*0.15)
tr=seg.iloc[:n-nt-nv]; va=seg.iloc[n-nt-nv:n-nt]; te=seg.iloc[n-nt:]
th=T.build_thresholds(tr["speed_loss_rate"])
y={k:T.cut_grades(d["speed_loss_rate"].to_numpy(),th) for k,d in [("Train",tr),("Valid",va),("Test",te)]}
imp=SimpleImputer(strategy="median"); imp.fit(tr[F49].astype("float64"))
spl=SplineTransformer(n_knots=6,degree=3,knots="quantile",include_bias=False,extrapolation="constant")
spl.fit(imp.transform(tr[F49].astype("float64")))
ohe=OneHotEncoder(handle_unknown="ignore",sparse_output=True); ohe.fit(grid_id(tr))
Z={k:design(d,F49,imp,spl,ohe) for k,d in [("Train",tr),("Valid",va),("Test",te)]}
clf=SGDClassifier(loss="log_loss",penalty="l2",alpha=1e-5,learning_rate="optimal",random_state=42,n_jobs=-1)
classes=np.arange(5); best=(1e9,None,0); wait=0
rng=np.random.RandomState(42)
for ep in range(1,13):   # 固定 12 轮：与 gamm49.py 的早停终点一致，保证两份材料同源
    o=rng.permutation(Z["Train"].shape[0])
    for b0 in range(0,len(o),200_000):
        i=o[b0:b0+200_000]; clf.partial_fit(Z["Train"][i],y["Train"][i],classes=classes)
    pv=clf.predict_proba(Z["Valid"]); ll=-np.mean(np.log(np.clip(pv[np.arange(len(y["Valid"])),y["Valid"]],1e-9,1)))
    print("  epoch",ep,"valid logloss %.4f"%ll, flush=True)
out={}
for k in ("Train","Valid","Test"):
    pred=clf.predict(Z[k]); yt=y[k]
    out[k]=dict(acc=round(accuracy_score(yt,pred),4), f1=round(f1_score(yt,pred,average="macro"),4),
                qwk=round(qwk(yt,pred),4), w1=round(float((np.abs(yt-pred)<=1).mean()),4))
print("epochs",ep,"best",best[1]); print(json.dumps(out))
json.dump(out,open("figs/gamm_3sets.json","w"),indent=2)
