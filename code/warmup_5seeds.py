"""Warm-up robustness check across all five plain-MLP seeds.

Retrains the plain MLP exactly as synth_experiment.py does (same data, same
architecture, same 600 epochs, seeds 42-46), keeps the per-timestep position
error, and recomputes RMSE with a leading warm-up fraction excluded.
Answers: is the plain MLP's error a start-of-arc transient, or broad?
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ.setdefault("OMP_NUM_THREADS","4")
import json, math, time
import numpy as np, torch, torch.nn as nn
from scipy.integrate import solve_ivp

R="results"
src=open("synth_experiment.py",encoding="utf-8").read()
cut=src.index("# ================= FOURIER PINN")
g={}
exec(compile(src[:cut],"synth_setup","exec"), g)      # data + plain-MLP definitions

truth_t=g["truth_t"]; truth_state=g["truth_state"]
train_plain=g["train_plain"] if "train_plain" in g else None
if train_plain is None:
    raise SystemExit("plain trainer not found; keys: "+", ".join(k for k in g if "plain" in k.lower()))

T=truth_t[-1]; FRACS=(0.0,0.02,0.05,0.10,0.20,0.30,0.50)
per_seed={}; errs=[]
for sd in (42,43,44,45,46):
    t0=time.time()
    pr,vr,pred = train_plain(sd)
    e=np.linalg.norm(pred[:,:3]-truth_state[:,:3],axis=1)
    errs.append(e)
    per_seed[sd]={f"{int(f*100)}%":float(np.sqrt(np.mean(e[truth_t>=f*T]**2))) for f in FRACS}
    print(f"[plain seed={sd}] full={pr:.1f} km  excl50%={per_seed[sd]['50%']:.1f} km  ({time.time()-t0:.0f}s)",flush=True)

errs=np.array(errs)
summary={}
for f in FRACS:
    m=truth_t>=f*T
    vals=np.sqrt(np.mean(errs[:,m]**2,axis=1))
    summary[f"{int(f*100)}%"]=dict(mean=float(vals.mean()),sd=float(vals.std()),n_kept=int(m.sum()))
    print(f"  exclude first {f*100:4.0f}%  n={m.sum():4d}   RMSE = {vals.mean():7.1f} +/- {vals.std():.1f} km")

dec=[]
for i in range(10):
    m=(truth_t>=i*0.1*T)&(truth_t<(i+1)*0.1*T)
    v=np.sqrt(np.mean(errs[:,m]**2,axis=1)); dec.append([float(v.mean()),float(v.std())])
share=[float(np.sum(e[truth_t<0.1*T]**2)/np.sum(e**2)) for e in errs]
print("\n  decile RMSE (mean +/- SD over 5 seeds), km:")
for i,(mu,sd_) in enumerate(dec,1): print(f"    {i:2d}  {mu:7.1f} +/- {sd_:.1f}")
print(f"\n  share of squared error in first 10% of arc: {100*np.mean(share):.1f}% +/- {100*np.std(share):.1f}%")

out=dict(per_seed=per_seed, summary=summary, decile_mean_sd=dec,
         share_first_decile_mean=float(np.mean(share)), share_first_decile_sd=float(np.std(share)))
prev=json.load(open(R+"/victoria_r3.json")); prev["warmup_5seed"]=out
json.dump(prev,open(R+"/victoria_r3.json","w"),indent=2)
print("\nsaved to victoria_r3.json")
