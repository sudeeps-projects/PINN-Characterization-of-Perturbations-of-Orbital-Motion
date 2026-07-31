"""Continuous (Brent-refined) drag estimator to remove the grid-quantization
artifact Victoria flagged, so the across-noise-draw variance is genuine.
Also runs the observation-window sweep with the same refined estimator."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ.setdefault("OMP_NUM_THREADS","4")
import json, numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

RESULTS="results"
MU=398600.4418; R=6378.1363; J2=1.08262668e-3; TRUE=3.245e-5; H=60.0
def norm(v): return np.linalg.norm(v)
def dyn(t,s,c):
    r,v=s[:3],s[3:]; rn=norm(r); a=-MU*r/rn**3
    x,y,z=r; f=1.5*J2*MU*R**2/rn**5; cm=5*z*z/rn**2
    a=a+np.array([f*x*(cm-1),f*y*(cm-1),f*z*(cm-3)])
    alt=max(rn-R,0.0); a=a-c*np.exp(-alt/H)*norm(v)*v
    return np.concatenate([v,a])
radius0=R+500.0; vc=np.sqrt(MU/radius0); inc=np.deg2rad(35.0)
state0=np.array([radius0,0,0,0,vc*np.cos(inc),vc*np.sin(inc)])
period=2*np.pi*np.sqrt(radius0**3/MU); t_final=8*period
t_eval=np.linspace(0,t_final,2500)
sol=solve_ivp(lambda t,y:dyn(t,y,TRUE),(0,t_final),state0,t_eval=t_eval,rtol=1e-9,atol=1e-9)
truth_t=sol.t; truth_state=sol.y.T
obs_idx=np.arange(0,len(truth_t),35); obs_t_full=truth_t[obs_idx]
init=truth_state[0].copy()
def make_noisy(seed):
    rng=np.random.default_rng(seed); o=truth_state[obs_idx].copy()
    o[:,:3]+=rng.normal(0,0.5,o[:,:3].shape); o[:,3:]+=rng.normal(0,0.002,o[:,3:].shape); return o
def sim(c,ts):
    s=solve_ivp(lambda t,y:dyn(t,y,c),(float(ts[0]),float(ts[-1])),init,t_eval=ts,rtol=1e-9,atol=1e-9); return s.y[:3].T
def estimate(ts,obs_pos):
    def mse(c): pp=sim(c,ts); return float(np.mean(np.sum((pp-obs_pos)**2,axis=1)))
    r=minimize_scalar(mse,bounds=(0.2e-5,4.0e-5),method='bounded',options={'xatol':1e-9})
    return float(r.x)

SEEDS=[42,43,44,45,46]
# full-window continuous estimate over 5 draws
best=[]
for sd in SEEDS:
    ob=make_noisy(sd); b=estimate(obs_t_full,ob[:,:3]); best.append(b)
    print(f"[refine full] seed={sd} best={b:.6e}")
best=np.array(best); rel=100*np.abs(best-TRUE)/TRUE
out={}
out["full"]=dict(best_mean=float(best.mean()),best_sd=float(best.std()),
                 relerr_mean=float(rel.mean()),relerr_sd=float(rel.std()),
                 best_seed42=float(best[0]))
print(f"full: best={best.mean():.6e} ± {best.std():.2e}  relerr={rel.mean():.3f} ± {rel.std():.3f} %")
# window sweep
for Wh in [2,4,8,12]:
    mask=obs_t_full<=Wh*3600.0; ts=obs_t_full[mask]; rr=[]
    for sd in SEEDS:
        ob=make_noisy(sd)[mask]; rr.append(100*abs(estimate(ts,ob[:,:3])-TRUE)/TRUE)
    rr=np.array(rr); out[f"{Wh}h"]=dict(relerr_mean=float(rr.mean()),relerr_sd=float(rr.std()),nobs=int(mask.sum()))
    print(f"W={Wh}h nobs={int(mask.sum())} relerr={rr.mean():.3f} ± {rr.std():.3f} %")
json.dump(out,open(RESULTS+"/grid_refine.json","w"),indent=2)
print(json.dumps(out,indent=2))
