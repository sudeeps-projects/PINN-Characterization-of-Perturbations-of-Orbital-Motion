"""Observation-window sweep for the classical grid-search drag estimator,
over five noise draws with the finer 400-point grid. Fixes the zero-variance
artifact from the original coarse grid."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ.setdefault("OMP_NUM_THREADS","4")
import json, numpy as np
from scipy.integrate import solve_ivp

RESULTS="results"
MU=398600.4418; R=6378.1363; J2=1.08262668e-3; TRUE=3.245e-5; H=60.0
def norm(v): return np.linalg.norm(v)
def dyn(t,s,c):
    r,v=s[:3],s[3:]; rn=norm(r); a=-MU*r/rn**3
    x,y,z=r; f=1.5*J2*MU*R**2/rn**5; cm=5*z*z/rn**2
    a=a+np.array([f*x*(cm-1),f*y*(cm-1),f*z*(cm-3)])
    alt=max(rn-R,0.0); a=a-c*np.exp(-alt/H)*norm(v)*v
    return np.concatenate([v,a])
# regenerate the NB1 truth
radius0=R+500.0; vc=np.sqrt(MU/radius0); inc=np.deg2rad(35.0)
state0=np.array([radius0,0,0,0,vc*np.cos(inc),vc*np.sin(inc)])
period=2*np.pi*np.sqrt(radius0**3/MU); t_final=8*period
t_eval=np.linspace(0,t_final,2500)
sol=solve_ivp(lambda t,y:dyn(t,y,TRUE),(0,t_final),state0,t_eval=t_eval,rtol=1e-9,atol=1e-9)
truth_t=sol.t; truth_state=sol.y.T
obs_idx=np.arange(0,len(truth_t),35); obs_t=truth_t[obs_idx]
def make_noisy(seed):
    rng=np.random.default_rng(seed); o=truth_state[obs_idx].copy()
    o[:,:3]+=rng.normal(0,0.5,o[:,:3].shape); o[:,3:]+=rng.normal(0,0.002,o[:,3:].shape); return o
GRID=np.linspace(0.2e-5,4.0e-5,400); init=truth_state[0].copy()
def sim(c,ts):
    s=solve_ivp(lambda t,y:dyn(t,y,c),(float(ts[0]),float(ts[-1])),init,t_eval=ts,rtol=1e-8,atol=1e-8); return s.y[:3].T
out={}
for Wh in [2,4,8,12]:
    mask=obs_t<=Wh*3600.0; ts=obs_t[mask]
    rel=[]
    for sd in [42,43,44,45,46]:
        ob=make_noisy(sd)[mask]
        losses=[np.mean(np.sum((sim(c,ts)-ob[:,:3])**2,axis=1)) for c in GRID]
        best=GRID[int(np.argmin(losses))]; rel.append(100*abs(best-TRUE)/TRUE)
    rel=np.array(rel); out[f"{Wh}h"]=[float(rel.mean()),float(rel.std()),int(mask.sum())]
    print(f"W={Wh}h nobs={int(mask.sum())} relerr={rel.mean():.2f} ± {rel.std():.2f} %")
json.dump(out,open(RESULTS+"/window_sweep.json","w"),indent=2)
print(json.dumps(out,indent=2))
