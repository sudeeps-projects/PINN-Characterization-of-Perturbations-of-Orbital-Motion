"""Victoria round-3 science checks.

(a) Warm-up robustness of the spectral-bias claim: is the plain-MLP error an
    initial transient, or persistent across the arc?
(b) Formal paired tests of Fourier-PINN vs classical integration (ISS) and
    Fourier-PINN vs plain MLP (synthetic), including trimmed variants.
Reuses the exact ISS setup from iss_experiment.py so the baseline matches Table 1.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ.setdefault("OMP_NUM_THREADS","4")
import json, numpy as np, pandas as pd
from scipy.integrate import solve_ivp
from scipy import stats
from sgp4.api import Satrec, jday

RESULTS="results"
MU=3.986004418e14; RE=6378136.3; J2=1.08262668e-3; OMEGA=7.292115e-5
RHO0=3.5e-12; H=60e3; CD=2.2; AREA=1000.0; MASS=420000.0
PRIOR_B=CD*AREA/MASS

L1="1 25544U          26169.80206886  .00008200  00000-0  15439-3 0    00"
L2="2 25544  51.6331 292.1003 0004667 202.2920 157.7867 15.49302471    05"
sat=Satrec.twoline2rv(L1,L2); epoch_jd=sat.jdsatepoch+sat.jdsatepochF
def jd_to_ts(jd): return pd.to_datetime((jd-2440587.5)*86400.0,unit="s",utc=True)
def ts_to_jday(ts):
    ts=pd.Timestamp(ts).tz_convert("UTC")
    return jday(ts.year,ts.month,ts.day,ts.hour,ts.minute,ts.second+ts.microsecond/1e6)
epoch_ts=jd_to_ts(epoch_jd)
mins=np.arange(0,12*60+5,5)
times=epoch_ts+pd.to_timedelta(mins,unit="m")
rows=[]
for ts in times:
    jd,fr=ts_to_jday(ts); e,r,v=sat.sgp4(jd,fr)
    if e==0: rows.append(dict(t=(ts-times[0]).total_seconds(),
                              x=r[0]*1e3,y=r[1]*1e3,z=r[2]*1e3,
                              vx=v[0]*1e3,vy=v[1]*1e3,vz=v[2]*1e3))
dense=pd.DataFrame(rows)
t_dense=dense.t.values
truth_pos=dense[["x","y","z"]].values/1000.0

H_REF=float(np.linalg.norm(dense[["x","y","z"]].values[0])-RE)
def rho_np(alt):
    alt=np.maximum(alt,120000.0); return RHO0*np.exp(-(alt-H_REF)/H)
def rhs(t,s,B):
    r=s[:3]; v=s[3:]; rn=np.linalg.norm(r); a=-MU*r/rn**3
    x,yy,z=r; f=1.5*J2*MU*RE**2/rn**5; cm=5*z*z/(rn*rn)
    a=a+np.array([f*x*(cm-1),f*yy*(cm-1),f*z*(cm-3)])
    om=np.array([0,0,OMEGA]); vr=v-np.cross(om,r); vrn=np.linalg.norm(vr)+1e-12
    a=a-0.5*rho_np(rn-RE)*B*vrn*vr
    return np.concatenate([v,a])

# sparse noisy observations, identical construction to iss_experiment.py
rng=np.random.default_rng(42)
sp=dense.iloc[::6].copy().reset_index(drop=True)
POS_N=0.20; VEL_N=0.0002
for c in ["x","y","z"]:    sp[c]=sp[c]/1000.0+rng.normal(0,POS_N,len(sp))
for c in ["vx","vy","vz"]: sp[c]=sp[c]/1000.0+rng.normal(0,VEL_N,len(sp))
y_obs=sp[["x","y","z","vx","vy","vz"]].values*1000.0

sol=solve_ivp(lambda t,y:rhs(t,y,PRIOR_B),(float(t_dense[0]),float(t_dense[-1])),y_obs[0],
              t_eval=t_dense,rtol=1e-8,atol=1e-8,method="RK45")
base_pos=sol.y.T[:,:3]/1000.0
base_err=np.linalg.norm(base_pos-truth_pos,axis=1)
base_rmse=float(np.sqrt(np.mean(base_err**2)))
print(f"[classical ISS] RMSE={base_rmse:.2f} km  (Table 1 says 16.57)")

four_err=np.load(RESULTS+"/iss_arrays.npz")["four_poserr"]
print(f"[fourier  ISS] RMSE={np.sqrt(np.mean(four_err**2)):.2f} km  (Table 1 says 5.68)")

out={}
def paired(a,b,na,nb,tag):
    """a,b are per-timestep position errors on the same grid."""
    w=stats.wilcoxon(a,b,alternative="less")
    t=stats.ttest_rel(a,b)
    diff=a-b; n=len(a)
    dz=float(np.mean(diff)/np.std(diff,ddof=1))
    res=dict(n=int(n), mean_a=float(a.mean()), mean_b=float(b.mean()),
             rmse_a=float(np.sqrt(np.mean(a**2))), rmse_b=float(np.sqrt(np.mean(b**2))),
             wilcoxon_stat=float(w.statistic), wilcoxon_p=float(w.pvalue),
             ttest_t=float(t.statistic), ttest_p=float(t.pvalue), cohens_dz=dz,
             pct_timesteps_a_better=float(100*np.mean(a<b)))
    out[tag]=res
    print(f"\n=== {tag}: {na} vs {nb} (n={n}) ===")
    print(f"  mean |err|  {na}={a.mean():9.3f} km   {nb}={b.mean():9.3f} km")
    print(f"  RMSE        {na}={res['rmse_a']:9.3f} km   {nb}={res['rmse_b']:9.3f} km")
    print(f"  Wilcoxon signed-rank (one-sided, {na}<{nb}): W={w.statistic:.0f}  p={w.pvalue:.3e}")
    print(f"  paired t: t={t.statistic:.2f}  p={t.pvalue:.3e}   Cohen's dz={dz:.2f}")
    print(f"  {na} closer at {res['pct_timesteps_a_better']:.1f}% of timesteps")
    return res

paired(four_err, base_err, "Fourier", "classical", "iss_fourier_vs_classical")

# synthetic: Fourier vs plain MLP, full and trimmed
d=np.load(RESULTS+"/synth_arrays.npz")
T=d['truth_state']; P=d['plain_pred42']; F=d['four_pred42']; hrs=d['truth_t']/3600.0
ep=np.linalg.norm(P[:,:3]-T[:,:3],axis=1); ef=np.linalg.norm(F[:,:3]-T[:,:3],axis=1)
paired(ef, ep, "Fourier", "plainMLP", "synth_fourier_vs_plain_full")
m=hrs>=hrs.max()*0.10
paired(ef[m], ep[m], "Fourier", "plainMLP", "synth_fourier_vs_plain_drop10pct")
# trim the largest 10% of plain-MLP errors, as Victoria asked
keep=ep<=np.percentile(ep,90)
paired(ef[keep], ep[keep], "Fourier", "plainMLP", "synth_fourier_vs_plain_trim_top10")

# warm-up profile
seg={}
for lo,hi in [(0,10),(10,25),(25,50),(50,75),(75,100)]:
    mm=(hrs>=hrs.max()*lo/100)&(hrs<hrs.max()*hi/100)
    seg[f"{lo}-{hi}%"]=dict(plain_rmse=float(np.sqrt(np.mean(ep[mm]**2))),
                            four_rmse=float(np.sqrt(np.mean(ef[mm]**2))))
drop={}
for w in [0,5,10,20,30]:
    mm=hrs>=hrs.max()*w/100
    drop[f"drop_{w}pct"]=dict(plain_rmse=float(np.sqrt(np.mean(ep[mm]**2))),
                              four_rmse=float(np.sqrt(np.mean(ef[mm]**2))))
out["warmup_segments"]=seg; out["warmup_drop"]=drop
out["plain_frac_sqerr_first10pct"]=float(100*np.sum(ep[hrs<hrs.max()*0.1]**2)/np.sum(ep**2))
out["plain_pct_timesteps_over_1000km"]=float(100*np.mean(ep>1000))
out["classical_iss_rmse"]=base_rmse
json.dump(out,open(RESULTS+"/stats_tests.json","w"),indent=2)
print("\nsaved stats_tests.json")
