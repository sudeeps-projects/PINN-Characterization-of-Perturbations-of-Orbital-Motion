# -*- coding: utf-8 -*-
"""THE FAIR COMPARATOR (JHSS reviewer's required experiment).

Batch weighted least-squares orbit determination: fit the initial state AND the
drag parameter to the SAME sparse noisy observations the PINN received (all 72
synthetic / all 25 ISS points, positions and velocities, weighted by the known
noise sigmas), then propagate the known equations of motion. Evaluate on the
same dense grids with the same metrics as the PINN.

Synthetic: 5 independent noise draws (seeds 42-46), same as the paper.
ISS:       single draw (seed 42), same as the paper.
"""
import os, json, time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ.setdefault("OMP_NUM_THREADS","4")
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

RESULTS="results"
out={}

# ================= SYNTHETIC (identical to grid_refine.py / NB1) =================
MU=398600.4418; R=6378.1363; J2C=1.08262668e-3; TRUE=3.245e-5; H=60.0
def dyn(t,s,c):
    r,v=s[:3],s[3:]; rn=np.linalg.norm(r); a=-MU*r/rn**3
    x,y,z=r; f=1.5*J2C*MU*R**2/rn**5; cm=5*z*z/rn**2
    a=a+np.array([f*x*(cm-1),f*y*(cm-1),f*z*(cm-3)])
    alt=max(rn-R,0.0); a=a-c*np.exp(-alt/H)*np.linalg.norm(v)*v
    return np.concatenate([v,a])

radius0=R+500.0; vc=np.sqrt(MU/radius0); inc=np.deg2rad(35.0)
state0=np.array([radius0,0,0,0,vc*np.cos(inc),vc*np.sin(inc)])
period=2*np.pi*np.sqrt(radius0**3/MU); t_final=8*period
t_eval=np.linspace(0,t_final,2500)
sol=solve_ivp(lambda t,y:dyn(t,y,TRUE),(0,t_final),state0,t_eval=t_eval,rtol=1e-9,atol=1e-9)
truth_t=sol.t; truth_state=sol.y.T
obs_idx=np.arange(0,len(truth_t),35); obs_t=truth_t[obs_idx]
POS_S=0.5; VEL_S=0.002
def make_noisy(seed):
    rng=np.random.default_rng(seed); o=truth_state[obs_idx].copy()
    o[:,:3]+=rng.normal(0,POS_S,o[:,:3].shape); o[:,3:]+=rng.normal(0,VEL_S,o[:,3:].shape)
    return o

def sim_states(p,ts):
    s=solve_ivp(lambda t,y:dyn(t,y,p[6]),(float(ts[0]),float(ts[-1])),p[:6],
                t_eval=ts,rtol=1e-9,atol=1e-9)
    return s.y.T

W=np.concatenate([np.full(3,1.0/POS_S),np.full(3,1.0/VEL_S)])   # 1/sigma weights
def fit_synth(obs):
    def resid(p):
        pr=sim_states(p,obs_t)
        return ((pr-obs)*W).ravel()
    p0=np.concatenate([obs[0],[1e-5]])          # first noisy obs + PINN's initial drag guess
    r=least_squares(resid,p0,x_scale=[1,1,1,1e-3,1e-3,1e-3,1e-5],
                    bounds=(np.concatenate([p0[:6]-500,[0.0]]),
                            np.concatenate([p0[:6]+500,[1e-3]])),
                    xtol=1e-12,ftol=1e-12)
    return r

print("=== SYNTHETIC: weighted batch LSQ, 5 noise draws ===",flush=True)
sp=[];sv=[];sd=[];stime=[];snfev=[]
for seed in (42,43,44,45,46):
    t0=time.time(); r=fit_synth(make_noisy(seed)); el=time.time()-t0
    dense=sim_states(r.x,truth_t)
    prm=float(np.sqrt(np.mean(np.sum((dense[:,:3]-truth_state[:,:3])**2,axis=1))))
    vrm=float(np.sqrt(np.mean(np.sum((dense[:,3:]-truth_state[:,3:])**2,axis=1))))*1000.0
    dre=100*abs(r.x[6]-TRUE)/TRUE
    sp.append(prm);sv.append(vrm);sd.append(dre);stime.append(el);snfev.append(r.nfev)
    print(f"  seed {seed}: posRMSE={prm:8.4f} km  velRMSE={vrm:7.3f} m/s  "
          f"drag={r.x[6]:.4e} ({dre:6.3f}%)  [{el:5.1f}s, {r.nfev} evals]",flush=True)
sp,sv,sd=np.array(sp),np.array(sv),np.array(sd)
out["synthetic"]=dict(pos_rmse_mean=float(sp.mean()),pos_rmse_sd=float(sp.std()),
    vel_rmse_mean=float(sv.mean()),vel_rmse_sd=float(sv.std()),
    drag_relerr_mean=float(sd.mean()),drag_relerr_sd=float(sd.std()),
    fit_seconds_mean=float(np.mean(stime)),nfev_mean=float(np.mean(snfev)))
print(f"  SYNTH LSQ: pos {sp.mean():.4f} ± {sp.std():.4f} km | vel {sv.mean():.3f} ± {sv.std():.3f} m/s"
      f" | drag err {sd.mean():.4f} ± {sd.std():.4f} %   (Fourier PINN: 0.78 ± 0.03 km)",flush=True)

# ================= ISS (identical TLE + noise to iss_experiment.py) =================
import pandas as pd
from sgp4.api import Satrec, jday
L1="1 25544U          26169.80206886  .00008200  00000-0  15439-3 0    00"
L2="2 25544  51.6331 292.1003 0004667 202.2920 157.7867 15.49302471    05"
sat=Satrec.twoline2rv(L1,L2)
epoch_jd=sat.jdsatepoch+sat.jdsatepochF
epoch_ts=pd.to_datetime((epoch_jd-2440587.5)*86400.0,unit="s",utc=True)
def ts_to_jday(ts):
    ts=pd.Timestamp(ts).tz_convert("UTC")
    return jday(ts.year,ts.month,ts.day,ts.hour,ts.minute,
                ts.second+ts.microsecond/1e6+ts.nanosecond/1e9)
def propagate(minutes_step,total_hours):
    mins=np.arange(0,total_hours*60+minutes_step,minutes_step)
    rec=[]
    for m in mins:
        ts=epoch_ts+pd.to_timedelta(float(m),unit="m")
        jd,fr=ts_to_jday(ts); e,rr,vv=sat.sgp4(jd,fr)
        if e==0: rec.append([m*60.0,*rr,*vv])
    return np.array(rec)
A=propagate(5,12)                     # 145 x 7 : t, r(km), v(km/s)
t_dense=A[:,0]; y_dense=A[:,1:]
STRIDE=6; POS_N=0.20; VEL_N=0.0002
sp_i=A[::STRIDE].copy()
rng=np.random.default_rng(42)
sp_i[:,1:4]+=rng.normal(0,POS_N,sp_i[:,1:4].shape)
sp_i[:,4:7]+=rng.normal(0,VEL_N,sp_i[:,4:7].shape)
t_obs=sp_i[:,0]; y_obs=sp_i[:,1:]
print(f"\n=== ISS: dense {len(t_dense)}, sparse {len(t_obs)} ===",flush=True)

# SI dynamics from NB4 (two-body + J2 + rotating-atmosphere drag, exponential rho)
MUsi=3.986004418e14; REsi=6378136.3; OMEGA=7.2921150e-5
PRIOR_B=2.2*1000.0/420000.0
H_SCALE=60_000.0; RHO_REF=3.5e-12
H_REF=(np.linalg.norm(y_dense[0,:3])*1000.0)-REsi
def rhs_iss(t,s,B):
    r=s[:3]; v=s[3:]; rn=np.linalg.norm(r)
    a=-MUsi*r/rn**3
    x,y,z=r; f=1.5*J2C*MUsi*REsi**2/rn**5; cm=5*z*z/rn**2
    a=a+np.array([f*x*(cm-1),f*y*(cm-1),f*z*(cm-3)])
    alt=max(rn-REsi,120000.0)
    rho=RHO_REF*np.exp(-(alt-H_REF)/H_SCALE)
    vrel=v-np.cross([0,0,OMEGA],r)
    a=a-0.5*rho*B*np.linalg.norm(vrel)*vrel
    return np.concatenate([v,a])

def sim_iss(p,ts):
    s=solve_ivp(lambda t,y:rhs_iss(t,y,p[6]),(float(ts[0]),float(ts[-1])),p[:6],
                t_eval=ts,rtol=1e-9,atol=1e-6)
    return s.y.T
Wi=np.concatenate([np.full(3,1.0/(POS_N*1000.0)),np.full(3,1.0/(VEL_N*1000.0))])
obs_si=y_obs*1000.0
def resid_iss(p):
    pr=sim_iss(p,t_obs)
    return ((pr-obs_si)*Wi).ravel()
p0=np.concatenate([obs_si[0],[PRIOR_B]])
t0=time.time()
r=least_squares(resid_iss,p0,x_scale=[1e3,1e3,1e3,1,1,1,1e-3],
                bounds=(np.concatenate([p0[:6]-5e5,[0.0]]),
                        np.concatenate([p0[:6]+5e5,[1.0]])),
                xtol=1e-12,ftol=1e-12)
el=time.time()-t0
dense_fit=sim_iss(r.x,t_dense)/1000.0          # back to km, km/s
perr=np.linalg.norm(dense_fit[:,:3]-y_dense[:,:3],axis=1)
prm=float(np.sqrt(np.mean(perr**2)))
vrm=float(np.sqrt(np.mean(np.sum((dense_fit[:,3:]-y_dense[:,3:])**2,axis=1))))*1000.0
out["iss"]=dict(pos_rmse_km=prm,vel_rmse_ms=vrm,final_err_km=float(perr[-1]),
    fitted_B=float(r.x[6]),prior_B=PRIOR_B,fit_seconds=el,nfev=int(r.nfev))
np.savez(RESULTS+"/lsq_arrays.npz",iss_dense_fit=dense_fit,iss_perr=perr,t_dense=t_dense)
print(f"  ISS LSQ: posRMSE={prm:.4f} km  velRMSE={vrm:.3f} m/s  final={perr[-1]:.4f} km  "
      f"B={r.x[6]:.4e} (prior {PRIOR_B:.4e})  [{el:.1f}s, {r.nfev} evals]",flush=True)
print(f"  (Fourier PINN ISS: 5.68 km / 8.0 m/s / final 1.44 km; old single-point baseline: 16.57 km)",flush=True)

json.dump(out,open(RESULTS+"/lsq_baseline.json","w"),indent=2)
print("\nsaved lsq_baseline.json",flush=True)
