"""ISS branch rerun (mirrors Notebook 4). Uses the reconstructed exact TLE
(epoch 2026-06-18 19:14 UTC). Single run, seed 42, as the paper reports.
Reproduces plain-MLP, Fourier-PINN and classical baseline; saves arrays for Fig 3/Fig 6."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "4")
import json, math, time
import numpy as np, pandas as pd
from scipy.integrate import solve_ivp
import torch, torch.nn as nn
from sgp4.api import Satrec, jday
import matplotlib; matplotlib.use("Agg")

RESULTS="results"
SEED=42; np.random.seed(SEED); torch.manual_seed(SEED)

L1="1 25544U          26169.80206886  .00008200  00000-0  15439-3 0    00"
L2="2 25544  51.6331 292.1003 0004667 202.2920 157.7867 15.49302471    05"
sat=Satrec.twoline2rv(L1,L2)
epoch_jd=sat.jdsatepoch+sat.jdsatepochF
def jd_to_ts(jd): return pd.to_datetime((jd-2440587.5)*86400.0,unit="s",utc=True)
epoch_ts=jd_to_ts(epoch_jd)

def ts_to_jday(ts):
    ts=pd.Timestamp(ts).tz_convert("UTC")
    sec=ts.second+ts.microsecond/1e6+ts.nanosecond/1e9
    return jday(ts.year,ts.month,ts.day,ts.hour,ts.minute,sec)

R_EARTH_KM=6378.1363
def propagate(minutes_step,total_hours):
    mins=np.arange(0,total_hours*60+minutes_step,minutes_step)
    times=epoch_ts+pd.to_timedelta(mins,unit="m")
    rec=[]
    for ts in times:
        jd,fr=ts_to_jday(ts); e,r,v=sat.sgp4(jd,fr)
        if e==0:
            rec.append([ (ts-times[0]).total_seconds(), r[0],r[1],r[2],v[0],v[1],v[2] ])
    a=np.array(rec)
    df=pd.DataFrame(a,columns=["t_sec","x_km","y_km","z_km","vx_kms","vy_kms","vz_kms"])
    df["radius_km"]=np.linalg.norm(df[["x_km","y_km","z_km"]].to_numpy(),axis=1)
    df["altitude_km"]=df["radius_km"]-R_EARTH_KM
    return df

# dense 12h / 5min -> 145 ; altitude 24h for Fig 6
dense=propagate(5,12)
alt24=propagate(10,24)
print("dense rows",len(dense),"alt range",dense.altitude_km.min(),dense.altitude_km.max())

# sparse: stride 6 (30 min) -> 25, seed 42 noise
STRIDE=6; POS_N=0.20; VEL_N=0.0002
sparse=dense.iloc[::STRIDE].copy().reset_index(drop=True)
rng=np.random.default_rng(SEED)
for c in ["x_km","y_km","z_km"]: sparse[c]+=rng.normal(0,POS_N,len(sparse))
for c in ["vx_kms","vy_kms","vz_kms"]: sparse[c]+=rng.normal(0,VEL_N,len(sparse))
print("sparse rows",len(sparse))

# ---------- SI physics (NB4) ----------
MU=3.986004418e14; RE=6378136.3; J2=1.08262668e-3; OMEGA=7.2921150e-5
mass=420000.0; area=1000.0; cd=2.2
PRIOR_B=cd*area/mass
H_SCALE=60_000.0; RHO_REF=3.5e-12
def to_si(df):
    a=df[["x_km","y_km","z_km","vx_kms","vy_kms","vz_kms"]].to_numpy(float).copy()
    a[:,:3]*=1000; a[:,3:]*=1000; return a
t_dense=dense["t_sec"].to_numpy(float); y_dense=to_si(dense)
t_obs=sparse["t_sec"].to_numpy(float); y_obs=to_si(sparse)
H_REF=float(dense["altitude_km"].iloc[0]*1000.0)

def rho_np(alt):
    alt=np.maximum(alt,120000.0); return RHO_REF*np.exp(-(alt-H_REF)/H_SCALE)
def rhs(t,s,B=PRIOR_B):
    r=s[:3];v=s[3:];rn=np.linalg.norm(r); a=-MU*r/rn**3
    x,y,z=r; f=1.5*J2*MU*RE**2/rn**5; cm=5*z*z/rn**2
    a=a+np.array([f*x*(cm-1),f*y*(cm-1),f*z*(cm-3)])
    om=np.array([0,0,OMEGA]); vr=v-np.cross(om,r); vrn=np.linalg.norm(vr)+1e-12
    a=a-0.5*rho_np(rn-RE)*B*vrn*vr
    return np.concatenate([v,a])

# ---------- normalize ----------
T0=float(t_dense.min()); TSC=float(t_dense.max()-t_dense.min())
SM=y_dense.mean(0); SS=y_dense.std(0); SS[SS<1e-9]=1.0
def nt(t): return ((np.asarray(t)-T0)/TSC).reshape(-1,1)
def nstate(y): return (np.asarray(y)-SM)/SS
def dstate(y): return np.asarray(y)*SS+SM
smean_T=torch.tensor(SM,dtype=torch.float32).reshape(1,6)
sscale_T=torch.tensor(SS,dtype=torch.float32).reshape(1,6)
def denorm_t(yn): return yn*sscale_T+smean_T
t_obs_T=torch.tensor(nt(t_obs),dtype=torch.float32)
y_obs_T=torch.tensor(nstate(y_obs),dtype=torch.float32)
ic_T=torch.tensor(nstate(y_obs)[0:1],dtype=torch.float32)

def acc_t(r,v,logB):
    rn=torch.linalg.norm(r,dim=1,keepdim=True); a=-MU*r/(rn**3+1e-12)
    x=r[:,0:1];y=r[:,1:2];z=r[:,2:3]; f=1.5*J2*MU*RE**2/(rn**5+1e-12); cm=5*z*z/(rn**2+1e-12)
    a=a+torch.cat([f*x*(cm-1),f*y*(cm-1),f*z*(cm-3)],dim=1)
    om=torch.tensor([0.,0.,OMEGA]).reshape(1,3).repeat(r.shape[0],1)
    vr=v-torch.cross(om,r,dim=1); vrn=torch.linalg.norm(vr,dim=1,keepdim=True)+1e-12
    alt=torch.clamp(rn-RE,min=120000.0); rho=RHO_REF*torch.exp(-(alt-H_REF)/H_SCALE)
    a=a-0.5*rho*torch.exp(logB)*vrn*vr
    return a

# ---------- plain MLP PINN (InitialConditionPINN, NB4) ----------
class MLP(nn.Module):
    def __init__(s,w=96,d=4):
        super().__init__(); L=[]; ind=1
        for _ in range(d): L+=[nn.Linear(ind,w),nn.Tanh()]; ind=w
        L+=[nn.Linear(w,6)]; s.net=nn.Sequential(*L)
    def forward(s,t): return s.net(t)
class ICPINN(nn.Module):
    def __init__(s,ic,w=96,d=4):
        super().__init__(); s.register_buffer("ic",ic.clone().detach()); s.net=MLP(w,d)
    def forward(s,t): return s.ic+t*s.net(t)

def train_plain():
    torch.manual_seed(SEED)
    m=ICPINN(ic_T,96,4); logB=nn.Parameter(torch.tensor(math.log(PRIOR_B),dtype=torch.float32))
    opt=torch.optim.Adam(list(m.parameters())+[logB],lr=1e-3)
    sch=torch.optim.lr_scheduler.StepLR(opt,step_size=max(500//3,1),gamma=0.5)
    tcol=torch.linspace(0,1,160).reshape(-1,1)
    W={"data":20,"ic":10,"physics":1,"prior":0.05}; logprior=math.log(PRIOR_B)
    for ep in range(500):
        opt.zero_grad()
        data=torch.mean((m(t_obs_T)-y_obs_T)**2)
        ic=torch.mean((m(torch.zeros(1,1))-ic_T)**2)
        tc=tcol.clone().detach().requires_grad_(True); ps=denorm_t(m(tc))
        gr=[torch.autograd.grad(ps[:,i:i+1],tc,torch.ones_like(ps[:,i:i+1]),create_graph=True,retain_graph=True)[0] for i in range(6)]
        dyt=torch.cat(gr,dim=1)/TSC; r=ps[:,:3];v=ps[:,3:]
        ea=acc_t(r,v,logB)
        phys=torch.mean(((dyt[:,:3]-v)/7600.0)**2)+torch.mean(((dyt[:,3:]-ea)/8.7)**2)
        prior=(logB-logprior)**2
        loss=W["data"]*data+W["ic"]*ic+W["physics"]*phys+W["prior"]*prior
        loss.backward(); torch.nn.utils.clip_grad_norm_(list(m.parameters())+[logB],1.0)
        opt.step(); sch.step()
    m.eval()
    with torch.no_grad(): pred=dstate(m(torch.tensor(nt(t_dense),dtype=torch.float32)).numpy())
    return pred,float(torch.exp(logB).detach())

# ---------- Fourier PINN (NB4 cell 43) ----------
K=24; freqs=torch.arange(1,K+1,dtype=torch.float32).reshape(1,-1)
def ff(t): ang=2*math.pi*t*freqs; return torch.cat([t,torch.sin(ang),torch.cos(ang)],dim=1)
class FPINN(nn.Module):
    def __init__(s,ed,w=128,d=3):
        super().__init__(); L=[nn.Linear(ed,w),nn.Tanh()]
        for _ in range(d-1): L+=[nn.Linear(w,w),nn.Tanh()]
        L+=[nn.Linear(w,6)]; s.net=nn.Sequential(*L)
    def forward(s,e): return s.net(e)
def train_fourier():
    torch.manual_seed(SEED)
    obs_emb=ff(t_obs_T); emb0=ff(torch.zeros(1,1)); col=torch.linspace(0,1,200).reshape(-1,1)
    m=FPINN(obs_emb.shape[1]); logB=nn.Parameter(torch.tensor(math.log(PRIOR_B),dtype=torch.float32))
    opt=torch.optim.Adam(list(m.parameters())+[logB],lr=2e-3)
    sch=torch.optim.lr_scheduler.StepLR(opt,step_size=2000,gamma=0.5)
    W={"data":20,"ic":10,"physics":1,"prior":0.05}; logprior=math.log(PRIOR_B)
    hist={"epoch":[],"total":[],"data":[],"physics":[]}
    for ep in range(1,5001):
        opt.zero_grad()
        data=torch.mean((m(obs_emb)-y_obs_T)**2)
        ic=torch.mean((m(emb0)-ic_T)**2)
        tc=col.clone().detach().requires_grad_(True); yc=denorm_t(m(ff(tc)))
        gr=[torch.autograd.grad(yc[:,i:i+1],tc,torch.ones_like(yc[:,i:i+1]),create_graph=True,retain_graph=True)[0] for i in range(6)]
        dyt=torch.cat(gr,dim=1)/TSC; r,v=yc[:,:3],yc[:,3:]
        ea=acc_t(r,v,logB)
        phys=torch.mean(((dyt[:,:3]-v)/7600.0)**2)+torch.mean(((dyt[:,3:]-ea)/8.7)**2)
        prior=(logB-logprior)**2
        total=W["data"]*data+W["ic"]*ic+W["physics"]*phys+W["prior"]*prior
        total.backward(); torch.nn.utils.clip_grad_norm_(list(m.parameters())+[logB],1.0)
        opt.step(); sch.step()
        if ep%500==0 or ep==1:
            hist["epoch"].append(ep); hist["total"].append(float(total.detach()))
            hist["data"].append(float(data.detach())); hist["physics"].append(float(phys.detach()))
    m.eval()
    with torch.no_grad(): pred=dstate(m(ff(torch.tensor(nt(t_dense),dtype=torch.float32))).numpy())
    return pred,float(torch.exp(logB).detach()),hist

truth_pos=dense[["x_km","y_km","z_km"]].to_numpy()
truth_vel=dense[["vx_kms","vy_kms","vz_kms"]].to_numpy()*1000.0
def metrics(pred):
    pp=pred[:,:3]/1000.0; pv=pred[:,3:]
    prmse=float(np.sqrt(np.mean(np.sum((pp-truth_pos)**2,axis=1))))
    vrmse=float(np.sqrt(np.mean(np.sum((pv-truth_vel)**2,axis=1))))
    final=float(np.linalg.norm(pp[-1]-truth_pos[-1]))
    return prmse,vrmse,final,pp

t0=time.time()
plain_pred,plainB=train_plain()
pl_prmse,pl_vrmse,pl_final,pl_pp=metrics(plain_pred)
print(f"[plain ISS] posRMSE={pl_prmse:.1f} km velRMSE={pl_vrmse:.1f} m/s ({time.time()-t0:.0f}s)")
four_pred,fourB,fhist=train_fourier()
f_prmse,f_vrmse,f_final,f_pp=metrics(four_pred)
print(f"[fourier ISS] posRMSE={f_prmse:.3f} km velRMSE={f_vrmse:.3f} m/s final={f_final:.3f} km ({time.time()-t0:.0f}s)")

# classical baseline
sol=solve_ivp(lambda t,y:rhs(t,y,PRIOR_B),(float(t_dense[0]),float(t_dense[-1])),y_obs[0],
              t_eval=t_dense,rtol=1e-8,atol=1e-8,method="RK45")
base_pos=sol.y.T[:,:3]/1000.0
base_rmse=float(np.sqrt(np.mean(np.sum((base_pos-truth_pos)**2,axis=1))))
base_final=float(np.linalg.norm(base_pos[-1]-truth_pos[-1]))
print(f"[classical ISS] posRMSE={base_rmse:.2f} km final={base_final:.2f} km")

M=dict(prior_B=PRIOR_B, plain_pos_rmse=pl_prmse, plain_vel_rmse=pl_vrmse,
       four_pos_rmse=f_prmse, four_vel_rmse=f_vrmse, four_final=f_final,
       four_B=fourB, four_B_over_prior=fourB/PRIOR_B,
       classical_rmse=base_rmse, classical_final=base_final,
       n_sparse=len(sparse), n_dense=len(dense),
       alt_min=float(dense.altitude_km.min()), alt_max=float(dense.altitude_km.max()))
json.dump(M,open(RESULTS+"/iss_metrics.json","w"),indent=2)
print("\n==== ISS METRICS ===="); print(json.dumps(M,indent=2))
np.savez(RESULTS+"/iss_arrays.npz",
    dense_x=dense.x_km.to_numpy(),dense_y=dense.y_km.to_numpy(),
    four_x=f_pp[:,0],four_y=f_pp[:,1],
    sparse_x=sparse.x_km.to_numpy(),sparse_y=sparse.y_km.to_numpy(),
    t_dense=t_dense, four_poserr=np.linalg.norm(f_pp-truth_pos,axis=1),
    alt24_t=alt24.t_sec.to_numpy(), alt24_alt=alt24.altitude_km.to_numpy())
print("saved ISS arrays")
