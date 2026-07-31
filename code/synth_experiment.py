"""
Synthetic-branch rerun: five network-initialization seeds for the plain-MLP and
Fourier-feature PINNs, plus a five-noise-draw finer grid search.
Faithfully mirrors Notebook 1 (data) and Notebook 3 (models); adds real statistics.
Data-generation seed is fixed at 42; the five training seeds are 42..46.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "4")
import json, math, time
import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = "results"
FIGS = "figs"

# ---------------- physics (km, s) -- identical to NB1/NB3 ----------------
MU_EARTH = 398600.4418
R_EARTH = 6378.1363
J2_EARTH = 1.08262668e-3
TRUE_DRAG_COEFF = 3.245e-5
H_DRAG = 60.0

def norm(v): return np.linalg.norm(v)
def two_body_acc(r): return -MU_EARTH * r / norm(r)**3
def j2_acc(r):
    x,y,z = r; rn = norm(r); factor = 1.5*J2_EARTH*MU_EARTH*R_EARTH**2/rn**5
    common = 5.0*z**2/rn**2
    return np.array([factor*x*(common-1), factor*y*(common-1), factor*z*(common-3)])
def drag_acc(r,v,c):
    alt = max(norm(r)-R_EARTH,0.0); dens = np.exp(-alt/H_DRAG)
    return -c*dens*norm(v)*v
def dyn(t,s,use_j2=True,use_drag=False,c=TRUE_DRAG_COEFF):
    r,v = s[:3],s[3:]; a = two_body_acc(r)
    if use_j2: a = a+j2_acc(r)
    if use_drag: a = a+drag_acc(r,v,c)
    return np.concatenate([v,a])
def circ_v(rad): return np.sqrt(MU_EARTH/rad)

# ---------------- generate synthetic dataset (NB1 config) ----------------
altitude_km=500.0; inclination_deg=35.0; num_orbits=8; n_eval=2500; observe_every=35
pos_noise=0.5; vel_noise=0.002
radius0 = R_EARTH+altitude_km
r0 = np.array([radius0,0,0]); vc = circ_v(radius0); inc = np.deg2rad(inclination_deg)
v0 = np.array([0.0, vc*np.cos(inc), vc*np.sin(inc)])
state0 = np.concatenate([r0,v0])
period = 2*np.pi*np.sqrt(radius0**3/MU_EARTH)
t_final = num_orbits*period
t_eval = np.linspace(0,t_final,n_eval)

sol_tb  = solve_ivp(lambda t,y:dyn(t,y,False,False),(0,t_final),state0,t_eval=t_eval,rtol=1e-9,atol=1e-9)
sol_j2  = solve_ivp(lambda t,y:dyn(t,y,True,False), (0,t_final),state0,t_eval=t_eval,rtol=1e-9,atol=1e-9)
sol_all = solve_ivp(lambda t,y:dyn(t,y,True,True,TRUE_DRAG_COEFF),(0,t_final),state0,t_eval=t_eval,rtol=1e-9,atol=1e-9)

truth_t = sol_all.t; truth_state = sol_all.y.T
r_tb=sol_tb.y[:3].T; r_j2=sol_j2.y[:3].T; r_all=sol_all.y[:3].T
resid_j2 = np.linalg.norm(r_j2-r_tb,axis=1)
resid_all= np.linalg.norm(r_all-r_tb,axis=1)
drift_j2 = float(resid_j2[-1]); drift_all=float(resid_all[-1]); drift_drag=drift_all-drift_j2
print(f"[perturb] J2 drift={drift_j2:.1f} km  J2+drag={drift_all:.1f} km  drag contrib={drift_drag:.1f} km")

# sparse obs, data seed 42 (fixed)
obs_idx = np.arange(0,len(truth_t),observe_every)
obs_t = truth_t[obs_idx]
def make_noisy(seed):
    rng = np.random.default_rng(seed)
    o = truth_state[obs_idx].copy()
    o[:,:3]+=rng.normal(0,pos_noise,o[:,:3].shape)
    o[:,3:]+=rng.normal(0,vel_noise,o[:,3:].shape)
    return o
obs_state_noisy = make_noisy(42)   # primary dataset
n_sparse = len(obs_t)
print(f"[data] dense={len(truth_t)} sparse={n_sparse}")

# ---------------- torch physics (km) ----------------
def tnorm(x): return torch.sqrt(torch.sum(x*x,dim=1,keepdim=True)+1e-12)
def tb_t(r): return -MU_EARTH*r/(tnorm(r)**3)
def j2_t(r):
    x=r[:,0:1];y=r[:,1:2];z=r[:,2:3];rn=tnorm(r)
    factor=1.5*J2_EARTH*MU_EARTH*R_EARTH**2/rn**5; common=5.0*z*z/(rn*rn)
    return torch.cat([factor*x*(common-1),factor*y*(common-1),factor*z*(common-3)],dim=1)
def drag_t(r,v,c):
    alt=torch.clamp(tnorm(r)-R_EARTH,min=0.0); dens=torch.exp(-alt/H_DRAG)
    return -c*dens*tnorm(v)*v

# normalization from dense truth
state_mean = truth_state.mean(0,keepdims=True)
state_std = truth_state.std(0,keepdims=True); state_std[state_std<1e-12]=1.0
t0=truth_t.min(); tsc=truth_t.max()-truth_t.min()
def ntime(a): return ((a-t0)/tsc).reshape(-1,1)
smean_T=torch.tensor(state_mean,dtype=torch.float32)
sstd_T=torch.tensor(state_std,dtype=torch.float32)
def denorm(yn): return yn*sstd_T+smean_T
ref_speed=float(np.mean(np.linalg.norm(truth_state[:,3:],axis=1)))
ref_acc=float(MU_EARTH/(np.mean(np.linalg.norm(truth_state[:,:3],axis=1))**2))

obs_y_norm=(obs_state_noisy-state_mean)/state_std
y0_norm=(truth_state[0:1]-state_mean)/state_std
obs_t_T=torch.tensor(ntime(obs_t),dtype=torch.float32)
obs_y_T=torch.tensor(obs_y_norm,dtype=torch.float32)
y0_T=torch.tensor(y0_norm,dtype=torch.float32)
eval_t_T=torch.tensor(ntime(truth_t),dtype=torch.float32)

# ================= PLAIN MLP PINN (NB3 cells 18-26) =================
class OrbitMLP(nn.Module):
    def __init__(s,w=64,d=3):
        super().__init__(); L=[]; ind=1
        for _ in range(d): L+=[nn.Linear(ind,w),nn.Tanh()]; ind=w
        L+=[nn.Linear(ind,6)]; s.net=nn.Sequential(*L)
    def forward(s,t): return s.net(t)

COLLO_PLAIN=160
cidx=np.unique(np.linspace(0,len(truth_t)-1,COLLO_PLAIN).astype(int))
colloc_t_plain=torch.tensor(ntime(truth_t)[cidx],dtype=torch.float32)
colloc_phys_plain=torch.tensor(truth_t[cidx].reshape(-1,1),dtype=torch.float32)

def fd_deriv(y,tp):
    d=torch.zeros_like(y)
    d[1:-1]=(y[2:]-y[:-2])/(tp[2:]-tp[:-2])
    d[0]=(y[1]-y[0])/(tp[1]-tp[0]); d[-1]=(y[-1]-y[-2])/(tp[-1]-tp[-2])
    return d

def train_plain(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model=OrbitMLP(64,3)
    logd=nn.Parameter(torch.tensor(math.log(1e-5),dtype=torch.float32))
    opt=torch.optim.Adam(list(model.parameters())+[logd],lr=2e-3)
    for ep in range(600):
        opt.zero_grad()
        data=torch.mean((model(obs_t_T)-obs_y_T)**2)
        ic=torch.mean((model(torch.zeros(1,1))-y0_T)**2)
        ps=denorm(model(colloc_t_plain)); pdot=fd_deriv(ps,colloc_phys_plain)
        r=ps[:,:3];v=ps[:,3:]; c=torch.exp(logd)
        acc=tb_t(r)+j2_t(r)+drag_t(r,v,c)
        kin=(pdot[:,:3]-v)/ref_speed; dyn_r=(pdot[:,3:]-acc)/ref_acc
        phys=torch.mean(kin**2)+torch.mean(dyn_r**2)
        loss=10*data+30*ic+1*phys
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters())+[logd],1.0)
        opt.step()
    model.eval()
    with torch.no_grad(): pred=denorm(model(eval_t_T)).numpy()
    prmse=float(np.sqrt(np.mean(np.sum((pred[:,:3]-truth_state[:,:3])**2,axis=1))))
    vrmse=float(np.sqrt(np.mean(np.sum((pred[:,3:]-truth_state[:,3:])**2,axis=1))))
    return prmse,vrmse*1000.0,pred  # vel in m/s

# ================= FOURIER PINN (NB3 cell 36) =================
K=24
freqs=torch.arange(1,K+1,dtype=torch.float32).reshape(1,-1)
def ff(t):
    ang=2*math.pi*t*freqs
    return torch.cat([t,torch.sin(ang),torch.cos(ang)],dim=1)
class FourierPINN(nn.Module):
    def __init__(s,ed,w=128,d=3):
        super().__init__(); L=[nn.Linear(ed,w),nn.Tanh()]
        for _ in range(d-1): L+=[nn.Linear(w,w),nn.Tanh()]
        L+=[nn.Linear(w,6)]; s.net=nn.Sequential(*L)
    def forward(s,e): return s.net(e)

obs_emb=ff(obs_t_T); emb0=ff(torch.zeros(1,1))
colloc_f=torch.linspace(0,1,400).reshape(-1,1)

def train_fourier(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m=FourierPINN(obs_emb.shape[1]).to("cpu")
    logd=nn.Parameter(torch.tensor(math.log(1e-5),dtype=torch.float32))
    opt=torch.optim.Adam(list(m.parameters())+[logd],lr=2e-3)
    sch=torch.optim.lr_scheduler.StepLR(opt,step_size=2000,gamma=0.5)
    hist={"epoch":[],"total":[],"data":[],"ic":[],"physics":[]}
    for ep in range(1,5001):
        opt.zero_grad()
        data=torch.mean((m(obs_emb)-obs_y_T)**2)
        ic=torch.mean((m(emb0)-y0_T)**2)
        tc=colloc_f.clone().detach().requires_grad_(True)
        yc=denorm(m(ff(tc)))
        grads=[torch.autograd.grad(yc[:,i:i+1],tc,torch.ones_like(yc[:,i:i+1]),
               create_graph=True,retain_graph=True)[0] for i in range(6)]
        dydt=torch.cat(grads,dim=1)/tsc
        r,v=yc[:,:3],yc[:,3:]; c=torch.exp(logd)
        acc=tb_t(r)+j2_t(r)+drag_t(r,v,c)
        phys=torch.mean(((dydt[:,:3]-v)/ref_speed)**2)+torch.mean(((dydt[:,3:]-acc)/ref_acc)**2)
        total=10*data+30*ic+20*phys
        total.backward()
        torch.nn.utils.clip_grad_norm_(list(m.parameters())+[logd],1.0)
        opt.step(); sch.step()
        if ep%500==0 or ep==1:
            hist["epoch"].append(ep)
            for k,val in [("total",total),("data",data),("ic",ic),("physics",phys)]:
                hist[k].append(float(val.detach()))
    m.eval()
    with torch.no_grad(): pred=denorm(m(ff(eval_t_T))).numpy()
    prmse=float(np.sqrt(np.mean(np.sum((pred[:,:3]-truth_state[:,:3])**2,axis=1))))
    vrmse=float(np.sqrt(np.mean(np.sum((pred[:,3:]-truth_state[:,3:])**2,axis=1))))
    drag_joint=float(torch.exp(logd).detach())
    # frozen-trajectory LSQ
    tg=torch.tensor(ntime(truth_t),dtype=torch.float32).requires_grad_(True)
    yg=denorm(m(ff(tg)))
    gg=[torch.autograd.grad(yg[:,i:i+1],tg,torch.ones_like(yg[:,i:i+1]),
        create_graph=False,retain_graph=True)[0] for i in range(6)]
    dvdt=(torch.cat(gg,dim=1)/tsc)[:,3:].detach()
    rr=yg[:,:3].detach(); vv=yg[:,3:].detach()
    alt=torch.clamp(tnorm(rr)-R_EARTH,min=0.0)
    phi=-torch.exp(-alt/H_DRAG)*tnorm(vv)*vv
    res=dvdt-tb_t(rr)-j2_t(rr)
    drag_lsq=float((torch.sum(phi*res)/torch.sum(phi*phi)))
    return dict(prmse=prmse,vrmse=vrmse*1000.0,drag_joint=drag_joint,drag_lsq=drag_lsq,
                pred=pred,hist=hist,logd_final=drag_joint)

# ---------------- grid search: 5 noise draws, finer grid ----------------
def sim_pos(c,ts,init):
    s=solve_ivp(lambda t,y:dyn(t,y,True,True,c),(float(ts[0]),float(ts[-1])),init,
                t_eval=ts,rtol=1e-8,atol=1e-8)
    return s.y[:3].T
GRID=np.linspace(0.2e-5,4.0e-5,400)   # finer than the original 40-point grid
init_known=truth_state[0].copy()
grid_best=[]; grid_curve_seed42=None
for i,sd in enumerate([42,43,44,45,46]):
    ob=make_noisy(sd); losses=[]
    for c in GRID:
        pp=sim_pos(c,obs_t,init_known)
        losses.append(np.mean(np.sum((pp-ob[:,:3])**2,axis=1)))
    losses=np.array(losses); bi=int(np.argmin(losses))
    grid_best.append(float(GRID[bi]))
    if sd==42: grid_curve_seed42=(GRID.copy(),losses.copy(),float(GRID[bi]))
    print(f"[grid] draw seed={sd} best={GRID[bi]:.6e}")
grid_best=np.array(grid_best)
grid_relerr=100*np.abs(grid_best-TRUE_DRAG_COEFF)/TRUE_DRAG_COEFF

# ================= run 5 seeds =================
SEEDS=[42,43,44,45,46]
t_start=time.time()
plain_p=[];plain_v=[];plain_pred42=None
for sd in SEEDS:
    p,v,pred=train_plain(sd); plain_p.append(p); plain_v.append(v)
    if sd==42: plain_pred42=pred
    print(f"[plain] seed={sd} posRMSE={p:.1f} km velRMSE={v:.1f} m/s  ({time.time()-t_start:.0f}s)")
four_p=[];four_v=[];four_dj=[];four_dl=[];four42=None
for sd in SEEDS:
    r=train_fourier(sd); four_p.append(r['prmse']); four_v.append(r['vrmse'])
    four_dj.append(r['drag_joint']); four_dl.append(r['drag_lsq'])
    if sd==42: four42=r
    print(f"[fourier] seed={sd} posRMSE={r['prmse']:.3f} km velRMSE={r['vrmse']:.3f} m/s "
          f"dragJoint={r['drag_joint']:.4e} dragLSQ={r['drag_lsq']:.4e}  ({time.time()-t_start:.0f}s)")

def ms(a): a=np.array(a,dtype=float); return float(a.mean()),float(a.std())
M={}
M['drift_j2_km']=drift_j2; M['drift_all_km']=drift_all; M['drift_drag_km']=drift_drag
M['n_sparse']=n_sparse; M['n_dense']=len(truth_t); M['true_drag']=TRUE_DRAG_COEFF
M['plain_pos_mean'],M['plain_pos_sd']=ms(plain_p)
M['plain_vel_mean'],M['plain_vel_sd']=ms(plain_v)
M['four_pos_mean'],M['four_pos_sd']=ms(four_p)
M['four_vel_mean'],M['four_vel_sd']=ms(four_v)
M['four_dragjoint_mean'],M['four_dragjoint_sd']=ms(four_dj)
M['four_draglsq_mean'],M['four_draglsq_sd']=ms(four_dl)
dj_rel=100*np.abs(np.array(four_dj)-TRUE_DRAG_COEFF)/TRUE_DRAG_COEFF
dl_rel=100*np.abs(np.array(four_dl)-TRUE_DRAG_COEFF)/TRUE_DRAG_COEFF
M['four_dragjoint_relerr_mean'],M['four_dragjoint_relerr_sd']=ms(dj_rel)
M['four_draglsq_relerr_mean'],M['four_draglsq_relerr_sd']=ms(dl_rel)
M['grid_best_mean'],M['grid_best_sd']=ms(grid_best)
M['grid_relerr_mean'],M['grid_relerr_sd']=ms(grid_relerr)
M['grid_best_seed42']=grid_curve_seed42[2]
M['seeds']=SEEDS
json.dump(M,open(RESULTS+"/synth_metrics.json","w"),indent=2)
print("\n==== SYNTH METRICS ===="); print(json.dumps(M,indent=2))

# save arrays for figures
np.savez(RESULTS+"/synth_arrays.npz",
    truth_state=truth_state, truth_t=truth_t, obs_state_noisy=obs_state_noisy, obs_t=obs_t,
    plain_pred42=plain_pred42, four_pred42=four42['pred'],
    resid_j2=resid_j2, resid_all=resid_all,
    grid_x=grid_curve_seed42[0], grid_loss=grid_curve_seed42[1], grid_best42=grid_curve_seed42[2],
    four_hist_epoch=np.array(four42['hist']['epoch']),
    four_hist_total=np.array(four42['hist']['total']),
    four_hist_data=np.array(four42['hist']['data']),
    four_hist_ic=np.array(four42['hist']['ic']),
    four_hist_physics=np.array(four42['hist']['physics']),
    four_dj=np.array(four_dj), four_dl=np.array(four_dl), grid_best=grid_best)
print("saved arrays")
