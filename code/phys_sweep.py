"""Physics-loss-weight sweep for the Fourier-feature PINN.

Tests the claim in Results that increasing the physics weight relative to the data
weight does not expose the weak drag signal. Everything except PHYS_W is identical
to synth_experiment.py: same trajectory, same data seed (42), same architecture,
same 5,000 epochs, same optimizer/scheduler. Data weight stays 10, IC weight 30.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("OMP_NUM_THREADS", "4")
import json, math, time
import numpy as np
from scipy.integrate import solve_ivp
import torch, torch.nn as nn

RESULTS = "results"
MU_EARTH = 398600.4418; R_EARTH = 6378.1363; J2_EARTH = 1.08262668e-3
TRUE_DRAG_COEFF = 3.245e-5; H_DRAG = 60.0

def norm(v): return np.linalg.norm(v)
def two_body_acc(r): return -MU_EARTH*r/norm(r)**3
def j2_acc(r):
    x,y,z=r; rn=norm(r); f=1.5*J2_EARTH*MU_EARTH*R_EARTH**2/rn**5; cm=5.0*z**2/rn**2
    return np.array([f*x*(cm-1), f*y*(cm-1), f*z*(cm-3)])
def drag_acc(r,v,c):
    return -c*np.exp(-max(norm(r)-R_EARTH,0.0)/H_DRAG)*norm(v)*v
def dyn(t,s,use_j2=True,use_drag=False,c=TRUE_DRAG_COEFF):
    r,v=s[:3],s[3:]; a=two_body_acc(r)
    if use_j2: a=a+j2_acc(r)
    if use_drag: a=a+drag_acc(r,v,c)
    return np.concatenate([v,a])

# ---- identical synthetic dataset (NB1 config, data seed 42) ----
altitude_km=500.0; inclination_deg=35.0; num_orbits=8; n_eval=2500; observe_every=35
pos_noise=0.5; vel_noise=0.002
radius0=R_EARTH+altitude_km; vc=np.sqrt(MU_EARTH/radius0); inc=np.deg2rad(inclination_deg)
state0=np.concatenate([np.array([radius0,0,0]), np.array([0.0,vc*np.cos(inc),vc*np.sin(inc)])])
period=2*np.pi*np.sqrt(radius0**3/MU_EARTH); t_final=num_orbits*period
t_eval=np.linspace(0,t_final,n_eval)
sol=solve_ivp(lambda t,y:dyn(t,y,True,True,TRUE_DRAG_COEFF),(0,t_final),state0,
              t_eval=t_eval,rtol=1e-9,atol=1e-9)
truth_t=sol.t; truth_state=sol.y.T
obs_idx=np.arange(0,len(truth_t),observe_every); obs_t=truth_t[obs_idx]
rng=np.random.default_rng(42); obs=truth_state[obs_idx].copy()
obs[:,:3]+=rng.normal(0,pos_noise,obs[:,:3].shape)
obs[:,3:]+=rng.normal(0,vel_noise,obs[:,3:].shape)
print(f"[data] dense={len(truth_t)} sparse={len(obs_t)}", flush=True)

def tnorm(x): return torch.sqrt(torch.sum(x*x,dim=1,keepdim=True)+1e-12)
def tb_t(r): return -MU_EARTH*r/(tnorm(r)**3)
def j2_t(r):
    x=r[:,0:1];y=r[:,1:2];z=r[:,2:3];rn=tnorm(r)
    f=1.5*J2_EARTH*MU_EARTH*R_EARTH**2/rn**5; cm=5.0*z*z/(rn*rn)
    return torch.cat([f*x*(cm-1),f*y*(cm-1),f*z*(cm-3)],dim=1)
def drag_t(r,v,c):
    return -c*torch.exp(-torch.clamp(tnorm(r)-R_EARTH,min=0.0)/H_DRAG)*tnorm(v)*v

state_mean=truth_state.mean(0,keepdims=True)
state_std=truth_state.std(0,keepdims=True); state_std[state_std<1e-12]=1.0
t0=truth_t.min(); tsc=truth_t.max()-truth_t.min()
ntime=lambda a: ((a-t0)/tsc).reshape(-1,1)
smean_T=torch.tensor(state_mean,dtype=torch.float32)
sstd_T=torch.tensor(state_std,dtype=torch.float32)
denorm=lambda yn: yn*sstd_T+smean_T
ref_speed=float(np.mean(np.linalg.norm(truth_state[:,3:],axis=1)))
ref_acc=float(MU_EARTH/(np.mean(np.linalg.norm(truth_state[:,:3],axis=1))**2))
obs_t_T=torch.tensor(ntime(obs_t),dtype=torch.float32)
obs_y_T=torch.tensor((obs-state_mean)/state_std,dtype=torch.float32)
y0_T=torch.tensor((truth_state[0:1]-state_mean)/state_std,dtype=torch.float32)
eval_t_T=torch.tensor(ntime(truth_t),dtype=torch.float32)

K=24; freqs=torch.arange(1,K+1,dtype=torch.float32).reshape(1,-1)
def ff(t):
    ang=2*math.pi*t*freqs
    return torch.cat([t,torch.sin(ang),torch.cos(ang)],dim=1)
class FourierPINN(nn.Module):
    def __init__(s,ed,w=128,d=3):
        super().__init__(); L=[nn.Linear(ed,w),nn.Tanh()]
        for _ in range(d-1): L+=[nn.Linear(w,w),nn.Tanh()]
        L+=[nn.Linear(w,6)]; s.net=nn.Sequential(*L)
    def forward(s,e): return s.net(e)
obs_emb=ff(obs_t_T); emb0=ff(torch.zeros(1,1)); colloc=torch.linspace(0,1,400).reshape(-1,1)

def train(seed, phys_w, data_w=10.0, ic_w=30.0, epochs=5000):
    torch.manual_seed(seed); np.random.seed(seed)
    m=FourierPINN(obs_emb.shape[1])
    logd=nn.Parameter(torch.tensor(math.log(1e-5),dtype=torch.float32))
    opt=torch.optim.Adam(list(m.parameters())+[logd],lr=2e-3)
    sch=torch.optim.lr_scheduler.StepLR(opt,step_size=2000,gamma=0.5)
    for ep in range(1,epochs+1):
        opt.zero_grad()
        data=torch.mean((m(obs_emb)-obs_y_T)**2)
        ic=torch.mean((m(emb0)-y0_T)**2)
        tc=colloc.clone().detach().requires_grad_(True)
        yc=denorm(m(ff(tc)))
        g=[torch.autograd.grad(yc[:,i:i+1],tc,torch.ones_like(yc[:,i:i+1]),
           create_graph=True,retain_graph=True)[0] for i in range(6)]
        dydt=torch.cat(g,dim=1)/tsc
        r,v=yc[:,:3],yc[:,3:]; c=torch.exp(logd)
        acc=tb_t(r)+j2_t(r)+drag_t(r,v,c)
        phys=torch.mean(((dydt[:,:3]-v)/ref_speed)**2)+torch.mean(((dydt[:,3:]-acc)/ref_acc)**2)
        (data_w*data+ic_w*ic+phys_w*phys).backward()
        torch.nn.utils.clip_grad_norm_(list(m.parameters())+[logd],1.0)
        opt.step(); sch.step()
    m.eval()
    with torch.no_grad(): pred=denorm(m(ff(eval_t_T))).numpy()
    prmse=float(np.sqrt(np.mean(np.sum((pred[:,:3]-truth_state[:,:3])**2,axis=1))))
    drag_joint=float(torch.exp(logd).detach())
    # frozen-trajectory least squares, same as the main run
    tg=torch.tensor(ntime(truth_t),dtype=torch.float32).requires_grad_(True)
    yg=denorm(m(ff(tg)))
    gg=[torch.autograd.grad(yg[:,i:i+1],tg,torch.ones_like(yg[:,i:i+1]),
        create_graph=False,retain_graph=True)[0] for i in range(6)]
    dvdt=(torch.cat(gg,dim=1)/tsc)[:,3:].detach()
    rr=yg[:,:3].detach(); vv=yg[:,3:].detach()
    phi=-torch.exp(-torch.clamp(tnorm(rr)-R_EARTH,min=0.0)/H_DRAG)*tnorm(vv)*vv
    res=dvdt-tb_t(rr)-j2_t(rr)
    drag_lsq=float(torch.sum(phi*res)/torch.sum(phi*phi))
    return prmse, drag_joint, drag_lsq

WEIGHTS=[50.0,100.0]; SEEDS=[42,43]
out={}; t_start=time.time()
for w in WEIGHTS:
    pr=[];dj=[];dl=[]
    for sd in SEEDS:
        t0_=time.time(); p,j,l=train(sd,w)
        pr.append(p); dj.append(j); dl.append(l)
        print(f"[phys_w={w:6.1f} seed={sd}] prmse={p:9.3f} km  drag_joint={j:.4e} "
              f"({100*abs(j-TRUE_DRAG_COEFF)/TRUE_DRAG_COEFF:5.1f}%)  drag_lsq={l:.4e} "
              f"({100*abs(l-TRUE_DRAG_COEFF)/TRUE_DRAG_COEFF:5.1f}%)  [{time.time()-t0_:.0f}s]", flush=True)
    pr=np.array(pr); dj=np.array(dj); dl=np.array(dl)
    rj=100*np.abs(dj-TRUE_DRAG_COEFF)/TRUE_DRAG_COEFF
    rl=100*np.abs(dl-TRUE_DRAG_COEFF)/TRUE_DRAG_COEFF
    out[str(w)]=dict(prmse_mean=float(pr.mean()),prmse_sd=float(pr.std()),
        drag_joint_mean=float(dj.mean()),drag_joint_sd=float(dj.std()),
        relerr_joint_mean=float(rj.mean()),relerr_joint_sd=float(rj.std()),
        drag_lsq_mean=float(dl.mean()),relerr_lsq_mean=float(rl.mean()),relerr_lsq_sd=float(rl.std()))
    print(f"  == phys_w={w}: prmse={pr.mean():.3f}±{pr.std():.3f} km | "
          f"joint drag err={rj.mean():.1f}±{rj.std():.1f}% | lsq err={rl.mean():.1f}±{rl.std():.1f}%\n", flush=True)
    json.dump(out,open(RESULTS+"/phys_sweep_high.json","w"),indent=2)
print(f"TOTAL {time.time()-t_start:.0f}s")
print(json.dumps(out,indent=2))
