"""Plain-MLP diagnostic + hyperparameter sweep (Victoria round 4, items 2 and 3).

(a) DIAGNOSTIC: is the collapsed output anchored at the initial condition?
    Measures how far the predicted arc stays from the initial state, its spatial
    extent, and how much of the orbit it spans - to test the hypothesis that an
    IC weight of 30 vs a data weight of 10 pins the network near t=0.

(b) SWEEP: does the plain MLP fail because it is under-tuned? Varies width,
    depth, epochs, learning rate and the IC/data weighting, and reports the best
    position RMSE achievable without a time embedding.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ.setdefault("OMP_NUM_THREADS","4")
import json, math, time
import numpy as np, torch, torch.nn as nn

src=open("synth_experiment.py",encoding="utf-8").read()
g={}; exec(compile(src[:src.index("# ================= FOURIER PINN")],"setup","exec"), g)
truth_t=g["truth_t"]; truth_state=g["truth_state"]
obs_t_T=g["obs_t_T"]; obs_y_T=g["obs_y_T"]; y0_T=g["y0_T"]; eval_t_T=g["eval_t_T"]
denorm=g["denorm"]; tsc=g["tsc"]; ref_speed=g["ref_speed"]; ref_acc=g["ref_acc"]
tb_t=g["tb_t"]; j2_t=g["j2_t"]; drag_t=g["drag_t"]; tnorm=g["tnorm"]
colloc_t_plain=g["colloc_t_plain"]; colloc_phys_plain=g["colloc_phys_plain"]
R_EARTH=g["R_EARTH"]; H_DRAG=g["H_DRAG"]
RESULTS="results"

class MLP(nn.Module):
    def __init__(s,w,d):
        super().__init__(); L=[]; i=1
        for _ in range(d): L+=[nn.Linear(i,w),nn.Tanh()]; i=w
        L+=[nn.Linear(i,6)]; s.net=nn.Sequential(*L)
    def forward(s,t): return s.net(t)

def fd(y,tp):
    d=torch.zeros_like(y)
    d[1:-1]=(y[2:]-y[:-2])/(tp[2:]-tp[:-2])
    d[0]=(y[1]-y[0])/(tp[1]-tp[0]); d[-1]=(y[-1]-y[-2])/(tp[-1]-tp[-2]); return d

def train(seed=42,w=64,d=3,epochs=600,lr=2e-3,dw=10.,iw=30.,pw=1.):
    torch.manual_seed(seed); np.random.seed(seed)
    m=MLP(w,d); logd=nn.Parameter(torch.tensor(math.log(1e-5),dtype=torch.float32))
    opt=torch.optim.Adam(list(m.parameters())+[logd],lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        data=torch.mean((m(obs_t_T)-obs_y_T)**2)
        ic=torch.mean((m(torch.zeros(1,1))-y0_T)**2)
        yc=denorm(m(colloc_t_plain)); dy=fd(yc,colloc_phys_plain)
        r,v=yc[:,:3],yc[:,3:]; c=torch.exp(logd)
        acc=tb_t(r)+j2_t(r)+drag_t(r,v,c)
        phys=torch.mean(((dy[:,:3]-v)/ref_speed)**2)+torch.mean(((dy[:,3:]-acc)/ref_acc)**2)
        (dw*data+iw*ic+pw*phys).backward()
        torch.nn.utils.clip_grad_norm_(list(m.parameters())+[logd],1.0); opt.step()
    m.eval()
    with torch.no_grad(): pred=denorm(m(eval_t_T)).numpy()
    rmse=float(np.sqrt(np.mean(np.sum((pred[:,:3]-truth_state[:,:3])**2,axis=1))))
    return rmse,pred

# ------------------------------------------------------------------ (a)
print("=== DIAGNOSTIC: is the plain MLP anchored at the initial condition? ===")
truth_p=truth_state[:,:3]; p0=truth_p[0]
truth_extent=float(np.max(np.linalg.norm(truth_p-truth_p.mean(0),axis=1)))
diag={}
for sd in (42,43,44,45,46):
    _,pred=train(seed=sd)
    pp=pred[:,:3]
    diag[sd]=dict(
        dist_from_IC_mean=float(np.mean(np.linalg.norm(pp-p0,axis=1))),
        pred_extent=float(np.max(np.linalg.norm(pp-pp.mean(0),axis=1))),
        frac_within_1000km_of_IC=float(np.mean(np.linalg.norm(pp-p0,axis=1)<1000)),
        start_err=float(np.linalg.norm(pp[0]-p0)))
    print(f"  seed {sd}: mean dist from IC={diag[sd]['dist_from_IC_mean']:7.0f} km | "
          f"arc extent={diag[sd]['pred_extent']:7.0f} km | start err={diag[sd]['start_err']:6.1f} km")
print(f"  TRUE orbit extent for comparison: {truth_extent:.0f} km")

# ------------------------------------------------------------------ (b)
print("\n=== SWEEP: can the plain MLP be tuned into working? ===")
CONFIGS=[
 dict(w=64,d=3,epochs=600,lr=2e-3,iw=30.,tag="published baseline"),
 dict(w=64,d=3,epochs=5000,lr=2e-3,iw=30.,tag="epochs 600->5000"),
 dict(w=256,d=3,epochs=5000,lr=2e-3,iw=30.,tag="width 256"),
 dict(w=256,d=6,epochs=5000,lr=2e-3,iw=30.,tag="width 256, depth 6"),
 dict(w=512,d=4,epochs=5000,lr=2e-3,iw=30.,tag="width 512, depth 4"),
 dict(w=64,d=3,epochs=5000,lr=1e-2,iw=30.,tag="lr 1e-2"),
 dict(w=64,d=3,epochs=5000,lr=5e-4,iw=30.,tag="lr 5e-4"),
 dict(w=64,d=3,epochs=5000,lr=2e-3,iw=1.,tag="IC weight 30->1"),
 dict(w=64,d=3,epochs=5000,lr=2e-3,iw=0.,tag="IC weight 30->0"),
 dict(w=256,d=4,epochs=5000,lr=2e-3,iw=1.,tag="width 256, depth 4, IC weight 1"),
 dict(w=256,d=4,epochs=5000,lr=1e-2,iw=0.,tag="width 256, depth 4, lr 1e-2, IC 0"),
]
sweep=[]
for cfg in CONFIGS:
    tag=cfg.pop("tag"); t0=time.time()
    rmse,_=train(seed=42,**cfg)
    sweep.append(dict(tag=tag,rmse=rmse,**cfg))
    print(f"  {tag:38s} RMSE = {rmse:9.1f} km   [{time.time()-t0:.0f}s]",flush=True)
best=min(sweep,key=lambda r:r["rmse"])
print(f"\n  BEST over {len(sweep)} configurations: {best['rmse']:.1f} km  ({best['tag']})")
print(f"  Fourier-feature PINN for comparison: 0.78 km")

out=dict(diagnostic=diag,true_extent_km=truth_extent,sweep=sweep,best=best)
prev=json.load(open(RESULTS+"/victoria_r3.json")); prev["r4_plain_mlp"]=out
json.dump(prev,open(RESULTS+"/victoria_r3.json","w"),indent=2)
print("\nsaved")
