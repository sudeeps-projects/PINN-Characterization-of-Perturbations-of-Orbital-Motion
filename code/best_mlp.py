"""Victoria round 5: report the BEST tuned plain MLP as the baseline, not the
matched-capacity one.

(a) SYNTHETIC: best configuration from the round-4 sweep (width 256, depth 3,
    5,000 epochs, lr 2e-3, IC weight 30) over the same five seeds, so the
    baseline is reported as mean +/- SD like every other number in the paper.
(b) ISS: the plain MLP there was never tuned. Sweep it, then run the best
    configuration and report it.
Also saves the tuned synthetic prediction so Figure 2 can be regenerated.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; os.environ.setdefault("OMP_NUM_THREADS","4")
import json, math, time
import numpy as np, torch, torch.nn as nn

RES="results"
out={}

# ================================================================= SYNTHETIC
src=open("synth_experiment.py",encoding="utf-8").read()
g={}; exec(compile(src[:src.index("# ================= FOURIER PINN")],"setup","exec"), g)
truth_t=g["truth_t"]; truth_state=g["truth_state"]
obs_t_T=g["obs_t_T"]; obs_y_T=g["obs_y_T"]; y0_T=g["y0_T"]; eval_t_T=g["eval_t_T"]
denorm=g["denorm"]; tsc=g["tsc"]; ref_speed=g["ref_speed"]; ref_acc=g["ref_acc"]
tb_t=g["tb_t"]; j2_t=g["j2_t"]; drag_t=g["drag_t"]
colloc_t_plain=g["colloc_t_plain"]; colloc_phys_plain=g["colloc_phys_plain"]

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

def train_syn(seed,w=256,d=3,epochs=5000,lr=2e-3,dw=10.,iw=30.,pw=1.):
    torch.manual_seed(seed); np.random.seed(seed)
    m=MLP(w,d); logd=nn.Parameter(torch.tensor(math.log(1e-5),dtype=torch.float32))
    opt=torch.optim.Adam(list(m.parameters())+[logd],lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        data=torch.mean((m(obs_t_T)-obs_y_T)**2)
        ic=torch.mean((m(torch.zeros(1,1))-y0_T)**2)
        yc=denorm(m(colloc_t_plain)); dy=fd(yc,colloc_phys_plain)
        r,v=yc[:,:3],yc[:,3:]
        acc=tb_t(r)+j2_t(r)+drag_t(r,v,torch.exp(logd))
        phys=torch.mean(((dy[:,:3]-v)/ref_speed)**2)+torch.mean(((dy[:,3:]-acc)/ref_acc)**2)
        (dw*data+iw*ic+pw*phys).backward()
        torch.nn.utils.clip_grad_norm_(list(m.parameters())+[logd],1.0); opt.step()
    m.eval()
    with torch.no_grad(): pred=denorm(m(eval_t_T)).numpy()
    p=float(np.sqrt(np.mean(np.sum((pred[:,:3]-truth_state[:,:3])**2,axis=1))))
    v=float(np.sqrt(np.mean(np.sum((pred[:,3:]-truth_state[:,3:])**2,axis=1))))*1000.0
    return p,v,pred

print("=== SYNTHETIC: best config (w=256,d=3,5000ep) over 5 seeds ===",flush=True)
P=[];V=[];preds={}
for sd in (42,43,44,45,46):
    t0=time.time(); p,v,pred=train_syn(sd); P.append(p); V.append(v); preds[sd]=pred
    print(f"  seed {sd}: pos={p:8.2f} km  vel={v:8.2f} m/s   [{time.time()-t0:.0f}s]",flush=True)
P=np.array(P); V=np.array(V)
out["synthetic_best"]=dict(config="width 256, depth 3, 5000 epochs, lr 2e-3, IC weight 30",
    pos_mean=float(P.mean()),pos_sd=float(P.std()),vel_mean=float(V.mean()),vel_sd=float(V.std()),
    per_seed_pos=[float(z) for z in P])
print(f"  => {P.mean():.2f} +/- {P.std():.2f} km | {V.mean():.2f} +/- {V.std():.2f} m/s\n",flush=True)
np.savez_compressed(RES+"/best_mlp_synth.npz",pred42=preds[42],truth=truth_state,t=truth_t)

# ======================================================================= ISS
src2=open("iss_experiment.py",encoding="utf-8").read()
h={}; exec(compile(src2[:src2.index("# ---------- Fourier PINN")],"iss_setup","exec"), h)
ic_T=h["ic_T"]; t_obs_T=h["t_obs_T"]; y_obs_T=h["y_obs_T"]; denorm_t=h["denorm_t"]
acc_t=h["acc_t"]; TSC=h["TSC"]; nt=h["nt"]; dstate=h["dstate"]
t_dense=h["t_dense"]; y_dense=h["y_dense"]; PRIOR_B=h["PRIOR_B"]; ICPINN=h["ICPINN"]

def train_iss(seed=42,w=96,d=4,epochs=500,lr=1e-3):
    torch.manual_seed(seed)
    m=ICPINN(ic_T,w,d); logB=nn.Parameter(torch.tensor(math.log(PRIOR_B),dtype=torch.float32))
    opt=torch.optim.Adam(list(m.parameters())+[logB],lr=lr)
    sch=torch.optim.lr_scheduler.StepLR(opt,step_size=max(epochs//3,1),gamma=0.5)
    tcol=torch.linspace(0,1,160).reshape(-1,1)
    W={"data":20,"ic":10,"physics":1,"prior":0.05}; lp=math.log(PRIOR_B)
    for ep in range(epochs):
        opt.zero_grad()
        data=torch.mean((m(t_obs_T)-y_obs_T)**2)
        ic=torch.mean((m(torch.zeros(1,1))-ic_T)**2)
        tc=tcol.clone().detach().requires_grad_(True); ps=denorm_t(m(tc))
        gr=[torch.autograd.grad(ps[:,i:i+1],tc,torch.ones_like(ps[:,i:i+1]),
            create_graph=True,retain_graph=True)[0] for i in range(6)]
        dyt=torch.cat(gr,dim=1)/TSC; r=ps[:,:3]; v=ps[:,3:]
        phys=torch.mean(((dyt[:,:3]-v)/7600.0)**2)+torch.mean(((dyt[:,3:]-acc_t(r,v,logB))/8.7)**2)
        (W["data"]*data+W["ic"]*ic+W["physics"]*phys+W["prior"]*(logB-lp)**2).backward()
        torch.nn.utils.clip_grad_norm_(list(m.parameters())+[logB],1.0); opt.step(); sch.step()
    m.eval()
    with torch.no_grad(): pred=dstate(m(torch.tensor(nt(t_dense),dtype=torch.float32)).numpy())
    p=float(np.sqrt(np.mean(np.sum((pred[:,:3]-y_dense[:,:3])**2,axis=1))))/1000.0
    v=float(np.sqrt(np.mean(np.sum((pred[:,3:]-y_dense[:,3:])**2,axis=1))))
    return p,v,pred

print("=== ISS: plain-MLP tuning sweep ===",flush=True)
CFG=[dict(w=96,d=4,epochs=500,lr=1e-3,tag="published baseline"),
     dict(w=96,d=4,epochs=5000,lr=1e-3,tag="epochs 500->5000"),
     dict(w=256,d=3,epochs=5000,lr=1e-3,tag="width 256, depth 3"),
     dict(w=256,d=4,epochs=5000,lr=2e-3,tag="width 256, depth 4, lr 2e-3"),
     dict(w=512,d=3,epochs=5000,lr=1e-3,tag="width 512, depth 3")]
sweep=[]
for c in CFG:
    tag=c.pop("tag"); t0=time.time(); p,v,_=train_iss(**c)
    sweep.append(dict(tag=tag,pos_km=p,vel_ms=v,**c))
    print(f"  {tag:30s} pos={p:9.2f} km  vel={v:8.1f} m/s  [{time.time()-t0:.0f}s]",flush=True)
best=min(sweep,key=lambda r:r["pos_km"])
out["iss_sweep"]=sweep; out["iss_best"]=best
print(f"\n  BEST ISS plain MLP: {best['pos_km']:.2f} km ({best['tag']})")
print( "  ISS Fourier PINN for comparison: 5.68 km")

json.dump(out,open(RES+"/best_mlp.json","w"),indent=2)
print("\nsaved results/best_mlp.json")
