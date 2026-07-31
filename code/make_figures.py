import os, json
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

R="results"
OUT="newfigs"
os.makedirs(OUT,exist_ok=True)
S=np.load(R+"/synth_arrays.npz"); I=np.load(R+"/iss_arrays.npz")
GR=json.load(open(R+"/grid_refine.json")); SM=json.load(open(R+"/synth_metrics.json"))
TRUE=3.245e-5
plt.rcParams.update({"axes.grid":True,"font.size":11})

def sci_y(ax):
    f=ScalarFormatter(useMathText=True); f.set_powerlimits((0,0))
    ax.yaxis.set_major_formatter(f)
def sci_x(ax):
    f=ScalarFormatter(useMathText=True); f.set_powerlimits((0,0))
    ax.xaxis.set_major_formatter(f)
def save(fig,name,w_in,h_in):
    fig.set_size_inches(w_in*2.4,h_in*2.4)
    fig.savefig(OUT+"/"+name,dpi=150)  # no bbox='tight' -> exact aspect
    plt.close(fig)

# ---- Figure 1: classical inverse baseline curve (image1) ----
fig,ax=plt.subplots(constrained_layout=True)
ax.plot(S["grid_x"],S["grid_loss"])
ax.axvline(TRUE,ls="--",label="True drag coeff")
ax.axvline(GR["full"]["best_seed42"],ls="--",label="Best baseline estimate")
ax.set_xlabel("Candidate Drag Coefficient"); ax.set_ylabel("Position MSE on Sparse Observations")
sci_y(ax); sci_x(ax); ax.legend()
save(fig,"image1.png",3.6724,2.3853)

# ---- Figure 2a: plain MLP synthetic reconstruction (image2) ----
ts=S["truth_state"]; ob=S["obs_state_noisy"]; pp=S["plain_pred42"]
fig,ax=plt.subplots(constrained_layout=True)
ax.plot(ts[:,0],ts[:,1],label="Dense truth")
ax.plot(pp[:,0],pp[:,1],label="PINN reconstruction")
ax.scatter(ob[:,0],ob[:,1],s=16,label="Sparse noisy observations")
ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)"); ax.axis("equal"); ax.legend()
save(fig,"image2.png",2.8137,2.7444)

# ---- Figure 2b: Fourier synthetic reconstruction (image3) ----
fp=S["four_pred42"]
fig,ax=plt.subplots(constrained_layout=True)
ax.plot(ts[:,0],ts[:,1],label="Dense truth")
ax.plot(fp[:,0],fp[:,1],"--",label="Fourier-PINN reconstruction")
ax.scatter(ob[:,0],ob[:,1],s=16,color="k",zorder=5,label="Sparse noisy obs")
ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)"); ax.axis("equal"); ax.legend()
save(fig,"image3.png",3.2508,3.1707)

# ---- Figure 3: ISS Fourier reconstruction (image4) ----
fig,ax=plt.subplots(constrained_layout=True)
ax.plot(I["dense_x"],I["dense_y"],label="Dense SGP4 benchmark")
ax.plot(I["four_x"],I["four_y"],"--",label="Fourier-PINN reconstruction")
ax.scatter(I["sparse_x"],I["sparse_y"],s=16,color="k",zorder=5,label="Sparse obs")
ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)"); ax.axis("equal"); ax.legend()
save(fig,"image4.png",2.8264,2.7567)

# ---- Figure 4: training history + drag error bars across seeds (image5) ----
fig,ax=plt.subplots(1,2,constrained_layout=True)
lab={"total":"Total loss","data":"Data loss","ic":"Initial-condition loss","physics":"Physics loss"}
for k in ["total","data","ic","physics"]:
    ax[0].semilogy(S["four_hist_epoch"],S["four_hist_"+k],label=lab[k])
ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss"); ax[0].legend(fontsize=9)
# right: mean +/- SD across seeds for the three estimators
dj=S["four_dj"]; dl=S["four_dl"]
means=[dj.mean(), dl.mean(), GR["full"]["best_mean"]]
sds=[dj.std(), dl.std(), GR["full"]["best_sd"]]
xs=[0,1,2]; labels=["Jointly\nlearned","Frozen-LSQ","Grid\nsearch"]
ax[1].errorbar(xs,means,yerr=sds,fmt="o",capsize=5,color="C0",label="Estimate (mean ± SD)")
ax[1].axhline(TRUE,ls="--",color="k",label="True value")
ax[1].set_xticks(xs); ax[1].set_xticklabels(labels)
ax[1].set_ylabel("Drag Coefficient"); sci_y(ax[1]); ax[1].legend(fontsize=9)
ax[1].set_xlim(-0.5,2.5)
save(fig,"image5.png",6.1944,1.7302)

# ---- Figure 5: perturbation growth (image6) ----
th=S["truth_t"]/3600.0
fig,ax=plt.subplots(constrained_layout=True)
ax.plot(th,S["resid_j2"],label="|r(J2) - r(two-body)|")
ax.plot(th,S["resid_all"],label="|r(J2+drag) - r(two-body)|")
ax.set_xlabel("Time (hours)"); ax.set_ylabel("Position Residual (km)"); ax.legend()
save(fig,"image6.png",2.9097,1.9378)

# ---- Figure 6: ISS altitude over 24 h (image7) ----
fig,ax=plt.subplots(constrained_layout=True)
ax.plot(I["alt24_t"]/3600.0,I["alt24_alt"])
ax.set_xlabel("Time Since TLE Epoch (hours)"); ax.set_ylabel("Altitude (km)")
save(fig,"image7.png",2.8542,1.9125)

print("figures written to",OUT)
for f in sorted(os.listdir(OUT)):
    from PIL import Image
    im=Image.open(OUT+"/"+f); print(f, im.size, "ratio=%.3f"%(im.size[0]/im.size[1]))
