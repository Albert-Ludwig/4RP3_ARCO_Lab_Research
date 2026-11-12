"""Diagnostic figures only; no ROS execution, tuning or model changes."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"result"/json.loads((ROOT/"result/route_diagnosis_latest.json").read_text())["run"]
d=json.loads((OUT/"geometry.json").read_text())
v=json.loads((OUT/"verification.json").read_text())
names=["straight","previous_bent","lower_bent","vertical"]
labels=["Horizontal","Previous bent","Lower bent","Vertical"]
fig,axes=plt.subplots(2,2,figsize=(11,7),layout="constrained")
for name,label in zip(names,labels):
 a=np.array(d["fitted_kernel"][name]["events"])
 axes[0,0].plot(a[:,0],a[:,1],label=label,alpha=.8)
axes[0,0].set(xlim=(0,97),ylim=(0,64),xlabel="Field x (m)",ylabel="Field y (m)",title="Same first 180 sensing ticks",aspect="equal")
axes[0,0].legend(fontsize=8)
x=np.arange(len(names));width=.35
for offset,cell in ((-width/2,10),(width/2,5)):
 scores=[v["scores"][f"fitted_kernel_{cell}m"][n] for n in names]
 axes[0,1].bar(x+offset,scores,width,label=f"{cell} m scoring cells")
axes[0,1].set(xticks=x,xticklabels=labels,ylabel="Weighted variance reduction (degC squared)",title="Fitted RBF; identical event count")
axes[0,1].tick_params(axis="x",labelrotation=15);axes[0,1].legend(fontsize=8)
dist=np.linspace(0,40,300)
for i,label in ((0,"x"),(1,"y")):
 ell=d["initial_length"][i]
 axes[1,0].plot(dist,np.exp(-.5*(dist/ell)**2),label=f"{label}: length = {ell:.2f} m")
axes[1,0].set(xlabel="Same-time spatial separation (m)",ylabel="Prior kernel correlation",title="Fitted directional correlation");axes[1,0].legend()
axes[1,1].bar(x,[v["scores"]["axis_swapped_counterfactual_10m"][n] for n in names])
axes[1,1].set(xticks=x,xticklabels=labels,ylabel="Weighted variance reduction (degC squared)",title="Counterfactual: x/y lengths swapped")
axes[1,1].tick_params(axis="x",labelrotation=15)
fig.suptitle("Route geometry diagnostic; not a policy comparison or field-accuracy test")
fig.savefig(OUT/"route_diagnosis.png",dpi=160)
plt.close(fig)
print("SAVED",OUT/"route_diagnosis.png")
