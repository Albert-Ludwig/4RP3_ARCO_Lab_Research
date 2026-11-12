"""Scientific figures for measured-data replay and GP-only field estimates."""

import csv
import json
from pathlib import Path
from data import ROOT
from gp import GPModel
from navigation import GridMap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


def rows(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def main():
    folder=ROOT/"10.18_result/baseline/intel_lab/replay"
    figures=folder/"figures";figures.mkdir(exist_ok=True)
    config=json.loads((folder/"config.json").read_text());grid=GridMap(config["map"])
    initial=GPModel.load(ROOT/"10.18_result/baseline/intel_lab/forecast/initial_gp.pt")
    final=GPModel.load(folder/"final_gp.pt")
    axis=np.arange(.375,30,.75);xx,yy=np.meshgrid(axis,axis)
    xy=np.column_stack([xx.ravel(),yy.ravel()]);mask=np.array([grid.is_reachable(p) for p in xy])
    query=np.column_stack([xy[mask],np.full(mask.sum(),72.25)])
    before,after=initial.predict(query),final.predict(query)
    np.savez_compressed(folder/"field_predictions.npz",query=query,xx=xx,yy=yy,mask=mask,
                        initial_mean=before["mean"],final_mean=after["mean"],
                        initial_latent_var=before["latent_var"],final_latent_var=after["latent_var"])
    for key,name,label in [("mean","temperature_maps.png","Temperature (°C)"),
                            ("latent_var","uncertainty_maps.png","Latent standard deviation (°C)")]:
        values=[p[key] if key=="mean" else np.sqrt(p[key]) for p in [before,after]]
        low=min(v.min() for v in values);high=max(v.max() for v in values)
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
        for ax,v,title in zip(axes,values,["History only","After recorded online sampling"]):
            field=np.full(len(xy),np.nan);field[mask]=v
            mesh=ax.pcolormesh(xx,yy,field.reshape(xx.shape),shading="nearest",vmin=low,vmax=high,cmap="viridis")
            ax.set(title=title,xlabel="x (m)",ylabel="y (m)",aspect="equal")
        fig.colorbar(mesh,ax=axes,label=label)
        fig.suptitle("GP prediction at 72.25 h; dense real ground truth unavailable",fontsize=10)
        fig.savefig(figures/name,dpi=160);plt.close(fig)
    trajectory=rows(folder/"robot_trajectory.csv");candidates=rows(folder/"candidates.csv")
    fig,ax=plt.subplots(figsize=(6,5))
    for polygon in config["map"]["obstacles"]:ax.add_patch(Polygon(polygon,color="grey",alpha=.7))
    ax.plot([float(r["x_m"]) for r in trajectory],[float(r["y_m"]) for r in trajectory],lw=1.2,label="Executed path")
    ax.scatter([float(r["x_m"]) for r in candidates],[float(r["y_m"]) for r in candidates],s=20,label="Sensor candidates")
    for r in candidates:ax.annotate(r["sensor_id"],(float(r["x_m"]),float(r["y_m"])),fontsize=6,xytext=(3,3),textcoords="offset points")
    ax.scatter(*config["map"]["home_m"],marker="*",s=120,label="Home: sensor 16")
    ax.set(xlim=(0,30),ylim=(0,30),xlabel="x (m)",ylabel="y (m)",aspect="equal",title="Intel replay on the existing simulated map")
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(figures/"navigation.png",dpi=160);plt.close(fig)
    metrics=rows(folder/"metrics.csv")
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for ax,key,label in zip(axes,["rmse_c","mean_variance_c2"],["Held-out RMSE (°C)","Mean latent variance (°C²)"]):
        ax.plot([int(r["measurement_count"]) for r in metrics],[float(r[key]) for r in metrics],"o-")
        ax.set(xlabel="Executed measurements",ylabel=label)
    fig.tight_layout();fig.savefig(figures/"reconstruction_metrics.png",dpi=160);plt.close(fig)
    depth=json.loads((ROOT/"10.18_result/grg_depth_benchmark/intel_lab/depth_summary.json").read_text())
    successful=[r for r in depth if r["status"]=="SUCCESS"]
    fig,axes=plt.subplots(1,3,figsize=(11,3.3))
    for ax,key,label in zip(axes,["final_rmse_c","mi_nats","grg_runtime_s"],["Final held-out RMSE (°C)","Executed joint MI (nats)","Total GRG wall time (s)"]):
        ax.bar([str(r["depth"]) for r in successful],[r[key] for r in successful])
        ax.set(xlabel="Recursive depth",ylabel=label)
    fig.tight_layout()
    target=ROOT/"10.18_result/grg_depth_benchmark/intel_lab/figures";target.mkdir(exist_ok=True)
    fig.savefig(target/"depth_comparison.png",dpi=160);plt.close(fig)


if __name__=="__main__":main()
