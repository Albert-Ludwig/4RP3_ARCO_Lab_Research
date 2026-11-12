import site
site.addsitedir('/usr/lib/python3/dist-packages')
"""Freeze real-data mapping, source configuration and the obstacle-free scene."""
from pathlib import Path
import csv,json,importlib.util,os,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; REPO=ROOT.parents[2]
OUT=Path(os.environ["CL_OUTPUT"]); OUT.mkdir(parents=True,exist_ok=True)
def save(name,value): (OUT/name).write_text(json.dumps(value,indent=2)+"\n")
rows=list(csv.DictReader((REPO/"Data/Real_Data_Integration/result/gp_samples.csv").open()))
x=np.array([[float(r["x_east_m"]),float(r["y_north_m"]),float(r["t_s"])] for r in rows])
offset=x[:,:2].min(axis=0); epoch=x[:,2].max()+1
x[:,:2]-=offset; x[:,2]-=epoch
y=np.array([float(r["temperature_c"]) for r in rows])
kroot=REPO/"Code/ROS_simulation/Trained_Real_Data_Kernel/result"
hp=json.loads((kroot/json.loads((kroot/"latest.json").read_text())["run"]/"hyperparameters.json").read_text())
save("initial.json",dict(x=x.tolist(),y=y.tolist(),length=hp["lengthscales_m_m_s"],variance=hp["output_variance_c2"],noise=hp["noise_variance_c2"],mean=hp["mean_c"]))
save("mapping.json",dict(field_xy_equals_enu_minus=offset.tolist(),world_xy_equals_field_minus=[48.5,32],mission_zero_real_relative_s=epoch,
description="Temporary planar translation; not surveyed ROS map registration. Historical distances and intervals preserved. Mission follows last real measurement by 1 second.",
history_count=len(y),historical_time_range_s=[x[:,2].min(),x[:,2].max()]))
cfg=json.loads((ROOT/"testing_code/field_config.json").read_text())
save("field.json",cfg)
protocol=json.loads((ROOT/"testing_code/protocol.json").read_text())
from navigation_profile import profile
protocol["navigation_nominal_speed_m_s"]=profile()["nominal_speed_m_s"]
save("protocol.json",protocol)
points=[[48.5,32]]+[[float(round(v)),float(round(w))] for v in np.linspace(4,93,6) for w in np.linspace(4,60,5)]
save("candidates.json",points)
nav=ROOT.parent/"Nav2_test/testing_code/src/well_e_nav2_test"
spec=importlib.util.spec_from_file_location("nav_launch",nav/"launch/nav2_test.launch.py"); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
scenario=json.loads((nav/"config/scenario.json").read_text())
scenario.update(map_bounds_m=[-48.5,48.5,-32,32])
scenario["obstacle"].update(center_m=[-100,-100])
from ament_index_python.packages import get_package_share_directory
mod.generate_scene(Path(get_package_share_directory("well_e_bunker_sim")),OUT,scenario)
mod.generate_parameters(OUT,scenario)
# The original fixture adds an out-of-area obstacle to use its generator; remove it entirely.
import xml.etree.ElementTree as ET
tree=ET.parse(OUT/"world.sdf"); world=tree.getroot().find("world")
world.remove(world.find("model[@name='central_obstacle']")); tree.write(OUT/"world.sdf")
from navigation_profile import apply, profile
apply(OUT,scenario)
scenario["navigation_profile"]="Capstone-derived limits/smoother, Bunker RPP at up to 1 m/s"
save("scene.json",scenario)
print("Prepared",len(y),"real observations and",len(points)-1,"candidates",flush=True)
