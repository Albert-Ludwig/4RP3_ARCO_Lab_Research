"""Freeze a shared synthetic history, RBF parameters, scene and evaluation grid."""
import site
site.addsitedir('/usr/lib/python3/dist-packages')
from pathlib import Path
import importlib.util,sys,os,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[2];CODE=ROOT/'testing_code'
OUT=Path(os.environ['SP_COMMON']);OUT.mkdir(parents=True,exist_ok=False)
VER=REPO/'Code/Verification tests';OLD=VER/'ROS_closed_loop_test/testing_code'
sys.path.insert(0,str(VER/'SRF_test/testing_code'));sys.path.insert(0,str(OLD))
from srf_field import TemperatureField
from navigation_profile import apply,profile
def save(n,v):(OUT/n).write_text(json.dumps(v,indent=2)+'\n')
p=json.loads((CODE/'protocol.json').read_text());save('protocol.json',p)
f=json.loads((OLD/'field_config.json').read_text());f.update(working_extent_m=p['extent_m'],mission_interval_s=[p['history_interval_s'][0],300],field_seed=p['field_seeds'][0])
f['status']='Controlled synthetic comparison; common SRF history and online truth; field scales uncalibrated'
save('field.json',f);field=TemperatureField(f)
kroot=REPO/'Code/ROS_simulation/Trained_Real_Data_Kernel/result'
kernel_file=kroot/json.loads((kroot/'latest.json').read_text())['run']/'hyperparameters.json'
hp=json.loads(kernel_file.read_text())
initial=dict(length=hp['lengthscales_m_m_s'],variance=hp['output_variance_c2'],noise=hp['noise_variance_c2'],mean=hp['mean_c'])
rng=np.random.default_rng(p['history_seed']);x=rng.random((p['history_count'],3));x[:,:2]*=p['extent_m'];lo,hi=p['history_interval_s'];x[:,2]=lo+(hi-lo)*x[:,2]
y=field.measure(x,[f'history-{i}' for i in range(len(x))]);initial.update(x=x.tolist(),y=y.tolist());save('initial.json',initial)
points=[p['home_m']]+[[a,b] for a in (3,9,15,21,27) for b in (4,16)];save('candidates.json',points)
xy=np.array([[x,y] for x in np.arange(1,30,2) for y in np.arange(1,20,2)])
q=np.vstack([np.column_stack((xy,np.full(len(xy),t))) for t in np.arange(0,301,5)])
np.savez_compressed(OUT/'evaluation_grid.npz',queries=q,truth=field.truth(q))
nav=VER/'Nav2_test/testing_code/src/well_e_nav2_test'
spec=importlib.util.spec_from_file_location('nav_scene',nav/'launch/nav2_test.launch.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
scenario=json.loads((nav/'config/scenario.json').read_text());scenario.update(map_bounds_m=[-15,15,-10,10]);scenario['obstacle'].update(center_m=[-100,-100])
from ament_index_python.packages import get_package_share_directory
mod.generate_scene(Path(get_package_share_directory('well_e_bunker_sim')),OUT,scenario);mod.generate_parameters(OUT,scenario)
import xml.etree.ElementTree as ET
tree=ET.parse(OUT/'world.sdf');world=tree.getroot().find('world');world.remove(world.find("model[@name='central_obstacle']"));tree.write(OUT/'world.sdf')
apply(OUT,scenario);save('scene.json',scenario)
save('provenance.json',dict(conditioning='980 synthetic past observations of the SAME SRF, not interpolated real temperatures',
  hyperparameter_source=str(kernel_file),hyperparameter_source_sha256=hashlib.sha256(kernel_file.read_bytes()).hexdigest(),
  generator='GSTools Matérn finite-mode SRF',kernel='Fixed real-data-fitted ARD RBF; no MLL refit in this experiment',
  history_interval_s=p['history_interval_s'],evaluation_queries=len(q),common_files_sha256={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ('initial.json','field.json','candidates.json','evaluation_grid.npz')}))
print('Prepared shared synthetic history and',len(q),'independent space-time evaluation queries',flush=True)
