"""Check 2 s WIVR target times; measurement events stay at 1 Hz. No ROS run."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/"Continuous_sensing_test/testing_code"))
sys.path.insert(0,str(ROOT.parent/"Scoring_policy_test/testing_code"))
from components import BatchGP
from scoring import Grid,AdaptiveWeights,Objective
OUT=ROOT/"result"/datetime.now(timezone.utc).strftime("grid_check_%Y%m%dT%H%M%SZ")
OUT.mkdir(parents=True)
protocol=json.loads((ROOT/"testing_code/protocol.json").read_text())
assert protocol["scoring_time_step_s"]==2 and protocol["sensor_interval_s"]==1
assert protocol["reference_resolution"]["time_s"]==.5
prior=ROOT/"result"/json.loads((ROOT/"result/batch_check_latest.json").read_text())["run"]
reference=json.loads((prior/"1_batched.json").read_text())
baseline=ROOT/"result/run_20261111T212025Z"
subprocess.run(["/tmp/cl_install/well_e_closed_loop/lib/well_e_closed_loop/score_benchmark",
                "grid2",str(baseline),str(OUT/"cpp_grid2.json")],check=True)
current=json.loads((OUT/"cpp_grid2.json").read_text())
checks=[]
for old,new in zip(reference,current):
    assert new["scoring_inputs"]==70*151
    assert old["events"]==new["events"],"Target-time changes must not discard sensor events"
    events=np.array(new["events"]).reshape(-1,3)
    assert np.allclose(np.diff(events[:,2]),1.)
    j=json.loads((baseline/f'gp_{new["case"]}.json').read_text())
    hp=dict(lengthscales_m_m_s=j["length"],output_variance_c2=j["variance"],
            noise_variance_c2=j["noise"],mean_c=j["mean"])
    gp=BatchGP(np.array(j["x"]),np.array(j["y"]),hp,[str(i) for i in range(len(j["y"]))])
    grid=Grid(extent=(97,64),cell=10,obstacle=None,times=np.arange(0,301,2.))
    w=AdaptiveWeights(grid).refresh(gp,new["elapsed_s"])
    score=Objective("wivr",gp,grid,w).score(events)
    error=abs(score-new["scores"][0])
    assert error<1e-10+1e-8*abs(score)
    checks.append(dict(case=new["case"],event_count=len(events),cpp_score_2s=new["scores"][0],
        python_score_2s=score,numerical_absolute_error=error,score_05s=old["scores"][0],
        relative_score_change=abs(new["scores"][0]-old["scores"][0])/max(1e-30,abs(old["scores"][0]))))
result=dict(passed=True,scoring_dt_s=2,scoring_inputs=10570,sensor_interval_s=1,srf_changed=False,
    checks=checks,max_numerical_absolute_error=max(c["numerical_absolute_error"] for c in checks),
    max_relative_score_change_vs_05s=max(c["relative_score_change"] for c in checks),
    scope="Three fixed saved workloads; independent Python 2 s WIVR and unchanged 1 Hz event sets. This is a configuration/numerical check, not a full ROS rerun or route-ranking convergence study.",
    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        ROOT/"testing_code/src/core.hpp",ROOT/"testing_code/src/nodes.cpp",ROOT/"testing_code/protocol.json"]})
(OUT/"summary.json").write_text(json.dumps(result,indent=2))
(ROOT/"result/grid_check_latest.json").write_text(json.dumps(dict(run=OUT.name)))
print(json.dumps(result),flush=True)
