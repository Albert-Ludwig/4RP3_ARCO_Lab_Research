"""Independent Python route-score and quadrature-sensitivity diagnostics."""
from pathlib import Path
import sys,json,numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"result"/json.loads((ROOT/"result/route_diagnosis_latest.json").read_text())["run"]
sys.path.insert(0,str(ROOT.parent/"Continuous_sensing_test/testing_code"))
sys.path.insert(0,str(ROOT.parent/"Scoring_policy_test/testing_code"))
from components import BatchGP
from scoring import Grid,AdaptiveWeights,Objective
source=ROOT/"result/run_20261111T233622Z"
initial=json.loads((source/"initial.json").read_text())
d=json.loads((OUT/"geometry.json").read_text())
values={};errors=[]
for mode in ("fitted_kernel","axis_swapped_counterfactual"):
 length=list(initial["length"])
 if mode!="fitted_kernel":length[0],length[1]=length[1],length[0]
 parameters=dict(lengthscales_m_m_s=length,output_variance_c2=initial["variance"],
   noise_variance_c2=initial["noise"],mean_c=initial["mean"])
 gp=BatchGP(np.array(initial["x"]),np.array(initial["y"]),parameters,[str(i) for i in range(len(initial["y"]))])
 for cell in ((10,5) if mode=="fitted_kernel" else (10,)):
  grid=Grid(extent=(97,64),cell=cell,obstacle=None,times=np.arange(0,301,2))
  weights=AdaptiveWeights(grid).refresh(gp,d["elapsed_s"])
  objective=Objective("wivr",gp,grid,weights)
  scores={}
  for name,row in d[mode].items():
   events=np.array(row["events"]);assert len(events)==180 and np.array_equal(events[:,2],np.arange(1,181))
   score=objective.score(events);scores[name]=score
   if cell==10:
    errors.append(abs(score-row["score"]));assert abs(score-row["score"])<1e-10
  if mode=="fitted_kernel" and cell==10:
   completed=json.loads((OUT/"complete.json").read_text())["search"]
   complete_score=objective.score(np.array(completed["events"]))
   complete_error=abs(complete_score-completed["score"])
   assert complete_error<1e-10
  values[f"{mode}_{cell}m"]=scores
  print(mode,cell,scores,flush=True)
result=dict(passed=True,max_cpp_python_error=max(errors),complete_search_score_error=complete_error,scores=values,
  spatial_correlation_at_5m=dict(x=float(np.exp(-.5*(5/initial["length"][0])**2)),
                                 y=float(np.exp(-.5*(5/initial["length"][1])**2))),
  interpretation="Matched-route diagnostic, not policy comparison or real-field accuracy. Swapped axes are a counterfactual and do not modify the saved/deployed GP. A 5 m grid is diagnostic only; production remains 10 m.",
  source_sha256=__import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/"verification.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
