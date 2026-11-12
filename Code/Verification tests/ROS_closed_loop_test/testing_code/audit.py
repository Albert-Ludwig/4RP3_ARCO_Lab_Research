"""Independent Python numerical checks and offline GP estimation on saved snapshots."""
from pathlib import Path
import json,sys,os
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUN=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/"result"/json.loads((ROOT/"result/latest.json").read_text())["run"]
sys.path.insert(0,str(ROOT.parent/"Continuous_sensing_test/testing_code"))
sys.path.insert(0,str(ROOT.parent/"Scoring_policy_test/testing_code"))
sys.path.insert(0,str(ROOT.parent/"SRF_test/testing_code"))
from components import BatchGP
from scoring import Grid,AdaptiveWeights,Objective
from srf_field import TemperatureField
def gp(j):
 p=dict(lengthscales_m_m_s=j["length"],output_variance_c2=j["variance"],noise_variance_c2=j["noise"],mean_c=j["mean"])
 return BatchGP(np.array(j["x"]),np.array(j["y"]),p,[str(i) for i in range(len(j["y"]))])
protocol=json.loads((RUN/"protocol.json").read_text()) if (RUN/"protocol.json").exists() else {}
scoring_dt=protocol.get("scoring_time_step_s",.5)
duration=json.loads((RUN/"field.json").read_text())["mission_interval_s"][1]
initial=gp(json.loads((RUN/"initial.json").read_text()))
cpp=json.loads((RUN/"cpp_audit.json").read_text());mu,var=initial.posterior(np.array(cpp["queries"]))
checks=dict(mean_max_error=float(np.max(np.abs(mu-cpp["mean"]))),variance_max_error=float(np.max(np.abs(var-cpp["variance"]))))
checks["open_route_regression"]=json.loads((RUN/"open_route_check.json").read_text())["passed"] if (RUN/"open_route_check.json").exists() else None
checks["gp_equivalent"]=checks["mean_max_error"]<1e-8 and checks["variance_max_error"]<1e-8
audits=[]
for p in sorted(RUN.glob("plan_*.json"),key=lambda p:int(p.stem.split("_")[1])):
 plan=json.loads(p.read_text());g=gp(json.loads((RUN/f'gp_{plan["round"]}.json').read_text()))
 grid=Grid(extent=(97,64),cell=10,obstacle=None,times=np.arange(0,duration+scoring_dt/2,scoring_dt))
 weights=AdaptiveWeights(grid).refresh(g,plan["t"])
 score=Objective("wivr",g,grid,weights).score(np.array(plan["events"]).reshape(-1,3))
 audits.append(dict(round=plan["round"],cpp_score=plan["score"],python_score=score,absolute_error=abs(score-plan["score"]),passed=abs(score-plan["score"])<1e-8))
field=TemperatureField(json.loads((RUN/"field.json").read_text()))
plans=[json.loads(p.read_text()) for p in sorted(RUN.glob("plan_*.json"),key=lambda p:int(p.stem.split("_")[1]))]
evaluation=[]
gridxy=np.array([[min(x+5,(x+97)/2),min(y+5,(y+64)/2)] for x in range(0,97,10) for y in range(0,64,10)])
for t in np.linspace(0,duration,5):
 eligible=sorted([p for p in plans if p.get("gp_committed_t",p["t"])<=t],key=lambda p:(p["gp_version"],p.get("gp_committed_t",p["t"]),p["round"]))
 model=gp(json.loads((RUN/f'gp_{eligible[-1]["round"]}.json').read_text())) if eligible else initial
 q=np.column_stack((gridxy,np.full(len(gridxy),t)));truth=field.truth(q)
 pred,variance=model.posterior(q);base,_=initial.posterior(q)
 evaluation.append(dict(time_s=t,rmse_c=float(np.sqrt(np.mean((pred-truth)**2))),mae_c=float(np.mean(np.abs(pred-truth))),history_only_rmse_c=float(np.sqrt(np.mean((base-truth)**2))),mean_std_c=float(np.sqrt(variance).mean()),observations=len(model.x)))
lookahead=[p for p in plans if p.get("lookahead",False)]
lookahead_checks=dict(search_count=len(lookahead),
 no_uncommitted_observations_used=all(p["batch_size"]==0 for p in lookahead),
 gp_available_before_search=all(p.get("gp_committed_t",0)<=p.get("planning_started_t",0)+.1 for p in lookahead))
assert lookahead_checks["no_uncommitted_observations_used"] and lookahead_checks["gp_available_before_search"]
result=dict(gp_checks=checks,returned_route_score_checks=audits,lookahead_checks=lookahead_checks,evaluation=evaluation,
 evaluation_scope="70 fixed cell representatives at five evenly spaced mission times, using the posterior committed by that time; synthetic current-field estimates, not real-field accuracy or a policy ranking.")
(RUN/"audit_and_estimation.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
assert checks["gp_equivalent"] and all(a["passed"] for a in audits)

# Pre-assimilation estimates: choose the GP committed before each sample timestamp.
records=[]
sensor_file=RUN/"sampler.jsonl"
if sensor_file.exists():
    records=[json.loads(s) for s in sensor_file.read_text().splitlines() if s.strip()]
    records=[r for r in records if "temperature" in r]
prequential=[]
models={p["round"]:gp(json.loads((RUN/f'gp_{p["round"]}.json').read_text())) for p in plans}
for r in records:
    eligible=sorted([p for p in plans if p.get("gp_committed_t",p["t"])<=r["t"]],key=lambda p:(p["gp_version"],p.get("gp_committed_t",p["t"]),p["round"]))
    model=models[eligible[-1]["round"]] if eligible else initial
    point=np.array([[r["x"],r["y"],r["t"]]])
    mu,v=model.posterior(point);truth=float(field.truth(point)[0])
    prequential.append(dict(id=r["id"],t=r["t"],mean=float(mu[0]),latent_truth=truth,
        measured=r["temperature"],variance=float(v[0]),observations=len(model.x)))
if prequential:
    pred=np.array([p["mean"] for p in prequential])
    truth=np.array([p["latent_truth"] for p in prequential])
    observed=np.array([p["measured"] for p in prequential])
    result["pre_assimilation"]=dict(count=len(pred),latent_rmse_c=float(np.sqrt(np.mean((pred-truth)**2))),
        observed_rmse_c=float(np.sqrt(np.mean((pred-observed)**2))),latent_mae_c=float(np.mean(abs(pred-truth))))
    (RUN/"pre_assimilation_predictions.json").write_text(json.dumps(prequential,indent=2))
result["audit_source_sha256"]=__import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest()
(RUN/"audit_and_estimation.json").write_text(json.dumps(result,indent=2))
