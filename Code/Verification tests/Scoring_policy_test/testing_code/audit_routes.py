"""Independent final-route reference formulas, outside timed GRG searches."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[2]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit():
    run=ROOT/"result"/json.loads((ROOT/"result/latest.json").read_text())["run"]
    summary=json.loads((run/"summary.json").read_text())
    assert summary["passed"]
    protocol=json.loads((run/"protocol.json").read_text())
    provenance=json.loads((run/"provenance.json").read_text())
    entries=provenance["references"]
    for entry in entries.values():assert sha(REPO/entry["path"])==entry["sha256"]
    spec=importlib.util.spec_from_file_location("independent_route_gp",REPO/entries["legacy_gp"]["path"])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    initial=np.load(REPO/entries["gp_initial"]["path"])
    params=json.loads((run/"gp_parameters.json").read_text())
    gp=module.GaussianProcess(initial["x"],initial["y"],params)
    grid=np.load(run/"scoring_grid.npz")
    q=grid["queries"];area_q=np.tile(grid["areas"],len(grid["times"]))
    current_variance=gp.posterior(np.column_stack([grid["xy"],np.zeros(len(grid["xy"]))]))[1]
    spatial=grid["areas"] if np.all(current_variance<=protocol["near_zero_variance_c2"]) else grid["areas"]*current_variance
    adaptive_q=np.tile(spatial,len(grid["times"]))
    rows=[]
    for count in protocol["candidate_counts"]:
        for policy in ["mi","emse","wivr"]:
            plan=json.loads((run/f"plan_{count}_{policy}.json").read_text())
            events=np.array(plan["events"],dtype=float).reshape(-1,3)
            covariance=gp.covariance(events);system=covariance+gp.noise*np.eye(len(events))
            if policy=="mi":
                sign,logdet=np.linalg.slogdet(np.eye(len(events))+covariance/gp.noise)
                assert sign>0
                reference=float(.5*logdet)
            else:
                cross=gp.covariance(q,events)
                reductions=np.sum(cross*np.linalg.solve(system,cross.T).T,axis=1)
                weights=area_q if policy=="emse" else adaptive_q
                reference=float(weights@reductions/weights.sum())
            error=abs(reference-plan["gain"])
            passed=bool(np.isclose(reference,plan["gain"],atol=protocol["numerical_absolute_tolerance"],
                                  rtol=protocol["numerical_relative_tolerance"]))
            rows.append(dict(checkpoints=count,policy=policy,events=len(events),logged_gain=plan["gain"],
                             independent_gain=reference,absolute_error=error,passed=passed))
            print("REFERENCE",count,policy,passed,error,flush=True)
    result=dict(passed=all(r["passed"] for r in rows),checks=len(rows),rows=rows,
                maximum_absolute_error=max(r["absolute_error"] for r in rows),
                reference_source_sha256=entries["legacy_gp"]["sha256"],audit_source_sha256=sha(__file__),
                scope="Uncached independent NumPy conditioning formulas; offline acceptance only, not performance timing or policy ranking.")
    (run/"independent_route_checks.json").write_text(json.dumps(result,indent=2)+"\n")
    assert result["passed"],"Independent route score verification failed"
    print(json.dumps({k:result[k] for k in ["passed","checks","maximum_absolute_error"]}))


if __name__=="__main__":audit()
