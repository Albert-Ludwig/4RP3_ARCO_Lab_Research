"""Matched-workload correctness and timing check; no ROS or SRF truth."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,hashlib,statistics,os
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"result"/datetime.now(timezone.utc).strftime("batch_check_%Y%m%dT%H%M%SZ")
OUT.mkdir(parents=True)
baseline=ROOT/"result/run_20261111T212025Z"
exe="/tmp/cl_install/well_e_closed_loop/lib/well_e_closed_loop/score_benchmark"
os.environ.update(OPENBLAS_NUM_THREADS="2",OMP_NUM_THREADS="2")
trials=[]
for i,mode in enumerate(("scalar","batched","batched","scalar")):
    path=OUT/f"{i}_{mode}.json"
    subprocess.run([exe,mode,str(baseline),str(path)],check=True)
    data=json.loads(path.read_text())
    trials.append(dict(mode=mode,cases=data))
    print(mode,"cold_score_seconds",round(sum(c["timings_s"][0] for c in data),4),flush=True)
reference=trials[0]["cases"]
errors=[]
for trial in trials:
    for a,b in zip(reference,trial["cases"]):
        assert a["events"]==b["events"]
        for x,y in zip(a["scores"],b["scores"]):
            err=abs(x-y);errors.append(err)
            assert err<=1e-10+1e-8*abs(x),(trial["mode"],a["case"],x,y)
cold={mode:statistics.median(sum(c["timings_s"][0] for c in t["cases"]) for t in trials if t["mode"]==mode) for mode in ("scalar","batched")}
result=dict(passed=True,max_score_absolute_error=max(errors),cases_per_trial=3,trials=len(trials),
    scope="Same saved GPs, route events, weights, grid, noise and full WIVR; 2 trials per mode in ABBA order; cold event-cache times exclude GP/query setup.",
    median_total_cold_score_s=cold,cold_score_speedup=cold["scalar"]/cold["batched"],
    source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/"testing_code/src/core.hpp",ROOT/"testing_code/src/score_benchmark.cpp",baseline/"source_snapshot/src/core.hpp"]})
(OUT/"summary.json").write_text(json.dumps(result,indent=2))
(ROOT/"result/batch_check_latest.json").write_text(json.dumps(dict(run=OUT.name)))
print(json.dumps(result),flush=True)
