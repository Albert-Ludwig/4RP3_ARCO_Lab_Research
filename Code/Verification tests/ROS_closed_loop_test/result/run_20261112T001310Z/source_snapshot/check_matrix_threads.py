"""Isolate OpenMP thread count on unchanged 2 s-grid WIVR workloads."""
from pathlib import Path
from datetime import datetime, timezone
import json, subprocess, statistics, os, hashlib
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"result"/datetime.now(timezone.utc).strftime("matrix_threads_%Y%m%dT%H%M%SZ")
OUT.mkdir(parents=True)
exe="/tmp/cl_install/well_e_closed_loop/lib/well_e_closed_loop/score_benchmark"
baseline=ROOT/"result/run_20261111T212025Z"
trials=[]
for index,threads in enumerate((1,2,4,4,2,1)):
    target=OUT/f"{index}_{threads}.json"
    env=dict(os.environ, CL_MATRIX_THREADS=str(threads), OMP_NUM_THREADS=str(threads),
             OPENBLAS_NUM_THREADS="1", OMP_DYNAMIC="FALSE")
    subprocess.run([exe,"grid2",str(baseline),str(target)],env=env,check=True)
    cases=json.loads(target.read_text())
    assert all(c["matrix_threads"]["eigen_effective"]==threads for c in cases)
    assert all(c["matrix_threads"]["openmp_version"]>0 for c in cases)
    trial=dict(threads=threads,cases=cases,
               setup_s=sum(c["setup_s"] for c in cases),
               cold_score_s=sum(c["timings_s"][0] for c in cases),
               all_score_s=sum(sum(c["timings_s"]) for c in cases))
    trials.append(trial)
    print(json.dumps({k:v for k,v in trial.items() if k!="cases"}),flush=True)
errors=[]
for trial in trials:
    for a,b in zip(trials[0]["cases"],trial["cases"]):
        assert a["events"]==b["events"] and a["scoring_inputs"]==b["scoring_inputs"]==10570
        for x,y in zip(a["scores"],b["scores"]):
            errors.append(abs(x-y))
            assert abs(x-y)<=1e-10+1e-8*abs(x)
medians={str(n):{k:statistics.median(t[k] for t in trials if t["threads"]==n)
                  for k in ("setup_s","cold_score_s","all_score_s")} for n in (1,2,4)}
# Include query preparation and all four cache workloads, not just one multiply.
chosen=min((1,2,4),key=lambda n:medians[str(n)]["setup_s"]+medians[str(n)]["all_score_s"])
result=dict(passed=True,selected_threads=chosen,medians=medians,
            max_score_absolute_error=max(errors),trials=trials,
            scope="Two repetitions per count, balanced 1/2/4/4/2/1 order; same three GPs, routes, 1 Hz events, 2 s targets, weights and noise. Timing is fixed-workload, not full recursive search.",
            source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (ROOT/"testing_code/src/core.hpp",ROOT/"testing_code/src/matrix_threads.hpp",
                        ROOT/"testing_code/src/CMakeLists.txt",ROOT/"testing_code/src/score_benchmark.cpp")})
(OUT/"summary.json").write_text(json.dumps(result,indent=2))
(ROOT/"result/matrix_threads_latest.json").write_text(json.dumps(dict(run=OUT.name)))
print(json.dumps({k:v for k,v in result.items() if k!="trials"}),flush=True)
