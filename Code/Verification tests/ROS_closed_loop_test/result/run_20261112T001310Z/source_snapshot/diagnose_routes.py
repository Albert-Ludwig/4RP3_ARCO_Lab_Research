"""Read saved state, compare route geometry, and measure uncapped full GRG."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,os,hashlib
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"result"/datetime.now(timezone.utc).strftime("route_diagnosis_%Y%m%dT%H%M%SZ")
OUT.mkdir(parents=True)
base=ROOT/"result/run_20261111T233622Z"
env=dict(os.environ,CL_MATRIX_THREADS="4",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="1")
exe="/tmp/cl_route_diagnostics"
subprocess.run(["g++","-O3","-Wall","-std=c++17","-fopenmp","-I/usr/include/eigen3",
                str(ROOT/"testing_code/src/route_diagnostics.cpp"),"-o",exe],check=True)
(ROOT/"result/route_diagnosis_latest.json").write_text(json.dumps(dict(run=OUT.name)))
for mode in ("geometry","complete"):
 with (OUT/(mode+".log")).open("w") as stream:
  subprocess.run([exe,mode,str(base),str(OUT/(mode+".json"))],env=env,stdout=stream,check=True)
 v=json.loads((OUT/(mode+".json")).read_text())
 if mode=="geometry":
  print({k:{name:row["score"] for name,row in v[k].items()} for k in ("fitted_kernel","axis_swapped_counterfactual")},flush=True)
 else:
  s=v["search"];print({k:s.get(k) for k in ("complete_wall_s","timeout","calls","route","score")},flush=True)
(OUT/"sources.json").write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"testing_code/src/core.hpp",ROOT/"testing_code/src/route_diagnostics.cpp")},indent=2))
print("SAVED",OUT,flush=True)
