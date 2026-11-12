"""Compile and test deterministic handover transitions without ROS."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"result"/datetime.now(timezone.utc).strftime("handover_check_%Y%m%dT%H%M%SZ")
OUT.mkdir(parents=True)
subprocess.run(["g++","-O2","-Wall","-std=c++17",str(ROOT/"testing_code/src/handover_test.cpp"),"-o","/tmp/cl_handover_test"],check=True)
r=subprocess.run(["/tmp/cl_handover_test"],capture_output=True,text=True,check=True)
result=dict(passed=True,output=r.stdout.strip(),source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"testing_code/src/handover.hpp",ROOT/"testing_code/src/handover_test.cpp")})
(OUT/"summary.json").write_text(json.dumps(result,indent=2))
(ROOT/"result/handover_check_latest.json").write_text(json.dumps(dict(run=OUT.name)))
print(json.dumps(result),flush=True)
