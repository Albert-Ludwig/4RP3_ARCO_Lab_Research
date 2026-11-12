"""Compile and test deterministic handover transitions without ROS."""
from pathlib import Path
import json,subprocess,hashlib,sys
ROOT=Path(__file__).resolve().parents[1]
subprocess.run(["g++","-O2","-Wall","-std=c++17",str(ROOT/"testing_code/src/handover_test.cpp"),"-o","/tmp/cl_handover_test"],check=True)
r=subprocess.run(["/tmp/cl_handover_test"],capture_output=True,text=True,check=True)
result=dict(passed=True,output=r.stdout.strip(),source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"testing_code/src/handover.hpp",ROOT/"testing_code/src/handover_test.cpp")})
if len(sys.argv)>1:
    output=Path(sys.argv[1])
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
