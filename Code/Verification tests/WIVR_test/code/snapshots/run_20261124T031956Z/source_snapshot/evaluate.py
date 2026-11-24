from pathlib import Path
import sys,subprocess,json
ROOT=Path(__file__).resolve().parents[1];RUN=Path(sys.argv[1])
folder=RUN/'latest_gp'
subprocess.run([sys.executable,str(ROOT/'code/evaluate_case.py'),str(RUN),str(folder)],check=True)
summary=dict(purpose='Latest-GP scheduling validation; fixed WIVR, not a policy comparison',
 baseline_run='run_20261117T023229Z',cases=[json.loads((folder/'summary.json').read_text())])
(RUN/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
