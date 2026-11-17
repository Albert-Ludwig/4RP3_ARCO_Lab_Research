from pathlib import Path
import sys,subprocess,json
ROOT=Path(__file__).resolve().parents[1];RUN=Path(sys.argv[1]);rows=[]
for interval in (1,5):
    folder=RUN/f'update_{interval}s'
    subprocess.run([sys.executable,str(ROOT/'testing_code/evaluate_case.py'),str(RUN),str(folder)],check=True)
    rows.append(json.loads((folder/'summary.json').read_text()))
summary=dict(purpose='GP and GRG update interval comparison; WIVR fixed',cases=rows)
(RUN/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
