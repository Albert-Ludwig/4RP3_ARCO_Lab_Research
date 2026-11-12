"""Run the frozen experiment using the existing Docker GP environment."""
import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[2]
HERE=Path(__file__).resolve().parent
DATA=ROOT.parent/'Real_Data_Integration'/'result'/'gp_samples.csv'
MODEL=REPO/'Code/GP3D_baseline_test/10.16_testing_code/3D_time&spatial temp field rebuild & prediction/gp_model.py'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container',default='bunker-jazzy')
    parser.add_argument('--full-fit',action='store_true',help='Fit all observations and preserve earlier held-out runs')
    args=parser.parse_args()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    name=('full_fit_' if args.full_fit else 'run_')+stamp
    output_root=ROOT.parent/'Trained_Real_Data_Kernel' if args.full_fit else ROOT
    run=output_root/'result'/name; run.mkdir(parents=True,exist_ok=False)
    protocol=HERE/('full_fit_protocol.json' if args.full_fit else 'protocol.json')
    worker=HERE/('full_fit_worker.py' if args.full_fit else 'worker.py')
    sources=[(DATA,'gp_samples.csv'),(MODEL,'gp_model.py'),(protocol,'protocol.json'),(worker,'worker.py' if not args.full_fit else 'full_fit_worker.py')]
    if args.full_fit:
        sources += [(HERE/'worker.py','worker.py'),(DATA.parent/'summary.json','alignment_summary.json')]
    original={str(p):sha(p) for p,_ in sources}
    alignment_manifest=json.loads((DATA.parent/'manifest.json').read_text(encoding='utf-8'))
    if sha(DATA)!=alignment_manifest['outputs']['gp_samples.csv']:
        raise RuntimeError('Prepared data differ from the alignment manifest.')
    job='/tmp/minimal_gp_'+stamp
    docker=['docker','--context','desktop-linux']
    subprocess.run(docker+['exec','-u','ubuntu',args.container,'mkdir','-p',job],check=True)
    for p,target in sources:
        subprocess.run(docker+['cp',str(p),args.container+':'+job+'/'+target],check=True)
    command=docker+['exec','-u','ubuntu','-e','PYTHONDONTWRITEBYTECODE=1',args.container,
                    '/home/ubuntu/.venvs/policy-compare/bin/python','-B',job+'/'+worker.name]
    print('Running',name,flush=True)
    with (run/'execution.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        for line in process.stdout:
            print(line,end='',flush=True); log.write(line); log.flush()
        code=process.wait()
    if code:
        raise RuntimeError(f'Experiment failed with exit {code}. See {run / "execution.log"}')
    subprocess.run(docker+['cp',args.container+':'+job+'/result/.',str(run)],check=True)
    for path,expected in original.items():
        if sha(Path(path))!=expected:
            raise RuntimeError(f'Source changed during run: {path}')
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['input_sha256']==sha(DATA) and manifest['gp_model_sha256']==sha(MODEL)
    for path,expected in manifest['outputs'].items():
        assert sha(run/path)==expected,path
    shutil.copyfile(DATA,run/'input_samples.csv')
    provenance=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                    sources={Path(p).relative_to(REPO).as_posix():h for p,h in original.items()},
                    source_files_unchanged=True,container_job_directory=job,command=command,
                    alignment_manifest_sha256=sha(DATA.parent/'manifest.json'),
                    accepted_clock_assumption='Synchronized HOBO/GPS clocks, zero additional offset; accepted by user',
                    reused_model_source=MODEL.relative_to(REPO).as_posix())
    (run/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    (output_root/'result'/'latest.json').write_text(json.dumps({'run':name},indent=2)+'\n',encoding='utf-8')
    print('Saved',run,flush=True)


if __name__=='__main__':
    main()
