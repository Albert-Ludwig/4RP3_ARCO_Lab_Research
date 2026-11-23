"""Generate the English README, notebook and replay from a completed MI run."""
from pathlib import Path
import hashlib
import json
import shutil

from IPython.display import Video
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat as nb
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results'/json.loads((ROOT/'results/latest.json').read_text())['run']
CASE=RUN/'latest_gp'
MEDIA=ROOT/'recordings'/RUN.name
FIGURES=RUN/'figures'
FIGURES.mkdir(exist_ok=True)

def read(path):
    return json.loads(path.read_text(encoding='utf8'))

s=read(CASE/'summary.json')
m=s['mission']
w=s['worker']
reference=ROOT.parent/read(RUN/'fixture.json')['reference']
events=[json.loads(line) for line in (CASE/'events.jsonl').read_text().splitlines()]
cp=np.array(read(CASE/'candidates.json'))
trajectory=np.load(CASE/'trajectory.npy')
plans=sorted((read(p) for p in CASE.glob('plan_*.json')),key=lambda p:p['request']['id'])
assert plans and all(p['policy']=='mi' for p in plans)
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

checks=dict()
for name in ('initial.json','candidates.json','field.json','world.sdf','map.yaml','map.pgm'):
    checks[name]=digest(CASE/name)==digest(reference/'common'/name)
checks['evaluation_grid.npz']=digest(RUN/'common/evaluation_grid.npz')==digest(reference/'common/evaluation_grid.npz')
checks['graph.json']=read(CASE/'graph.json')==read(reference/'latest_gp/graph.json')
checks['navigation_profile.json']=digest(CASE/'navigation_profile.json')==digest(reference/'common/navigation_profile.json')
def nav_settings(path):
    return [line for line in path.read_text().splitlines() if 'yaml_filename:' not in line]
checks['nav2_settings']=nav_settings(CASE/'nav2_params.yaml')==nav_settings(reference/'common/nav2_params.yaml')
assert all(checks.values()),checks
assert all(digest(ROOT/'code/snapshots'/RUN.name/'source_snapshot'/name)==expected for name,expected in read(RUN/'manifest.json')['sources'].items())

fig,axes=plt.subplots(1,2,figsize=(11,4))
ax=axes[0]
ax.plot(trajectory[:,1],trajectory[:,2],color='#1769aa',lw=1.4)
ax.scatter(cp[1:,0],cp[1:,1],s=18,c='gray',label='Candidates')
ax.scatter(*cp[0],marker='*',s=110,c='#a00045',label='Home')
ax.set(xlim=(0,30),ylim=(0,20),aspect='equal',xlabel='Field x (m)',ylabel='Field y (m)',title='Recorded Gazebo trajectory')
ax.legend(loc='lower right')
ax=axes[1]
ax.plot([p['completion_t'] for p in plans],[p['total_compute_s'] for p in plans],color='#1769aa',lw=1)
ax.set(xlabel='Mission time (s)',ylabel='GRG compute per search (s)',title='Background MI search time')
for ax in axes:
    ax.grid(alpha=.25)
fig.tight_layout()
fig.savefig(FIGURES/'trajectory_and_compute.png',dpi=160)
plt.close(fig)

video=MEDIA/'robot_replay.mp4'
if not video.exists():
    from PIL import Image,ImageDraw,ImageFont
    import imageio_ffmpeg
    font=ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'),18)
    trajectory=np.load(CASE/'trajectory.npy')
    writer=imageio_ffmpeg.write_frames(str(video),(800,608),fps=10,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',output_params=['-crf','23','-threads','2']);writer.send(None)
    for frame in range(751):
        ts=min(300,frame*.4);img=Image.new('RGB',(800,608),'white');draw=ImageDraw.Draw(img)
        draw.text((25,15),f'Latest-GP background MI | {ts:05.1f} / 300 s | 4x',font=font,fill='#222222')
        past=[e for e in events if e['event_t']<=ts]
        ng=sum(e['event']=='gp_update' for e in past);nd=sum(e['event']=='grg_complete' for e in past)
        goal=next((e['target'] for e in reversed(past) if e['event']=='nav_goal'),None)
        draw.text((25,50),f'GP updates {ng} | completed GRG {nd} | active target {goal}',font=font,fill='#333333')
        def at(x,y): return 75+x*21,110+(20-y)*21
        for x in range(0,31,5):draw.line([at(x,0),at(x,20)],fill='#eeeeee')
        for y in range(0,21,5):draw.line([at(0,y),at(30,y)],fill='#eeeeee')
        draw.rectangle([at(0,20),at(30,0)],outline='#888888')
        for i,(x,y) in enumerate(cp):
            px,py=at(x,y);draw.ellipse((px-4,py-4,px+4,py+4),fill='#a00045' if i==0 else '#888888')
            draw.text((px+5,py-5),'H' if i==0 else str(i),font=font,fill='#444444')
        end=max(1,int(np.searchsorted(trajectory[:,0],ts,side='right')));trace=trajectory[:end:3]
        if len(trace)>1:draw.line([at(x,y) for _,x,y in trace],fill='#1769aa',width=3)
        row=trajectory[min(len(trajectory)-1,end-1)];px,py=at(row[1],row[2]);draw.ellipse((px-7,py-7,px+7,py+7),fill='#1769aa')
        draw.text((25,558),'Recorded MCAP pose replay; not a Gazebo GUI recording.',font=font,fill='#444444')
        writer.send(np.asarray(img))
    writer.close()


config="""30 × 20 m flat arena, 10 candidate waypoints plus Home, Bunker Mini in Gazebo with Nav2, and a 300 s mission. The test uses the map, directed Nav2 route graph, Matérn SRF realization, 980 synthetic history samples, fixed real-data-fitted RBF hyperparameters, and 9,150 evaluation queries from the saved latest-GP WIVR run. Sensing and GP conditioning run at 1 Hz. GRG uses depth 2, two time splits, a 270 s navigation budget and a 30 s return reserve."""

method="""For every candidate route, the existing MI objective scores the joint expected information in its 1 Hz sampling events given the current GP posterior. It uses the log determinant of the predictive measurement covariance relative to independent sensor noise and accounts for correlations between measurements. The worker finishes each search using a fixed GP snapshot and then takes the newest pending snapshot. Nav2 stays on its current goal until arrival; the remaining route supplies a fallback if no matching result is ready. All received observations enter the GP. The robot continues sampling until the mission window ends."""

audit=s['score_audits']
home=next((e['event_t'] for e in events if e['event']=='home_returned'),None)
early_home=next((e for e in events if e['event']=='arrival_selection' and e['next']==0
                 and len(e['remaining_route'])>2 and e['remaining_route'][1]==0),None)
home_note=''
if early_home and home is not None:
    planned=' → '.join('Home' if n==0 else str(n) for n in early_home['remaining_route'])
    home_note=(f"At {early_home['event_t']:.2f} s the arrival handoff selected {planned}. "
               f"The shared mission runner treats the next Home visit as terminal. "
               f"The robot reached Home at {home:.2f} s and sampled there for the rest of the 300 s window. "
               "That handoff shortened the executed sensing route and affects how this run's reconstruction error can be compared.")
metrics=[
    ('Mission duration (s)',f"{m['duration_s']:.3f}"),
    ('Measurements / missed ticks',f"{m['measurements']} / {m['missed_ticks']}"),
    ('GP updates',m['gp_updates']),
    ('Initial GRG compute (s)',f"{s['initial_plan_s']:.3f}"),
    ('Initial route (node IDs)',s['initial_route']),
    ('GRG requests started / completed within mission',f"{m['grg_requested']} / {m['grg_completed_at_stop']}"),
    ('GRG compute mean / P95 / max (s)', ' / '.join(f"{s['grg_compute_s'][k]:.3f}" for k in ('mean','p95','max'))),
    ('Cumulative GRG compute (s)',f"{s['grg_compute_total_s']:.2f}"),
    ('Peak pending slot / unfinished requests',f"{w['peak_queue']} / {w['unprocessed']}"),
    ('Search guard timeouts',s['grg_timeouts']),
    ('Arrival waiting (s)',f"{m['arrival_wait_s']:.3f}"),
    ('Handoffs with new plan / retained tail',f"{s['arrival_route_sources']['fresh_completed']} / {s['arrival_route_sources']['retained_tail']}"),
    ('Completed legs / distance (m)',f"{m['segments']} / {s['distance_m']:.2f}"),
    ('Returned Home / error (m)',f"{m['home_returned']} / {m['home_error_m']:.3f}"),
    ('Time of Home arrival (s)',f"{home:.2f}" if home is not None else 'Unavailable'),
    ('Mean process CPU (100% = one core)',f"{s['sampled_aggregate_cpu_mean_percent']:.1f}%"),
    ('Peak aggregate process RSS (MiB)',f"{s['sampled_process_peak_mib']:.1f}"),
    ('Real-time factor',f"{s['real_time_factor']:.5f}"),
    ('Final synthetic reconstruction RMSE (°C)',f"{s['final_reconstruction_rmse_c']:.5f}"),
    ('History-only reconstruction RMSE (°C)',f"{s['history_only_rmse_c']:.5f}"),
]
table='| Metric | MI |\n| --- | --- |\n'+'\n'.join(f'| {label} | {value} |' for label,value in metrics)
result=(f"Mission acceptance **{'PASS' if s['scheduler_acceptance'] else 'FAIL'}**. "
        f"The reference Python objective reproduced the initial score and {len(audit)-1} saved search scores "
        f"within the 1e-8 absolute and relative tolerances (maximum absolute error {max(a['absolute_error'] for a in audit):.3g}). "
        "The mission checks cover observation retention, arrival-only handoffs, return feasibility, and the 300 s window. "
        "The reconstruction RMSE summarizes the final GP posterior on the shared synthetic evaluation grid for this single run.")
trace='\n'.join(f'- {k.replace("_"," ")}: {v}' for k,v in s['checks'].items())
files="""| Location | Contents |
| --- | --- |
| `code/src/` | C++ GP, MI scoring, GRG search, latest-GP background scheduling and ROS/Nav2 interfaces |
| `code/protocol.json` | Mission and planner configuration |
| `code/run.sh`, `code/run.py` | ROS build and run, fixed input loading and MCAP recording |
| `code/scene.launch.py`, `code/srf_monitor.py` | Gazebo/Nav2 scene, SRF measurements and resource sampling |
| `code/evaluate_case.py`, `code/report.py` | MI score audits, reconstruction evaluation, report and replay |
| `results/<run>/` | Inputs, hashes, event logs, final GP state and metrics |
| `recordings/<run>/` | Raw rosbag and MP4 replay |
""".replace("BACKTICK",chr(96))
rel=RUN.relative_to(ROOT).as_posix()
vrel=video.relative_to(ROOT).as_posix()
root_md=(f"# MI closed-loop verification\n\nValidate the existing MI objective in the same latest-GP ROS 2 simulation used for WIVR.\n\n"
         f"## Configuration\n\n{config}\n\n{method}\n\n## Results\n\n{table}\n\n{result}\n\n{home_note}\n\n"
         f"[Notebook with plots and embedded replay](test_report.ipynb) · [Run summary]({rel}/latest_gp/summary.json)\n\n"
         f"## Files and dependencies\n\n{files}\n"
         "The runner uses the existing `bunker-jazzy` Docker container with the ROS 2 Jazzy/Nav2/Bunker Mini overlay and the SRF verification Python environment. "
         "Its fixture is read from the saved [WIVR run](../WIVR_test/results/run_20261120T145549Z/). "
         "The Python evaluator uses the repository's GP, MI and SRF reference modules.\n\n"
         "With the container running, invoke `docker exec bunker-jazzy bash '/workspace/Code/Verification tests/MI_Test/code/run.sh'`. "
         "Run `code/report.py` with the reporting Python environment to refresh the notebook.\n")
md=nb.v4.new_markdown_cell
cell=nb.v4.new_code_cell(
    f"from IPython.display import Video, display\n"
    f"display(Video(filename={vrel!r}, embed=True, width=800))",
    execution_count=1,outputs=[nb.v4.new_output('display_data',data={
        'text/html':Video(filename=str(video),embed=True,width=800)._repr_html_()})])
note=nb.v4.new_notebook(cells=[
    md(f'# MI closed-loop verification\n\n{RUN.name}\n\n{config}\n\n{method}'),
    md(f'## Results\n\n{table}\n\n{result}\n\n{home_note}'),
    md(f'## Trajectory and compute\n\n![]({rel}/figures/trajectory_and_compute.png)'),
    md('## MCAP pose replay\n\nPlayback at 4× speed.'),cell,
    md(f'## Validation\n\n{trace}\n\nAll shared numeric fixtures match the saved WIVR run.'),
])
nb.validate(note)
nb.write(note,ROOT/'test_report.ipynb')
(ROOT/'README.md').write_text(root_md,encoding='utf8')
(RUN/'artifact_validation.json').write_text(json.dumps(dict(shared_fixture_checks=checks,mission_acceptance=s['scheduler_acceptance'],policy='mi'),indent=2)+'\n')
report_snapshot=ROOT/'code/snapshots'/RUN.name/'report_source'
report_snapshot.mkdir(exist_ok=True)
shutil.copy2(__file__,report_snapshot/'report.py')
print(result)
