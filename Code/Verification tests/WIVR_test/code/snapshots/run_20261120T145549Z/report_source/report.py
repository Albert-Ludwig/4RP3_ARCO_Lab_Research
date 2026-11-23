"""Single English report: preserved FIFO baselines and one latest-GP ROS run."""
from pathlib import Path
import json, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat as nb
from IPython.display import Video

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results'/json.loads((ROOT/'results/latest.json').read_text())['run']
BASE=ROOT/'results/run_20261117T023229Z'
def read(p): return json.loads(p.read_text())
old=[read(BASE/f'update_{i}s/summary.json') for i in (1,5)]
new=read(RUN/'latest_gp/summary.json'); cases=old+[new]
labels=['FIFO 1 s GP','FIFO 5 s GP','Latest GP 1 s']
paths=[BASE/'update_1s',BASE/'update_5s',RUN/'latest_gp']
colors=['#a00045','#888888','#1769aa']
figdir=RUN/'figures';figdir.mkdir(exist_ok=True)
MEDIA=ROOT/'recordings'/RUN.name;MEDIA.mkdir(parents=True,exist_ok=True)

fields=[
 ('GP updates',lambda s:s['mission']['gp_updates']),
 ('GRG jobs completed by 300 s',lambda s:s['mission']['grg_completed_at_stop']),
 ('GRG jobs completed including drain',lambda s:s['worker']['completed_total']),
 ('Unprocessed accepted requests',lambda s:s['worker']['unprocessed']),
 ('Peak pending queue / state slot',lambda s:s['worker']['peak_queue']),
 ('Arrival waiting (s)',lambda s:f"{s['mission']['arrival_wait_s']:.3f}"),
 ('Completed navigation legs',lambda s:s['mission']['segments']),
 ('Distance (m)',lambda s:f"{s['distance_m']:.2f}"),
 ('GP update mean / P95 (ms)',lambda s:f"{1000*s['gp_update_s']['mean']:.2f} / {1000*s['gp_update_s']['p95']:.2f}"),
 ('GRG compute mean / P95 (s)',lambda s:f"{s['grg_compute_s']['mean']:.3f} / {s['grg_compute_s']['p95']:.3f}"),
 ('GRG compute total, including drain (s)',lambda s:f"{s['grg_compute_total_s']:.2f}"),
 ('Pending-state / queue wait P95 (s)',lambda s:f"{s['grg_queue_wait_s']['p95']:.3f}"),
 ('Newly adopted completed-plan age P95 (s)',lambda s:f"{s['consumed_plan_age_s']['p95']:.3f}" if s['consumed_plan_age_s'] else 'None'),
 ('Sampled aggregate CPU mean (%)',lambda s:f"{s['sampled_aggregate_cpu_mean_percent']:.1f}"),
 ('Sampled aggregate process RSS peak (MiB)',lambda s:f"{s['sampled_process_peak_mib']:.1f}"),
 ('Search-guard timeouts',lambda s:s['grg_timeouts']),
 ('Successful / missed measurements',lambda s:f"{s['mission']['measurements']} / {s['mission']['missed_ticks']}"),
 ('Home returned / error (m)',lambda s:f"{s['mission']['home_returned']} / {s['mission']['home_error_m']:.3f}"),
 ('Real-time factor',lambda s:f"{s['real_time_factor']:.5f}"),
 ('Final synthetic reconstruction RMSE (°C)',lambda s:f"{s['final_reconstruction_rmse_c']:.5f}"),
 ('History-Only RMSE (°C)',lambda s:f"{s['history_only_rmse_c']:.5f}"),
]
table='| Metric | '+' | '.join(labels)+' |\n| --- | --- | --- | --- |\n'
table+='\n'.join('| '+name+' | '+' | '.join(str(fn(s)) for s in cases)+' |' for name,fn in fields)

events=[[json.loads(l) for l in (p/'events.jsonl').read_text().splitlines() if l.strip()] for p in paths]
plans=[sorted([read(p) for p in folder.glob('plan_*.json')],key=lambda p:p['request']['id']) for folder in paths]
cp=np.array(read(RUN/'common/candidates.json'))
fig,axes=plt.subplots(1,3,figsize=(13,3.6),sharex=True,sharey=True)
for ax,folder,label,color in zip(axes,paths,labels,colors):
    t=np.load(folder/'trajectory.npy');ax.plot(t[:,1],t[:,2],color=color,lw=1.5)
    ax.scatter(cp[1:,0],cp[1:,1],s=15,c='#777777');ax.scatter(*cp[0],marker='*',s=110,c='#a00045')
    ax.set(title=label,xlabel='Field x (m)',xlim=(0,30),ylim=(0,20),aspect='equal');ax.grid(alpha=.2)
axes[0].set_ylabel('Field y (m)');fig.tight_layout();fig.savefig(figdir/'trajectories.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(13,3.6))
bars=axes[0].bar(labels,[s['mission']['arrival_wait_s'] for s in cases],color=colors);axes[0].bar_label(bars,labels=[f"{s['mission']['arrival_wait_s']:.3f}" for s in cases],padding=3);axes[0].set_ylabel('Arrival waiting (s)');axes[0].tick_params(axis='x',rotation=20)
for pp,label,color in zip(plans,labels,colors):
    axes[1].plot([p['completion_t'] for p in pp],[p['total_compute_s'] for p in pp],label=label,color=color)
axes[1].set(xlabel='Mission time (s)',ylabel='Per-search compute (s)');axes[1].legend(fontsize=8)
for folder,label,color in zip(paths,labels,colors):
    b=np.load(folder/'backlog.npy');axes[2].step(b[:,0],b[:,1],where='post',color=color,label=label)
axes[2].set(xlabel='Mission time (s)',ylabel='Accepted unfinished jobs*');axes[2].legend(fontsize=8)
fig.tight_layout();fig.savefig(figdir/'performance.png',dpi=160);plt.close(fig)

video=MEDIA/'robot_replay.mp4'
if not video.exists():
    from PIL import Image,ImageDraw,ImageFont
    import imageio_ffmpeg
    font=ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'),18)
    trajectory=np.load(paths[-1]/'trajectory.npy')
    writer=imageio_ffmpeg.write_frames(str(video),(800,608),fps=10,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',output_params=['-crf','23','-threads','2']);writer.send(None)
    for frame in range(751):
        ts=min(300,frame*.4);img=Image.new('RGB',(800,608),'white');draw=ImageDraw.Draw(img)
        draw.text((25,15),f'Latest-GP background WIVR | {ts:05.1f} / 300 s | 4x',font=font,fill='#222222')
        past=[e for e in events[-1] if e['event_t']<=ts]
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

wait_drop=100*(1-new['mission']['arrival_wait_s']/old[0]['mission']['arrival_wait_s'])
acceptance=new['scheduler_acceptance']
source_counts=new['arrival_route_sources']
config=('30 × 20 m flat arena; ten candidate waypoints plus Home; Bunker Mini, Gazebo and Nav2; '
 '300 s mission; 1 Hz continuous sensing and GP conditioning. Fixed real-data-fitted ARD RBF parameters, '
 '980 synthetic historical observations from the same Matérn SRF realization. WIVR uses area × posterior variance '
 'at the frozen GP sample time; 24 spatial cells × 151 times = 3,624 scoring targets. Predicted sensing is 1 Hz; '
 'GRG depth 2, two splits, 180 s emergency search guard; 270 s navigation budget and 30 s Home reserve. '
 'Reuse the exact baseline Nav2 graph, SRF seed, initial observations, kernel and 9,150-point reconstruction evaluation grid. '
 'Initial planning is outside the mission clock; all actual 1 Hz measurements, including at Home, enter the final GP.')
method=('The ROS executor assimilates each temperature observation and offers its coherent data-only GP snapshot '
 'plus the active goal endpoint and epoch to a single replaceable pending-state slot. The C++ worker finishes its '
 'current full GRG search, then takes the latest slot and immediately starts the next search; intermediate signals '
 'are coalesced without dropping measurements. A new GP update does not cancel the current search. '
 'Each search keeps its GP and adaptive WIVR weights frozen for numerical consistency. '
 'The active executable route retains its unexecuted tail to Home. On Nav2 arrival, a completed result can replace '
 'that tail only if its goal epoch and starting waypoint match; otherwise the existing tail supplies the next goal. '
 'Every next leg is checked with its return cost and reserve. Goals are never switched mid-leg. '
 'A goal change offers the latest GP with the new endpoint. Committing to Home clears the pending state and stops '
 'new exploration searches; sensing and GP updates continue until 300 s. An in-flight search may drain after stop. '
 'There is no hard 10 s age limit, no restart on every sample, and no intermediate partial-search route publication.')
conclusion=(f"Scheduler acceptance: **{acceptance}**. Arrival waiting changed from "
 f"{old[0]['mission']['arrival_wait_s']:.3f} s (FIFO 1 s) to {new['mission']['arrival_wait_s']:.3f} s "
 f"({wait_drop:.2f}% reduction). The old 5 s baseline waited {old[1]['mission']['arrival_wait_s']:.3f} s. "
 f"The latest-state slot peaked at {new['worker']['peak_queue']}; "
 f"{new['worker']['coalesced_state_signals']} pending state signals were deliberately replaced. "
 f"Arrival choices used {source_counts['fresh_completed']} completed matching results and "
 f"{source_counts['retained_tail']} retained tails. "
 f"GP snapshot age at search start P95 was {new['grg_snapshot_age_at_start_s']['p95']:.3f} s. "
 f"Retained-tail fallback coverage: {source_counts['retained_tail']} observed handoffs. Zero means this mission did not exercise the fallback failure path. "
 'This removes obsolete-request accumulation and avoids waiting solely for a new completed search. '
 'It does not make an individual WIVR/GRG search inherently cheaper. Different visited positions, start times, '
 'remaining budgets and the Home phase change the completed-search set; per-search means and totals are descriptive, '
 'not a matched algorithm speedup. CPU is sampled aggregate process utilization (100% = one core); RSS is the sampled '
 'process sum, not a hardware maximum. Compute overlaps travel and must not be added to mission time. '
 'Freshness at search start does not guarantee a result is under 10 s old at adoption. A retained tail may originate '
 'from an older GP. No GUI FPS was measured. One deterministic synthetic field with ideal localization does not '
 'establish real-robot behavior or policy superiority; RMSE is retrospective reconstruction, not forecasting. '
 'Runtime log notes: four startup map-transform waits, two default-checker warnings on the first goal, '
 'and two helper nodes requiring SIGTERM during shutdown. No missed-control-loop warning or in-mission navigation failure was recorded; '
 'these startup/shutdown messages are preserved in scene.log.')
checks='\n'.join(f'- `{k}`: **{v}**' for k,v in new['checks'].items())
audits=new['score_audits'];maxerr=max(a['absolute_error'] for a in audits)
validation=(f"All 299 expected observations must arrive and be assimilated exactly, all started jobs must have saved "
 f"results, at most one pending state is permitted, no mid-leg goal preemption is permitted, Home error must be ≤0.3 m, "
 f"and the simulation must cover 300–300.5 s. Independent Python audits of {len(audits)} saved WIVR routes use "
 f"atol=rtol=1e-8; maximum score error {maxerr:.3g}. Final GP inputs and labels are checked against all logged "
 'measurements; SRF labels are reproduced independently. A 1 s full-GRG completion deadline is not the acceptance '
 'criterion for this completion-driven scheduler.\n\n'+checks)
files='''| File / folder | Purpose |
| --- | --- |
| `code/src/nodes.cpp` | 1 Hz GP updates, latest-state slot, background worker, retained-route cursor and arrival-only handoff |
| `code/src/core.hpp` | Fixed-parameter exact RBF posterior, route sensing events, WIVR and depth-2 GRG; unchanged |
| `code/protocol.json` | Recorded scheduling, geometry, GP, scoring and time-budget configuration |
| `code/prepare.py` | Shared synthetic history, candidates, map, SRF and evaluation queries |
| `code/scene.launch.py`, `navigation_profile.*` | Existing Bunker Mini / Nav2 scene and speed settings |
| `code/srf_monitor.py` | Python SRF temperature source and process resource monitoring |
| `code/run.py`, `run.sh`, `Run-ClosedLoop.ps1` | Build, one 300 s run, MCAP recording and worker drain |
| `code/evaluate.py`, `evaluate_case.py` | Data, independent score, mission and scheduling audits |
| `code/report.py` | This README, single notebook, figures and embedded replay |
| `code/snapshots/<run>/` | Frozen source and report sources for each experiment |
| `results/<run>/common/` | Exact inputs and common evaluation grid |
| `results/<run>/latest_gp/` | Events, complete plans, final GP, metrics and trajectory |
| `recordings/<run>/` | MP4 and `latest_gp/rosbag/` MCAP recording |
'''
rel=RUN.relative_to(ROOT).as_posix();vrel=video.relative_to(ROOT).as_posix()
md=nb.v4.new_markdown_cell
video_cell=nb.v4.new_code_cell("from IPython.display import Video, display\n"+f"display(Video(filename={vrel!r}, embed=True, width=800))",execution_count=1,
 outputs=[nb.v4.new_output('display_data',data={'text/html':Video(filename=str(video),embed=True,width=800)._repr_html_(),'text/plain':'<IPython.core.display.Video object>'})])
notebook=nb.v4.new_notebook(cells=[
 md('# Latest-GP background GRG: 300 s ROS validation\n\n'+RUN.name+'\n\n'+config),
 md('## Implementation and control flow\n\n'+method),
 md('## Results against preserved baselines\n\n'+table+'\n\nPrimary comparator: FIFO 1 s GP, which keeps the same GP update frequency. FIFO 5 s is contextual. GRG job counts mean periodic accepted requests in the old runs and actually started searches in the new run; replaced state signals are not accepted jobs.'),
 md('## Performance and actual paths\n\n![]('+rel+'/figures/performance.png)\n\n*The latest-GP curve counts started unfinished searches, excluding its single replaceable state slot. Old FIFO curves include accepted queued requests.\n\n![]('+rel+'/figures/trajectories.png)'),
 md('## Recorded replay\n\n4× MCAP pose replay, embedded directly below. Raw rosbag is under `'+vrel.rsplit('/',1)[0]+'/latest_gp/rosbag/`.'),
 video_cell,md('## Validation\n\n'+validation),md('## Conclusion\n\n'+conclusion)
],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
nb.write(notebook,ROOT/'test_report.ipynb')
readme='# ROS closed loop: latest-GP background GRG\n\n## Purpose\n\nReduce arrival waiting and obsolete planning backlog while preserving an executable route. Run one 300 s WIVR mission; compare scheduling performance against the saved 1 s / 5 s FIFO runs. No MI/EMSE/WIVR ranking is performed.\n\n## Configuration\n\n'+config+'\n\n## Architecture\n\n'+method+'\n\n## Results\n\n'+table+'\n\n'+conclusion+'\n\n## Files\n\n'+files+'\nLayout: `code/` contains all source and snapshots; `results/` contains runtime inputs, data and metrics; `recordings/` contains videos and rosbags. The single notebook contains results, figures, validation and an embedded video.\n\nDependencies: existing `bunker-jazzy` Docker container, Jazzy/Nav2/Gazebo overlay, trained real-data RBF parameters, SRF_test temperature source, Continuous_sensing_test and Scoring_policy_test Python references. The WELL-E logic document, trained kernel and previous run artifacts are preserved.\n\n```powershell\npowershell -ExecutionPolicy Bypass -File .\\code\\Run-ClosedLoop.ps1\n```\n\nLatest results: `'+rel+'`. Historical baselines remain under `results/'+BASE.name+'`. [Notebook](test_report.ipynb); [new 4× replay]('+vrel+').\n'
(ROOT/'README.md').write_text(readme,encoding='utf8')
report_source=ROOT/'code/snapshots'/RUN.name/'report_source';report_source.mkdir(parents=True,exist_ok=True)
for name in ('evaluate.py','evaluate_case.py','report.py'):shutil.copy2(ROOT/'code'/name,report_source/name)
(RUN/'scheduler_comparison.json').write_text(json.dumps(dict(labels=labels,cases=cases,arrival_wait_reduction_percent=wait_drop,baseline_run=BASE.name),indent=2)+'\n')
print(json.dumps(dict(run=RUN.name,wait_s=new['mission']['arrival_wait_s'],wait_reduction_percent=wait_drop,acceptance=acceptance,video=str(video),notebook_bytes=(ROOT/'test_report.ipynb').stat().st_size),indent=2))
