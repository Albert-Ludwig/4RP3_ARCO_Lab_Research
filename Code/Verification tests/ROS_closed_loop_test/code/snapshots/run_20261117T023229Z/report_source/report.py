"""One English notebook/README comparing GP+GRG update intervals, fixed WIVR."""
from pathlib import Path
import json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat as nb
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'results'/json.loads((ROOT/'results/latest.json').read_text())['run']
CODE=ROOT/'code';MEDIA=ROOT/'recordings'/RUN.name;MEDIA.mkdir(parents=True,exist_ok=True)
cases=json.loads((RUN/'comparison.json').read_text())['cases'];figdir=RUN/'figures';figdir.mkdir(exist_ok=True)
def table():
    headers=['Metric','GP + GRG every 1 s','GP + GRG every 5 s']
    fields=[('Successful measurements',lambda s:s['mission']['measurements']),('GP updates (including terminal partial batch)',lambda s:s['mission']['gp_updates']),
      ('Pre-mission initial GRG (s)',lambda s:f"{s['initial_plan_s']:.3f}"),('GRG requests',lambda s:s['mission']['grg_requested']),('GRG completed before stop',lambda s:s['mission']['grg_completed_at_stop']),('GRG completed including in-flight drain',lambda s:s['worker']['completed_total']),
      ('Unprocessed requests',lambda s:s['worker']['unprocessed']),('GRG compute mean / P95 (s)',lambda s:f"{s['grg_compute_s']['mean']:.3f} / {s['grg_compute_s']['p95']:.3f}"),
      ('GP update mean / P95 (ms)',lambda s:f"{1000*s['gp_update_s']['mean']:.2f} / {1000*s['gp_update_s']['p95']:.2f}"),('Total GP update compute (s)',lambda s:f"{s['gp_update_total_s']:.2f}"),('Total completed GRG compute (s)',lambda s:f"{s['grg_compute_total_s']:.2f}"),('GRG queue wait P95 (s)',lambda s:f"{s['grg_queue_wait_s']['p95']:.2f}"),
      ('GRG end-to-end latency P95 (s)',lambda s:f"{s['grg_latency_s']['p95']:.2f}"),('Requests meeting update deadline',lambda s:s['grg_deadlines_met']),('Search guard timeouts',lambda s:s['grg_timeouts']),
      ('Peak queued requests',lambda s:s['worker']['peak_queue']),('Sampled process peak RSS (MiB)',lambda s:f"{s['sampled_process_peak_mib']:.0f}"),('Sampled aggregate CPU mean / peak (%)',lambda s:f"{s['sampled_aggregate_cpu_mean_percent']:.0f} / {s['sampled_aggregate_cpu_peak_percent']:.0f}"),('Consumed plan age P95 (s)',lambda s:f"{s['consumed_plan_age_s']['p95']:.2f}" if s['consumed_plan_age_s'] else 'No plan consumed'),('Arrival waiting (s)',lambda s:f"{s['mission']['arrival_wait_s']:.2f}"),
      ('Completed navigation legs',lambda s:s['mission']['segments']),('Distance (m)',lambda s:f"{s['distance_m']:.1f}"),('Returned Home',lambda s:s['mission']['home_returned']),('Missed sampling ticks',lambda s:s['mission']['missed_ticks']),
      ('Real-time factor',lambda s:f"{s['real_time_factor']:.5f}"),('Final synthetic RMSE (°C)',lambda s:f"{s['final_reconstruction_rmse_c']:.5f}"),('History Only RMSE (°C)',lambda s:f"{s['history_only_rmse_c']:.5f}"),
      ('All GP+GRG requests met interval deadline',lambda s:s['update_interval_feasible'])]
    return '| '+' | '.join(headers)+' |\n| --- | --- | --- |\n'+'\n'.join('| '+str(label)+' | '+' | '.join(str(fn(s)) for s in cases)+' |' for label,fn in fields)
metrics=table();colors=['#a00045','#1769aa']
if not all((figdir/name).exists() for name in ('performance.png','trajectories.png')):
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for s,color in zip(cases,colors):
        dt=s['mission']['update_interval_s'];folder=RUN/f'update_{dt}s';b=np.load(folder/'backlog.npy')
        axes[0].step(b[:,0],b[:,1],where='post',color=color,label=f'{dt} s updates')
        p=sorted([json.loads(f.read_text()) for f in folder.glob('plan_*.json')],key=lambda j:j['request']['id'])
        axes[1].plot([j['completion_t'] for j in p],[j['total_compute_s'] for j in p],color=color,label=f'{dt} s updates')
    axes[0].set(xlabel='Mission time (s)',ylabel='Queued + in-flight GRG requests');axes[1].set(xlabel='Mission time (s)',ylabel='Per-request compute (s)');axes[1].axhline(1,c='#999',ls='--');axes[1].axhline(5,c='#999',ls=':')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(figdir/'performance.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),sharex=True,sharey=True);cp=np.array(json.loads((RUN/'common/candidates.json').read_text()))
    for ax,s,color in zip(axes,cases,colors):
        dt=s['mission']['update_interval_s'];t=np.load(RUN/f'update_{dt}s/trajectory.npy');ax.plot(t[:,1],t[:,2],color=color);ax.scatter(cp[1:,0],cp[1:,1],c='#999',s=14);ax.scatter(*cp[0],marker='*',c='#a00045',s=90);ax.set(title=f'GP + GRG every {dt} s',xlim=(0,30),ylim=(0,20),xlabel='Field x (m)',aspect='equal');ax.grid(alpha=.2)
    axes[0].set_ylabel('Field y (m)');fig.tight_layout();fig.savefig(figdir/'trajectories.png',dpi=160);plt.close(fig)
cp=np.array(json.loads((RUN/'common/candidates.json').read_text()))
def make_replay():
    """Side-by-side 4x MCAP pose replay; not a Gazebo GUI recording."""
    from PIL import Image,ImageDraw,ImageFont
    import imageio_ffmpeg
    output=MEDIA/'robot_replay.mp4'
    if output.exists():return
    font=ImageFont.truetype(str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'),18)
    data=[]
    for item in cases:
        dt=item['mission']['update_interval_s'];folder=RUN/f'update_{dt}s'
        events=[json.loads(l) for l in (folder/'events.jsonl').read_text().splitlines() if l.strip()]
        data.append((dt,np.load(folder/'trajectory.npy'),np.array([e['event_t'] for e in events if e['event']=='gp_update']),np.array([e['event_t'] for e in events if e['event']=='gp_update' and not e['terminal']]),np.array([e['event_t'] for e in events if e['event']=='grg_complete'])))
    writer=imageio_ffmpeg.write_frames(str(output),(1120,480),fps=10,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',output_params=['-crf','23','-threads','2'])
    writer.send(None)
    for frame in range(751):
        time_s=min(300,frame*.4);img=Image.new('RGB',(1120,480),'white');draw=ImageDraw.Draw(img)
        draw.text((25,10),f'Actual Gazebo pose replay | mission {time_s:05.1f} / 300 s | 4x speed',font=font,fill='#222222')
        for index,(dt,path,gp_times,request_times,done_times) in enumerate(data):
            left=35+index*560;top=100;sx=16;sy=16
            def at(x,y):return (left+x*sx,top+(20-y)*sy)
            draw.text((left,45),f'GP + GRG every {dt} s (WIVR fixed)',font=font,fill='#222222')
            ng=int(np.searchsorted(gp_times,time_s,side='right'));nr=int(np.searchsorted(request_times,time_s,side='right'));nd=int(np.searchsorted(done_times,time_s,side='right'))
            draw.text((left,72),f'GP {ng} | GRG {nd}/{nr} | pending {nr-nd}',font=font,fill='#333333')
            for x in range(0,31,5):draw.line([at(x,0),at(x,20)],fill='#eeeeee')
            for y in range(0,21,5):draw.line([at(0,y),at(30,y)],fill='#eeeeee')
            draw.rectangle([at(0,20),at(30,0)],outline='#888888',width=1)
            for x,y in cp[1:]:
                px,py=at(x,y);draw.ellipse((px-3,py-3,px+3,py+3),fill='#aaaaaa')
            h=at(*cp[0]);draw.rectangle((h[0]-4,h[1]-4,h[0]+4,h[1]+4),fill='#a00045')
            end=int(np.searchsorted(path[:,0],time_s,side='right'));trace=path[:max(1,end):3]
            color=colors[index]
            if len(trace)>1:draw.line([at(x,y) for _,x,y in trace],fill=color,width=2)
            row=path[max(0,min(len(path)-1,end-1))];px,py=at(row[1],row[2]);draw.ellipse((px-6,py-6,px+6,py+6),fill=color)
        draw.text((25,447),'Blue/maroon dot: robot | grey: candidates | square: Home | positions decoded from recorded MCAP',font=font,fill='#444444')
        writer.send(np.asarray(img))
    writer.close()
make_replay()

cfg='30 × 20 m flat map, ten candidate waypoints plus Home, Bunker Mini/Gazebo/Nav2, 300 s missions, 1 Hz continuous temperature sampling in BOTH cases. Fixed real-fitted RBF parameters, 980 same-SRF synthetic historical observations; independent Matérn SRF truth. Fixed WIVR scoring method, depth 2/two splits; 24 spatial cells × 151 target times = 3,624 scoring targets. Predicted sensing remains 1 Hz. The only intended configuration change is the GP + GRG update interval: 1 s versus 5 s.'
method='The main C++ node buffers every received temperature record. On each configured update it assimilates all buffered observations, freezes a data-only GP snapshot, and enqueues one complete GRG request. The worker rebuilds that snapshot, evaluates WIVR with area × sample-time posterior variance, and searches from the active goal endpoint at its estimated arrival time. The current Nav2 goal is never replaced mid-leg. At arrival, only a completed result with the matching goal epoch can supply the next goal. A delayed GRG result can use an older GP version; latency/version lag and waiting are recorded. FIFO requests are neither coalesced nor silently dropped. Initial planning is before the 300 s mission clock; the Home reserve is 30 s. Sensing continues until 300 s, including any terminal Home wait.'
terminal='At stop, any residual observations are committed to the final GP without requesting post-mission navigation planning. In the 5 s case this normally means four residual records after the 295 s periodic update. This terminal partial-batch operation is separate from the periodic GRG deadline test. The worker finishes only its in-flight calculation; queued requests remain saved and are not counted as executed.'
conclusion='Five-second updates reduce request pressure, but not the intrinsic cost of a matched GRG search. Late-mission searches have fewer feasible branches as the remaining budget shrinks; the two per-request means summarize different completed request sets and are not an algorithm speedup factor. Computation overlaps navigation and must not be added to mission duration. The results distinguish functional success (sensing, exact GP assimilation, unchanged active goals and Home return) from timing acceptance. A complete search that exceeds its configured update interval is late even without hitting the 180 s guard. This is an implementation performance test of GP/GRG scheduling with one fixed scoring method, not a comparison of MI/EMSE/WIVR or proof about their reconstruction ranking.'
for s in cases:
    dt=s['mission']['update_interval_s'];conclusion+=f" GP + GRG every {dt} s: deadline acceptance = {s['update_interval_feasible']}; unfinished requests = {s['worker']['unprocessed']}."
rel=RUN.relative_to(ROOT).as_posix();md=lambda t:nb.v4.new_markdown_cell(t)
from IPython.display import Video
video_path=MEDIA/'robot_replay.mp4'
video_relative=video_path.relative_to(ROOT).as_posix()
video_source=("from pathlib import Path\nfrom IPython.display import Video, display\n"
              +"video = Path("+repr(video_relative)+")\n"
              +"display(Video(filename=str(video), embed=True, width=1000))")
video_cell=nb.v4.new_code_cell(source=video_source,execution_count=1,
    outputs=[nb.v4.new_output('display_data',data={
        'text/html':Video(filename=str(video_path),embed=True,width=1000)._repr_html_(),
        'text/plain':'<IPython.core.display.Video object>'})])
notebook=nb.v4.new_notebook(cells=[
    md('# GP–GRG update interval performance\n\n'+RUN.name+'\n\n'+cfg),
    md('## Implementation\n\n'+method+'\n\n'+terminal),
    md('## Results\n\n'+metrics),
    md('## Compute and backlog\n\n![]('+rel+'/figures/performance.png)'),
    md('## Actual Gazebo trajectories\n\n![]('+rel+'/figures/trajectories.png)'),
    md('## Recorded replay\n\nEmbedded side-by-side replay at 4× speed. The video output is stored inside this notebook and can be played directly. It is generated from recorded MCAP poses, not a Gazebo GUI capture.'),
    video_cell,
    md('## Checks and conclusion\n\n'+conclusion+'\n\nChecks use raw MCAP poses, all successful temperature records, exact final-GP input/label equality, reproducible SRF measurements and independent first/middle/last WIVR score audits for each case (atol=rtol=1e-8). RMSE uses the final posterior on the shared 9,150-point 0–300 s synthetic grid (2 m cell centres, 5 s time slices; these do not restrict continuous SRF queries); it is descriptive here. One paired run, ideal localization and a flat map do not establish real-world timing or GUI frame rate.')
],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
nb.write(notebook,ROOT/'test_report.ipynb')
files='| File | Purpose |\n| --- | --- |\n| code/protocol.json | Shared configuration; 1 s / 5 s case overrides saved per run |\n| code/src/core.hpp | Exact GP, WIVR, continuous sensing events and depth-2 GRG |\n| code/src/nodes.cpp | Configured GP batching, FIFO GRG worker, goal-epoch checks and arrival-only goal switching |\n| code/prepare.py | Common history, SRF, map, candidates and evaluation grid |\n| code/scene.launch.py, navigation_profile.* | Existing Bunker/Nav2 launch and speed limits |\n| code/srf_monitor.py | Existing Python temperature source and resource monitor |\n| code/run.py, run.sh, Run-ClosedLoop.ps1 | Paired launch, MCAP recording, stop and worker drain |\n| code/evaluate.py, evaluate_case.py | Independent numerical, data and performance checks |\n| code/report.py | Single README and notebook |\n| results/'+RUN.name+' | Runtime inputs, logs, queued requests, final GP models, metrics and figures |\n| code/snapshots/'+RUN.name+' | Original runtime/analysis sources and current report source |\n| recordings/'+RUN.name+' | Video replay and update_1s/update_5s rosbag folders |'
text='# ROS closed loop: GP–GRG update interval\n\n## Purpose\n\nMeasure the compute cost and feasibility of GP + GRG updates every 1 s versus every 5 s, with **WIVR fixed in both cases**. \n\n## Folder layout\n\n```text\nROS_closed_loop_test/\n  code/          # Source, configuration, launchers and source snapshots\n  results/       # Runtime data, metrics, models and figures\n  recordings/    # MP4 replay and rosbag recordings\n  README.md\n  test_report.ipynb\n```\n\n## Configuration and architecture\n\n'+cfg+'\n\n'+method+'\n\n'+terminal+'\n\n## Executed results\n\n'+metrics+'\n\n'+conclusion+'\n\n[Figures and details](test_report.ipynb). [4× MCAP pose replay]('+video_relative+').\n\n## Files and computations\n\n'+files+'\n\n`GP::append/factor/posterior` implements the exact fixed-parameter RBF posterior. `Search::events` places predicted 1 Hz events on Nav2 polylines. `Search::score` computes joint posterior variance reduction, weighted by area × variance from the request GP; weights are fixed during each search. `Search::recurse` performs depth-2 GRG. `Experiment::update_gp` implements the 1/5 s batching; `process_jobs` computes GRG; `dispatch/poll` enforce arrival-only goal switching and remaining-time/Home checks. All actual samples use simulator pose and timestamp.\n\nDependencies: existing Docker bunker-jazzy, Jazzy/Nav2/Gazebo overlay, trained RBF hyperparameters, Nav2_test scene generation, SRF_test TemperatureField, Continuous_sensing_test GP reference and Scoring_policy_test objectives. WELL-E logic is unchanged.\n\n```powershell\npowershell -ExecutionPolicy Bypass -File .\\code\\Run-ClosedLoop.ps1\n```\n'
(ROOT/'README.md').write_text(text,encoding='utf8')
analysis=CODE/'snapshots'/RUN.name/'report_source';analysis.mkdir(parents=True,exist_ok=True)
for name in ('evaluate.py','evaluate_case.py','report.py'):shutil.copy2(CODE/name,analysis/name)
print(ROOT/'test_report.ipynb')
