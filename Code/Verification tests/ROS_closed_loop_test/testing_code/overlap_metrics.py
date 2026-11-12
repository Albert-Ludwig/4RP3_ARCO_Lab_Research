"""Verify overlap from recorded physical poses, planning intervals and Nav2 results."""
import numpy as np
import matplotlib.pyplot as plt

def analyze(plans,manager,poses,origin,duration):
    dt=np.diff(poses[:,0]);mid=(poses[1:,0]+poses[:-1,0])/2
    valid=dt>0
    translation=np.zeros(len(dt));rotation=np.zeros(len(dt))
    translation[valid]=np.linalg.norm(np.diff(poses[:,1:3],axis=0),axis=1)[valid]/dt[valid]
    yaw=np.diff(poses[:,3])
    rotation[valid]=np.abs(np.arctan2(np.sin(yaw[valid]),np.cos(yaw[valid]))/dt[valid])
    moving=(translation>.05)|(rotation>.05)
    planning=np.zeros(len(dt),dtype=bool);future=planning.copy()
    spans=[]
    for p in plans:
        start=p.get("planning_started_t",p["finish_t"]-p["total_s"])
        stop=min(duration,p["finish_t"])
        if stop<=start:continue
        mask=(mid>=max(0,start))&(mid<=stop)
        planning|=mask
        if p.get("lookahead",False):future|=mask
        spans.append((max(0,start),stop,p.get("kind","search"),p.get("lookahead",False),p["round"]))
    navigation=[];goals=[];completions=[];open_goal=None
    for row in manager:
        t=row.get("sim",origin)-origin
        if row.get("event")=="goal":
            goals.append((t,row["target"]))
            if open_goal is not None:raise AssertionError("Nav2 goals overlap before a result")
            open_goal=(t,row["target"])
        elif row.get("event")=="navigation_result":
            if open_goal is not None:
                assert row["target"]==open_goal[1]
                navigation.append((open_goal[0],min(duration,t),row["target"],row["code"]))
                open_goal=None
            if row["code"]==4:completions.append(t)
    if open_goal is not None:navigation.append((open_goal[0],duration,open_goal[1],None))
    navmask=np.zeros(len(dt),dtype=bool)
    for a,b,_,_ in navigation:navmask|=(mid>=a)&(mid<=b)
    gaps=[]
    for end in completions:
        following=[start for start,_ in goals if start>=end]
        if following:gaps.append(min(following)-end)
    forecasts={r["goal_epoch"]:r["predicted_start_t"] for r in manager if r.get("event")=="lookahead_requested"}
    arrival_errors=[dict(goal_epoch=r["goal_epoch"],forecast_s=forecasts[r["goal_epoch"]],actual_s=r["sim"]-origin,error_s=r["sim"]-origin-forecasts[r["goal_epoch"]]) for r in manager if r.get("event")=="handover_arrived" and r["goal_epoch"] in forecasts]
    metrics=dict(arrival_forecast_errors_s=arrival_errors,
        planning_overlap_navigation_s=float(dt[planning&navmask].sum()),
        planning_overlap_translation_s=float(dt[planning&(translation>.05)].sum()),
        planning_overlap_physical_motion_s=float(dt[planning&moving].sum()),
        lookahead_overlap_physical_motion_s=float(dt[future&moving].sum()),
        planning_while_stationary_s=float(dt[planning&~moving].sum()),
        lookahead_search_count=sum(p.get("lookahead",False) for p in plans),
        activation_count=sum(p.get("kind")=="activate" for p in plans),
        full_search_count=sum(p.get("kind","search")=="search" for p in plans),
        candidate_ready_while_navigating=sum(r.get("event")=="candidate_ready" and r.get("while_navigating",False) for r in manager),
        arrivals_waiting_for_search=sum(r.get("event")=="handover_arrived" and r.get("waiting_for_search",False) for r in manager),
        handover_gap_s=gaps,initial_dispatch_wait_s=goals[0][0] if goals else None,
        stale_plan_rejections=sum(r.get("event")=="stale_plan_rejected" for r in manager),
        late_goal_dispatches=[dict(t=start,target=target) for start,target in goals if start>duration+.1],
        goal_preemption_events=sum(r.get("event")=="navigation_result" and r.get("code")==5 and r["sim"]-origin<duration-.1 for r in manager),
        overlap_definition="Physical motion: position-derived speed >0.05 m/s OR wrapped yaw rate >0.05 rad/s. Time integrates positive rosbag pose intervals; GUI FPS is not measured.")
    return metrics,spans,navigation

def plot(path,spans,navigation,duration):
    fig,ax=plt.subplots(figsize=(10,3.2),layout="constrained")
    for a,b,target,status in navigation:
        ax.broken_barh([(a,max(0,b-a))],(0,0.7),facecolors="tab:green",alpha=.75)
        if b-a>3:ax.text((a+b)/2,.35,str(target),ha="center",va="center",fontsize=8)
    for a,b,kind,future,round_no in spans:
        y=1 if kind=="activate" else 2
        color="tab:orange" if kind=="activate" else "tab:blue"
        ax.broken_barh([(a,b-a)],(y,.7),facecolors=color,alpha=.8)
        if b-a>3:ax.text((a+b)/2,y+.35,str(round_no),ha="center",va="center",fontsize=8,color="white")
    ax.set(xlim=(0,duration),yticks=[.35,1.35,2.35],
           yticklabels=["Nav2 goals","GP update / handover","GRG search"],
           xlabel="Mission simulation time (s)",title="Recorded scheduling intervals; overlap is checked against physical poses")
    fig.savefig(path,dpi=150);plt.close(fig)
