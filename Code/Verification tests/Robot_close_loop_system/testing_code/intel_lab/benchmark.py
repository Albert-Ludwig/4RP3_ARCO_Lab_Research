"""Replay the same Intel checkpoint at depths 1/2/3 with bounded wall time."""

import argparse
from dataclasses import asdict
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
import time

from data import ROOT, PROCESSED
from replay import IntelPlanner, execute
from experiment import save_csv, save_json

sys.path.insert(0,str(ROOT / "testing_code/grg_depth_benchmark"))
from run_benchmark import supervised

RESULT = ROOT / "10.18_result/grg_depth_benchmark/intel_lab"
BASELINE = ROOT / "10.18_result/baseline/intel_lab/replay"


class LoggedPlanner(IntelPlanner):
    def __init__(self,*args,folder,**kwargs):
        super().__init__(*args,**kwargs)
        self.folder,self.calls=folder,[]

    def event(self,kind,**values):
        with (self.folder/"events.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(event=kind,wall_s=time.perf_counter(),**values))+"\n")

    def plan(self,gp,current,elapsed,remaining,visited=(),blocked=()):
        call=len(self.calls)
        self.event("plan_start",call=call,elapsed_s=elapsed,gp_version=gp.version)
        plan=super().plan(gp,current,elapsed,remaining,visited,blocked)
        samples=plan.route[1:-1]
        row=dict(call=call,elapsed_s=elapsed,**asdict(plan),sample_events=len(samples),
                 repeated_events=len(samples)-len(set(samples)))
        self.calls.append(row);self.event("plan_end",**row)
        save_csv(self.folder/"planning_runtime.csv",self.calls)
        return plan


def run_case(folder,depth,screen=False):
    if (folder/"execution.json").exists():
        previous=json.loads((folder/"execution.json").read_text())
        if previous["status"] in ["SUCCESS","TIMEOUT"]:
            print(f"Reuse depth {depth}: {folder}, {previous['status']}",flush=True)
            return previous
        archive=RESULT/"failed_attempts"/("screen_" if screen else "mission_")/f"depth_{depth}"
        if archive.exists():raise RuntimeError(f"Inspect earlier failed attempt: {archive}")
        archive.parent.mkdir(parents=True,exist_ok=True)
        folder.rename(archive)
    command=[sys.executable,str(Path(__file__).resolve()),"--worker","--depth",str(depth),"--output",str(folder)]
    if screen:command.append("--screen")
    outcome=supervised(command,folder,plan_limit=90,mission_limit=900)
    if outcome["status"]!="SUCCESS":save_json(folder/"summary.json",dict(depth=depth,**outcome))
    print(f"depth {depth} {'screen' if screen else 'mission'}: {outcome['status']}, {outcome['wall_s']:.2f} s",flush=True)
    return outcome


def main():
    RESULT.mkdir(parents=True,exist_ok=True)
    checkpoint=ROOT/"10.18_result/baseline/intel_lab/forecast/initial_gp.pt"
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    save_json(RESULT/"experiment.json",dict(depths=[1,2,3],depth5="NOT_RUN: explicitly excluded by task",
              initial_gp_sha256=digest,split=str(PROCESSED),seed=18,
              planning_wall_limit_s=90,mission_wall_limit_s=900,mission_budget_s=900,
              design="One fixed recorded dataset. Depth changes; history, initial GP, candidates, map, held-out reference and arrival matching are shared. Repeat depth 2 checks determinism; no stochastic significance claim."))
    summaries=[]
    for depth in [2,1,3]:
        outcome=run_case(RESULT/"screening"/f"depth_{depth}",depth,True)
        if outcome["status"]!="SUCCESS":
            summaries.append(dict(depth=depth,status="NOT_RUN_SCREEN_TIMEOUT" if outcome["status"]=="TIMEOUT" else "NOT_RUN_SCREEN_ERROR"))
            continue
        folder=BASELINE if depth==2 else RESULT/"runs"/f"depth_{depth}"
        outcome=run_case(folder,depth)
        summary=json.loads((folder/"summary.json").read_text())
        summaries.append(dict(depth=depth,status=summary["status"],wall_s=outcome["wall_s"],
                             measurements=summary.get("measurements"),
                             planning_calls=summary.get("planning_calls"),
                             grg_runtime_s=summary.get("grg_runtime_s"),
                             final_rmse_c=summary.get("final_metrics",{}).get("rmse_c"),
                             mean_variance_c2=summary.get("final_metrics",{}).get("mean_variance_c2"),
                             mi_nats=summary.get("information_gain_nats"),
                             elapsed_s=summary.get("elapsed_s"),returned_home=summary.get("returned_home")))
    if (BASELINE/"verification.json").exists():
        repeat=RESULT/"repeat_depth2"
        outcome=run_case(repeat,2)
        if outcome["status"]=="SUCCESS":
            comparisons={name:(BASELINE/name).read_bytes()==(repeat/name).read_bytes()
                         for name in ["online_measurements.csv","robot_trajectory.csv","metrics.csv"]}
            a=json.loads((BASELINE/"planned_routes.json").read_text());b=json.loads((repeat/"planned_routes.json").read_text())
            comparisons["routes_and_times"]=[(p["route"],p["arrivals_s"],p["finish_s"]) for p in a]==[(p["route"],p["arrivals_s"],p["finish_s"]) for p in b]
            save_json(RESULT/"repeat_verification.json",dict(passed=all(comparisons.values()),checks=comparisons))
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
    save_csv(RESULT/"depth_summary.csv",sorted(summaries,key=lambda r:r["depth"]))
    save_json(RESULT/"depth_summary.json",sorted(summaries,key=lambda r:r["depth"]))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--worker",action="store_true")
    parser.add_argument("--screen",action="store_true");parser.add_argument("--depth",type=int,default=2)
    parser.add_argument("--output",type=Path);args=parser.parse_args()
    if args.worker:
        execute(args.output,args.depth,args.screen,partial(LoggedPlanner,folder=args.output))
    else:main()
