"""Numerical, recursive-dispatch and scaled-search verification; no RMSE ranking."""
from contextlib import nullcontext
import csv
from datetime import datetime,timezone
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import numpy as np
from scoring import Grid,AdaptiveWeights,Objective,gp_token
from routing import PathGraph,make_candidates,make_planner,SearchTimeout,intersects_interior

TEST=Path(__file__).resolve().parents[1];REPO=TEST.parents[2];SOURCE=TEST/"testing_code"
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
def csv_save(path,rows):
    if not rows:Path(path).write_text("");return
    with Path(path).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def execute(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    config=json.loads((SOURCE/"protocol.json").read_text());save(output/"protocol.json",config)
    continuous=REPO/config["continuous_test_directory"]
    previous_name=json.loads((continuous/"result/latest.json").read_text())["run"]
    previous=continuous/"result"/previous_name
    assert json.loads((previous/"summary.json").read_text())["passed"]
    sys.path.insert(0,str(continuous/"testing_code"))
    from components import BatchGP,MemoryProbe
    references={"continuous_components":continuous/"testing_code/components.py",
                "gp_initial":previous/"gp_initial.npz","gp_parameters":previous/"gp_parameters.json",
                "grg_source":REPO/config["grg_source"],
                "legacy_gp":REPO/"Code/Verification tests/GRG grading policy compare verification/testing_code/nav2_policy/gp.py"}
    protected={name:sha(path) for name,path in references.items()}
    params=json.loads(references["gp_parameters"].read_text())
    initial=np.load(references["gp_initial"])
    gp=BatchGP(initial["x"],initial["y"],params,initial["record_ids"].tolist())
    save(output/"gp_parameters.json",params)
    times=np.arange(0,120.00001,config["scoring_time_step_s"])
    grid=Grid(config["extent_m"],config["scoring_cell_m"],config["obstacle_bounds_m"],times)
    np.savez_compressed(output/"scoring_grid.npz",cells=grid.cells,xy=grid.xy,areas=grid.areas,times=grid.times,queries=grid.q)
    checks=[];math_rows=[]
    def check(name,passed,**details):
        checks.append(dict(name=name,passed=bool(passed),**details))
        save(output/"checks.json",checks);print(("PASS " if passed else "FAIL ")+name,flush=True)
    atol=config["numerical_absolute_tolerance"];rtol=config["numerical_relative_tolerance"]
    def equal(name,a,b):
        error=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
        check(name,np.allclose(a,b,atol=atol,rtol=rtol),maximum_absolute_error=error)
    check("10m_grid_has_valid_area_and_representatives",len(grid.xy)==9 and np.isclose(grid.areas.sum(),884) and
          all(not (13<=x<=17 and 13<=y<=17) for x,y in grid.xy),
          cells=len(grid.xy),effective_area_m2=float(grid.areas.sum()))
    check("online_query_times_are_0_5s",grid.q.shape==(2169,3) and len(grid.times)==241 and np.allclose(np.diff(grid.times),.5))
    boundary=Grid([27,23],10,[12,16,11,15],times=[0,60,120])
    check("incomplete_boundary_cell_areas_exact",np.isclose(boundary.areas.sum(),27*23-16) and len(np.unique(boundary.areas))>1)
    excluded=Grid([20,20],10,[0,10,0,10],times=[0,1])
    check("zero_valid_area_cells_excluded",len(excluded.xy)==3 and np.isclose(excluded.areas.sum(),300))
    five=Grid([30,30],5,config["obstacle_bounds_m"],times=[0,60,120])
    check("5m_sensitivity_grid_area_consistent",np.isclose(five.areas.sum(),grid.areas.sum()) and len(five.xy)>len(grid.xy),
          note="Geometry check only; reconstruction sensitivity comparison remains pending")
    unit_x=np.array([[2.,2.,-2],[11.,6.,-1],[7.,16.,-3],[23.,25.,-1],[17.,8.,-2],[26.,5.,-4]])
    unit_y=params["mean_c"]+np.array([.05,-.02,.03,.01,-.03,.02])
    unit_ids=[f"unit-history-{i}" for i in range(len(unit_x))]
    unit=BatchGP(unit_x,unit_y,params,unit_ids)
    spec=importlib.util.spec_from_file_location("scoring_reference_gp",references["legacy_gp"])
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    reference=legacy.GaussianProcess(unit_x,unit_y,params)
    events=np.array([[3.1,4.2,1.],[4.3,5.1,2.],[6.4,6.2,3.],[8.1,7.4,4.]])
    weights=AdaptiveWeights(grid,config["near_zero_variance_c2"]);snapshot=weights.refresh(unit,0.)
    equal("adaptive_weight_is_effective_area_times_current_variance",snapshot.spatial,grid.areas*snapshot.variance)
    check("higher_variance_has_higher_per_area_importance",
          np.all(np.diff((snapshot.spatial/grid.areas)[np.argsort(snapshot.variance)])>=-atol))
    equal("no_separate_temporal_weighting",snapshot.query.reshape(len(grid.times),len(grid.xy)),np.tile(snapshot.spatial,(len(grid.times),1)))
    check("weight_snapshot_arrays_are_readonly",not snapshot.spatial.flags.writeable and not snapshot.query.flags.writeable)
    repeated=weights.refresh(unit,10.)
    check("time_advance_without_gp_update_does_not_refresh_weights",repeated is snapshot and weights.generations==1)
    blocked=False
    try:
        with weights.freeze():weights.refresh(unit,0.)
    except RuntimeError:blocked=True
    check("refresh_forbidden_inside_frozen_search",blocked)
    class ZeroVariance:
        version=0;parameters=params
        def posterior(self,q):return np.zeros(len(q)),np.zeros(len(q))
    zero=AdaptiveWeights(grid).refresh(ZeroVariance(),0.)
    check("near_zero_variance_falls_back_to_area",zero.fallback and np.array_equal(zero.spatial,grid.areas))
    objectives={name:Objective(name,unit,grid,snapshot if name=="wivr" else None) for name in ["mi","emse","wivr"]}
    covariance=reference.covariance(events)
    cross=reference.covariance(grid.q,events)
    reduction=np.sum(cross*np.linalg.solve(covariance+unit.noise*np.eye(len(events)),cross.T).T,axis=1)
    expected={
        "mi":float(.5*np.linalg.slogdet(np.eye(len(events))+covariance/unit.noise)[1]),
        "emse":float(grid.area_q@reduction/grid.area_q.sum()),
        "wivr":float(snapshot.query@reduction/snapshot.query.sum())}
    for policy,obj in objectives.items():
        actual=obj.score(events)
        equal(policy+"_matches_independent_direct_matrix_formula",actual,expected[policy])
        math_rows.append(dict(policy=policy,implemented=actual,direct_reference=expected[policy],absolute_error=abs(actual-expected[policy])))
        equal(policy+"_event_order_invariant",obj.score(events[::-1]),actual)
        check(policy+"_empty_route_zero",obj.score(np.empty((0,3)))==0)
        before=obj.cache_hits;equal(policy+"_cached_score_exact",obj.score(events),actual)
        check(policy+"_repeated_events_hit_cache",obj.cache_hits>before)
    one=objectives["mi"].score(events[:1]);duplicated=np.repeat(events[:1],2,axis=0)
    check("mi_preserves_joint_redundancy",objectives["mi"].score(duplicated)<2*one)
    uniform=Objective("wivr",unit,grid,snapshot,explicit_weights=grid.area_q)
    equal("uniform_per_area_wivr_equals_area_emse",uniform.score(events),objectives["emse"].score(events))
    normalized=Objective("wivr",unit,grid,snapshot,explicit_weights=snapshot.query/snapshot.query.sum())
    equal("weight_normalization_applied_once",normalized.score(events),objectives["wivr"].score(events))
    partial=np.array(snapshot.query);partial[::2]=0.
    partial_obj=Objective("wivr",unit,grid,snapshot,explicit_weights=partial)
    equal("zero_individual_weights_supported",partial_obj.score(events),partial@reduction/partial.sum())
    bad_weights=False
    try:Objective("wivr",unit,grid,snapshot,explicit_weights=np.zeros(len(grid.q)))
    except ValueError:bad_weights=True
    check("all_zero_explicit_weights_rejected",bad_weights)
    covariance_one=reference.covariance(events[:1])[0,0]+unit.noise
    cross_one=reference.covariance(grid.q,events[:1])[:,0]
    equal("single_event_wivr_gain_formula",objectives["wivr"].score(events[:1]),
          snapshot.query@(cross_one**2/covariance_one)/snapshot.query.sum())
    hypothetical=BatchGP(unit_x,unit_y,params,unit_ids)
    hypothetical_rows=[dict(record_id=f"hyp-{i}",x_m=float(e[0]),y_m=float(e[1]),t_s=float(e[2]),temperature_c=unit.mean)
                       for i,e in enumerate(events)]
    hypothetical.append_batch(hypothetical_rows)
    actual_delta=reference.posterior(grid.q)[1]-hypothetical.posterior(grid.q)[1]
    equal("emse_equals_before_after_posterior_variance",objectives["emse"].score(events),grid.area_q@actual_delta/grid.area_q.sum())
    alt=BatchGP(unit_x,unit_y,params,unit_ids)
    alt.append_batch([dict(r,temperature_c=unit.mean+1.) for r in hypothetical_rows])
    equal("gain_independent_of_future_temperature_labels",hypothetical.posterior(grid.q)[1],alt.posterior(grid.q)[1])
    check("hypothetical_conditioning_does_not_refresh_live_weights",weights.generations==1)
    # Force accidental cache sharing: full keys must still isolate policy/grid/weights.
    objectives["mi"].cache=objectives["emse"].cache
    equal("shared_cache_cannot_mix_mi_and_emse",objectives["mi"].score(events),expected["mi"])
    different=Objective("emse",unit,boundary);different.cache=objectives["emse"].cache
    equal("shared_cache_cannot_mix_query_grids",different.score(events),Objective("emse",unit,boundary).score(events))
    alternate=Objective("wivr",unit,grid,snapshot,explicit_weights=grid.area_q)
    alternate.cache=objectives["wivr"].cache
    equal("shared_cache_cannot_mix_weight_configurations",alternate.score(events),expected["emse"])
    bounded=Objective("emse",unit,grid,feature_entries=2,score_entries=2)
    for i in range(4):bounded.score(events[i:i+1])
    check("feature_and_score_caches_are_bounded",len(bounded.features.cache)<=2 and len(bounded.cache)<=2 and bounded.features.evictions>0)
    save(output/"weights_initial.json",dict(time_s=0,areas_m2=grid.areas.tolist(),variance_c2=snapshot.variance.tolist(),
         spatial_weights=snapshot.spatial.tolist(),generation=snapshot.generation))
    old_objective=objectives["wivr"]
    unit.append_batch(hypothetical_rows[:1])
    stale=False
    try:old_objective.score(events)
    except RuntimeError:stale=True
    check("gp_update_rejects_stale_scoring_round",stale)
    stale_weights=False
    try:Objective("wivr",unit,grid,snapshot)
    except RuntimeError:stale_weights=True
    check("gp_update_rejects_stale_weight_snapshot",stale_weights)
    fresh=weights.refresh(unit,1.)
    equal("weights_refreshed_from_updated_gp",fresh.spatial,grid.areas*unit.posterior(np.column_stack([grid.xy,np.ones(len(grid.xy))]))[1])
    check("actual_gp_update_creates_new_weight_generation",fresh.generation==2 and fresh.token!=snapshot.token)
    save(output/"weights_updated.json",dict(time_s=1,areas_m2=grid.areas.tolist(),variance_c2=fresh.variance.tolist(),
         spatial_weights=fresh.spatial.tolist(),generation=fresh.generation))
    fresh_obj=Objective("wivr",unit,grid,fresh)
    check("new_round_scores_after_real_update",np.isfinite(fresh_obj.score(events)))
    # Small complete searches verify every recursive level, including return-edge events.
    tiny_points=np.array([[3.,3.],[8.,3.],[8.,8.],[3.,8.],[6.,6.]])
    tiny=PathGraph(tiny_points,config["obstacle_bounds_m"],config["path_clearance_m"],1.)
    small_config=dict(config,search_wall_limit_s=15.)
    dispatch=[]
    for policy in ["mi","emse","wivr"]:
        manager=AdaptiveWeights(grid);state=manager.refresh(unit,1.) if policy=="wivr" else None
        obj=Objective(policy,unit,grid,state)
        planner=make_planner(references["grg_source"],tiny,obj,small_config)
        with manager.freeze() if policy=="wivr" else nullcontext():
            plan=planner.plan(unit,0,0,28,blocked=[0])
        final_events,arrivals,finish=planner.schedule(plan.route,0)
        check("recursive_dispatch_"+policy,obj.calls>1 and planner.score_policies=={policy} and obj.policy_trace=={policy}
              and not hasattr(unit,"covariance_many"))
        check("recursive_weights_frozen_"+policy,policy!="wivr" or manager.generations==1)
        equal("returned_route_gain_recomputes_"+policy,plan.gain,obj.score(final_events))
        check("small_search_feasible_"+policy,finish<=28+1e-8 and plan.route[-1]==0)
        dispatch.append(dict(policy=policy,route=plan.route,events=len(final_events),gain=plan.gain,finish_s=finish,
                             objective_calls=obj.calls,weights_generations=manager.generations))
    save(output/"recursive_dispatch.json",dispatch)
    obj=Objective("mi",unit,grid);planner=make_planner(references["grg_source"],tiny,obj,small_config)
    e,a,f=planner.schedule([0,1,0],0)
    check("return_edge_continuously_sampled",len(e)==10 and np.allclose(e[-1,:2],tiny_points[0]))
    check("continuous_schedule_has_no_checkpoint_dwell",f==10 and planner.sample_s==0)
    check("route_events_are_global_1hz_without_junction_duplicates",np.array_equal(e[:,2],np.arange(1,11)))
    old_events=e.copy();old_path=tiny.paths[0,1].copy()
    tiny.replace_edge(0,1,old_path,10.)
    changed=planner.schedule([0,1,0],0)[0]
    check("travel_time_change_invalidates_route_event_cache",len(changed)==15 and not np.array_equal(changed,old_events))
    tiny.replace_edge(0,1,None,np.inf)
    check("unreachable_edge_not_replaced_by_straight_line",not np.isfinite(planner.schedule([0,1,0],0)[2]))
    # A matching-length geometric detour must also alter the predicted positions.
    tiny.replace_edge(0,1,np.array([[3,3],[5,5],[8,3]],dtype=float),10.)
    rerouted=planner.schedule([0,1,0],0)[0]
    check("path_geometry_change_invalidates_sampling_events",len(rerouted)==15 and not np.allclose(rerouted[:10,:2],changed[:10,:2]))
    # Exact regression against the scalar implementation preserved with its timing run.
    old_path=TEST/"result"/config["previous_timing_run"]/"source_snapshot/routing.py"
    spec=importlib.util.spec_from_file_location("scalar_event_reference",old_path)
    old_module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old_module;spec.loader.exec_module(old_module)
    sample_points=make_candidates(10)
    scalar_graph=old_module.PathGraph(sample_points)
    vector_graph=PathGraph(sample_points)
    stub=Objective("mi",unit,grid)
    scalar_planner=old_module.make_planner(references["grg_source"],scalar_graph,stub,small_config)
    vector_planner=make_planner(references["grg_source"],vector_graph,stub,small_config)
    rng=np.random.default_rng(731);max_difference=0.;same=True
    for i in range(100):
        route=[0,*rng.integers(1,len(sample_points),size=3).tolist(),0]
        elapsed=[0.,.25,23.4][i%3]
        left=scalar_planner.schedule(route,elapsed);right=vector_planner.schedule(route,elapsed)
        same=same and left[0].shape==right[0].shape and np.isclose(left[2],right[2],atol=1e-9)
        if left[0].shape==right[0].shape and len(left[0]):
            max_difference=max(max_difference,float(np.max(np.abs(left[0]-right[0]))))
    check("vectorized_events_match_scalar_reference",same and max_difference<1e-9,
          cases=100,maximum_absolute_error=max_difference)
    correctness=all(c["passed"] for c in checks)
    csv_save(output/"mathematical_errors.csv",math_rows)
    save(output/"correctness.json",dict(passed=correctness,checks=len(checks),
                                       failed_checks=[c["name"] for c in checks if not c["passed"]]))
    # Fresh cold-cache 30/50-checkpoint searches. No ranking or accuracy evaluation.
    scales=[]
    for count in config["candidate_counts"]:
        built=time.perf_counter()
        points=make_candidates(count,config["extent_m"],config["obstacle_bounds_m"],
                               config["checkpoint_clearance_m"],config["home_xy_m"])
        graph=PathGraph(points,config["obstacle_bounds_m"],config["path_clearance_m"],config["fixture_speed_m_s"])
        graph_s=time.perf_counter()-built
        np.savez_compressed(output/f"graph_{count}.npz",points=points,travel_s=graph.travel)
        for policy in ["mi","emse","wivr"]:
            print("SEARCH_START",count,policy,flush=True);gc.collect()
            manager=AdaptiveWeights(grid);started=time.perf_counter()
            status="completed";plan=None;message=""
            with MemoryProbe() as memory:
                state=manager.refresh(gp,0.) if policy=="wivr" else None
                obj=Objective(policy,gp,grid,state,config["feature_cache_entries"],config["score_cache_entries"])
                prep=time.perf_counter()-started
                planner=make_planner(references["grg_source"],graph,obj,config)
                search_start=time.perf_counter()
                try:
                    with manager.freeze() if policy=="wivr" else nullcontext():
                        plan=planner.plan(gp,0,0,config["motion_search_budget_s"],blocked=[0])
                except SearchTimeout as exc:
                    status="wall_limit";message=str(exc)
                search_s=time.perf_counter()-search_start
            score_confirmed=False;event_count=None;finish=None;route=[]
            if plan is not None:
                events,_,finish=planner.schedule(plan.route,0)
                score_confirmed=bool(np.isclose(plan.gain,obj.score(events),atol=atol,rtol=rtol))
                event_count=len(events);route=plan.route
                save(output/f"plan_{count}_{policy}.json",dict(route=route,finish_s=finish,events=events.tolist(),
                     objective=policy,gain=plan.gain,score_recomputed=score_confirmed,
                     weights_generations=manager.generations))
            row=dict(checkpoints=count,policy=policy,status=status,history_count=len(gp.x),queries=len(grid.q),
                     geometry_build_s=graph_s,round_preparation_s=prep,search_s=search_s,total_component_s=prep+search_s,
                     event_schedule_generation_s=planner.schedule_generation_s,score_function_s=planner.score_function_s,
                     route_events=event_count,estimated_motion_s=finish,score_recomputed=score_confirmed,
                     objective_calls=obj.calls,score_cache_hits=obj.cache_hits,feature_cache_hits=obj.features.hits,
                     feature_cache_misses=obj.features.misses,feature_cache_evictions=obj.features.evictions,
                     max_route_events_scored=planner.maximum_events,weight_generations=manager.generations,
                     nominal_budget_balance_s=None if finish is None else config["mission_budget_s"]-(prep+search_s+finish),
                     **memory.values(),note=message)
            scales.append(row);save(output/"search_benchmarks.json",scales)
            print("SEARCH_RESULT",count,policy,status,round(prep+search_s,3),"s",obj.calls,"scores",flush=True)
            check("scale_search_completed_"+str(count)+"_"+policy,status=="completed" and score_confirmed and finish<=config["motion_search_budget_s"]+1e-8)
            del obj,planner;gc.collect()
    csv_save(output/"search_benchmarks.csv",scales)
    check("shared_gp_grg_and_previous_test_files_unchanged",all(sha(path)==protected[name] for name,path in references.items()))
    summary=dict(passed=all(c["passed"] for c in checks),correctness_passed=correctness,
                 checks=len(checks),correctness_checks=json.loads((output/"correctness.json").read_text())["checks"],
                 failed_checks=[c["name"] for c in checks if not c["passed"]],
                 extent_m=config["extent_m"],grid_cell_m=10,scoring_spatial_cells=len(grid.xy),
                 scoring_time_step_s=.5,scoring_queries=len(grid.q),effective_area_m2=float(grid.areas.sum()),
                 history_count=len(gp.x),continuous_events=True,checkpoint_dwell_s=0,
                 completed_searches=sum(r["status"]=="completed" for r in scales),requested_searches=len(scales),
                 search_wall_limit_s=config["search_wall_limit_s"],
                 policy_comparison_performed=False,ros_executed=False,reconstruction_accuracy_evaluated=False,
                 complete_mission_budget_validated=False,
                 performance_ready=all(r["status"]=="completed" and r["nominal_budget_balance_s"]>=0 for r in scales),
                 maximum_numerical_error=max((c.get("maximum_absolute_error",0) for c in checks),default=0),
                 scope="Exact scoring/recursive correctness and single-run component timings with temporary geometry. Performance readiness is only nominal component+motion arithmetic, never full ROS acceptance.")
    save(output/"summary.json",summary)
    save(output/"provenance.json",dict(previous_continuous_run=previous_name,
         references={name:dict(path=p.relative_to(REPO).as_posix(),sha256=protected[name]) for name,p in references.items()},
         geometry_note=config["graph_note"],gp_note="Synthetic initial conditioning from stage 2; frozen physical-unit RBF parameters. No unseen SRF truth queried."))
    save(output/"manifest.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),
         packages={name:importlib.metadata.version(name) for name in ["numpy","scipy","psutil"]},
         thread_environment={k:os.environ.get(k) for k in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"]},
         source_hashes={p.name:sha(p) for p in SOURCE.iterdir() if p.is_file() and p.suffix in (".py",".json")}))
    save(TEST/"result/latest.json",dict(run=output.name))
    print(json.dumps(summary),flush=True)
    return summary


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--output");a=p.parse_args()
    output=Path(a.output) if a.output else TEST/"result"/("run_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    try:result=execute(output)
    except Exception:
        if output.exists():(output/"failure.txt").write_text(traceback.format_exc())
        raise
    raise SystemExit(0 if result["passed"] else 2)
