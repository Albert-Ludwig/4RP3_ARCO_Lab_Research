"""Execute the continuous-sensing functional protocol; no GRG or ROS run."""
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
from components import Polyline,TimedRoute,predicted_events,GlobalSampler,BatchGP,MeasurementBuffer,MemoryProbe

TEST=Path(__file__).resolve().parents[1]
REPO=TEST.parents[2]
SOURCE=TEST/"testing_code"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
def csv_save(path,rows):
    rows=list(rows)
    if not rows:
        Path(path).write_text("");return
    with Path(path).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((SOURCE/"protocol.json").read_text())
    save(output/"protocol.json",protocol)
    srf_root=REPO/protocol["srf_test_directory"]
    srf_name=json.loads((srf_root/"result/latest.json").read_text())["run"]
    srf_run=srf_root/"result"/srf_name
    srf_summary=json.loads((srf_run/"summary.json").read_text())
    assert srf_summary["passed"] and srf_summary["config"]["working_extent_m"]==protocol["extent_m"]
    assert srf_summary["config"]["mission_interval_s"]==protocol["mission_interval_s"]
    srf_config=json.loads((srf_run/"field_config.json").read_text())
    sys.path.insert(0,str(srf_root/"testing_code"))
    from srf_field import TemperatureField
    references={
        "srf_field.py":srf_root/"testing_code/srf_field.py",
        "srf_field_config.json":srf_run/"field_config.json",
        "kernel_parameters.json":REPO/protocol["kernel_parameter_file"],
        "nav2_planned_path.json":REPO/protocol["path_source"],
        "reference_gp.py":REPO/"Code/Verification tests/GRG grading policy compare verification/testing_code/nav2_policy/gp.py"}
    protected={name:sha(p) for name,p in references.items()}
    saved_kernel=json.loads(references["kernel_parameters.json"].read_text())
    parameters={k:saved_kernel[k] for k in ["lengthscales_m_m_s","output_variance_c2","noise_variance_c2","mean_c"]}
    assert saved_kernel["training_observations"]==protocol["history_count"]
    save(output/"gp_parameters.json",parameters)
    save(output/"srf_config.json",srf_config)
    field=TemperatureField(srf_config)
    history_config=dict(srf_config,mission_interval_s=[protocol["history_interval_s"][0],protocol["mission_interval_s"][1]])
    historical_field=TemperatureField(history_config)
    history_rng=np.random.default_rng(protocol["history_seed"])
    max_count=max(protocol["scale_history_counts"])
    history_x=history_rng.random((max_count,3))
    history_x[:,:2]*=np.array(protocol["extent_m"])
    lo,hi=protocol["history_interval_s"];history_x[:,2]=lo+history_x[:,2]*(hi-lo)
    history_ids=[f"synthetic-history-{i}" for i in range(max_count)]
    history_y=historical_field.measure(history_x,history_ids)
    np.savez_compressed(output/"synthetic_history.npz",x=history_x,y=history_y,record_ids=np.asarray(history_ids))
    checks=[]
    def check(name,passed,**details):
        row=dict(name=name,passed=bool(passed),**details)
        checks.append(row);save(output/"checks.json",checks)
        print(("PASS " if passed else "FAIL ")+name,flush=True)
    mean_tol=protocol["mean_tolerance_c"];var_tol=protocol["variance_tolerance_c2"]
    def equal(name,a,b,tolerance):
        error=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
        check(name,np.allclose(a,b,atol=tolerance,rtol=1e-9),maximum_absolute_error=error,tolerance=tolerance)
    spec=importlib.util.spec_from_file_location("reference_fixed_gp",references["reference_gp.py"])
    reference_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference_module)
    LegacyGP=reference_module.GaussianProcess
    q_rng=np.random.default_rng(741)
    queries=q_rng.random((125,3))*np.array([30.,30.,120.])
    initial_n=protocol["history_count"]
    gp=BatchGP(history_x[:initial_n],history_y[:initial_n],parameters,history_ids[:initial_n])
    gp.save(output/"gp_initial.npz")
    reference=LegacyGP(history_x[:18],history_y[:18],parameters)
    small=BatchGP(history_x[:18],history_y[:18],parameters,history_ids[:18])
    equal("rbf_kernel_matches_existing_implementation",small.kernel(history_x[:18],queries),reference.kernel(history_x[:18],queries),var_tol)
    equal("initial_mean_matches_existing_implementation",small.posterior(queries)[0],reference.posterior(queries)[0],mean_tol)
    equal("initial_variance_matches_existing_implementation",small.posterior(queries)[1],reference.posterior(queries)[1],var_tol)
    check("synthetic_history_is_past_and_sized_explicitly",len(gp.x)==980 and np.max(history_x[:,2])<0,
          conditioning_source="SRF synthetic history; actual real observations are not remapped",future_scale_proxy=max_count)
    raw=np.asarray(json.loads(references["nav2_planned_path.json"].read_text()),dtype=float)[:,:2]
    shifted=raw-raw[0]+np.array(protocol["fixture_translation_m"])
    forward=Polyline(shifted);backward=Polyline(shifted[::-1])
    paths=[forward,backward,forward,backward]
    actual=TimedRoute(paths,protocol["actual_segment_end_s"],actual=True)
    planned=TimedRoute(paths,np.cumsum(protocol["nominal_segment_duration_s"]))
    events=predicted_events(paths,protocol["nominal_segment_duration_s"])
    csv_save(output/"predicted_initial_events.csv",events)
    check("uses_saved_nav2_detour_geometry",forward.length>np.linalg.norm(forward.xy[-1]-forward.xy[0])+.2,
          path_length_m=forward.length,chord_length_m=float(np.linalg.norm(forward.xy[-1]-forward.xy[0])))
    check("planned_events_include_edges_not_only_checkpoints",len(events)>len(paths) and
          sum(not any(np.linalg.norm([e["x_m"]-p.xy[-1,0],e["y_m"]-p.xy[-1,1]])<1e-8 for p in paths) for e in events)>len(events)/2,
          planned_event_count=len(events),path_segment_count=len(paths))
    split_events=[];offset=0.
    for path,duration in zip(paths,protocol["nominal_segment_duration_s"]):
        split_events.extend(predicted_events([path],[duration],start_s=offset))
        offset+=duration
    equal("predicted_junction_events_not_duplicated",[e["t_s"] for e in events],[e["t_s"] for e in split_events],1e-9)
    equal("predicted_segment_split_preserves_positions",
          [[e["x_m"],e["y_m"]] for e in events],[[e["x_m"],e["y_m"]] for e in split_events],1e-9)
    changed=predicted_events(paths,np.array(protocol["nominal_segment_duration_s"])*.8)
    check("predicted_timing_changes_scheduled_events",len(changed)!=len(events))
    sampler=GlobalSampler(identity="main");buffer=MeasurementBuffer(protocol["extent_m"],protocol["mission_interval_s"])
    trajectory=[];measurements=[];updates=[];buffer_trace=[];remaining_events=[]
    ticks=np.linspace(0,120,1201)
    fixed_before=gp.posterior(queries)[1]
    fixed_parameters=json.dumps(gp.parameters,sort_keys=True)
    ended=set()
    for t in ticks:
        t=round(float(t),9);xy=actual.at(t)
        row=sampler.poll(t,xy,field)
        if row:
            row["segment"]=actual.segment(t)
            row["predicted_x_m"],row["predicted_y_m"]=map(float,planned.at(t))
            row["pose_difference_m"]=float(np.linalg.norm(xy-planned.at(t)))
            assert buffer.offer(row)
            measurements.append(row)
        trajectory.append(dict(t_s=t,x_m=float(xy[0]),y_m=float(xy[1]),segment=actual.segment(t)))
        buffer_trace.append(dict(t_s=t,pending=len(buffer.pending),gp_count=len(gp.x),event="before_update"))
        for i,end in enumerate(protocol["actual_segment_end_s"]):
            if abs(t-end)<1e-8 and i not in ended:
                pending=len(buffer.pending);before_count=len(gp.x)
                started=time.perf_counter()
                with MemoryProbe() as memory:batch=buffer.flush(gp)
                elapsed=time.perf_counter()-started
                after=gp.posterior(queries)[1]
                check("fixed_query_variance_nonincreasing_batch_"+str(i+1),np.all(after<=fixed_before+var_tol))
                fixed_before=after
                snap=f"gp_batch_{i+1:02d}.npz";gp.save(output/snap)
                updates.append(dict(segment=i,time_s=t,batch_count=pending,before_count=before_count,after_count=len(gp.x),
                    gp_version=gp.version,wall_update_s=elapsed,gp_file=snap,
                    mean_fixed_query_variance_c2=float(after.mean()),**memory.values()))
                ended.add(i)
                buffer_trace.append(dict(t_s=t,pending=len(buffer.pending),gp_count=len(gp.x),event="after_update"))
                if i+1<len(paths):
                    for e in predicted_events(paths[i+1:],protocol["nominal_segment_duration_s"][i+1:],start_s=t):
                        remaining_events.append(dict(gp_version=gp.version,**e))
    csv_save(output/"trajectory.csv",trajectory)
    csv_save(output/"measurements.csv",measurements)
    csv_save(output/"gp_updates.csv",updates)
    csv_save(output/"buffer_trace.csv",buffer_trace)
    csv_save(output/"predicted_remaining_events.csv",remaining_events)
    save(output/"gp_updates.json",updates)
    gp.save(output/"gp_final.npz")
    check("exactly_120_actual_measurements",len(measurements)==120)
    check("global_phase_survives_noninteger_segment_boundaries",
          np.allclose([r["t_s"] for r in measurements],np.arange(1,121)) and not sampler.missed_ticks)
    check("measurement_identities_unique",len({r["record_id"] for r in measurements})==120)
    observed=np.array([[r["x_m"],r["y_m"],r["t_s"]] for r in measurements])
    equal("recorded_positions_are_actual_positions",observed[:,:2],np.array([actual.at(t) for t in observed[:,2]]),1e-9)
    check("actual_measurements_not_replaced_with_predicted_events",max(r["pose_difference_m"] for r in measurements)>.1,
          maximum_pose_difference_m=max(r["pose_difference_m"] for r in measurements))
    equal("temperatures_queried_at_actual_inputs",
          np.array([r["temperature_c"] for r in measurements]),
          field.measure(observed,[r["record_id"] for r in measurements]),mean_tol)
    check("physical_positions_not_snapped_to_0_1m_grid",np.any(np.abs(observed[:,:2]/.1-np.round(observed[:,:2]/.1))>1e-4))
    stationary=np.linalg.norm(np.diff(observed[:,:2],axis=0),axis=1)<1e-10
    check("stationary_samples_keep_distinct_time_and_identity",np.any(stationary))
    times=np.array([r["t_s"] for r in trajectory]);xy=np.array([[r["x_m"],r["y_m"]] for r in trajectory])
    speeds=np.linalg.norm(np.diff(xy,axis=0),axis=1)/np.diff(times)
    check("fixture_speed_within_declared_limit",speeds.max()<=protocol["max_fixture_speed_m_s"]+1e-9,
          maximum_sampled_speed_m_s=float(speeds.max()))
    check("batch_boundaries_and_counts_correct",[r["batch_count"] for r in updates]==[23,28,32,37])
    check("gp_count_and_version_advance_once_per_batch",len(gp.x)==1100 and gp.version==4 and not buffer.pending)
    check("kernel_hyperparameters_frozen",json.dumps(gp.parameters,sort_keys=True)==fixed_parameters)
    equal("gp_only_contains_history_and_valid_observed_inputs",gp.x,np.vstack([history_x[:980],observed]),1e-9)
    equal("gp_only_contains_observed_temperature_values",gp.y,np.r_[history_y[:980],[r["temperature_c"] for r in measurements]],mean_tol)
    check("remaining_event_rebuild_preserves_global_clock",
          all(e["t_s"]>protocol["actual_segment_end_s"][e["gp_version"]-1] and abs(e["t_s"]-round(e["t_s"]))<1e-9
              for e in remaining_events))
    check("return_endpoint_is_fixture_home",np.linalg.norm(actual.at(120)-forward.xy[0])<1e-9,
          scope="Offline geometry only; not Nav2 return execution")
    reference_full=LegacyGP(gp.x,gp.y,parameters)
    equal("final_batch_mean_matches_existing_gp",gp.posterior(queries)[0],reference_full.posterior(queries)[0],mean_tol)
    equal("final_batch_variance_matches_existing_gp",gp.posterior(queries)[1],reference_full.posterior(queries)[1],var_tol)
    # Independent sequential/partition conditioning identities on a small model.
    small_rows=measurements[:12]
    sequential=LegacyGP(history_x[:18],history_y[:18],parameters)
    for r in small_rows:sequential.append([r["x_m"],r["y_m"],r["t_s"]],r["temperature_c"])
    batch=BatchGP(history_x[:18],history_y[:18],parameters,history_ids[:18]);batch.append_batch(small_rows)
    equal("batch_equals_sequential_mean",batch.posterior(queries)[0],sequential.posterior(queries)[0],mean_tol)
    equal("batch_equals_sequential_variance",batch.posterior(queries)[1],sequential.posterior(queries)[1],var_tol)
    partitioned=BatchGP(history_x[:18],history_y[:18],parameters,history_ids[:18])
    for rows in [small_rows[:3],small_rows[3:8],small_rows[8:]]:partitioned.append_batch(rows)
    equal("batch_partition_independent_mean",batch.posterior(queries)[0],partitioned.posterior(queries)[0],mean_tol)
    equal("batch_partition_independent_variance",batch.posterior(queries)[1],partitioned.posterior(queries)[1],var_tol)
    # Quality, duplicate and transactional-buffer branches.
    test_buffer=MeasurementBuffer(protocol["extent_m"],protocol["mission_interval_s"])
    test_gp=BatchGP(history_x[:4],history_y[:4],parameters,history_ids[:4])
    valid=dict(measurements[0],record_id="unit-valid")
    assert test_buffer.offer(valid)
    check("buffer_rejects_duplicate_before_update",not test_buffer.offer(valid))
    test_buffer.flush(test_gp)
    check("buffer_rejects_duplicate_after_update",not test_buffer.offer(valid))
    invalid=[dict(valid,record_id="quality",valid=False),dict(valid,record_id="nan",temperature_c=float("nan")),
             dict(valid,record_id="outside",x_m=31.),dict(valid,record_id="missing")]
    del invalid[-1]["temperature_c"]
    check("bad_quality_nonfinite_outside_missing_rejected",not any(test_buffer.offer(r) for r in invalid))
    check("invalid_records_never_enter_gp",len(test_gp.x)==5 and len(test_buffer.pending)==0)
    recovery_row=dict(measurements[1],record_id="transaction-retry")
    assert test_buffer.offer(recovery_row)
    previous_x=test_gp.x.copy();version=test_gp.version
    original_factorize=test_gp._factorize
    def fail(*args):raise RuntimeError("Injected linear-algebra failure")
    test_gp._factorize=fail
    try:test_buffer.flush(test_gp)
    except RuntimeError:pass
    test_gp._factorize=original_factorize
    check("failed_update_preserves_buffer_and_gp",len(test_buffer.pending)==1 and test_gp.version==version
          and np.array_equal(test_gp.x,previous_x))
    test_buffer.flush(test_gp)
    check("retry_commits_once_and_clears_buffer",len(test_gp.x)==6 and not test_buffer.pending)
    # Navigation interruption: retain measurements from the uncompleted segment.
    failure_gp=BatchGP(history_x[:980],history_y[:980],parameters,history_ids[:980])
    failure_buffer=MeasurementBuffer(protocol["extent_m"],protocol["mission_interval_s"])
    partial=[r for r in measurements if r["t_s"]<=protocol["failure_time_s"]]
    for r in partial:
        failure_buffer.offer(r)
        if r["t_s"]==23:failure_buffer.flush(failure_gp)
    pending_before=len(failure_buffer.pending);failure_buffer.flush(failure_gp)
    check("partial_failed_segment_retains_valid_measurements",pending_before==19 and len(failure_gp.x)==protocol["history_count"]+len(partial),
          valid_measurements=len(partial),partial_batch_count=pending_before,
          failure_t_s=protocol["failure_time_s"])
    check("failure_does_not_fabricate_later_samples",len(partial)==42 and max(r["t_s"] for r in partial)<=protocol["failure_time_s"])
    csv_save(output/"failure_measurements.csv",partial)
    save(output/"failure_case.json",dict(failure_time_s=protocol["failure_time_s"],completed_batch_count=23,
         partial_batch_count=pending_before,total_retained=len(partial),new_gp_count=len(failure_gp.x),
         rule="Flush valid interrupted-edge samples; no backfill after failure."))
    # Delayed callbacks preserve actual timestamps and record missed ticks.
    jitter=GlobalSampler(identity="jitter");late=[]
    for t in [0.,1.08,2.04,5.03,6.10]:
        r=jitter.poll(t,actual.at(t),field)
        if r:late.append(r)
    check("late_callbacks_use_actual_timestamps",np.allclose([r["t_s"] for r in late],[1.08,2.04,5.03,6.10]))
    check("missed_ticks_logged_without_backfill",jitter.missed_ticks==[3,4] and [r["tick"] for r in late]==[1,2,5,6])
    csv_save(output/"delayed_callback_measurements.csv",late)
    # Scale check: keep model and observed batch fixed, vary only synthetic history count.
    scale=[]
    del reference_full,failure_gp,small,batch,partitioned,test_gp
    gc.collect()
    for n in protocol["scale_history_counts"]:
        started=time.perf_counter()
        with MemoryProbe() as mem:
            scaled=BatchGP(history_x[:n],history_y[:n],parameters,history_ids[:n])
        init=time.perf_counter()-started
        init_memory=mem.values()
        started=time.perf_counter()
        with MemoryProbe() as mem:
            scaled.append_batch(measurements[:protocol["scale_batch_count"]])
        update=time.perf_counter()-started
        mu,var=scaled.posterior(queries[:16])
        row=dict(history_count=n,batch_count=120,final_count=len(scaled.x),
                 initial_factorization_s=init,batch_update_s=update,
                 initialization_sampled_peak_rss_mib=init_memory["sampled_peak_rss_mib"],**mem.values(),
                 label="current-count proxy" if n==980 else "hypothetical 4x-count proxy")
        scale.append(row)
        check("finite_scale_check_"+str(n),len(scaled.x)==n+120 and np.isfinite(mu).all() and np.isfinite(var).all() and np.all(var>=0))
        print("SCALE",n,"+ 120:",round(update,4),"s, peak",round(mem.values()["sampled_peak_rss_mib"],2),"MiB",flush=True)
        del scaled;gc.collect()
    csv_save(output/"scale_benchmarks.csv",scale);save(output/"scale_benchmarks.json",scale)
    check("shared_sources_not_modified",all(sha(p)==protected[name] for name,p in references.items()))
    save(output/"provenance.json",dict(srf_run=srf_name,references={name:dict(path=p.relative_to(REPO).as_posix(),sha256=protected[name])
         for name,p in references.items()},gp_conditioning="Synthetic SRF observations only; real RBF parameters reused without fitting",
         history_note=protocol["history_note"],virtual_time_note=protocol["scheduling_note"]))
    passed=all(c["passed"] for c in checks)
    summary=dict(passed=passed,checks=len(checks),failed_checks=[c["name"] for c in checks if not c["passed"]],
        extent_m=protocol["extent_m"],virtual_duration_s=120,sensor_interval_s=1,measurements=len(measurements),
        batch_sizes=[r["batch_count"] for r in updates],gp_updates=len(updates),initial_gp_count=980,final_gp_count=len(gp.x),
        missed_main_ticks=sampler.missed_ticks,maximum_fixture_speed_m_s=float(speeds.max()),
        total_batch_wall_s=sum(r["wall_update_s"] for r in updates),
        maximum_batch_wall_s=max(r["wall_update_s"] for r in updates),
        scale_benchmarks=scale,ros_executed=False,grg_executed=False,policy_comparison_performed=False,
        complete_120s_mission_budget_validated=False,
        scope="Offline functional trace and component timings; no GP hyperparameter fitting or real-map registration.")
    save(output/"summary.json",summary)
    save(output/"manifest.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),
         packages={name:importlib.metadata.version(name) for name in ["numpy","scipy","gstools","psutil"]},
         thread_environment={k:os.environ.get(k) for k in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"]},
         source_hashes={p.name:sha(p) for p in SOURCE.iterdir() if p.is_file() and p.suffix in (".py",".json")}))
    if passed:save(TEST/"result/latest.json",dict(run=output.name))
    print(json.dumps(summary),flush=True)
    return summary


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--output");a=p.parse_args()
    out=Path(a.output) if a.output else TEST/"result"/("run_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    try:result=run(out)
    except Exception:
        if out.exists():(out/"failure.txt").write_text(traceback.format_exc())
        raise
    raise SystemExit(0 if result["passed"] else 1)
