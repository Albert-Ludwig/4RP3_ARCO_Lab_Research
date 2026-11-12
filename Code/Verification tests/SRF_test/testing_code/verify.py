"""Execute the frozen SRF protocol and save auditable functional checks."""
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import traceback
import numpy as np
import psutil
from srf_field import TemperatureField

TEST = Path(__file__).resolve().parents[1]
REPO = TEST.parents[2]
SOURCE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


class MemoryProbe:
    """Sample process RSS including native allocations; not tracemalloc alone."""
    def __enter__(self):
        self.process = psutil.Process()
        self.base = self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        def sample():
            while not self.stop.wait(.005):
                self.peak = max(self.peak, self.process.memory_info().rss)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self
    def __exit__(self, *args):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()
    @property
    def increment_mib(self):
        return max(0, self.peak-self.base)/2**20


def execute(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((SOURCE/"protocol.json").read_text())
    save(output/"protocol.json", protocol)
    width,height=map(float,protocol["working_extent_m"])
    t_start,t_end=map(float,protocol["mission_interval_s"])
    duration=t_end-t_start
    assert width>0 and height>0 and duration>1
    domain_scale=np.array([width,height,duration])
    domain_offset=np.array([0.,0.,t_start])
    history = REPO/protocol["history_file"]
    original_hash = digest(history)
    with history.open(encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    temperatures = np.array([float(row["temperature_c"]) for row in records])
    if len(temperatures) < 2 or not np.isfinite(temperatures).all():
        raise ValueError("Invalid historical temperature reference")
    config = dict(protocol, mean_c=float(temperatures.mean()),
                  variance_c2=float(temperatures.var(ddof=1)))
    config["historical_reference"] = dict(file=protocol["history_file"], sha256=original_hash,
        observations=len(records), minimum_c=float(temperatures.min()), maximum_c=float(temperatures.max()))
    save(output/"field_config.json", config)
    checks = []
    def check(name, passed, **detail):
        row = dict(name=name, passed=bool(passed), **detail)
        checks.append(row)
        save(output/"checks.json", checks)
        print(("PASS " if passed else "FAIL ")+name, flush=True)
    atol = protocol["absolute_tolerance_c"]
    rtol = protocol["relative_tolerance"]
    def equal(name, actual, expected):
        error = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
        check(name, np.allclose(actual, expected, atol=atol, rtol=rtol), maximum_absolute_error_c=error)
    rng = np.random.default_rng(7001)
    queries = rng.random((protocol["query_test_count"],3))*domain_scale+domain_offset
    np.save(output/"query_inputs.npy",queries)
    started = time.perf_counter()
    field = TemperatureField(config)
    initialization_s = time.perf_counter()-started
    truth = field.truth(queries)
    np.save(output/"query_truth.npy",truth)
    if protocol.get("baseline_run"):
        baseline_config=json.loads((TEST/"result"/protocol["baseline_run"]/"field_config.json").read_text())
        same_parameters=["scaled_coordinate_scales_m_m_s","matern_nu","fourier_modes","field_seed",
                         "mean_c","variance_c2","trend_c_per_s","noise_std_c","noise_seed","batch_size","threads"]
        assert all(config[key]==baseline_config[key] for key in same_parameters)
        equal("same_field_on_common_baseline_domain",truth,TemperatureField(baseline_config).truth(queries))
    check("finite_physical_unit_queries", truth.shape==(len(queries),) and np.isfinite(truth).all())
    check("explicit_three_dimensional_temporal_model",
          field.model.dim==3 and field.model.spatial_dim==2 and field.model.temporal)
    equal("repeated_queries", field.truth(queries), truth)
    equal("reverse_query_order",field.truth(queries[::-1])[::-1],truth)
    order = rng.permutation(len(queries))
    equal("shuffled_query_order",field.truth(queries[order])[np.argsort(order)],truth)
    for size in [1,17,131]:
        actual = np.concatenate([field.truth(queries[i:i+size]) for i in range(0,len(queries),size)])
        equal("external_batch_partition_"+str(size),actual,truth)
    equal("fresh_generator",TemperatureField(config).truth(queries),truth)
    equal("saved_configuration_reload",TemperatureField(json.loads((output/"field_config.json").read_text())).truth(queries),truth)
    subprocess.run([sys.executable,str(SOURCE/"srf_field.py"),"--config",str(output/"field_config.json"),
                    "--inputs",str(output/"query_inputs.npy"),"--output",str(output/"fresh_process_truth.npy")],
                   check=True,env=os.environ.copy())
    equal("fresh_process_restart",np.load(output/"fresh_process_truth.npy"),truth)
    other = TemperatureField(dict(config,field_seed=config["field_seed"]+1)).truth(queries)
    check("different_seed_changes_field", not np.allclose(other,truth,atol=atol,rtol=rtol),
          rms_difference_c=float(np.sqrt(np.mean((other-truth)**2))))
    offgrid = np.array([[.1234567,.7654321,.2345678]])
    delta = float(abs(field.truth(offgrid)[0]-field.truth(np.array([[0.,1.,0.]]))[0]))
    check("continuous_off_grid_query_not_snapped", delta>atol, difference_from_rounded_query_c=delta)
    dimensionless = queries/field.scales
    residual = field.srf(dimensionless.T,mesh_type="unstructured",store=False)
    equal("physical_coordinate_scaling",truth,config["mean_c"]+residual)
    temporal_lags=np.array([0.,1.,duration/2,duration])
    correlations = field.model.correlation(temporal_lags/config["scaled_coordinate_scales_m_m_s"][2])
    check("temporal_scale_is_explicit_not_independent_redraws",
          correlations[0]==1 and 0<correlations[-1]<correlations[2]<correlations[1]<1,
          lags_s=temporal_lags.tolist(),model_correlations=correlations.tolist())
    temporal_probes=np.column_stack([np.full(4,width/2),np.full(4,height/2),t_start+temporal_lags])
    check("fixed_location_varies_over_time",np.ptp(field.truth(temporal_probes))>atol)
    shifted = TemperatureField(dict(config,mean_c=config["mean_c"]+.25,trend_c_per_s=.001))
    equal("deterministic_mean_and_trend_plumbing",shifted.truth(queries)-truth,.25+.001*queries[:,2])
    invalid = [np.array([[np.nan,0,t_start]]),np.array([[width+1,0,t_start]]),
               np.array([[0,height+1,t_start]]),np.array([[0,0,t_end+1]]),np.array([[1,2]])]
    rejected = 0
    for bad in invalid:
        try: field.truth(bad)
        except ValueError: rejected += 1
    check("invalid_shape_nonfinite_and_domain_rejected", rejected==len(invalid))
    check("empty_query_supported",field.truth(np.empty((0,3))).shape==(0,))
    ids = [f"measurement-{i}" for i in range(len(queries))]
    measured = field.measure(queries,ids)
    equal("measurement_identity_replay",field.measure(queries,ids),measured)
    equal("measurement_noise_query_order",field.measure(queries[order],[ids[i] for i in order])[np.argsort(order)],measured)
    field.truth(queries[::-1])
    field.truth(np.array([[9.,8.,7.]]))
    equal("offline_truth_queries_do_not_consume_sensor_noise",field.measure(queries,ids),measured)
    alternate_noise = TemperatureField(dict(config,noise_seed=config["noise_seed"]+1))
    equal("sensor_noise_seed_does_not_change_truth",alternate_noise.truth(queries),truth)
    check("noise_seed_changes_observations",not np.allclose(alternate_noise.measure(queries,ids),measured,atol=atol))
    check("distinct_measurement_ids_have_distinct_noise",
          not np.allclose(field.noise(ids),field.noise([i+"-repeat" for i in ids]),atol=atol))
    zero = TemperatureField(dict(config,noise_std_c=0.))
    equal("zero_sensor_noise_equals_truth",zero.measure(queries,ids),truth)
    noise = field.noise([f"noise-audit-{i}" for i in range(protocol["noise_test_count"])])
    noise_mean,noise_std = float(noise.mean()),float(noise.std(ddof=1))
    sigma = config["noise_std_c"]
    check("keyed_noise_distribution_sanity",abs(noise_mean)<5*sigma/np.sqrt(len(noise))
          and .9*sigma<noise_std<1.1*sigma,mean_c=noise_mean,sample_std_c=noise_std,
          note="Seeded functional sanity check, not real sensor calibration")
    np.save(output/"noise_samples.npy",noise)
    check("no_generated_field_stored",field.srf.field_names==[],
          storage=field.storage_summary())
    timed = []
    for i in range(protocol["timing_query_count"]):
        start = time.perf_counter()
        field.truth(queries[i:i+1])
        timed.append((time.perf_counter()-start)*1000)
    p95 = float(np.percentile(timed,95))
    check("isolated_query_fits_initial_1hz_interval",p95<protocol["acceptance"]["p95_single_query_ms"],
          median_ms=float(np.median(timed)),p95_ms=p95,
          maximum_ms=float(max(timed)),note="Isolated source only; not ROS deadline or sensor-response validation")
    benchmarks = []
    for count in protocol["benchmark_query_counts"]:
        generator = np.random.default_rng(7002)
        count_done,total,total_squared = 0,0.,0.
        start = time.perf_counter()
        with MemoryProbe() as memory:
            for start_index in range(0,count,config["batch_size"]):
                n = min(config["batch_size"],count-start_index)
                chunk = generator.random((n,3))*domain_scale+domain_offset
                values = field.truth(chunk)
                count_done += len(values)
                total += float(values.sum())
                total_squared += float(values@values)
        seconds = time.perf_counter()-start
        row=dict(query_count=count_done,elapsed_s=seconds,queries_per_s=count_done/seconds,
                 baseline_rss_mib=memory.base/2**20,sampled_peak_rss_mib=memory.peak/2**20,
                 sampled_rss_increment_mib=memory.increment_mib,streamed_mean_c=total/count_done,
                 streamed_variance_c2=total_squared/count_done-(total/count_done)**2,
                 **field.storage_summary())
        benchmarks.append(row)
        print("BENCHMARK",count_done,round(seconds,3),"s",flush=True)
    save(output/"benchmarks.json",benchmarks)
    with (output/"benchmarks.csv").open("w",newline="") as f:
        keys=["query_count","elapsed_s","queries_per_s","baseline_rss_mib","sampled_peak_rss_mib","sampled_rss_increment_mib"]
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        w.writerows([{k:r[k] for k in keys} for r in benchmarks])
    max_memory = max(row["sampled_rss_increment_mib"] for row in benchmarks)
    check("bounded_sampled_native_memory",max_memory<protocol["acceptance"]["peak_rss_increment_mib"],
          maximum_sampled_rss_increment_mib=max_memory,sampling_period_s=.005,
          note="RSS sampled at 5 ms; very short allocation peaks may be missed")
    sizes = {row["generator_array_bytes"] for row in benchmarks}
    check("no_field_or_query_history_accumulation",len(sizes)==1 and
          all(row["retained_position_count"]<=config["batch_size"] and not row["stored_field_names"] for row in benchmarks),
          generator_array_bytes=list(sizes)[0],max_retained_query_count=max(row["retained_position_count"] for row in benchmarks))
    streamed = []
    for block in field.truth_chunks(queries): streamed.extend(block)
    equal("streamed_output_matches_direct_queries",np.asarray(streamed),truth)
    # Query each requested fine-grid point directly in bounded batches.
    # Save only three requested 2D slices and time traces, never a full 3D volume.
    step = float(protocol["visualization"]["space_m"])
    temporal_step = float(protocol["visualization"]["time_step_s"])
    assert step == protocol["reference_resolution"]["space_m"] == 0.1
    assert temporal_step == protocol["reference_resolution"]["time_s"] == 0.5
    xs = np.linspace(0,width,int(round(width/step))+1)
    ys = np.linspace(0,height,int(round(height/step))+1)
    check("spatial_reference_resolution_0_1m",
          len(xs)==int(round(width/step))+1 and len(ys)==int(round(height/step))+1 and
          np.allclose(np.diff(xs),.1,atol=atol,rtol=rtol) and
          np.allclose(np.diff(ys),.1,atol=atol,rtol=rtol),
          x_points=len(xs),y_points=len(ys),points_per_slice=len(xs)*len(ys),spacing_m=step)
    snapshots=[]
    resolution_benchmarks=[]
    point_count=len(xs)*len(ys)
    for stamp in protocol["visualization"]["snapshot_times_s"]:
        start=time.perf_counter()
        with MemoryProbe() as memory:
            values=np.empty(point_count,dtype=float)
            for first in range(0,point_count,config["batch_size"]):
                last=min(first+config["batch_size"],point_count)
                indices=np.arange(first,last)
                z=np.column_stack([xs[indices % len(xs)],ys[indices // len(xs)],
                                   np.full(last-first,stamp)])
                values[first:last]=field.truth(z)
            snapshots.append(values.reshape(len(ys),len(xs)))
        seconds=time.perf_counter()-start
        row=dict(time_s=stamp,query_count=point_count,spacing_m=step,
                 elapsed_s=seconds,queries_per_s=point_count/seconds,
                 baseline_rss_mib=memory.base/2**20,sampled_peak_rss_mib=memory.peak/2**20,
                 sampled_rss_increment_mib=memory.increment_mib,
                 max_query_batch=config["batch_size"],**field.storage_summary())
        resolution_benchmarks.append(row)
        print("FINE_GRID",stamp,"s:",point_count,"direct queries,",round(seconds,3),"wall s",flush=True)
    save(output/"resolution_benchmarks.json",resolution_benchmarks)
    max_memory=max(max_memory,max(row["sampled_rss_increment_mib"] for row in resolution_benchmarks))
    check("fine_grid_batch_memory_bounded",
          max_memory<protocol["acceptance"]["peak_rss_increment_mib"] and
          all(not row["stored_field_names"] and row["retained_position_count"]<=config["batch_size"]
              for row in resolution_benchmarks),
          maximum_sampled_rss_increment_mib=max_memory,
          direct_queries_total=point_count*len(snapshots),no_spatial_interpolation=True)
    stamps=np.linspace(t_start,t_end,int(round(duration/temporal_step))+1)
    check("temporal_reference_resolution_0_5s",
          len(stamps)==int(round(duration/temporal_step))+1 and np.allclose(np.diff(stamps),.5,atol=atol,rtol=rtol),
          points_per_trace=len(stamps),spacing_s=temporal_step)
    probes=np.array([[.15,.15],[.5,.5],[.85,.85]])*np.array([width,height])
    traces=np.array([field.truth(np.column_stack([np.full(len(stamps),x),
                    np.full(len(stamps),y),stamps])) for x,y in probes])
    np.savez_compressed(output/"visualization.npz",x=xs,y=ys,spatial_slices=np.array(snapshots),
        snapshot_times_s=protocol["visualization"]["snapshot_times_s"],times_s=stamps,probes_xy_m=probes,traces_c=traces)
    check("visualization_queries_do_not_change_realization",np.allclose(field.truth(queries),truth,atol=atol,rtol=rtol))
    check("historical_source_not_modified",digest(history)==original_hash)
    dense_shape=[len(xs),len(ys),len(stamps)]
    summary=dict(passed=all(row["passed"] for row in checks),checks=len(checks),
        failed_checks=[row["name"] for row in checks if not row["passed"]],
        initialization_s=initialization_s,query_p95_ms=p95,
        max_sampled_rss_increment_mib=max_memory,config=config,
        spatial_resolution_m=step,temporal_resolution_s=temporal_step,
        spatial_points_per_slice=point_count,
        direct_fine_grid_queries=point_count*len(snapshots),
        fine_grid_total_elapsed_s=sum(row["elapsed_s"] for row in resolution_benchmarks),
        maximum_sampled_absolute_rss_mib=max(row["sampled_peak_rss_mib"] for row in benchmarks+resolution_benchmarks),
        historical_temperature_range_c=[float(temperatures.min()),float(temperatures.max())],
        sampled_fixture_range_c=[float(np.min(snapshots)),float(np.max(snapshots))],
        dense_reference_shape=dense_shape,dense_reference_float64_gib=float(np.prod(dense_shape)*8/2**30),
        dense_reference_materialized=False,full_real_calibration_complete=False,
        ros_integration_complete=False,policy_comparison_performed=False,
        scope="Functional source verification with empirical single-run temperature level/variance; metric Matern scales, smoothness, trend and noise are provisional.",
        finite_mode_note="RandMeth uses a fixed 1024-mode approximation; Matérn covariance is the target model, not a guarantee of exact empirical covariance in one realization.")
    save(output/"summary.json",summary)
    versions={name:importlib.metadata.version(name) for name in ["numpy","scipy","gstools","gstools-cython","psutil"]}
    save(output/"manifest.json",dict(created_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version,platform=platform.platform(),packages=versions,
        history_sha256=original_hash,source_hashes={p.name:digest(p) for p in SOURCE.iterdir()
            if p.is_file() and p.suffix in (".py",".json",".txt")},
        references=["https://geostat-framework.readthedocs.io/projects/gstools/en/stable/api/gstools.field.SRF.html",
                    "https://geostat-framework.readthedocs.io/projects/gstools/en/stable/api/gstools.covmodel.Matern.html",
                    "https://geostat-framework.readthedocs.io/projects/gstools/en/stable/api/gstools.field.generator.RandMeth.html"]))
    freeze=subprocess.run([sys.executable,"-m","pip","freeze"],text=True,capture_output=True,check=True)
    (output/"requirements-lock.txt").write_text(freeze.stdout)
    print(json.dumps({k:summary[k] for k in ["passed","checks","failed_checks","initialization_s","query_p95_ms","max_sampled_rss_increment_mib"]}),flush=True)
    if summary["passed"]:
        save(TEST/"result/latest.json",dict(run=output.name))
    return summary


if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--output")
    args=p.parse_args()
    output=Path(args.output) if args.output else TEST/"result"/("run_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    try:
        result=execute(output)
    except Exception:
        if output.exists():
            (output/"failure.txt").write_text(traceback.format_exc())
        raise
    raise SystemExit(0 if result["passed"] else 1)
