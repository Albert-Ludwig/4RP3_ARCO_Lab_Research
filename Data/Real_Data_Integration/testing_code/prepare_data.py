"""Prepare read-only HOBO / GPS inputs as auditable GP samples.

Clock offset is an explicit assumption, never inferred from sampling phase.
"""
from __future__ import annotations
import argparse
import base64
import csv
import hashlib
import json
import math
import platform
import sqlite3
import subprocess
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import openpyxl

EXPERIMENT = Path(__file__).resolve().parents[1]
REPOSITORY = EXPERIMENT.parents[2]
UTC = timezone.utc
EST = timezone(timedelta(hours=-5), "EST")
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
TOPIC = "/calian_gnss/gps"

# Pass only selected small messages through stdin, avoiding dependence on a
# Docker bind mount. Use the actual ROS definition to deserialize CDR.
ROS_DECODER = r'''
import base64, json
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import NavSatFix
rows = []
for item in PAYLOAD:
    m = deserialize_message(base64.b64decode(item['cdr']), NavSatFix)
    row = {k: item[k] for k in ('message_id', 'bag_timestamp_ns')}
    row.update(header_timestamp_ns=int(m.header.stamp.sec)*10**9+int(m.header.stamp.nanosec),
               frame_id=m.header.frame_id, latitude_deg=float(m.latitude),
               longitude_deg=float(m.longitude), altitude_m=float(m.altitude),
               status=int(m.status.status), service=int(m.status.service),
               covariance_type=int(m.position_covariance_type))
    row.update({f'cov_{i}_m2': float(v) for i, v in enumerate(m.position_covariance)})
    rows.append(row)
print(json.dumps(rows, allow_nan=True))
'''


def utc_ns(local: datetime, offset_s: float = 0.0) -> int:
    if local.tzinfo is not None:
        raise ValueError("Expected the workbook's naive Date-Time (EST) value")
    delta = local.replace(tzinfo=EST).astimezone(UTC) - EPOCH
    return ((delta.days * 86400 + delta.seconds) * 10**9
            + delta.microseconds * 1000 + round(offset_s * 10**9))


def iso_ns(value: int) -> str:
    sec, ns = divmod(int(value), 10**9)
    return (EPOCH + timedelta(seconds=sec)).strftime("%Y-%m-%dT%H:%M:%S") + f".{ns:09d}Z"


def enu_from_geodetic(llh, origin):
    """WGS84 geodetic degrees/metres -> ECEF -> fixed local east/north/up."""
    def ecef(a):
        a = np.asarray(a, dtype=float)
        lat, lon, h = np.deg2rad(a[..., 0]), np.deg2rad(a[..., 1]), a[..., 2]
        e2 = 6.6943799901413165e-3
        n = 6378137.0 / np.sqrt(1.0 - e2 * np.sin(lat)**2)
        return np.stack(((n+h)*np.cos(lat)*np.cos(lon),
                         (n+h)*np.cos(lat)*np.sin(lon),
                         (n*(1-e2)+h)*np.sin(lat)), axis=-1)
    lat, lon = np.deg2rad(origin[:2])
    rotation = np.array([[-np.sin(lon), np.cos(lon), 0.0],
                         [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
                         [np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]])
    return (ecef(llh) - ecef(origin)) @ rotation.T


def bracket(times: np.ndarray, target: int, valid: np.ndarray, max_gap_s: float):
    """Never extrapolate or bridge invalid fixes / excessive gaps."""
    if target < times[0]:
        return None, "before_gps"
    if target > times[-1]:
        return None, "after_gps"
    right = int(np.searchsorted(times, target, side="left"))
    if times[right] == target:
        if not valid[right]:
            return None, "invalid_gps_endpoint"
        return (right, right, 0.0, 0.0, 0.0), "matched"
    left = right - 1
    if not (valid[left] and valid[right]):
        return None, "invalid_gps_endpoint"
    gap = (int(times[right]) - int(times[left])) / 1e9
    if gap > max_gap_s:
        return None, "gps_gap_too_large"
    alpha = (target - int(times[left])) / (int(times[right]) - int(times[left]))
    nearest = min(target-int(times[left]), int(times[right])-target)/1e9
    return (left, right, alpha, gap, nearest), "matched"


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def describe(values):
    a = np.asarray(values, dtype=float)
    return dict(min=float(a.min()), median=float(np.median(a)),
                p95=float(np.quantile(a, .95)), max=float(a.max()))


def one_file(root, pattern):
    paths = sorted(root.rglob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one {pattern} under {root}, found {len(paths)}. Select a single run with --source.")
    return paths[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=REPOSITORY / "Data/Real_spatialtemporal_data")
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "result")
    parser.add_argument("--container", default="bunker-jazzy")
    parser.add_argument("--ros-python", help="Local Python with sourced rclpy, instead of Docker")
    parser.add_argument("--temperature-offset-s", type=float, default=0.0,
                        help="Added to HOBO UTC timestamps; default 0, unverified")
    parser.add_argument("--max-gps-gap-s", type=float, default=2.0)
    args = parser.parse_args()
    if not math.isfinite(args.temperature_offset_s) or not (0 < args.max_gps_gap_s < math.inf):
        raise ValueError("Offset must be finite; maximum GPS gap must be positive and finite")
    source, output = args.source.resolve(), args.output.resolve()
    if output == source or source in output.parents:
        raise ValueError("Write derived products outside the source directory")
    xlsx, db, meta = [one_file(source, p) for p in ("*.xlsx", "*.db3", "metadata.yaml")]
    inputs = [xlsx, db, meta]
    source_stats = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in inputs}
    print("Reading HOBO workbook and GPS messages (read-only).", flush=True)
    workbook = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    data = list(workbook["Data"].values)
    if data[0][1] != "Date-Time (EST)" or "Temperature" not in data[0][2] or "°C" not in data[0][2]:
        raise ValueError("Unexpected temperature column/timezone/unit")
    events = [[v.isoformat() if isinstance(v, datetime) else v for v in row]
              for row in workbook["Events"].values]
    details = [list(row) for row in workbook["Details"].values]
    workbook.close()
    temperature = []
    for excel_row, (sample, time, value) in enumerate(data[1:], start=2):
        if not isinstance(time, datetime):
            raise ValueError(f"Invalid timestamp at Excel row {excel_row}")
        raw_ns, corrected_ns = utc_ns(time), utc_ns(time, args.temperature_offset_s)
        value = float(value) if value is not None else float("nan")
        temperature.append(dict(sample_id=sample, excel_row=excel_row,
                                source_local_time=time.isoformat(), source_timezone="EST (UTC-05:00)",
                                original_utc_ns=raw_ns, timestamp_utc_ns=corrected_ns,
                                timestamp_utc=iso_ns(corrected_ns),
                                temperature_c=value if math.isfinite(value) else None))
    tt = np.array([r["timestamp_utc_ns"] for r in temperature], dtype=np.int64)
    if not np.all(np.diff(tt)>0) or len({r['sample_id'] for r in temperature}) != len(temperature):
        raise ValueError("Temperature timestamps/IDs must be unique and increasing")
    con = sqlite3.connect(db.as_uri()+"?mode=ro&immutable=1", uri=True)
    try:
        topics = con.execute("SELECT id,name,type,serialization_format FROM topics ORDER BY id").fetchall()
        topic = next(t for t in topics if t[1] == TOPIC)
        if topic[2:] != ("sensor_msgs/msg/NavSatFix", "cdr"):
            raise ValueError(f"Unsupported topic: {topic}")
        raw = con.execute("SELECT id,timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp", (topic[0],)).fetchall()
    finally:
        con.close()
    payload = [dict(message_id=i, bag_timestamp_ns=t, cdr=base64.b64encode(b).decode("ascii")) for i,t,b in raw]
    program = "PAYLOAD = " + repr(payload) + "\n" + ROS_DECODER
    if args.ros_python:
        command = [args.ros_python, "-B", "-"]
    else:
        command = ["docker", "--context", "desktop-linux", "exec", "-i", "-u", "ubuntu",
                   "-e", "PYTHONDONTWRITEBYTECODE=1", args.container, "bash", "-lc",
                   "source /opt/ros/jazzy/setup.bash && exec /usr/bin/python3 -B -"]
    decoded = subprocess.run(command, input=program, text=True, capture_output=True, check=True)
    gps = json.loads(decoded.stdout)
    if len(gps) != len(raw) or len(gps) < 2:
        raise ValueError("GPS decode count mismatch or insufficient data")
    gt = np.array([r['header_timestamp_ns'] for r in gps], dtype=np.int64)
    if not np.all(np.diff(gt)>0):
        raise ValueError("GPS acquisition timestamps must increase; inspect clock resets/duplicates")
    frames = sorted({r['frame_id'] for r in gps})
    if len(frames) != 1:
        raise ValueError(f"GPS frame changed during run: {frames}")
    llh = np.array([[r['latitude_deg'],r['longitude_deg'],r['altitude_m']] for r in gps])
    valid = (np.isfinite(llh).all(axis=1) & (np.abs(llh[:,0]) <= 90) &
             (np.abs(llh[:,1]) <= 180) & np.array([r['status']>=0 for r in gps]))
    if valid.sum() < 2:
        raise ValueError("Insufficient finite valid fixes")
    origin_index = int(np.flatnonzero(valid)[0])
    origin = llh[origin_index]
    xyz = enu_from_geodetic(llh, origin)
    for row, position, ok in zip(gps, xyz, valid):
        row.update(timestamp_utc=iso_ns(row['header_timestamp_ns']), valid_for_position=bool(ok),
                   receipt_minus_header_s=(row['bag_timestamp_ns']-row['header_timestamp_ns'])/1e9,
                   x_east_m=float(position[0]) if ok else None,
                   y_north_m=float(position[1]) if ok else None,
                   z_up_m=float(position[2]) if ok else None)
        for k,v in row.items():
            if isinstance(v,float) and not math.isfinite(v):
                row[k] = None
    aligned, samples = [], []
    for row in temperature:
        result = dict(row)
        match, reason = bracket(gt, row['timestamp_utc_ns'], valid, args.max_gps_gap_s)
        if row['temperature_c'] is None:
            match, reason = None, "invalid_temperature"
        result.update(run_id=db.parent.name, alignment_status=reason, clock_alignment_verified=False)
        if match is not None:
            left, right, alpha, gap, nearest = match
            pos = (1-alpha)*xyz[left] + alpha*xyz[right]
            result.update(t_s=(row['timestamp_utc_ns']-int(gt[origin_index]))/1e9,
                          x_east_m=float(pos[0]), y_north_m=float(pos[1]), z_up_m=float(pos[2]),
                          gps_left_message_id=gps[left]['message_id'], gps_right_message_id=gps[right]['message_id'],
                          gps_left_utc_ns=int(gt[left]), gps_right_utc_ns=int(gt[right]),
                          interpolation_fraction=float(alpha), gps_bracket_gap_s=float(gap),
                          nearest_gps_time_distance_s=float(nearest))
            samples.append({k:result[k] for k in ('run_id','sample_id','excel_row','timestamp_utc',
                           'timestamp_utc_ns','t_s','x_east_m','y_north_m','temperature_c',
                           'clock_alignment_verified')})
        aligned.append(result)
    if not samples:
        raise ValueError("No matched samples. Inspect timestamps before assuming another offset.")
    matched = [r for r in aligned if r['alignment_status']=='matched']
    adjacent_valid = valid[1:] & valid[:-1]
    distance = np.linalg.norm(np.diff(xyz[:,:2],axis=0),axis=1)[adjacent_valid]
    dt = np.diff(gt)[adjacent_valid]/1e9
    sensitivity = []
    for delta in (-2., -1., 1., 2.):
        shifts=[]
        for row in matched:
            m,_ = bracket(gt,row['timestamp_utc_ns']+int(delta*1e9),valid,args.max_gps_gap_s)
            if m:
                l,r,a,_,_ = m
                p=(1-a)*xyz[l]+a*xyz[r]
                shifts.append(float(np.linalg.norm(p[:2]-[row['x_east_m'],row['y_north_m']])))
        sensitivity.append(dict(assumed_extra_offset_s=delta, comparable_samples=len(shifts),
                                horizontal_displacement_m=describe(shifts) if shifts else None))
    summary = dict(
        run_id=db.parent.name, status="prepared_with_unverified_clock_alignment",
        temperature_count=len(temperature), gps_count=len(gps), valid_gps_count=int(valid.sum()),
        matched_count=len(samples), alignment_counts=dict(Counter(r['alignment_status'] for r in aligned)),
        temperature_start_utc=temperature[0]['timestamp_utc'], temperature_end_utc=temperature[-1]['timestamp_utc'],
        gps_start_utc=iso_ns(gt[0]), gps_end_utc=iso_ns(gt[-1]),
        matched_start_utc=samples[0]['timestamp_utc'], matched_end_utc=samples[-1]['timestamp_utc'],
        temperature_interval_s=describe(np.diff(tt)/1e9), gps_interval_s=describe(np.diff(gt)/1e9),
        temperature_c=describe([r['temperature_c'] for r in temperature if r['temperature_c'] is not None]),
        gps_bracket_gap_s=describe([r['gps_bracket_gap_s'] for r in matched]),
        nearest_gps_time_distance_s=describe([r['nearest_gps_time_distance_s'] for r in matched]),
        receipt_minus_header_s=describe([r['receipt_minus_header_s'] for r in gps]),
        gps_status_counts=dict(Counter(str(r['status']) for r in gps)),
        gps_covariance_type_counts=dict(Counter(str(r['covariance_type']) for r in gps)),
        gps_frame_ids=frames, gps_extent_east_north_m=np.ptp(xyz[valid,:2],axis=0).tolist(),
        horizontal_path_length_m=float(distance.sum()), horizontal_speed_m_s=describe(distance/dt),
        coordinate_system=dict(name="local_enu_at_first_valid_gps", datum="WGS84",
                               origin_latitude_deg=float(origin[0]), origin_longitude_deg=float(origin[1]),
                               origin_altitude_m=float(origin[2]), origin_gps_message_id=gps[origin_index]['message_id'],
                               time_origin_utc=iso_ns(gt[origin_index]), time_origin_utc_ns=int(gt[origin_index]),
                               axes="x east, y north, z up", position_reference="GPS antenna; HOBO lever arm unknown",
                               ros_map_transform="not established"),
        assumptions=dict(temperature_timezone="Fixed EST UTC-05:00, as declared by workbook",
                         temperature_offset_s=args.temperature_offset_s, clock_alignment_verified=False,
                         gps_time="NavSatFix header timestamp interpreted as Unix UTC",
                         max_gps_gap_s=args.max_gps_gap_s,
                         interpolation="Linear ENU interpolation between consecutive valid fixes; no extrapolation",
                         sensor_response_delay="unknown; no correction applied",
                         common_synchronization_event="not identified"),
        offset_sensitivity=sensitivity,
        limitations=["Sample phase and bag receipt latency do not establish HOBO/GPS clock synchronization.",
                     "NavSatStatus=2 indicates GBAS; it does not alone establish an RTK fixed solution or centimetre accuracy.",
                     "GPS antenna position is a proxy for temperature sensor position; mounting offset is unmeasured.",
                     "One mobile run confounds spatial and temporal variation. No long-term temporal validation is possible here.",
                     "No temperature detrending, smoothing, calibration, response correction or kernel fitting was performed.",
                     "This trajectory provides sampled locations, not an occupancy or traversability map.",
                     "Only selected GPS messages were deserialized; no whole-bag integrity or image-stream validation."],
        source_workbook_events=events, source_workbook_details=details,
        source_topics=[dict(id=i,name=n,type=t,serialization_format=f) for i,n,t,f in topics])
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'gps.csv',gps)
    write_csv(output/'temperature_alignment.csv',aligned)
    write_csv(output/'gp_samples.csv',samples)
    write_json(output/'summary.json',summary)
    print(f"Prepared {len(samples)} matched samples. Hashing original inputs for provenance...", flush=True)
    provenance=[]
    for p in inputs:
        print(f"SHA-256: {p.name}",flush=True)
        provenance.append(dict(path=p.relative_to(REPOSITORY).as_posix() if REPOSITORY in p.parents else str(p),
                               size_bytes=p.stat().st_size,sha256=digest(p)))
    unchanged = all(source_stats[p] == (p.stat().st_size,p.stat().st_mtime_ns) for p in inputs)
    if not unchanged:
        raise RuntimeError("Source size/mtime changed during processing; rerun on a stable recording")
    write_json(output/'manifest.json',dict(created_at_utc=datetime.now(UTC).isoformat(),
        inputs=provenance, source_size_and_mtime_unchanged=unchanged,
        selected_cdr_sha256=hashlib.sha256(b''.join(b for _,_,b in raw)).hexdigest(),
        script_sha256=digest(Path(__file__)),
        runtime=dict(python=platform.python_version(),numpy=np.__version__,openpyxl=openpyxl.__version__,
                     decoder="rclpy.serialization.deserialize_message(sensor_msgs.msg.NavSatFix)",
                     decoder_command=command),
        outputs={p.name:digest(p) for p in sorted(output.iterdir()) if p.suffix=='.csv' or p.name=='summary.json'}))
    print(json.dumps({k:summary[k] for k in ('matched_count','alignment_counts','gps_extent_east_north_m')},indent=2))


if __name__ == '__main__':
    main()
