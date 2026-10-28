"""Check independent geometric fixtures and recompute recorded navigation metrics."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

STAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STAGE/'testing_code/src/well_e_nav2_test/scripts'))
from geometry import clearance, footprint, rectangle


def verify():
    square = rectangle([0, 0], [2, 2])
    assert math.isclose(clearance(square, rectangle([4, 0], [2, 2])), 2.0)
    assert math.isclose(clearance(square, rectangle([3, 3], [2, 2])), math.sqrt(2))
    assert clearance(square, rectangle([1.9, 0], [2, 2])) == 0
    assert clearance(square, footprint([2.2, 0, math.pi/4], square)) == 0
    latest = json.loads((STAGE/'result/latest.json').read_text())
    run = STAGE/'result'/latest['run']
    record = json.loads((run/'manifest.json').read_text())
    scenario = record['scenario']
    for name, expected in record['artifact_sha256'].items():
        assert hashlib.sha256((run/name).read_bytes()).hexdigest() == expected, name
    for name, expected in record['source_sha256'].items():
        assert hashlib.sha256((STAGE.parent/name).read_bytes()).hexdigest() == expected, name
    metrics = json.loads((run/'metrics.json').read_text())
    with (run/'trajectory.csv').open() as stream:
        rows = [row for row in csv.DictReader(stream) if row['phase'] in ('navigating', 'settling')]
    obstacle = rectangle(scenario['obstacle']['center_m'], scenario['obstacle']['size_m'])
    distances = [clearance(footprint([float(row['x_m']), float(row['y_m']), float(row['yaw_rad'])],
                                     scenario['footprint_m']), obstacle) for row in rows]
    assert math.isclose(min(distances), metrics['minimum_footprint_clearance_m'], abs_tol=1e-10)
    assert sum(value == 0 for value in distances) == metrics['sampled_footprint_overlap_count']
    last = rows[-1]
    error = math.dist([float(last['x_m']), float(last['y_m'])], scenario['goal'][:2])
    assert math.isclose(error, metrics['final_position_error_m'], abs_tol=1e-10)
    assert metrics['status'] == 'passed' and all(metrics['checks'].values())
    result = {'geometry_fixtures': 4, 'physical_pose_samples': len(rows),
              'artifact_and_source_hashes_match': True, 'recorded_metrics_recomputed': True}
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    verify()
