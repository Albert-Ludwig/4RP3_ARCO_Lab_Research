# Real Data Integration — Step 1: Temperature and GPS Alignment

## Purpose and dataset

Prepare traceable `(x, y, t, T)` observations from Julian's **2026-11-03** lawnmower recording near the baseball field. This stage covers timestamp conversion, GPS decoding, local metric coordinates and data-quality checks. GP training and ROS 2 navigation are subsequent stages.

**Result:** 980 temperature observations have interpolated GPS positions under an **unverified zero additional clock-offset assumption**. Processing checks passed; physical sensor synchronization is still unresolved.

## Inputs and dependencies

Original inputs remain in `../../../Data/Real_spatialtemporal_data/` relative to this folder:

- `lawnmower_5m_test_2026-11-03 14_41_39 EST (Data EST).xlsx`: HOBO temperature, `Data!A1:C1101`; `Events` and `Details` preserve logging metadata.
- `2026-11-03-welle-baseball-lawnmower/nov-3-baseball-1/nov-3-baseball-1_0.db3` and `metadata.yaml`: ROS recording. Only `/calian_gnss/gps` (`sensor_msgs/msg/NavSatFix`) is decoded.

Processing uses Python, NumPy and openpyxl for read-only Excel access, plus ROS 2 Jazzy `rclpy` and `sensor_msgs` in the existing `bunker-jazzy` Docker container. Selected GPS messages are streamed to ROS through stdin, so the 25.4 GB bag need not be copied or mounted. Report generation uses Matplotlib, nbformat, nbclient and an installed Python Jupyter kernel. See [requirements.txt](testing_code/requirements.txt).

## Method and assumptions

1. Convert the workbook's explicitly labelled **EST (UTC−05:00)** timestamps to UTC. Preserve original timestamps, sample IDs and Excel row numbers. Use the GPS **header** timestamp for position matching; retain bag receipt time separately.
2. Convert WGS84 latitude/longitude/altitude to ECEF and then a fixed local **ENU** frame. Origin: the first valid GPS fix, latitude `43.265474`, longitude `-79.928666`, altitude `45.3434 m`. `x` is east and `y` is north, in metres. `t_s` is seconds since `2026-11-03T19:24:15.325545540Z`.
3. Interpolate ENU position between consecutive valid GPS fixes at each temperature timestamp. Require valid finite positions and a bracket interval ≤2 s. This gap threshold is a processing choice, not a calibrated accuracy bound. Do not extrapolate, bridge an invalid fix or invent a position for rejected records.
4. Retain all temperature rows with match/exclusion reasons. Temperature values are not interpolated, smoothed or detrended. Preserve decoded GPS status, covariance and message IDs for review.

The approximately 0.325 s nearest-sample separation is **sampling phase**, not evidence of clock bias. No common HOBO/GPS synchronization event was identified. The script accepts an explicit `--temperature-offset-s` (positive adds time to HOBO), but does not estimate or certify it. Sensor response delay and the HOBO-to-GPS mounting offset are also unknown. Local ENU is not yet registered to a ROS `map` frame.

## Results

| Measure | Result |
|---|---:|
| HOBO records / interval | 1100 / 1 s |
| Valid GPS records | 981 of 981 |
| Matched temperature records | 980 |
| Excluded before / after GPS support | 63 / 57 |
| Matched local interval (EST) | 14:24:16–14:40:35 |
| Full recorded temperature range | 26.08–27.75 °C |
| GPS extent, east × north | 97.51 × 63.55 m |
| Sum of consecutive horizontal GPS distances | 1028.37 m |
| Largest GPS interval | 1.188 s |
| Median nearest GPS time distance | 0.326 s |
| Automated processing checks | 8 passed |

The recorded trajectory resembles the supplied lawnmower route. Pass spacing is approximately 5 m according to the collector; this pipeline does not treat that statement as a surveyed calibration. GPS status 2 denotes GBAS and does not by itself establish an RTK fixed solution or centimetre accuracy. A single moving run cannot isolate spatial variation from temporal drift or validate multi-day prediction. The data describe sampled locations, not occupancy or traversability.

## Files and data contract

| File | Role |
|---|---|
| [test_report.ipynb](test_report.ipynb) | Executed summary, sample preview, figures, clock sensitivity and conclusions |
| [result/gp_samples.csv](result/gp_samples.csv) | Matched observations for the next GP stage; `x_east_m`, `y_north_m`, `t_s`, `temperature_c` plus source identity and clock flag |
| [result/temperature_alignment.csv](result/temperature_alignment.csv) | All original temperature rows; match status, source and UTC timestamps, bracketing GPS message IDs/times, interpolation fraction and gap |
| [result/gps.csv](result/gps.csv) | Decoded GPS headers/receipt times, WGS84 coordinates, ENU coordinates, status, covariance and validity |
| [result/summary.json](result/summary.json) | Origin, units, assumptions, source topic inventory, logging metadata and timing sensitivity |
| [result/manifest.json](result/manifest.json) | Full SHA-256 of original Excel/bag/metadata, selected CDR digest, script/runtime and derived-file hashes |
| [result/verification.json](result/verification.json) | Executed processing checks and their scope |

`timestamp_utc_ns` and GPS timestamps are integer Unix nanoseconds; keep them as integers, not spreadsheet floating-point timestamps. Covariance entries `cov_0_m2` through `cov_8_m2` retain the message's row-major ENU position covariance. Blank position fields mean an excluded temperature row, never a zero position. Every prepared row keeps `clock_alignment_verified=False`. For additional runs, reuse a common spatial/time origin before combining them; this single-run default resets to the first valid fix.

## Reproduce

From this folder, with Docker Desktop and `bunker-jazzy` running:

```powershell
powershell -ExecutionPolicy Bypass -File .\testing_code\Run-Alignment.ps1
```

The launcher uses the existing managed extraction Python and reporting environment. Override `-DataPython` and `-ReportPython` for other installations. It regenerates derived results, checks, this README and the single Notebook. For a ROS-sourced Linux Python, run `prepare_data.py --ros-python /usr/bin/python3` with a host Python that has NumPy/openpyxl. Source files are opened read-only. Hashing the complete bag records provenance, not a full bag integrity certification. No Git commit or upload is performed.

Next: confirm HOBO/GPS clock setup or a common event, and sensor response/mounting details. Then use blocked held-out route/time data for initial GP validation. Until synchronization is confirmed, treat the prepared positions as provisional and retain the documented assumption in any experiment.
