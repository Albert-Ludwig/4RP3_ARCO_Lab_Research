# Interpolated Data

This dataset contains measured HOBO temperatures with GPS positions aligned to the temperature timestamps. It prepares `(x, y, t, T)` observations for GP modelling. [Original Data](<../Original Data/README.md>) contains the acquisition files and raw record counts.

| Collection | Temperature records | Matched | Before GPS coverage | After GPS coverage |
| --- | ---: | ---: | ---: | ---: |
| [2026-11-03](result/gp_samples.csv) | 1,100 | 980 | 63 | 57 |
| [2026-11-17 to 2026-11-21](result/2026-11-17-johnson-baseball/README.md) | 5,616 | 5,468 | 14 | 134 |
| **Total** | **6,716** | **6,448** | **77** | **191** |

The 18 November workbook has a 60 s logging interval. The processed output contains its 24 measured temperatures at the original timestamps.

## Interpolation method

1. Read the HOBO temperature and `Date-Time (EST)` columns from the workbook's `Data` sheet. Convert fixed EST (UTC-05:00) to integer Unix nanoseconds in UTC. Preserve the logger timestamp, sample ID, Excel row, and temperature value. Read `/calian_gnss/gps` as `sensor_msgs/msg/NavSatFix` from each ROS bag. Use the GPS message **header** timestamp for alignment and keep the bag receipt timestamp for auditing.
2. Convert WGS84 latitude, longitude, and altitude to Earth-centred Cartesian coordinates (ECEF), then rotate to local east, north, and up (ENU). All runs share the first valid 3 November GPS fix as their spatial origin, at latitude `43.265474` degrees north, longitude `-79.928666` degrees, and altitude `45.3434 m`. The `x_east_m` and `y_north_m` columns are in metres. The coordinates can be overlaid across days. Registration to ROS `map` is a separate step.
3. Find the consecutive GPS fixes with header times `t0 <= t <= t1` around each HOBO timestamp `t`. For valid fixes and a bracket of at most 2 s, set `alpha = (t - t0) / (t1 - t0)` and compute `p(t) = (1 - alpha) * p0 + alpha * p1` in ENU. An exact timestamp match uses that valid fix directly. A fix must have finite latitude, longitude, and altitude within the coordinate bounds, with a nonnegative NavSatFix status. Positions are omitted when the temperature timestamp is outside GPS coverage, an endpoint is invalid, or the GPS interval exceeds 2 s. The 2 s threshold is a processing choice.
4. Keep all original temperature rows in the alignment audit, along with match or exclusion reason, bracketing GPS IDs and times, interpolation fraction, and gap. The GP CSV contains only matched rows with a measured temperature. The temperature values are copied without resampling, smoothing, or detrending.

### Time reference and limitations

- For each run, `t_s` is elapsed seconds since that run's first valid GPS fix. The 3 November time origin is `2026-11-03T19:24:15.325545540Z`. The combined November 17-21 file also contains `run_date` and `t_campaign_s` measured from the first valid GPS fix in the five-day campaign. Use `timestamp_utc_ns` or `t_campaign_s` when comparing sample times across days. Keep Unix nanosecond values as integers.
- The alignment applies zero extra clock offset. No common synchronization event was identified. A positive `--temperature-offset-s` adds time to the HOBO timestamp in UTC. The roughly 0.326 s median nearest-fix separation on 3 November measures sampling phase. Every row keeps `clock_alignment_verified=False`.
- The GPS antenna position is used as the temperature sensor position. The mounting offset and temperature sensor response delay are unmeasured. GPS status 2 indicates GBAS. RTK fixed status and centimetre accuracy require separate evidence. GPS coverage describes the driven path. Occupancy and traversability require separate mapping.
- The 3 November GPS track spans approximately 97.51 m east by 63.55 m north. The matched local interval was 14:24:16-14:40:35 EST, and the original temperatures ranged from 26.08 to 27.75 degrees Celsius. The individual [campaign summaries](result/2026-11-17-johnson-baseball/summary.csv) record the later runs' extents, gaps, and temperature ranges.

## Outputs

| File | Contents |
| --- | --- |
| [3 November GP samples](result/gp_samples.csv) | 980 matched observations |
| [17-21 November GP samples](result/2026-11-17-johnson-baseball/gp_samples.csv) | 5,468 matched observations with run and campaign times |
| `temperature_alignment.csv` | All original temperature rows and alignment decisions. Blank positions mark excluded records. |
| `gps.csv` | GPS timestamps, geodetic and ENU positions, decoded status, covariance and validity. Covariance fields `cov_0_m2` through `cov_8_m2` preserve row-major ENU order. |
| `summary.json` and `manifest.json` | Run counts, assumptions, clock-offset sensitivity, source metadata, and input/output checksums. |
| [Campaign source archive map](result/2026-11-17-johnson-baseball/source_archive.json) | Outer ZIP checksum, retained original files, and nested bag member names. |
| [Campaign verification](result/2026-11-17-johnson-baseball/verification.json) | Shared ENU origin, verified brackets, reproduced positions and unchanged measured temperatures. |

## Reproduce

Run [prepare_data.py](testing_code/prepare_data.py) from the repository root with Python, NumPy, and openpyxl. The default ROS 2 decoder uses `rclpy` and `sensor_msgs` in the existing `bunker-jazzy` Docker container. A ROS-sourced Linux environment can instead pass `--ros-python /usr/bin/python3`. Only selected GPS messages go to the decoder.

For a November 17-21 run, extract its `.tar.gz` into a working folder, place that day's HOBO workbook in the same folder, and pass `--source` and `--output` using quoted paths. Pass `--enu-reference-summary "Data/Interpolated Data/result/summary.json"` to match the 3 November spatial origin. Choose a separate output folder when reproducing results. The legacy run's defaults point to `Data/Original Data/2026-11-03` and `Data/Interpolated Data/result`.

The alignment script reads source data without modifying it. Its hashes establish which files and GPS messages were used. Verification after migration confirmed that the two GP input CSVs and the retained 3 November GPS bag kept their original bytes.
