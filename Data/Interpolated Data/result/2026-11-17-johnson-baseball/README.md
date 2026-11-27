# WELL-E baseball field temperature and GPS alignment, 17-21 November 2026

Processed five lawnmower runs supplied in `2026-11-17-johnson-baseball.zip`. The aligned dataset contains **5,468 of 5,616 measured temperatures**. Each matched temperature retains its original value and receives a GPS-derived position.

| Run date | HOBO interval (s) | HOBO records | Valid GPS / total | Matched | Unmatched |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-11-17 | 1 | 1,386 | 1,362 / 1,362 | 1,360 | 26 |
| 2026-11-18 | 60 | 24 | 1,420 / 1,420 | 24 | 0 |
| 2026-11-19 | 1 | 1,366 | 1,371 / 1,371 | 1,365 | 1 |
| 2026-11-20 | 1 | 1,404 | 1,359 / 1,359 | 1,358 | 46 |
| 2026-11-21 | 1 | 1,436 | 1,362 / 1,362 | 1,361 | 75 |

November 18 has one measured temperature per minute. It contributes 24 observations at that original cadence. Across the campaign, unmatched rows fall before or after GPS coverage. All original temperature rows and exclusion reasons remain in the daily audit files.

## Alignment and coordinates

HOBO timestamps were converted from fixed EST (UTC-05:00) to UTC and matched to the header timestamps of `/calian_gnss/gps` (`sensor_msgs/msg/NavSatFix`). The pipeline converts WGS84 to local ENU and linearly interpolates position between consecutive valid fixes, with a maximum gap of 2 s. Temperature values are kept as measured. The time alignment assumes zero additional HOBO-to-GPS offset. Physical synchronization and temperature sensor response delay remain unverified.

All runs use the 3 November WGS84 ENU origin `(43.265474, -79.928666, 45.3434 m)`, so the coordinates can be overlaid directly. The combined CSV preserves integer `timestamp_utc_ns`, `t_s` since each run's first valid GPS fix, and `t_campaign_s` since the first run's first valid GPS fix. Use the absolute timestamp or campaign time if comparing observation times across days.

## Files

- [gp_samples.csv](gp_samples.csv) contains the combined observations, with run dates and IDs.
- [summary.csv](summary.csv) lists run-level counts, timing gaps, temperature ranges, and ENU extents.
- Each date folder contains `gp_samples.csv`, `temperature_alignment.csv`, `gps.csv`, `summary.json`, and `manifest.json`.
- [source_archive.json](source_archive.json) maps the original ZIP and nested TAR members to the source hashes. Daily manifests capture the temporary extraction paths used during processing.
- [verification.json](verification.json) records the position and temperature checks. Regenerating the 3 November observations produced a byte-identical GP input CSV.

To reproduce a run after extracting the corresponding workbook and bag, run [prepare_data.py](../../testing_code/prepare_data.py) with `--source <day-folder> --output <output-folder> --enu-reference-summary "Data/Interpolated Data/result/summary.json"` from the repository root.

Processing ends at temperature/GPS alignment. The datasets can be split by run for subsequent training or independent evaluation.
