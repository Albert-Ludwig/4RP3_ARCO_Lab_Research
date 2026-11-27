# Original Data

WELL-E collected the lawnmower runs at the baseball field. The HOBO workbooks record air temperature in EST (UTC-05:00). The GPS counts below are the raw messages on `/calian_gnss/gps` (`sensor_msgs/msg/NavSatFix`). HOBO counts include rows outside the GPS recording window.

| Collection date | Raw files | HOBO temperature records | HOBO interval | GPS messages |
| --- | --- | ---: | ---: | ---: |
| 2026-11-03 | [Original workbook, GPS-only bag, and route image](2026-11-03/) | 1,100 | 1 s | 981 |
| 2026-11-17 | [Day 1, Tuesday](2026-11-17-johnson-baseball/Day_1_Tuesday/) | 1,386 | 1 s | 1,362 |
| 2026-11-18 | [Day 2, Wednesday](2026-11-17-johnson-baseball/Day_2_Wednesday/) | 24 | 60 s | 1,420 |
| 2026-11-19 | [Day 3, Thursday](2026-11-17-johnson-baseball/Day_3_Thursday/) | 1,366 | 1 s | 1,371 |
| 2026-11-20 | [Day 4, Friday](2026-11-17-johnson-baseball/Day_4_Friday/) | 1,404 | 1 s | 1,359 |
| 2026-11-21 | [Day 5, Saturday](2026-11-17-johnson-baseball/Day_5_Saturday/) | 1,436 | 1 s | 1,362 |
| **Total** | | **6,716** | | **7,855** |

The 3 November GPS-only bag retains all 981 original GPS messages, with their serialized bytes, IDs, and receipt timestamps. It is the retained subset of the original multi-topic bag. See the [retention notes](2026-11-03/README.md) and [retention manifest](2026-11-03/retention_manifest.json).

Each day from 17 to 21 November contains the original HOBO workbook and the received ROS 2 bag `.tar.gz` as supplied in the outer ZIP. The 17 November bag is the collector's filtered version. The collector's [README](2026-11-17-johnson-baseball/README.md) describes the route, camera data, and 18 November's one-minute HOBO setting.
