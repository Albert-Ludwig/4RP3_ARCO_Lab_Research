# Data

| Folder | Contents |
| --- | --- |
| [real_data](real_data/) | Retained original HOBO temperature workbook, all original GPS messages in `gps_only.db3`, route image and acquisition metadata. |
| [interpolated_data](interpolated_data/) | Temperature/GPS alignment outputs, processing code and report. The GP input is [gp_samples.csv](interpolated_data/result/gp_samples.csv). |

Only GPS positions are interpolated to temperature timestamps; temperature values remain measured values. Moving these directories does not change data values or trained kernel parameters.

Historical saved manifests/results retain their original path names: `Data/Real_spatialtemporal_data` now corresponds to `Data/real_data`, and `Data/Real_Data_Integration` to `Data/interpolated_data`. Original checksums remain unchanged. The earlier full ROS bag was already reduced to the retained GPS subset; this reorganization does not restore discarded topics.
