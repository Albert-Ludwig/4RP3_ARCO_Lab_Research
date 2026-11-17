# Retained real-data inputs

The full ROS recording was reduced to the exact `/calian_gnss/gps` input used by the current temperature/GPS alignment pipeline. `gps_only.db3` retains all 981 GPS messages with their original serialized bytes, message IDs, receipt timestamps and topic metadata. The temperature Excel and collection route image remain unchanged.

The full bag was deleted locally after verifying its original SHA-256 and checking the GPS subset byte for byte. Camera, IMU, VIO and all other topics require downloading the original recording again if needed. `metadata.full_recording.yaml` preserves the original topic inventory; `metadata.yaml` now describes only the GPS subset.

Aligned data, alignment audit files and the trained kernel are unchanged. Existing result manifests describe the historical full recording; `retention_manifest.json` records the subset and deletion checks. The retained raw GPS and temperature workbook allow alignment to be regenerated without the discarded topics. No clock correction or temperature interpolation was introduced by this cleanup.
