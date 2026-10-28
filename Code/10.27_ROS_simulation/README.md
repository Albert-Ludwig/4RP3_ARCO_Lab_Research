# 10.27 — ROS 2 Simulation

The work is organized into three stages. Implemented stages contain `testing_code/`, `result/`, `README.md` and `test_report.ipynb`. `Integration_test` is reserved only; it has not started and contains no experiment report or results.

| Stage | Purpose | Current status |
|---|---|---|
| [BunkerMini_drive_test](BunkerMini_drive_test/README.md) | Mini model, tracked drive, velocity commands, odom/TF and command timeout | Executed; see the stage report |
| [Nav2_test](Nav2_test/README.md) | Point-to-point navigation around one known static obstacle | Passed with ground-truth localization; live perception and additional cases pending |
| [Integration_test](Integration_test/README.md) | GP/GRG → Nav2 → measurement → GP update and replanning | Not implemented or run |

Experiment results and captured launch logs are saved under each stage's `result/`. Gazebo asset caches, build artifacts and raw ROS runtime logs remain in `C:\Ubuntu\ROS`. The existing 10.16 and 10.18 experiments and GP/GRG implementations are unchanged.

Windows entry point: [Start-BunkerMini.ps1](BunkerMini_drive_test/testing_code/Start-BunkerMini.ps1). The existing `C:\Ubuntu\Bunker-Mini-Sim.cmd` and `Test-Bunker-Mini.cmd` shortcuts use the updated paths.

Generate the implemented-stage READMEs and report notebooks from saved artifacts, without rerunning ROS. The reserved Integration stage is skipped:

```powershell
& C:\Ubuntu\ROS\reporting-venv\Scripts\python.exe .\BunkerMini_drive_test\testing_code\tools\report.py
```

The reporting environment is at `C:\Ubuntu\ROS\reporting-venv`; dependencies are listed in [requirements-report.txt](BunkerMini_drive_test/testing_code/tools/requirements-report.txt). The drive and Nav2 notebooks replay saved execution records. Integration is excluded from automatic report generation; GP/GRG has not been integrated with ROS yet.

Nav2 Windows shortcuts: `C:\Ubuntu\Test-Nav2.cmd` runs the automatic obstacle test with a GUI; `C:\Ubuntu\Nav2-Sim.cmd` opens the field. The first Nav2 case uses a known obstacle map and simulator ground-truth localization, not live obstacle sensing or AMCL.

Automatic Nav2 tests record ROS topics with rosbag2 under `Nav2_test/result/run_.../rosbag/` (MCAP). CSV/JSON and figures remain analysis outputs. Each new bag is read back and replayed on isolated ROS domain 73; the run report includes its path, topic counts and replay commands.
