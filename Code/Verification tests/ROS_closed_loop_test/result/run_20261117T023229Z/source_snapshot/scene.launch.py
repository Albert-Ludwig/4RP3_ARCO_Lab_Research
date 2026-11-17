"""Bunker fixture with Capstone-derived limits and official velocity smoothing."""
import os
from pathlib import Path
import json
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node, SetParameter
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    out = Path(os.environ["CL_OUTPUT"])
    cfg = json.loads((out/"navigation_profile.json").read_text())
    drive = Path(get_package_share_directory("well_e_bunker_sim"))
    packages = [("nav2_map_server","map_server"),("nav2_planner","planner_server"),
                ("nav2_controller","controller_server"),("nav2_behaviors","behavior_server"),
                ("nav2_bt_navigator","bt_navigator"),("nav2_velocity_smoother","velocity_smoother")]
    nodes = []
    for pkg,name in packages:
        remap = []
        if name in ("controller_server","behavior_server"):
            remap = [("cmd_vel","/cl/cmd_vel_nav")]
        if name == "velocity_smoother":
            remap = [("cmd_vel","/cl/cmd_vel_nav"),("cmd_vel_smoothed","/cmd_vel")]
        nodes.append(Node(package=pkg, executable=name, name=name,
                          parameters=[str(out/"nav2_params.yaml")], remappings=remap))
    return LaunchDescription([
        GroupAction([
            # Existing watchdog declares these parameters; override its earlier
            # 0.35 m/s and 0.6 rad/s fixture limits in this launch scope only.
            SetParameter(name="max_linear_m_s",value=float(cfg["body_forward_max_m_s"])),
            SetParameter(name="max_angular_rad_s",value=float(cfg["body_yaw_max_rad_s"])),
            IncludeLaunchDescription(PythonLaunchDescriptionSource(str(drive/"launch/sim.launch.py")),
                launch_arguments={"gui":"false","world_file":str(out/"world.sdf")}.items())
        ]),
        Node(package="well_e_nav2_test",executable="truth_localization.py",parameters=[{"use_sim_time":True}]),
        *nodes,
        Node(package="nav2_lifecycle_manager",executable="lifecycle_manager",name="closed_loop_lifecycle",
             parameters=[{"use_sim_time":True,"autostart":True,"node_names":[n for _,n in packages]}])
    ])
