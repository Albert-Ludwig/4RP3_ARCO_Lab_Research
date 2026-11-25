"""Launch the verified Bunker Mini navigation fixture from this directory."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter


ROOT = Path(__file__).resolve().parents[1]
NAV2_NODES = (
    ("nav2_map_server", "map_server"),
    ("nav2_planner", "planner_server"),
    ("nav2_controller", "controller_server"),
    ("nav2_behaviors", "behavior_server"),
    ("nav2_bt_navigator", "bt_navigator"),
    ("nav2_velocity_smoother", "velocity_smoother"),
)


def launch_scene(context):
    runtime = Path(os.environ.get("ROS_HOME", "/ros-data")).expanduser().resolve()
    mesh = runtime / "assets/bunker_mini/base_link.STL"
    if not mesh.is_file():
        raise RuntimeError("Bunker Mini mesh missing. Run tools/fetch_asset.py as described in README.md.")

    with (ROOT / "config/navigation_profile.json").open(encoding="utf-8") as file:
        profile = json.load(file)
    nav2_params = str(ROOT / "config/nav2_params.yaml")
    fixture = LaunchConfiguration("fixture_dir").perform(context)
    fixture_dir = Path(fixture).resolve() if fixture else None
    map_path = str(fixture_dir / "map.yaml" if fixture_dir else ROOT / "maps/map.yaml")
    world_template = fixture_dir / "world.sdf" if fixture_dir else ROOT / "worlds/world.sdf"
    template = world_template.read_text(encoding="utf-8")
    run = TemporaryDirectory(prefix="well-e-nav2-")
    world = Path(run.name) / "world.sdf"
    world.write_text(template.replace("__BUNKER_MESH_URI__", mesh.as_uri()), encoding="utf-8")

    # Keep the generated world available until launch shuts down.
    def clean_world(_context):
        run.cleanup()

    drive_share = Path(get_package_share_directory("well_e_bunker_sim"))
    actions = [
        RegisterEventHandler(OnShutdown(on_shutdown=[OpaqueFunction(function=clean_world)])),
        GroupAction([
            SetParameter(name="max_linear_m_s", value=float(profile["body_forward_max_m_s"])),
            SetParameter(name="max_angular_rad_s", value=float(profile["body_yaw_max_rad_s"])),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(drive_share / "launch/sim.launch.py")),
                launch_arguments={
                    "gui": LaunchConfiguration("gui").perform(context),
                    "world_file": str(world),
                }.items(),
            ),
        ]),
        Node(
            package="well_e_nav2_test",
            executable="truth_localization.py",
            parameters=[{"use_sim_time": True}],
        ),
    ]
    for package, name in NAV2_NODES:
        parameters = [nav2_params]
        remappings = []
        if name == "map_server":
            parameters.append({"yaml_filename": map_path})
        if name in ("controller_server", "behavior_server"):
            remappings = [("cmd_vel", "/cl/cmd_vel_nav")]
        if name == "velocity_smoother":
            remappings = [("cmd_vel", "/cl/cmd_vel_nav"), ("cmd_vel_smoothed", "/cmd_vel")]
        actions.append(Node(package=package, executable=name, name=name,
                            parameters=parameters, remappings=remappings))

    actions.append(Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="closed_loop_lifecycle",
        parameters=[{"use_sim_time": True, "autostart": True,
                     "node_names": [name for _, name in NAV2_NODES]}],
    ))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="false", description="Open the Gazebo GUI"),
        DeclareLaunchArgument("fixture_dir", default_value="",
                              description="Prepared fixture containing map.yaml and world.sdf"),
        OpaqueFunction(function=launch_scene),
    ])