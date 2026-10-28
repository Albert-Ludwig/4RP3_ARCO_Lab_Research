import hashlib
import importlib.util
import json
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory('well_e_bunker_sim'))
    spec = importlib.util.spec_from_file_location('bunker_model', share / 'launch/model.py')
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    runtime = Path(os.environ.get('ROS_HOME', '/ros-data'))
    mesh = runtime / 'assets/bunker_mini/base_link.STL'
    drive_config = json.loads((share / 'config/drive.json').read_text(encoding='utf-8'))
    expected = 'af09e88d1b29b3a436083443f6030529a8e95b07a91ed23975e58799cf250f52'
    if not mesh.exists() or hashlib.sha256(mesh.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Run testing_code/tools/fetch_asset.py before launching')
    output = runtime / 'generated/well_e_bunker_sim'
    output.mkdir(parents=True, exist_ok=True)
    world = output / 'world.sdf'
    world.write_text(model.make_world(mesh, drive_config['steering_efficiency']), encoding='utf-8')
    (output / 'robot.urdf').write_text(model.make_description(mesh), encoding='utf-8')
    server = ExecuteProcess(cmd=['gz', 'sim', '-r', '-s', '-v', '3',
                                 LaunchConfiguration('world_file')], output='screen')
    gui = ExecuteProcess(cmd=['gz', 'sim', '-g'],
                         condition=IfCondition(LaunchConfiguration('gui')), output='screen')
    bridge = Node(package='ros_gz_bridge', executable='parameter_bridge',
                  parameters=[{'config_file': str(share / 'config/bridge.yaml')}], output='screen')
    watchdog = Node(package='well_e_bunker_sim', executable='cmd_watchdog.py',
                    parameters=[{'use_sim_time': True}], output='screen')
    odometry = Node(package='well_e_bunker_sim', executable='odometry_adapter.py',
                    parameters=[{'use_sim_time': True, 'yaw_scale': drive_config['odometry_yaw_scale']}],
                    output='screen')
    critical_handlers = [RegisterEventHandler(OnProcessExit(target_action=process, on_exit=[
        EmitEvent(event=Shutdown(reason='Required simulation process exited'))]))
        for process in (server, bridge, watchdog, odometry, gui)]
    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='false'),
        DeclareLaunchArgument('world_file', default_value=str(world)),
        server,
        *critical_handlers,
        gui,
        bridge,
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': model.make_description(mesh), 'use_sim_time': True}]),
        watchdog,
        odometry,
    ])
