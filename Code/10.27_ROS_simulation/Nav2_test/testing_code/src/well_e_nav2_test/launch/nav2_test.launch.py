import importlib.util
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml


def add(parent, tag, text=None, **attrs):
    node = ET.SubElement(parent, tag, attrs)
    if text is not None:
        node.text = str(text)
    return node


def generate_scene(drive_share, output, scenario):
    spec = importlib.util.spec_from_file_location('drive_model', drive_share/'launch/model.py')
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    runtime = Path(os.environ.get('ROS_HOME', '/ros-data'))
    config = json.loads((drive_share/'config/drive.json').read_text())
    sdf = ET.fromstring(model.make_world(runtime/'assets/bunker_mini/base_link.STL',
                                        config['steering_efficiency']))
    world = sdf.find('world')
    obstacle = scenario['obstacle']
    x, y = obstacle['center_m']
    dx, dy = obstacle['size_m']
    xmin, xmax, ymin, ymax = scenario['map_bounds_m']
    boxes = [('central_obstacle', x, y, dx, dy, obstacle['height_m']),
             ('west_wall', xmin-0.05, (ymin+ymax)/2, 0.1, ymax-ymin+0.2, 1.0),
             ('east_wall', xmax+0.05, (ymin+ymax)/2, 0.1, ymax-ymin+0.2, 1.0),
             ('south_wall', (xmin+xmax)/2, ymin-0.05, xmax-xmin, 0.1, 1.0),
             ('north_wall', (xmin+xmax)/2, ymax+0.05, xmax-xmin, 0.1, 1.0)]
    for name, bx, by, sx, sy, height in boxes:
        item = add(world, 'model', name=name)
        add(item, 'static', 'true')
        add(item, 'pose', f'{bx} {by} {height/2} 0 0 0')
        link = add(item, 'link', name='body')
        for tag in ('collision', 'visual'):
            element = add(link, tag, name=tag)
            add(add(add(element, 'geometry'), 'box'), 'size', f'{sx} {sy} {height}')
            if tag == 'visual':
                material = add(element, 'material')
                add(material, 'ambient', '0.7 0.2 0.15 1')
                add(material, 'diffuse', '0.7 0.2 0.15 1')
    ET.indent(sdf)
    (output/'world.sdf').write_text(ET.tostring(sdf, encoding='unicode'))
    resolution = scenario['resolution_m']
    width, height = round((xmax-xmin)/resolution), round((ymax-ymin)/resolution)
    pixels = bytearray()
    for row in range(height):
        py = ymax-(row+0.5)*resolution
        for col in range(width):
            px = xmin+(col+0.5)*resolution
            occupied = (col == 0 or col == width-1 or row == 0 or row == height-1 or
                        (abs(px-x) <= dx/2 and abs(py-y) <= dy/2))
            pixels.append(0 if occupied else 254)
    (output/'map.pgm').write_bytes(f'P5\n{width} {height}\n255\n'.encode()+pixels)
    (output/'map.yaml').write_text(yaml.safe_dump({
        'image': 'map.pgm', 'resolution': resolution, 'origin': [xmin, ymin, 0.0],
        'negate': 0, 'occupied_thresh': 0.65, 'free_thresh': 0.25}))


def generate_parameters(output, scenario):
    defaults = Path(get_package_share_directory('nav2_bringup'))/'params/nav2_params.yaml'
    data = yaml.safe_load(defaults.read_text())
    for node in data.values():
        if isinstance(node, dict) and 'ros__parameters' in node:
            node['ros__parameters']['use_sim_time'] = True
    controller = data['controller_server']['ros__parameters']
    controller.update(enable_stamped_cmd_vel=False, controller_frequency=20.0,
                      failure_tolerance=0.5)
    controller['progress_checker'].update(required_movement_radius=0.1, movement_time_allowance=20.0)
    controller['general_goal_checker'].update(xy_goal_tolerance=scenario['goal_xy_tolerance_m'],
        yaw_goal_tolerance=scenario['goal_yaw_tolerance_rad'])
    controller['FollowPath'] = {
        'plugin': 'nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController',
        'desired_linear_vel': 0.25, 'lookahead_dist': 0.5,
        'use_velocity_scaled_lookahead_dist': False, 'transform_tolerance': 0.3,
        'min_approach_linear_velocity': 0.04, 'approach_velocity_scaling_dist': 0.6,
        'use_collision_detection': True, 'max_allowed_time_to_collision_up_to_carrot': 1.5,
        'use_regulated_linear_velocity_scaling': True, 'use_cost_regulated_linear_velocity_scaling': True,
        'cost_scaling_dist': 0.6, 'cost_scaling_gain': 1.0, 'inflation_cost_scaling_factor': 2.8,
        'regulated_linear_scaling_min_radius': 0.9, 'regulated_linear_scaling_min_speed': 0.05,
        'use_rotate_to_heading': True, 'rotate_to_heading_angular_vel': 0.45,
        'rotate_to_heading_min_angle': 0.5, 'max_angular_accel': 0.8,
        'allow_reversing': False, 'max_robot_pose_search_dist': 10.0,
    }
    for name in ('global_costmap', 'local_costmap'):
        params = data[name][name]['ros__parameters']
        params.update(use_sim_time=True, global_frame='map', robot_base_frame='base_link',
                      transform_tolerance=0.3, resolution=scenario['resolution_m'],
                      plugins=['static_layer', 'inflation_layer'],
                      footprint=json.dumps(scenario['footprint_m']),
                      footprint_padding=scenario['footprint_padding_m'],
                      always_send_full_costmap=True, track_unknown_space=False,
                      update_frequency=10.0, publish_frequency=2.0)
        params.pop('robot_radius', None)
        params.pop('voxel_layer', None)
        params.pop('obstacle_layer', None)
        params['static_layer'] = {'plugin': 'nav2_costmap_2d::StaticLayer',
                                  'map_subscribe_transient_local': True}
        params['inflation_layer'] = {'plugin': 'nav2_costmap_2d::InflationLayer',
                                    'inflation_radius': 0.9, 'cost_scaling_factor': 2.8}
        if name == 'local_costmap':
            params.update(rolling_window=True, width=5, height=5)
    planner = data['planner_server']['ros__parameters']
    planner['GridBased'].update(tolerance=0.1, use_astar=True, allow_unknown=False)
    behavior = data['behavior_server']['ros__parameters']
    behavior.update(enable_stamped_cmd_vel=False, global_frame='map', local_frame='map',
                    robot_base_frame='base_link', transform_tolerance=0.3,
                    max_rotational_vel=0.5, min_rotational_vel=0.15, rotational_acc_lim=0.8)
    navigator = data['bt_navigator']['ros__parameters']
    navigator.update(global_frame='map', robot_base_frame='base_link',
                     navigators=['navigate_to_pose'], transform_tolerance=0.3,
                     default_server_timeout=1000, bt_loop_duration=20)
    data['map_server'] = {'ros__parameters': {'use_sim_time': True, 'yaml_filename': str(output/'map.yaml')}}
    used = {'map_server', 'planner_server', 'controller_server', 'behavior_server',
            'bt_navigator', 'global_costmap', 'local_costmap'}
    data = {name: parameters for name, parameters in data.items() if name in used}
    (output/'nav2_params.yaml').write_text(yaml.safe_dump(data, sort_keys=False))


def generate_launch_description():
    share = Path(get_package_share_directory('well_e_nav2_test'))
    drive = Path(get_package_share_directory('well_e_bunker_sim'))
    scenario = json.loads((share/'config/scenario.json').read_text())
    output = Path(os.environ.get('ROS_HOME', '/ros-data'))/'generated/Nav2_test'
    output.mkdir(parents=True, exist_ok=True)
    generate_scene(drive, output, scenario)
    generate_parameters(output, scenario)
    params = str(output/'nav2_params.yaml')
    nodes = [('nav2_map_server', 'map_server'), ('nav2_planner', 'planner_server'),
             ('nav2_controller', 'controller_server'), ('nav2_behaviors', 'behavior_server'),
             ('nav2_bt_navigator', 'bt_navigator')]
    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='false'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(drive/'launch/sim.launch.py')),
            launch_arguments={'gui': LaunchConfiguration('gui'),
                              'world_file': str(output/'world.sdf')}.items()),
        Node(package='well_e_nav2_test', executable='truth_localization.py',
             parameters=[{'use_sim_time': True}], output='screen'),
        *[Node(package=package, executable=name, name=name, parameters=[params], output='screen')
          for package, name in nodes],
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_nav2_test', output='screen',
             parameters=[{'use_sim_time': True, 'autostart': True,
                          'node_names': [name for _, name in nodes]}]),
    ])
