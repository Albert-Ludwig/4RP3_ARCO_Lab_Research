"""Reuse the tested Bunker Mini physics and Nav2 configuration, in a fresh world."""
import importlib.util
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    online=Path(os.environ['POLICY_ONLINE']);output=Path(os.environ['POLICY_GENERATED'])
    config=json.loads((online/'config.json').read_text());scene=config['scene']
    output.mkdir(parents=True,exist_ok=True)
    share=Path(get_package_share_directory('well_e_nav2_test'))
    drive=Path(get_package_share_directory('well_e_bunker_sim'))
    spec=importlib.util.spec_from_file_location('reused_nav2_launch',share/'launch/nav2_test.launch.py')
    reused=importlib.util.module_from_spec(spec);spec.loader.exec_module(reused)
    reused.generate_scene(drive,output,scene)
    tree=ET.parse(output/'world.sdf');world=tree.getroot().find('world')
    robot=world.find("model[@name='bunker_mini']")
    if robot is None: raise RuntimeError('Expected Bunker Mini physics model missing')
    robot.find('pose').text=f"{scene['start'][0]} {scene['start'][1]} 0.28 0 0 {scene['start'][2]}"
    if config['mode']=='real_recorded':
        # A known-map static 2D proxy, NOT a reconstruction of real 3D terrain.
        import yaml
        import numpy as np
        from PIL import Image
        meta=yaml.safe_load((online/config['map_yaml']).read_text())
        if abs(meta['origin'][2])>1e-9: raise ValueError('Rotated occupancy-map origins require explicit coordinate conversion')
        raster=np.asarray(Image.open(online/meta['image']).convert('L'))/255.
        probability=raster if meta.get('negate',0) else 1-raster
        occupied=probability>=meta['occupied_thresh'];res=meta['resolution'];ox,oy,_=meta['origin']
        for item in list(world.findall('model')):
            if item.get('name') in ('central_obstacle','west_wall','east_wall','south_wall','north_wall'):world.remove(item)
        counter=0
        for row,values in enumerate(occupied):
            edges=np.diff(np.r_[False,values,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
            for a,b in zip(starts,ends):
                item=reused.add(world,'model',name=f'occupancy_{counter}');counter+=1
                reused.add(item,'static','true');reused.add(item,'pose',f'{ox+(a+b)*res/2} {oy+(len(occupied)-row-.5)*res} .5 0 0 0')
                link=reused.add(item,'link',name='body')
                for tag in ('collision','visual'):
                    el=reused.add(link,tag,name=tag);reused.add(reused.add(reused.add(el,'geometry'),'box'),'size',f'{(b-a)*res} {res} 1')
        from shutil import copy2
        copy2(online/config['map_yaml'],output/'map.yaml');copy2(online/meta['image'],output/meta['image'])
    tree.write(output/'world.sdf',encoding='unicode')
    reused.generate_parameters(output,scene)
    params=str(output/'nav2_params.yaml')
    nodes=[('nav2_map_server','map_server'),('nav2_planner','planner_server'),
           ('nav2_controller','controller_server'),('nav2_behaviors','behavior_server'),('nav2_bt_navigator','bt_navigator')]
    return LaunchDescription([
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(drive/'launch/sim.launch.py')),
            launch_arguments={'gui':'false','world_file':str(output/'world.sdf')}.items()),
        Node(package='well_e_nav2_test',executable='truth_localization.py',parameters=[{'use_sim_time':True}],output='screen'),
        *[Node(package=package,executable=name,name=name,parameters=[params],output='screen') for package,name in nodes],
        Node(package='nav2_lifecycle_manager',executable='lifecycle_manager',name='lifecycle_manager_policy',
             parameters=[{'use_sim_time':True,'autostart':True,'node_names':[n for _,n in nodes]}],output='screen')])
