"""Prepare an existing ROS sampling fixture at a different Home waypoint."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET


def home_inputs(reference: Path, home_m: list) -> dict:
    """Move index 0 and the spawn together, preserving graph geometry."""
    common=reference/'common'
    original=json.loads((common/'candidates.json').read_text())
    # Swap indices so the two runs use the same physical waypoint set.
    index=original.index(home_m)
    reordered=original.copy()
    reordered[0],reordered[index]=reordered[index],reordered[0]
    old_to_new=list(range(len(original)))
    old_to_new[0],old_to_new[index]=index,0
    graph=json.loads((reference/'latest_gp/graph.json').read_text())
    for edge in graph:
        edge['from']=old_to_new[edge['from']]
        edge['to']=old_to_new[edge['to']]
    graph.sort(key=lambda edge:(edge['from'],edge['to']))

    # Field coordinates are relative to the map's lower-left corner.
    scene=json.loads((common/'scene.json').read_text())
    xmin,_,ymin,_=scene['map_bounds_m']
    x,y=home_m[0]+xmin,home_m[1]+ymin
    world=ET.fromstring((common/'world.sdf').read_text())
    pose=world.find("./world/model[@name='bunker_mini']/pose")
    if pose is None or pose.text is None:
        raise ValueError('Missing Bunker Mini spawn pose')
    old_pose=pose.text.split()
    pose.text=' '.join([str(x),str(y),*old_pose[2:]])
    scene['start']=[x,y,float(old_pose[5])]
    return {
        'candidates.json': json.dumps(reordered,indent=2)+'\n',
        'scene.json': json.dumps(scene,indent=2)+'\n',
        'world.sdf': ET.tostring(world,encoding='unicode')+'\n',
        'graph.json': json.dumps(graph,indent=2)+'\n',
    }
