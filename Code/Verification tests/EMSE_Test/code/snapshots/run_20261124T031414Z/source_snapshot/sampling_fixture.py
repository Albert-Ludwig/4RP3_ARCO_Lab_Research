"""Build the origin Home fixture while retaining the sensing coordinates."""
from pathlib import Path
import json
import xml.etree.ElementTree as ET


def home_inputs(reference: Path, home_m: list, clearance_m: float) -> dict[str, bytes]:
    """Replace the reference Home and move the two lower arena boundaries."""
    common=reference/'common'
    original=json.loads((common/'candidates.json').read_text())
    candidates=[home_m,*original[1:]]
    if len({tuple(xy) for xy in candidates})!=len(candidates):
        raise ValueError('Home must have its own location')
    scene=json.loads((common/'scene.json').read_text())
    xmin,xmax,ymin,ymax=scene['map_bounds_m']
    x,y=home_m[0]+xmin,home_m[1]+ymin
    resolution=scene['resolution_m']
    if clearance_m<1.15 or abs(clearance_m/resolution-round(clearance_m/resolution))>1e-8:
        raise ValueError('Origin Home clearance must span whole map cells and exceed inflation')
    nav_xmin,nav_ymin=xmin-clearance_m,ymin-clearance_m

    tree=ET.fromstring((common/'world.sdf').read_bytes())
    model=tree.find("./world/model[@name='bunker_mini']/pose")
    old_pose=model.text.split()
    model.text=' '.join([str(x),str(y),*old_pose[2:]])
    span_x,span_y=xmax-nav_xmin,ymax-nav_ymin
    mid_x,mid_y=(nav_xmin+xmax)/2,(nav_ymin+ymax)/2
    walls={
        'west_wall':(nav_xmin-.05,mid_y,.1,span_y+.2),
        'east_wall':(xmax+.05,mid_y,.1,span_y+.2),
        'south_wall':(mid_x,nav_ymin-.05,span_x,.1),
        'north_wall':(mid_x,ymax+.05,span_x,.1),
    }
    for name,(bx,by,sx,sy) in walls.items():
        wall=tree.find(f"./world/model[@name='{name}']")
        wall.find('pose').text=f'{bx} {by} 0.5 0 0 0'
        for size in wall.findall('./link/*/geometry/box/size'):
            size.text=f'{sx} {sy} 1.0'

    width,height=round(span_x/resolution),round(span_y/resolution)
    row=b'\x00'+(b'\xfe'*(width-2))+b'\x00'
    border=bytes(width)
    pgm=f'P5\n{width} {height}\n255\n'.encode()+border+row*(height-2)+border
    map_yaml=(f'image: map.pgm\nresolution: {resolution}\norigin: [{nav_xmin}, {nav_ymin}, 0.0]\n'
              'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n')
    scene['start']=[x,y,float(old_pose[5])]
    scene['map_bounds_m']=[nav_xmin,xmax,nav_ymin,ymax]
    scene['field_origin_in_map_m']=[xmin,ymin]
    field=json.loads((common/'field.json').read_text())
    # The same random function can be evaluated in the navigable clearance.
    field['working_origin_m']=[-clearance_m,-clearance_m]
    def encoded(obj):
        return (json.dumps(obj,indent=2)+'\n').encode('utf8')
    return {
        'candidates.json': encoded(candidates),
        'scene.json': encoded(scene),
        'world.sdf': ET.tostring(tree,encoding='utf8')+b'\n',
        'map.pgm': pgm,
        'map.yaml': map_yaml.encode(),
        'field.json': encoded(field),
    }
