"""Record runtime versions without importing heavyweight ML packages into ROS."""
from importlib.metadata import version,PackageNotFoundError
import os
import platform
import subprocess
from nav2_policy.prepare import ROOT
from nav2_policy.storage import save_json
from nav2_policy.storage import digest
from ament_index_python.packages import get_package_share_directory,get_package_prefix

packages={}
for name in ('numpy','scipy','torch','gpytorch'):
    try:packages[name]=version(name)
    except PackageNotFoundError:packages[name]='not installed'
result=subprocess.run(['dpkg-query','-W','-f=${Package} ${Version}\n','ros-jazzy-nav2-bringup','ros-jazzy-nav2-msgs','ros-jazzy-gz-sim-vendor'],capture_output=True,text=True)
data=dict(python=platform.python_version(),platform=platform.platform(),ros_distro=os.environ.get('ROS_DISTRO'),
          packages=packages,system_packages=result.stdout.strip().splitlines(),
          docker_image_id=os.environ.get('DOCKER_IMAGE_ID'),
          note='Image ID supplied by host docker inspect; null if not supplied. No hard-coded version assumptions.')
dependencies={}
for package in ('well_e_bunker_sim','well_e_nav2_test'):
    share=__import__('pathlib').Path(get_package_share_directory(package))
    installed=__import__('pathlib').Path(get_package_prefix(package))/'lib'/package
    for base in (share,installed):
        for path in base.rglob('*'):
            if path.is_file() and path.suffix in ('.py','.json','.yaml'):
                dependencies[str(path)]=digest(path)
save_json(ROOT/'result/runtime_dependencies.json',dependencies)
save_json(ROOT/'result/environment.json',data);print(data)
