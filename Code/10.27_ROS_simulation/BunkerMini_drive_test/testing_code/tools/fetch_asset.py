#!/usr/bin/env python3
"""Fetch pinned upstream Mini mesh into ROS runtime storage, outside Git."""
import hashlib
import os
from pathlib import Path
import urllib.request

COMMIT = '27633a956c845903ee630538afeb17fe70afdd84'
SHA256 = 'af09e88d1b29b3a436083443f6030529a8e95b07a91ed23975e58799cf250f52'
URL = ('https://raw.githubusercontent.com/agilexrobotics/ugv_gazebo_sim/'
       f'{COMMIT}/bunker/bunker_mini/meshes/base_link.STL')


def fetch():
    target = Path(os.environ.get('ROS_HOME', '/ros-data')) / 'assets/bunker_mini/base_link.STL'
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != SHA256:
            raise RuntimeError('Downloaded mesh does not match pinned checksum')
        target.write_bytes(data)
    if hashlib.sha256(target.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError(f'Existing mesh has unexpected contents: {target}')
    print(f'Verified manufacturer Mini mesh: {target}')
    return target


if __name__ == '__main__':
    fetch()
