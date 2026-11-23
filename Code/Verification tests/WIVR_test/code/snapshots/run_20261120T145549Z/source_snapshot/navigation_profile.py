"""Reuse supported Capstone Nav2 overrides with Jazzy and Bunker-specific geometry."""
from pathlib import Path
import json, copy, math
import xml.etree.ElementTree as ET
import yaml
from ament_index_python.packages import get_package_share_directory

CODE = Path(__file__).resolve().parent

def profile():
    return json.loads((CODE / "navigation_profile.json").read_text())

def merge(base, patch):
    out = copy.deepcopy(base)
    for key, value in patch.items():
        out[key] = merge(out.get(key, {}), value) if isinstance(value, dict) else value
    return out

def apply(output, scenario):
    settings = profile()
    official = settings["manufacturer_limits"]["maximum_linear_speed_m_s"]
    assert 0 < settings["nominal_speed_m_s"] <= settings["body_forward_max_m_s"] <= official
    assert settings["body_yaw_max_rad_s"] <= .6 and settings["track_max_m_s"] <= official
    reference = yaml.safe_load((CODE / "capstone_nav2_reference.yaml").read_text())
    defaults = yaml.safe_load((Path(get_package_share_directory("nav2_bringup")) / "params/nav2_params.yaml").read_text())
    data = yaml.safe_load((output / "nav2_params.yaml").read_text())
    controller = data["controller_server"]["ros__parameters"]
    source = reference["controller_server"]["ros__parameters"]
    # Lyrical's TrajectoryValidator and optimal-trajectory publication aren't
    # present in this installed Jazzy default; do not pretend they are active.
    supported = {k:v for k,v in source["FollowPath"].items()
                 if k not in ("TrajectoryValidator", "publish_optimal_trajectory")}
    if settings["controller"] == "mppi":
        controller["FollowPath"] = merge(defaults["controller_server"]["ros__parameters"]["FollowPath"], supported)
        controller["FollowPath"]["wz_max"] = settings["body_yaw_max_rad_s"]
        controller["FollowPath"]["az_max"] = settings["angular_accel_rad_s2"]
    else:
        # Keep the already exercised Bunker RPP. The donor MPPI parameters did
        # not meet the speed-transfer check, so they are not silently promoted.
        controller["FollowPath"].update(desired_linear_vel=settings["nominal_speed_m_s"],
            use_velocity_scaled_lookahead_dist=True,lookahead_time=1.,
            min_lookahead_dist=.5,max_lookahead_dist=1.5,
            max_angular_accel=settings["angular_accel_rad_s2"])
    controller.update(controller_frequency=source["controller_frequency"], enable_stamped_cmd_vel=False)
    controller["progress_checker"] = merge(controller["progress_checker"], source["progress_checker"])
    controller["general_goal_checker"] = merge(controller["general_goal_checker"], source["general_goal_checker"])
    for name in ("local_costmap", "global_costmap"):
        target = data[name][name]["ros__parameters"]
        src = reference[name][name]["ros__parameters"]
        for k in ("update_frequency", "publish_frequency", "width", "height"):
            if k in src: target[k] = src[k]
        target["inflation_layer"]["inflation_radius"] = src["inflation_layer"]["inflation_radius"]
        # Real Bunker polygon + padding, rather than the warehouse's square chassis.
        if name == "global_costmap":
            pad = scenario["footprint_padding_m"]
            radius = max(math.hypot(abs(x)+pad, abs(y)+pad) for x,y in scenario["footprint_m"])
            target.pop("footprint", None)
            target.update(robot_radius=math.ceil(radius*100)/100, footprint_padding=0.)
    smoother = merge(defaults["velocity_smoother"]["ros__parameters"],
                     reference["velocity_smoother"]["ros__parameters"])
    smoother["max_velocity"] = [settings["body_forward_max_m_s"],0.,settings["body_yaw_max_rad_s"]]
    smoother["min_velocity"] = [-settings["body_reverse_max_m_s"],0.,-settings["body_yaw_max_rad_s"]]
    smoother["max_accel"][2] = settings["angular_accel_rad_s2"]
    smoother["max_decel"][2] = -settings["angular_accel_rad_s2"]
    smoother.update(use_sim_time=True, enable_stamped_cmd_vel=False, odom_topic="/odom")
    data["velocity_smoother"] = {"ros__parameters": smoother}
    behavior = data["behavior_server"]["ros__parameters"]
    behavior.update(max_rotational_vel=settings["body_yaw_max_rad_s"], rotational_acc_lim=settings["angular_accel_rad_s2"])
    (output / "nav2_params.yaml").write_text(yaml.safe_dump(data,sort_keys=False))
    tree = ET.parse(output / "world.sdf")
    model = tree.getroot().find("world/model[@name='bunker_mini']")
    track_count = 0
    for plug in model.findall("plugin"):
        kind = plug.attrib.get("filename","")
        if "track-controller-system" in kind:
            plug.find("min_velocity").text = str(-settings["track_max_m_s"])
            plug.find("max_velocity").text = str(settings["track_max_m_s"])
            track_count += 1
        if "tracked-vehicle-system" in kind:
            plug.find("linear_velocity/min_velocity").text = str(-settings["body_reverse_max_m_s"])
            plug.find("linear_velocity/max_velocity").text = str(settings["body_forward_max_m_s"])
            plug.find("angular_velocity/min_velocity").text = str(-settings["body_yaw_max_rad_s"])
            plug.find("angular_velocity/max_velocity").text = str(settings["body_yaw_max_rad_s"])
    assert track_count == 2
    tree.write(output / "world.sdf")
    (output / "navigation_profile.json").write_text(json.dumps(settings,indent=2))
    (output / "nav2_adaptation.json").write_text(json.dumps(dict(
        source=settings["source"],controller=controller["FollowPath"]["plugin"],
        omitted_source_keys=["TrajectoryValidator","publish_optimal_trajectory"],
        source_components_not_launched=["AMCL","lidar obstacle layers","Collision Monitor","waypoint dwell"],
        reason="Current flat component fixture uses ideal localization and static map only; no lidar topic is supplied.",
        global_radius_m=data["global_costmap"]["global_costmap"]["ros__parameters"]["robot_radius"],
        chain="controller/behaviors -> /cl/cmd_vel_nav -> velocity_smoother -> /cmd_vel -> bounded watchdog -> /simulation/cmd_vel -> Gazebo tracks",
        footprint=scenario["footprint_m"],footprint_padding_m=scenario["footprint_padding_m"]
    ),indent=2))
