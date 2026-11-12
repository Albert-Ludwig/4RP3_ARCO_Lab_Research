"""Isolated Nav2-only transfer check; never overwrites the two GP/GRG run records."""
import site
site.addsitedir("/usr/lib/python3/dist-packages")
from pathlib import Path
from datetime import datetime, timezone
import os,sys,time,json,math,signal,subprocess,hashlib
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rcl_interfaces.srv import GetParameters
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

ROOT=Path(__file__).resolve().parents[1]; CODE=ROOT/"testing_code"
OUT=ROOT/"result"/datetime.now(timezone.utc).strftime("nav2_profile_%Y%m%dT%H%M%SZ")
OUT.mkdir(parents=True)
os.environ.update(CL_OUTPUT=str(OUT),ROS_DOMAIN_ID="81",GZ_PARTITION="well_e_nav2_profile",ROS_HOME="/ros-data")
import shutil
shutil.copytree(CODE,OUT/"source_snapshot",ignore=shutil.ignore_patterns("__pycache__"))
source_hashes={str(p.relative_to(CODE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in CODE.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
processes=[]

def start(name,args):
    log=(OUT/(name+".log")).open("w")
    p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    processes.append((name,p,log))

class Observer(Node):
    def __init__(self):
        super().__init__("navigation_profile_check")
        self.set_parameters([rclpy.parameter.Parameter("use_sim_time",value=True)])
        self.pose=None;self.positions=[];self.speeds=[];self.yaw_rates=[];self.cmds={};self.results=[]
        self.create_subscription(Odometry,"/simulation/ground_truth/odom",self.odometry,100)
        for topic in ("/cl/cmd_vel_nav","/cmd_vel","/simulation/cmd_vel"):
            self.cmds[topic]=[]
            self.create_subscription(Twist,topic,lambda m,t=topic:self.command(t,m),100)
        self.action=ActionClient(self,NavigateToPose,"/navigate_to_pose")
    def odometry(self,m):
        p=m.pose.pose.position;q=m.pose.pose.orientation;v=m.twist.twist.linear
        self.pose=(p.x,p.y,math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z)))
        self.positions.append([self.get_clock().now().nanoseconds/1e9,*self.pose])
        self.speeds.append(math.hypot(v.x,v.y));self.yaw_rates.append(abs(m.twist.twist.angular.z))
    def command(self,topic,m):
        self.cmds[topic].append([self.get_clock().now().nanoseconds/1e9,m.linear.x,m.angular.z])
    def wait(self,future,timeout=90):
        rclpy.spin_until_future_complete(self,future,timeout_sec=timeout)
        if not future.done():raise TimeoutError("ROS future timed out")
        return future.result()
    def goal(self,x,y,yaw):
        g=NavigateToPose.Goal();g.pose.header.frame_id="map";g.pose.header.stamp=self.get_clock().now().to_msg()
        g.pose.pose.position.x=float(x);g.pose.pose.position.y=float(y)
        g.pose.pose.orientation.z=math.sin(yaw/2);g.pose.pose.orientation.w=math.cos(yaw/2)
        begin=self.get_clock().now().nanoseconds/1e9
        h=self.wait(self.action.send_goal_async(g))
        if not h.accepted:raise RuntimeError("Nav2 rejected goal")
        result=self.wait(h.get_result_async())
        distance=math.hypot(self.pose[0]-x,self.pose[1]-y)
        angle=abs(math.atan2(math.sin(self.pose[2]-yaw),math.cos(self.pose[2]-yaw)))
        row=dict(goal=[x,y,yaw],status=result.status,elapsed_sim_s=self.get_clock().now().nanoseconds/1e9-begin,
                 actual_pose=list(self.pose),distance_error_m=distance,yaw_error_rad=angle)
        self.results.append(row);print(json.dumps(row),flush=True)
        if result.status!=4 or distance>.3 or angle>.3:raise RuntimeError("Goal acceptance failed")

report=dict(scope="Nav2-only two-leg profile check, no GP/GRG performance result",passed=False)
node=None
try:
    subprocess.run([sys.executable,str(CODE/"prepare.py")],check=True)
    start("scene",["ros2","launch",str(CODE/"scene.launch.py")])
    start("bag",["ros2","bag","record","-s","mcap","-o",str(OUT/"rosbag"),"/clock",
                 "/simulation/ground_truth/odom","/odom","/cl/cmd_vel_nav","/cmd_vel","/simulation/cmd_vel","/tf","/tf_static"])
    rclpy.init();node=Observer()
    deadline=time.monotonic()+90
    while not node.action.wait_for_server(timeout_sec=1) or node.pose is None:
        rclpy.spin_once(node,timeout_sec=.1)
        if time.monotonic()>deadline:raise TimeoutError("Nav2/pose startup timeout")
    for name in ("map_server","planner_server","controller_server","behavior_server","bt_navigator","velocity_smoother"):
        state=node.create_client(GetState,"/"+name+"/get_state")
        if not state.wait_for_service(timeout_sec=15):raise RuntimeError("Lifecycle service missing: "+name)
        deadline=time.monotonic()+60
        while node.wait(state.call_async(GetState.Request()),10).current_state.id!=3:
            rclpy.spin_once(node,timeout_sec=.1)
            if time.monotonic()>deadline:raise TimeoutError("Lifecycle activation: "+name)
    cli=node.create_client(GetParameters,"/bunker_mini_sim_watchdog/get_parameters")
    if not cli.wait_for_service(timeout_sec=10):raise RuntimeError("Watchdog parameter service missing")
    req=GetParameters.Request();req.names=["max_linear_m_s","max_angular_rad_s"]
    values=[v.double_value for v in node.wait(cli.call_async(req)).values]
    assert values==[1.,.6],values
    node.goal(8.,0.,0.)
    node.goal(8.,6.,math.pi/2)
    # Measure stopped state after the final goal, while the scene is still alive.
    until=time.monotonic()+1
    while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=.05)
    peaks={t:dict(max_forward_m_s=max(a[1] for a in rows),min_forward_m_s=min(a[1] for a in rows),
                  max_abs_yaw_rad_s=max(abs(a[2]) for a in rows),samples=len(rows)) for t,rows in node.cmds.items() if rows}
    speed=max(node.speeds)
    report.update(goals=node.results,commands=peaks,max_actual_speed_m_s=speed,max_actual_yaw_rate_rad_s=max(node.yaw_rates))
    assert .75<speed<=1.02,speed
    assert max(node.yaw_rates)<=.62,max(node.yaw_rates)
    assert all(v["max_forward_m_s"]<=1.00001 and v["max_abs_yaw_rad_s"]<=.60001 for t,v in peaks.items() if t!="/cl/cmd_vel_nav"),peaks
    assert peaks["/cl/cmd_vel_nav"]["max_forward_m_s"]<=1.00001,peaks
    report.update(passed=True,goals=node.results,watchdog_limits=values,max_actual_speed_m_s=speed,commands=peaks,
                  actual_pose_samples=len(node.positions),source=json.loads((OUT/"navigation_profile.json").read_text())["source"])
except (Exception,KeyboardInterrupt) as exc:
    report["error"]=str(exc) or type(exc).__name__
    if node:report["goals"]=node.results
    print("FAILURE",repr(exc),flush=True)
finally:
    if node:
        (OUT/"observations.json").write_text(json.dumps(dict(positions=node.positions,speeds=node.speeds,commands=node.cmds)))
        node.destroy_node()
    if rclpy.ok():rclpy.shutdown()
    for name,p,log in reversed(processes):
        if p.poll() is None:
            os.killpg(p.pid,signal.SIGINT)
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        log.close()
    report["process_exit_codes"]={n:p.returncode for n,p,_ in processes}
    report["source_hashes"]=source_hashes
    report["command_bound_scope"]="Published actuator commands after velocity_smoother and watchdog; raw controller requests are reported separately."
    (OUT/"verification.json").write_text(json.dumps(report,indent=2))
    (ROOT/"result/nav2_profile_latest.json").write_text(json.dumps(dict(run=OUT.name)))
    print("SAVED",OUT, json.dumps(report),flush=True)
if not report["passed"]:sys.exit(1)
