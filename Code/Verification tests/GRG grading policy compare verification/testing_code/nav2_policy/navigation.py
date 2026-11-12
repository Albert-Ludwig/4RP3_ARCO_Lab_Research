"""Actual ROS actions and independent physics telemetry; no pose-setting API."""
import math
import threading
import time
import numpy as np
import sys
sys.path.append('/usr/lib/python3/dist-packages')
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data,QoSProfile,DurabilityPolicy
from geometry_msgs.msg import PoseStamped,Twist
from nav_msgs.msg import Odometry
from nav2_msgs.action import ComputePathToPose,NavigateToPose
from rosgraph_msgs.msg import Clock
from lifecycle_msgs.srv import GetState
from std_msgs.msg import String
from tf2_ros import Buffer,TransformListener


def yaw(q):return math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
def angle(a):return math.atan2(math.sin(a),math.cos(a))


class Navigation(Node):
    def __init__(self,config,log):
        super().__init__('informative_sampling_mission')
        self.set_parameters([Parameter('use_sim_time',value=True)])
        self.config=config;self.log=log;self.clock_s=None;self.origin=None
        self.truth=None;self.trajectory=[];self.phase='startup';self.feedback={};self.active=None
        self.create_subscription(Clock,'/clock',self.on_clock,qos_profile_sensor_data)
        self.create_subscription(Odometry,'/simulation/ground_truth/odom',self.on_truth,10)
        self.tf=Buffer();self.listener=TransformListener(self.tf,self)
        self.planner=ActionClient(self,ComputePathToPose,'/compute_path_to_pose')
        self.navigator=ActionClient(self,NavigateToPose,'/navigate_to_pose')
        self.goals=self.create_publisher(PoseStamped,'/test/navigation_goal',QoSProfile(depth=10,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.events=self.create_publisher(String,'/policy/events',10)
        self.worker_executor=MultiThreadedExecutor(num_threads=2);self.worker_executor.add_node(self)
        self.thread=threading.Thread(target=self.worker_executor.spin,daemon=True);self.thread.start()
    def on_clock(self,msg):self.clock_s=msg.clock.sec+msg.clock.nanosec*1e-9
    def on_truth(self,msg):
        p=msg.pose.pose.position;q=msg.pose.pose.orientation;self.truth=[p.x,p.y,yaw(q)]
        if self.origin is not None:
            self.trajectory.append(dict(time_s=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9-self.origin,
                x_m=p.x,y_m=p.y,yaw_rad=yaw(q),phase=self.phase))
    def wait(self,future,seconds=20):
        end=time.monotonic()+seconds
        while not future.done():
            if time.monotonic()>end:raise TimeoutError('ROS action/service response timed out')
            time.sleep(.01)
        return future.result()
    def ready(self):
        end=time.monotonic()+75
        while self.clock_s is None or self.truth is None:
            if time.monotonic()>end:raise TimeoutError('Gazebo clock/physical pose not ready')
            time.sleep(.05)
        for name in ('map_server','planner_server','controller_server','bt_navigator'):
            client=self.create_client(GetState,f'/{name}/get_state')
            while time.monotonic()<end:
                if client.wait_for_service(timeout_sec=.2) and self.wait(client.call_async(GetState.Request()),5).current_state.id==3:break
                time.sleep(.1)
            else:raise TimeoutError(f'{name} not active')
        if not self.planner.wait_for_server(timeout_sec=5) or not self.navigator.wait_for_server(timeout_sec=5):
            raise TimeoutError('Nav2 actions not available')
        self.position()
    def start(self):self.origin=self.clock_s;self.phase='planning';return self.origin
    def now(self):return self.clock_s-self.origin
    def position(self):
        tr=self.tf.lookup_transform('map','base_link',rclpy.time.Time())
        return [tr.transform.translation.x,tr.transform.translation.y,yaw(tr.transform.rotation)]
    def pose(self,values):
        msg=PoseStamped();msg.header.frame_id='map';msg.header.stamp=self.get_clock().now().to_msg()
        msg.pose.position.x=float(values[0]);msg.pose.position.y=float(values[1])
        theta=values[2] if len(values)>2 else 0.
        msg.pose.orientation.z=math.sin(theta/2);msg.pose.orientation.w=math.cos(theta/2)
        return msg
    def path(self,start,goal):
        request=ComputePathToPose.Goal();request.start=self.pose(start);request.goal=self.pose(goal)
        request.use_start=True;request.planner_id='GridBased'
        handle=self.wait(self.planner.send_goal_async(request))
        if not handle.accepted:return None
        result=self.wait(handle.get_result_async())
        if result.status!=4 or not result.result.path.poses:return None
        return [[p.pose.position.x,p.pose.position.y] for p in result.result.path.poses]
    def estimate(self,start,goal):
        if math.dist(start[:2],goal[:2])<.05:return dict(seconds=0.,length_m=0.,path=[list(start[:2]),list(goal[:2])])
        path=self.path(start,goal)
        if path is None:return dict(seconds=math.inf,length_m=math.inf,path=[])
        xy=np.array(path);delta=np.diff(xy,axis=0);length=float(np.linalg.norm(delta,axis=1).sum())
        headings=np.arctan2(delta[:,1],delta[:,0]);headings=headings[np.linalg.norm(delta,axis=1)>1e-6]
        turns=0.
        if len(headings):
            turns=abs(angle(headings[0]-(start[2] if len(start)>2 else 0.)))+abs(angle((goal[2] if len(goal)>2 else 0.)-headings[-1]))
            turns+=sum(abs(angle(b-a)) for a,b in zip(headings,headings[1:]))
        c=self.config
        seconds=c['travel_safety_factor']*(length/c['nominal_speed_m_s']+turns/c['angular_speed_rad_s'])+c['leg_overhead_s']
        return dict(seconds=seconds,length_m=length,path=path)
    def navigate(self,goal,deadline_s,guard=None):
        request=NavigateToPose.Goal();request.pose=self.pose(goal);self.goals.publish(request.pose)
        def feedback(message):
            f=message.feedback
            self.feedback=dict(distance_remaining_m=float(f.distance_remaining),recoveries=int(f.number_of_recoveries))
        handle=self.wait(self.navigator.send_goal_async(request,feedback_callback=feedback))
        if not handle.accepted:return dict(success=False,status=0,error='goal_rejected')
        self.active=handle;future=handle.get_result_async();begin=self.now();wall=time.monotonic()
        reason=None
        while not future.done():
            if self.now()>=deadline_s or time.monotonic()-wall>max(120,2*(deadline_s-begin)):
                reason='navigation_deadline';break
            if guard is not None and not guard():reason='return_reserve_guard';break
            time.sleep(.025)
        if reason:
            self.wait(handle.cancel_goal_async(),5)
            try:self.wait(future,5)
            except TimeoutError:pass
            self.active=None
            return dict(success=False,status=5,error=reason,actual_travel_s=self.now()-begin)
        result=future.result();self.active=None
        error=math.dist(self.position()[:2],goal[:2])
        return dict(success=result.status==4 and error<=self.config['acquisition_space_tolerance_m'],
                    status=int(result.status),error_code=int(getattr(result.result,'error_code',0)),
                    position_error_m=error,actual_travel_s=self.now()-begin,**self.feedback)
    def wait_until(self,target,guard=None):
        while self.now()<target:
            if guard is not None and not guard():return False
            time.sleep(.005)
        return True
    def close(self):
        if self.active:
            try:self.wait(self.active.cancel_goal_async(),5)
            except Exception:pass
        self.worker_executor.shutdown(timeout_sec=5);self.thread.join(timeout=5);self.destroy_node()
