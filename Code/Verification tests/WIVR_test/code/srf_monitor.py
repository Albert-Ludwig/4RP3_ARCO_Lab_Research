"""Two independent Python ROS nodes: fixed SRF source and process monitoring."""
import site
site.addsitedir("/usr/lib/python3/dist-packages")
import sys,os,json,time
from pathlib import Path
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import psutil
ROOT=Path(__file__).resolve().parents[1];OUT=Path(os.environ["CL_OUTPUT"])
sys.path.insert(0,str(ROOT.parent/"SRF_test/testing_code"))
from srf_field import TemperatureField

class Source(Node):
    """Return only the temperature at the sampler's captured position/time."""
    def __init__(self):
        super().__init__("srf_source")
        self.field=TemperatureField(json.loads((OUT/"field.json").read_text()))
        self.pub=self.create_publisher(String,"/cl/temperature",100)
        self.sub=self.create_subscription(String,"/cl/query",self.query,100)
        self.log=(OUT/"source.jsonl").open("w",buffering=1)
    def query(self,msg):
        q=json.loads(msg.data);start=time.perf_counter()
        try:
            value=float(self.field.measure([[q["x"],q["y"],q["t"]]],[str(q["id"])])[0])
            q.update(temperature=value,source_ms=(time.perf_counter()-start)*1000,ok=True)
        except Exception as exc:
            q.update(ok=False,error=str(exc))
        self.log.write(json.dumps(q)+"\n")
        self.pub.publish(String(data=json.dumps(q)))

class Monitor(Node):
    """Observe only; do not block the SRF callback or modify mission state."""
    def __init__(self):
        super().__init__("runtime_monitor")
        self.log=(OUT/"resources.jsonl").open("w",buffering=1)
        self.create_timer(1,self.record)
    def record(self):
        processes=[]
        for process in psutil.Process(os.getppid()).children(recursive=True):
            try:
                info=process.as_dict(attrs=["pid","name","memory_info","cpu_times"])
                processes.append(dict(pid=process.pid,name=info["name"],rss=info["memory_info"].rss,
                                      cpu_s=sum(info["cpu_times"][:2])))
            except (psutil.NoSuchProcess,psutil.AccessDenied):
                pass
        self.log.write(json.dumps(dict(wall=time.time(),processes=processes,
            container_memory_bytes=int(Path("/sys/fs/cgroup/memory.current").read_text())))+"\n")

rclpy.init()
node=Monitor() if len(sys.argv)>1 and sys.argv[1]=="monitor" else Source()
try:
    rclpy.spin(node)
except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):
    pass
finally:
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
