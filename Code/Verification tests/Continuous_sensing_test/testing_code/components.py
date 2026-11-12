"""Continuous-sensing components and exact, fixed-parameter batch RBF GP."""
from collections import OrderedDict
from contextlib import AbstractContextManager
import threading
import time
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.spatial.distance import cdist
import psutil


class Polyline:
    def __init__(self, xy):
        xy=np.asarray(xy,dtype=float)
        if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<2 or not np.isfinite(xy).all():
            raise ValueError("Finite polyline required")
        keep=np.r_[True,np.linalg.norm(np.diff(xy,axis=0),axis=1)>1e-12]
        self.xy=xy[keep]
        self.distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(self.xy,axis=0),axis=1))]
        self.length=float(self.distance[-1])
        if self.length<=0: raise ValueError("Nonzero path required")
    def at(self,fraction):
        distance=float(np.clip(fraction,0,1))*self.length
        return np.array([np.interp(distance,self.distance,self.xy[:,axis]) for axis in range(2)])


class TimedRoute:
    """Offline planned or actual-profile fixture; no robot controller."""
    def __init__(self,paths,end_times,actual=False,start_s=0.):
        self.paths=paths;self.ends=np.asarray(end_times,dtype=float);self.start=float(start_s)
        if len(paths)!=len(end_times) or np.any(np.diff(np.r_[start_s,self.ends])<=0):
            raise ValueError("Positive route segment durations required")
        for a,b in zip(paths,paths[1:]):
            if not np.allclose(a.xy[-1],b.xy[0],atol=1e-10):raise ValueError("Disconnected path junction")
        self.actual=actual
    def segment(self,t):
        return min(int(np.searchsorted(self.ends,float(t),side="left")),len(self.paths)-1)
    def at(self,t):
        i=self.segment(t)
        start=self.start if i==0 else self.ends[i-1]
        u=float(np.clip((t-start)/(self.ends[i]-start),0,1))
        if self.actual:
            # Uneven movement; second segment includes a stop while sensing continues.
            if i==1:
                u=float(np.interp(u,[0,.30,.45,.75,1],[0,.40,.40,.80,1]))
            else:
                u=float(np.interp(u,[0,.25,.70,1],[0,.15,.75,1]))
        return self.paths[i].at(u)


def predicted_events(paths,durations,start_s=0.,interval_s=1.,phase_s=0.):
    route=TimedRoute(paths,start_s+np.cumsum(durations),start_s=start_s)
    first=max(1,int(np.floor((start_s-phase_s)/interval_s+1e-9))+1)
    last=int(np.floor((route.ends[-1]-phase_s)/interval_s+1e-9))
    events=[]
    for tick in range(first,last+1):
        t=phase_s+tick*interval_s
        x,y=route.at(t)
        events.append(dict(tick=tick,t_s=float(t),x_m=float(x),y_m=float(y),segment=route.segment(t)))
    return events


class GlobalSampler:
    """Emit at most one actual measurement per callback; never backfill missed ticks."""
    def __init__(self,interval_s=1.,origin_s=0.,identity="mission"):
        self.interval=float(interval_s);self.origin=float(origin_s);self.identity=identity
        self.next_tick=1;self.missed_ticks=[];self.last_callback=-np.inf
    def poll(self,now,actual_xy,field):
        now=float(now)
        if now<self.last_callback-1e-9:raise ValueError("Monotonic actual callback time required")
        self.last_callback=now
        due=int(np.floor((now-self.origin)/self.interval+1e-9))
        if due<self.next_tick:return None
        self.missed_ticks.extend(range(self.next_tick,due))
        self.next_tick=due+1
        x,y=map(float,actual_xy)
        identity=f"{self.identity}-tick-{due}"
        z=np.array([[x,y,now]])
        truth=float(field.truth(z)[0])
        observed=truth+float(field.noise([identity])[0])
        return dict(record_id=identity,tick=due,scheduled_t_s=self.origin+due*self.interval,
                    t_s=now,x_m=x,y_m=y,temperature_c=observed,truth_c=truth,valid=True)


class BatchGP:
    """Same exact 3D ARD RBF equations; dense Cholesky and triangular solves."""
    def __init__(self,x,y,parameters,record_ids):
        self.parameters=dict(parameters)
        self.length=np.asarray(parameters["lengthscales_m_m_s"],dtype=float)
        self.scale=float(parameters["output_variance_c2"])
        self.noise=float(parameters["noise_variance_c2"]);self.mean=float(parameters["mean_c"])
        if self.length.shape!=(3,) or np.any(self.length<=0) or self.scale<=0 or self.noise<=0:
            raise ValueError("Positive fixed GP parameters required")
        self.x=np.asarray(x,dtype=float).reshape(-1,3);self.y=np.asarray(y,dtype=float)
        self.record_ids=list(record_ids);self.version=0
        self._validate(self.x,self.y,self.record_ids)
        self.factor,self.alpha=self._factorize(self.x,self.y)
    @staticmethod
    def _validate(x,y,ids):
        if len(x)!=len(y) or len(x)!=len(ids) or len(ids)!=len(set(ids)):
            raise ValueError("Matching unique observation identities required")
        if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError("Finite observations required")
    def kernel(self,a,b):
        a=np.asarray(a,dtype=float).reshape(-1,3);b=np.asarray(b,dtype=float).reshape(-1,3)
        distance=cdist(a/self.length,b/self.length,metric="sqeuclidean")
        distance*= -.5;np.exp(distance,out=distance);distance*=self.scale
        return distance
    def _factorize(self,x,y):
        if not len(x):return np.empty((0,0)),np.empty(0)
        matrix=self.kernel(x,x);matrix.flat[::len(x)+1]+=self.noise
        factor=cholesky(matrix,lower=True,overwrite_a=True,check_finite=False)
        alpha=cho_solve((factor,True),y-self.mean,check_finite=False)
        return factor,alpha
    def posterior(self,queries):
        q=np.asarray(queries,dtype=float).reshape(-1,3)
        if not len(self.x):return np.full(len(q),self.mean),np.full(len(q),self.scale)
        cross=self.kernel(self.x,q)
        v=solve_triangular(self.factor,cross,lower=True,check_finite=False)
        return self.mean+cross.T@self.alpha,np.maximum(self.scale-np.sum(v*v,axis=0),0.)
    def append_batch(self,rows):
        if not rows:return 0
        ids=[r["record_id"] for r in rows]
        if len(ids)!=len(set(ids)) or set(ids)&set(self.record_ids):raise ValueError("Duplicate GP observations")
        x=np.array([[r["x_m"],r["y_m"],r["t_s"]] for r in rows],dtype=float)
        y=np.array([r["temperature_c"] for r in rows],dtype=float)
        self._validate(x,y,ids)
        xx=np.vstack([self.x,x]);yy=np.r_[self.y,y]
        # Factorize before commit: errors leave both GP and buffer untouched.
        factor,alpha=self._factorize(xx,yy)
        self.x,self.y,self.factor,self.alpha=xx,yy,factor,alpha
        self.record_ids.extend(ids);self.version+=1
        return len(rows)
    def save(self,path):
        np.savez_compressed(path,x=self.x,y=self.y,record_ids=np.asarray(self.record_ids),
                            version=np.array(self.version))


class MeasurementBuffer:
    def __init__(self,extent,interval):
        self.pending=OrderedDict();self.seen=set();self.rejected=[];self.extent=extent;self.interval=interval
    def offer(self,row):
        identity=row.get("record_id")
        reason=None
        if not isinstance(identity,str) or not identity:reason="missing_identity"
        elif identity in self.seen:reason="duplicate_identity"
        elif not row.get("valid",False):reason="invalid_quality"
        else:
            try:
                x,y,t,T=[float(row[k]) for k in ("x_m","y_m","t_s","temperature_c")]
                if not np.isfinite([x,y,t,T]).all():reason="nonfinite"
                elif not (0<=x<=self.extent[0] and 0<=y<=self.extent[1] and self.interval[0]<=t<=self.interval[1]):
                    reason="outside_domain"
            except (KeyError,TypeError,ValueError):reason="missing_or_invalid_value"
        if reason:
            self.rejected.append(dict(record_id=identity,reason=reason));return False
        self.pending[identity]=dict(row);self.seen.add(identity);return True
    def flush(self,gp):
        rows=list(self.pending.values())
        count=gp.append_batch(rows)
        self.pending.clear()
        return rows


class MemoryProbe(AbstractContextManager):
    def __enter__(self):
        self.process=psutil.Process();self.base=self.peak=self.process.memory_info().rss
        self.stop=threading.Event()
        def sample():
            while not self.stop.wait(.005):self.peak=max(self.peak,self.process.memory_info().rss)
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start()
        return self
    def __exit__(self,*args):
        self.peak=max(self.peak,self.process.memory_info().rss);self.stop.set();self.thread.join()
    def values(self):
        return dict(baseline_rss_mib=self.base/2**20,sampled_peak_rss_mib=self.peak/2**20,
                    sampled_rss_increment_mib=max(0,self.peak-self.base)/2**20)
