"""Temporary geometry and a continuous-event adapter for the unchanged GRG search."""
from collections import OrderedDict
import hashlib
import importlib.util
import sys
import time
import numpy as np
from scipy.sparse.csgraph import floyd_warshall


def intersects_interior(a,b,rectangle):
    # Open interior: helper corners and paths along the clearance boundary remain usable.
    xmin,xmax,ymin,ymax=rectangle;eps=1e-9
    lo=np.array([xmin+eps,ymin+eps]);hi=np.array([xmax-eps,ymax-eps])
    direction=np.asarray(b)-a;near,far=0.,1.
    for k in range(2):
        if abs(direction[k])<1e-14:
            if a[k]<=lo[k] or a[k]>=hi[k]:return False
        else:
            u,v=(lo[k]-a[k])/direction[k],(hi[k]-a[k])/direction[k]
            near=max(near,min(u,v));far=min(far,max(u,v))
            if near>=far:return False
    return near<far and far>0 and near<1


class PathGraph:
    def __init__(self,points,obstacle=(13,17,13,17),clearance=.75,speed=1.):
        self.points=np.asarray(points,dtype=float);self.speed=float(speed)
        a,b,c,d=obstacle;self.safe=(a-clearance,b+clearance,c-clearance,d+clearance)
        a,b,c,d=self.safe
        self.vertices=np.vstack([self.points,[[a,c],[a,d],[b,c],[b,d]]])
        n=len(self.vertices);adj=np.full((n,n),np.inf);np.fill_diagonal(adj,0)
        for i in range(n):
            for j in range(i):
                if not intersects_interior(self.vertices[i],self.vertices[j],self.safe):
                    adj[i,j]=adj[j,i]=np.linalg.norm(self.vertices[i]-self.vertices[j])
        distances,pred=floyd_warshall(adj,directed=False,return_predecessors=True)
        self.travel=distances[:len(points),:len(points)]/speed
        self.paths={}
        self.arcs={}
        for i in range(len(points)):
            for j in range(len(points)):
                if not np.isfinite(distances[i,j]):continue
                nodes=[j];node=j
                while node!=i:
                    node=int(pred[i,node])
                    if node<0:raise ValueError("Broken graph predecessor")
                    nodes.append(node)
                self.paths[i,j]=self.vertices[nodes[::-1]]
                path=self.paths[i,j]
                self.arcs[i,j]=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
        self.version=0
        self.key=hashlib.sha256(np.r_[self.points.ravel(),self.safe,speed].tobytes()).hexdigest()
    def at_many(self,a,b,fractions):
        path=self.paths[a,b];fractions=np.asarray(fractions,dtype=float)
        if len(path)==1:return np.tile(path[0],(len(fractions),1))
        arc=self.arcs[a,b]
        distances=np.clip(fractions,0,1)*arc[-1]
        return np.column_stack([np.interp(distances,arc,path[:,k]) for k in range(2)])
    def point(self,a,b,fraction):
        return self.at_many(a,b,np.array([fraction]))[0]
    def bump_version(self):self.version+=1
    def replace_edge(self,a,b,path,duration_s):
        if not np.isfinite(duration_s):
            self.travel[a,b]=np.inf;self.paths.pop((a,b),None);self.arcs.pop((a,b),None)
        else:
            path=np.asarray(path,dtype=float)
            if path.ndim!=2 or path.shape[1]!=2 or not np.isfinite(path).all():raise ValueError("Finite edge geometry required")
            if not np.allclose(path[0],self.points[a]) or not np.allclose(path[-1],self.points[b]):raise ValueError("Edge endpoints mismatch")
            self.travel[a,b]=float(duration_s);self.paths[a,b]=path
            self.arcs[a,b]=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
        self.version+=1


def make_candidates(count,extent=(30,30),obstacle=(13,17,13,17),clearance=1.,home=(3,3)):
    a,b,c,d=obstacle
    pool=np.array([[x,y] for x in np.arange(clearance,extent[0]-clearance+1e-9,1.)
                   for y in np.arange(clearance,extent[1]-clearance+1e-9,1.)
                   if not (a-clearance<=x<=b+clearance and c-clearance<=y<=d+clearance)
                   and (x,y)!=tuple(home)],dtype=float)
    selected=[np.asarray(home,dtype=float)]
    distances=np.linalg.norm(pool-selected[0],axis=1)
    for _ in range(count):
        i=int(np.argmax(distances));selected.append(pool[i].copy())
        distances=np.minimum(distances,np.linalg.norm(pool-pool[i],axis=1));distances[i]=-np.inf
    return np.asarray(selected)


class SearchTimeout(RuntimeError):pass


def make_planner(base_path,graph,objective,config):
    spec=importlib.util.spec_from_file_location("continuous_verified_grg_base",base_path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    Base=module.GRGPlanner
    class ContinuousGRG(Base):
        def __init__(self):
            super().__init__(graph.points,graph.travel,0.,1.,
                depth=config["grg_depth"],time_splits=config["grg_time_splits"],home=0)
            # The legacy constructor requires positive dwell. All continuous schedule
            # times below are motion-only; zero disables its endpoint split allowance.
            self.sample_s=0.
            self.event_cache=OrderedDict();self.search_deadline=np.inf
            self.maximum=config["schedule_cache_entries"]
            self.score_policies=set();self.maximum_events=0;self.schedule_hits=0;self.schedule_misses=0
            self.schedule_generation_s=0.;self.score_function_s=0.
        def schedule(self,route,elapsed,sample_last=False):
            key=(graph.key,graph.version,tuple(route),float(elapsed),config["sensor_interval_s"])
            if key in self.event_cache:
                self.schedule_hits+=1;self.event_cache.move_to_end(key)
                e,a,f=self.event_cache[key]
                return e.copy(),list(a),f
            self.schedule_misses+=1
            schedule_started=time.perf_counter()
            start=float(elapsed);arrivals=[];segments=[]
            for a,b in zip(route[:-1],route[1:]):
                duration=float(self.travel[a,b])
                if not np.isfinite(duration):return np.empty((0,3)),[],np.inf
                segments.append((a,b,elapsed,elapsed+duration));elapsed+=duration;arrivals.append(float(elapsed))
            interval=config["sensor_interval_s"]
            first=int(np.floor(start/interval+1e-9))+1
            last=int(np.floor(elapsed/interval+1e-9))
            stamps=np.arange(first,last+1,dtype=float)*interval
            assigned=np.zeros(len(stamps),dtype=bool);chunks=[]
            for a,b,left,right in segments:
                if right<=left:continue
                take=(~assigned)&(stamps<=right+1e-9)
                if not np.any(take):continue
                selected=stamps[take]
                xy=graph.at_many(a,b,(selected-left)/(right-left))
                chunks.append(np.column_stack([xy,selected]));assigned[take]=True
            events=np.vstack(chunks) if chunks else np.empty((0,3))
            self.schedule_generation_s+=time.perf_counter()-schedule_started
            self.event_cache[key]=(events.copy(),list(arrivals),float(elapsed))
            while len(self.event_cache)>self.maximum:self.event_cache.popitem(last=False)
            return events,arrivals,float(elapsed)
        def score_routes(self,routes,gp,elapsed,sample_last=False):
            scores=[]
            for route in routes:
                if time.perf_counter()>self.search_deadline:raise SearchTimeout("Predeclared search wall limit reached")
                events=self.schedule(route,elapsed,sample_last)[0]
                self.maximum_events=max(self.maximum_events,len(events));self.score_policies.add(objective.policy)
                score_started=time.perf_counter()
                scores.append(objective.score(events))
                self.score_function_s+=time.perf_counter()-score_started
            return np.asarray(scores)
        def plan(self,*args,**kwargs):
            self.search_deadline=time.perf_counter()+config["search_wall_limit_s"]
            return super().plan(*args,**kwargs)
    return ContinuousGRG()
