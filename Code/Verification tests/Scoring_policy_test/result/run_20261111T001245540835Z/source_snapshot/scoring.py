"""Exact joint objectives with area quadrature and frozen adaptive spatial weights."""
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from scipy.linalg import cholesky,solve_triangular


def signature(array):return hashlib.sha256(np.asarray(array,dtype=np.float64).tobytes()).hexdigest()
def gp_token(gp):return (id(gp),int(gp.version),json.dumps(gp.parameters,sort_keys=True))


class Grid:
    """Rectangle-minus-rectangle mask with exact clipped areas and valid representatives."""
    def __init__(self,extent=(30,30),cell=10.,obstacle=None,times=None):
        self.extent=np.asarray(extent,dtype=float);self.cell=float(cell);self.obstacle=obstacle
        rows=[]
        for x0 in np.arange(0,self.extent[0],cell):
            for y0 in np.arange(0,self.extent[1],cell):
                x1=min(x0+cell,self.extent[0]);y1=min(y0+cell,self.extent[1])
                pieces=[(x0,x1,y0,y1)]
                if obstacle is not None:
                    a,b,c,d=obstacle
                    ix0,ix1=max(x0,a),min(x1,b);iy0,iy1=max(y0,c),min(y1,d)
                    if ix0<ix1 and iy0<iy1:
                        pieces=[(x0,ix0,y0,y1),(ix1,x1,y0,y1),(ix0,ix1,y0,iy0),(ix0,ix1,iy1,y1)]
                pieces=[r for r in pieces if r[1]>r[0] and r[3]>r[2]]
                if not pieces:continue
                areas=np.array([(r[1]-r[0])*(r[3]-r[2]) for r in pieces])
                r=pieces[int(np.argmax(areas))]
                rows.append((x0,x1,y0,y1,float(areas.sum()),(r[0]+r[1])/2,(r[2]+r[3])/2))
        if not rows:raise ValueError("Nonempty valid scoring domain required")
        self.cells=np.array(rows);self.areas=self.cells[:,4].copy();self.xy=self.cells[:,5:7].copy()
        self.times=np.asarray([0,60,120] if times is None else times,dtype=float)
        if len(self.times)<2 or not np.all(np.diff(self.times)>0):raise ValueError("Increasing fixed scoring times required")
        self.q=np.vstack([np.column_stack([self.xy,np.full(len(self.xy),t)]) for t in self.times])
        self.area_q=np.tile(self.areas,len(self.times))
        self.key=signature(np.r_[self.cells.ravel(),self.times])
        for array in [self.cells,self.areas,self.xy,self.times,self.q,self.area_q]:array.setflags(write=False)


@dataclass(frozen=True)
class WeightSnapshot:
    token:tuple
    time_s:float
    variance:np.ndarray
    spatial:np.ndarray
    query:np.ndarray
    key:str
    fallback:bool
    generation:int


class AdaptiveWeights:
    def __init__(self,grid,near_zero=1e-12):
        self.grid=grid;self.near_zero=near_zero;self.current=None;self.generations=0;self.locked=0
    def refresh(self,gp,time_s):
        if self.locked:raise RuntimeError("Weight refresh forbidden during a recursive search")
        token=gp_token(gp)
        if self.current is not None and self.current.token==token:return self.current
        q=np.column_stack([self.grid.xy,np.full(len(self.grid.xy),time_s)])
        variance=np.asarray(gp.posterior(q)[1],dtype=float)
        if not np.isfinite(variance).all() or np.any(variance < -1e-10):raise ValueError("Invalid latent GP variance")
        variance=np.maximum(variance,0)
        fallback=bool(np.all(variance<=self.near_zero))
        spatial=self.grid.areas.copy() if fallback else self.grid.areas*variance
        query=np.tile(spatial,len(self.grid.times))
        for a in [variance,spatial,query]:a.setflags(write=False)
        self.generations+=1
        self.current=WeightSnapshot(token,float(time_s),variance,spatial,query,
                                    signature(query),fallback,self.generations)
        return self.current
    @contextmanager
    def freeze(self):
        self.locked+=1
        try:yield
        finally:self.locked-=1


class ConditionalEvents:
    """Bounded exact event feature cache; no rounding or event subsampling."""
    def __init__(self,gp,queries=None,max_entries=4096):
        self.gp=gp;self.token=gp_token(gp);self.queries=queries
        self.maximum=max_entries;self.cache=OrderedDict();self.hits=0;self.misses=0;self.evictions=0
        self.query_features=None
        if queries is not None:
            self.query_features=solve_triangular(gp.factor,gp.kernel(gp.x,queries),lower=True,check_finite=False)
    def assert_current(self):
        if gp_token(self.gp)!=self.token:raise RuntimeError("GP changed; create a new frozen planning round")
    def matrices(self,events):
        self.assert_current()
        keys=[row.tobytes() for row in events]
        missing={}
        for k,e in zip(keys,events):
            if k in self.cache:self.hits+=1
            elif k not in missing:missing[k]=e
        if missing:
            values=np.array(list(missing.values()))
            features=solve_triangular(self.gp.factor,self.gp.kernel(self.gp.x,values),lower=True,check_finite=False)
            cross=None if self.queries is None else self.gp.kernel(self.queries,values)-self.query_features.T@features
            for i,k in enumerate(missing):
                self.cache[k]=(features[:,i].copy(),None if cross is None else cross[:,i].copy())
            self.misses+=len(missing)
        z=np.column_stack([self.cache[k][0] for k in keys])
        cross=None if self.queries is None else np.column_stack([self.cache[k][1] for k in keys])
        for k in keys:self.cache.move_to_end(k)
        while len(self.cache)>self.maximum:self.cache.popitem(last=False);self.evictions+=1
        covariance=self.gp.kernel(events,events)-z.T@z
        return (covariance+covariance.T)*.5,cross


class Objective:
    def __init__(self,policy,gp,grid,snapshot=None,feature_entries=4096,score_entries=8192,explicit_weights=None):
        if policy not in ("mi","emse","wivr"):raise ValueError(policy)
        self.policy=policy;self.gp=gp;self.grid=grid;self.snapshot=snapshot
        self.token=gp_token(gp);self.weight_generation=None if snapshot is None else snapshot.generation
        if snapshot is not None and snapshot.token!=self.token:
            raise RuntimeError("Adaptive weights belong to a different GP state")
        if policy=="mi":self.weights=None
        elif policy=="emse":self.weights=grid.area_q.copy()
        elif explicit_weights is not None:self.weights=np.asarray(explicit_weights,dtype=float).copy()
        elif snapshot is not None:self.weights=snapshot.query.copy()
        else:raise ValueError("WIVR requires a frozen weight snapshot or explicit verification weights")
        if explicit_weights is not None:self.weights=np.asarray(explicit_weights,dtype=float).copy()
        if self.weights is not None:
            if len(self.weights)!=len(grid.q) or not np.isfinite(self.weights).all() or np.any(self.weights<0) or self.weights.sum()<=0:
                raise ValueError("Nonnegative nonzero quadrature weights required")
            self.weights.setflags(write=False)
        self.weight_key="none" if self.weights is None else signature(self.weights)
        self.features=ConditionalEvents(gp,None if policy=="mi" else grid.q,feature_entries)
        self.maximum_scores=score_entries;self.cache=OrderedDict()
        self.calls=0;self.cache_hits=0;self.policy_trace=set()
    def score(self,events):
        if gp_token(self.gp)!=self.token:raise RuntimeError("GP updated inside scoring round")
        events=np.ascontiguousarray(events,dtype=float).reshape(-1,3)
        if not np.isfinite(events).all():raise ValueError("Finite spatiotemporal events required")
        key=(self.policy,self.token,self.grid.key,self.weight_key,self.weight_generation,events.shape,events.tobytes())
        self.policy_trace.add(self.policy)
        if key in self.cache:
            self.cache_hits+=1;self.cache.move_to_end(key);return self.cache[key]
        self.calls+=1
        if not len(events):return 0.
        covariance,cross=self.features.matrices(events)
        matrix=covariance.copy();matrix.flat[::len(events)+1]+=self.gp.noise
        factor=cholesky(matrix,lower=True,check_finite=False)
        if self.policy=="mi":
            value=float(np.log(np.diag(factor)).sum()-.5*len(events)*np.log(self.gp.noise))
        else:
            solved=solve_triangular(factor,cross.T,lower=True,check_finite=False)
            reduction=np.sum(solved*solved,axis=0)
            value=float(self.weights@reduction/self.weights.sum())
        if value < -1e-8 or not np.isfinite(value):raise FloatingPointError("Invalid joint gain")
        value=max(value,0.)
        self.cache[key]=value
        while len(self.cache)>self.maximum_scores:self.cache.popitem(last=False)
        return value
