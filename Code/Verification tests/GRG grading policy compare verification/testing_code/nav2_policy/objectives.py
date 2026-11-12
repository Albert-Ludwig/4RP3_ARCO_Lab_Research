"""Joint route objectives on one fixed, temperature-independent space-time grid."""
import numpy as np


def query_grid(spatial_xy,times_s):
    xy=np.asarray(spatial_xy,dtype=float); t=np.asarray(times_s,dtype=float)
    if len(xy)==0 or len(t)<2 or not np.all(np.diff(t)>0):
        raise ValueError('Nonempty spatial grid and increasing temporal grid required')
    return np.vstack([np.column_stack([xy,np.full(len(xy),time)]) for time in t])


def priority_weights(q,roi):
    """A priori rectangular spatial ROI, independent of any measured temperature."""
    xmin,xmax,ymin,ymax=roi['bounds_m']; multiplier=float(roi['multiplier'])
    if multiplier<=0 or multiplier==1: raise ValueError('Distinct WIVR requires positive nonuniform weights')
    inside=((q[:,0]>=xmin)&(q[:,0]<=xmax)&(q[:,1]>=ymin)&(q[:,1]<=ymax))
    w=np.where(inside,multiplier,1.).astype(float)
    if np.ptp(w)==0: raise ValueError('ROI must include some, but not all, query locations')
    return w


class Objective:
    def __init__(self,policy,queries,weights):
        if policy not in ('mi','emse','wivr'): raise ValueError(policy)
        self.policy=policy; self.q=np.asarray(queries,dtype=float)
        self.weights=np.asarray(weights,dtype=float)
        if len(self.weights)!=len(self.q) or np.any(self.weights<=0): raise ValueError('Invalid query weights')
        self.cache={}; self.calls=0; self.cache_hits=0

    def score(self,gp,events):
        events=np.asarray(events,dtype=float).reshape(-1,3)
        key=(self.policy,gp.version,events.shape,events.tobytes())
        if key in self.cache: self.cache_hits+=1; return self.cache[key]
        self.calls+=1
        if len(events)==0: return 0.
        c=gp.covariance(events); c=(c+c.T)*.5
        l=np.linalg.cholesky(c+gp.noise*np.eye(len(events)))
        if self.policy=='mi':
            # I(y_S; f_S | D) in nats, matching the retained joint-MI baseline.
            value=float(np.log(np.diag(l)).sum()-.5*len(events)*np.log(gp.noise))
        else:
            # diag(C_QS (C_SS + sigma_n^2 I)^-1 C_SQ), no future labels.
            total=0.
            for start in range(0,len(self.q),256):
                cross=gp.covariance(events,self.q[start:start+256])
                v=np.linalg.solve(l,cross)
                reduction=np.sum(v*v,axis=0)
                w=self.weights[start:start+256] if self.policy=='wivr' else np.ones(len(reduction))
                total+=float(w@reduction)
            value=total/(self.weights.sum() if self.policy=='wivr' else len(self.q))
        if value < -1e-9 or not np.isfinite(value): raise FloatingPointError('Invalid route objective')
        value=max(value,0.); self.cache[key]=value
        return value

    def mean_variance(self,gp):
        return float(np.mean(np.concatenate([gp.posterior(self.q[i:i+256])[1] for i in range(0,len(self.q),256)])))
