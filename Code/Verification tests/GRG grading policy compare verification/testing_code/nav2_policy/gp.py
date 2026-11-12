"""Exact fixed-hyperparameter RBF posterior, physical inputs (m,m,seconds)."""
import numpy as np


class GaussianProcess:
    def __init__(self,x,y,parameters):
        self.parameters=dict(parameters)
        self.length=np.asarray(parameters['lengthscales_m_m_s'],dtype=float)
        self.scale=float(parameters['output_variance_c2'])
        self.noise=float(parameters['noise_variance_c2'])
        self.mean=float(parameters['mean_c'])
        if (self.length.shape!=(3,) or not np.isfinite(self.length).all() or
            not np.isfinite([self.scale,self.noise,self.mean]).all() or
            np.any(self.length<=0) or self.scale<=0 or self.noise<=0):
            raise ValueError('Positive physical-unit RBF lengthscales, variance and noise are required')
        self.x=np.asarray(x,dtype=float).reshape(-1,3); self.y=np.asarray(y,dtype=float)
        self.version=0
        self._refresh()

    def _refresh(self):
        if len(self.x)!=len(self.y) or not np.isfinite(self.x).all() or not np.isfinite(self.y).all():
            raise ValueError('Invalid GP observations')
        if len(self.x):
            self.factor=np.linalg.cholesky(self.kernel(self.x,self.x)+self.noise*np.eye(len(self.x)))
            self.alpha=np.linalg.solve(self.factor.T,np.linalg.solve(self.factor,self.y-self.mean))
        else:
            self.factor=np.empty((0,0)); self.alpha=np.empty(0)

    def kernel(self,a,b):
        a=np.asarray(a,dtype=float).reshape(-1,3); b=np.asarray(b,dtype=float).reshape(-1,3)
        d=(a[:,None,:]-b[None,:,:])/self.length
        return self.scale*np.exp(-.5*np.sum(d*d,axis=-1))

    def posterior(self,q):
        q=np.asarray(q,dtype=float).reshape(-1,3)
        cross=self.kernel(self.x,q)
        v=np.linalg.solve(self.factor,cross) if len(self.x) else cross
        mu=self.mean+cross.T@self.alpha
        variance=np.maximum(self.scale-np.sum(v*v,axis=0),0.)
        return mu,variance

    def covariance(self,a,b=None):
        a=np.asarray(a,dtype=float).reshape(-1,3)
        b=a if b is None else np.asarray(b,dtype=float).reshape(-1,3)
        ca=self.kernel(self.x,a); cb=self.kernel(self.x,b)
        if len(self.x):
            ca=np.linalg.solve(self.factor,ca); cb=np.linalg.solve(self.factor,cb)
        return self.kernel(a,b)-ca.T@cb

    def append(self,x,temperature):
        self.x=np.vstack([self.x,np.asarray(x,dtype=float).reshape(1,3)])
        self.y=np.r_[self.y,float(temperature)]
        self._refresh(); self.version+=1

    def save(self,path):
        np.savez_compressed(path,x=self.x,y=self.y,version=np.array(self.version))
