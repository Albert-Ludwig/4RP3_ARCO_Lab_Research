"""Public recorded acquisition pool only; this module never opens evaluation data."""
import math
import numpy as np
from .storage import read_csv


class Acquisition:
    def __init__(self,path,space_tolerance_m,time_tolerance_s):
        self.records=read_csv(path); self.used=set()
        self.space=float(space_tolerance_m); self.time=float(time_tolerance_s)
        for r in self.records:
            for k in ('x_m','y_m','t_s','temperature_c'): r[k]=float(r[k])
        self.records.sort(key=lambda r:(r['t_s'],r['record_id']))

    def next_time(self,xy,arrival_s,deadline_s=math.inf):
        for r in self.records:
            if r['record_id'] in self.used: continue
            if r['t_s']<arrival_s-self.time or r['t_s']>deadline_s: continue
            if math.hypot(r['x_m']-xy[0],r['y_m']-xy[1])<=self.space:
                return max(arrival_s,r['t_s'])
        return math.inf

    def acquire(self,xy,time_s):
        eligible=[r for r in self.records if r['record_id'] not in self.used
                  and abs(r['t_s']-time_s)<=self.time
                  and math.hypot(r['x_m']-xy[0],r['y_m']-xy[1])<=self.space]
        if not eligible: return None
        r=min(eligible,key=lambda r:(abs(r['t_s']-time_s),r['record_id']))
        self.used.add(r['record_id'])
        return dict(r,actual_x_m=float(xy[0]),actual_y_m=float(xy[1]),actual_t_s=float(time_s),
                    position_error_m=float(math.hypot(r['x_m']-xy[0],r['y_m']-xy[1])),
                    time_error_s=float(time_s-r['t_s']))
