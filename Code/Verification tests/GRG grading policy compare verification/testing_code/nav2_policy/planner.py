"""Unchanged GRG recursive search; override scoring and time-unit conversion."""
import importlib.util
from pathlib import Path
import sys
import numpy as np


def load_base(path):
    spec=importlib.util.spec_from_file_location('reused_time_aware_grg',path)
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module.GRGPlanner


def make_planner(base_path,points,travel_s,acquisition,objective,config):
    Base=load_base(base_path)
    class PolicyGRG(Base):
        def score_routes(self,routes,gp,elapsed,sample_last=False):
            # Base schedule returns hours; the physical GP uses seconds.
            scores=[]
            for route in routes:
                events=self.schedule(route,elapsed,sample_last)[0]
                if len(events): events[:,2]*=3600.
                scores.append(objective.score(gp,events))
            return np.asarray(scores)
    p=PolicyGRG(points,travel_s,start_h=0.,sample_s=config['sensing_s'],
                depth=config['grg_depth'],time_splits=config['grg_time_splits'],home=0,
                sampling_time=lambda node,t:acquisition.next_time(points[node],t,config['budget_s']-config['sensing_s']))
    return p
