"""Eight bounded correctness categories for time-separated sampling events."""

import copy
import json
import sys
import unittest
import numpy as np
import torch

from reference import ROOT, BASE, ReferenceMI, enumerate_routes, schedule, uncached_plan
from gp import GPModel
from grg import GRGPlanner
from interfaces import Measurement
from navigation import GridMap, PythonNavigationSimulator, travel_matrix
from failures import CandidateFailures

torch.set_num_threads(1)
EVIDENCE = {"exhaustive": [], "recursive_calls": [], "cache": [], "events": [], "mi_errors": [], "reproduction": {}}


def fixture(count=3):
    points = np.array([[2,2],[7,9],[20,24],[28,8],[8,28],[15,5],[25,28]],float)[:count+1]
    return points, np.linalg.norm(points[:,None]-points[None,:],axis=-1)/.35


class RecursiveChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template=GPModel.load(ROOT/"10.18_result/baseline/main/initial_gp.pt")

    def setUp(self):
        self.gp=copy.deepcopy(self.template)

    def compare(self,planner,route,elapsed=0,sample_last=False):
        query,arrivals,finish=schedule(planner.points,planner.travel,route,elapsed,
                                      dwell=planner.sample_s,sample_last=sample_last)
        q,a,f=planner.schedule(route,elapsed,sample_last)
        np.testing.assert_allclose(q,query,atol=1e-12)
        np.testing.assert_allclose(a,arrivals,atol=1e-9)
        self.assertAlmostEqual(f,finish,places=9)
        expected=float(ReferenceMI(self.gp).scores(query)) if len(query) else 0.
        actual=float(planner.score_routes([route],self.gp,elapsed,sample_last)[0])
        EVIDENCE["mi_errors"].append(abs(expected-actual))
        self.assertAlmostEqual(expected,actual,places=8)
        return query,arrivals,finish

    def test_01_bounded_exhaustive_repeated_routes(self):
        for count in [3,4,5,6]:
            points,travel=fixture(count);travel=travel/100
            # At 5 s dwell, 18 s permits at most three events, even at Home.
            routes=enumerate_routes(points,travel,0,37,18,max_events=3)
            values=ReferenceMI(self.gp).routes(points,travel,routes,37)
            for depth in [1,2,3]:
                planner=GRGPlanner(points,travel,12,5,depth)
                plan=planner.plan(self.gp,0,37,18,visited=set(range(count+1)))
                _,_,finish=self.compare(planner,plan.route,37)
                self.assertIn(plan.route,routes)
                self.assertLessEqual(finish,55+1e-8)
                self.assertAlmostEqual(plan.gain,values[routes.index(plan.route)],places=8)
                EVIDENCE["exhaustive"].append(dict(nodes=count,depth=depth,route=plan.route,
                    reference_routes=len(routes),event_bound=3,gain=plan.gain,
                    optimality_gap_diagnostic=float(max(values)-plan.gain)))

    def test_02_recursive_deadlines_prefixes_and_junctions(self):
        points,travel=fixture(3)
        reference,owner=ReferenceMI(self.gp),self
        class AuditedPlanner(GRGPlanner):
            def score_routes(self,routes,gp,elapsed,sample_last=False):
                actual=super().score_routes(routes,gp,elapsed,sample_last)
                expected=reference.routes(self.points,self.travel,routes,elapsed,sample_last=sample_last)
                EVIDENCE["mi_errors"].append(float(np.max(np.abs(actual-expected))))
                np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=1e-8)
                return actual
        planner=AuditedPlanner(points,travel,12,5,depth=3)
        records=[]
        def trace(frame,event,value):
            if frame.f_code.co_filename!=str(BASE/"grg.py") or frame.f_code.co_name!="search":return trace
            local=frame.f_locals
            if event=="call" and frame.f_back.f_code.co_name=="search":
                p=frame.f_back.f_locals
                owner.assertEqual(local["depth"],p["depth"]-1)
                owner.assertGreaterEqual(p["split"],p["lo"]-1e-8)
                owner.assertLessEqual(p["split"],p["hi"]+1e-8)
                if local["prefix"]==p["prefix"]:
                    owner.assertEqual(local["end"],p["v"])
                    owner.assertTrue(local["sample_end"])
                    owner.assertAlmostEqual(local["end_s"],p["split"])
                else:
                    owner.assertEqual(local["prefix"],p["left"])
                    owner.assertEqual(local["end"],p["end"])
                    owner.assertAlmostEqual(local["end_s"],p["end_s"])
            if event=="return" and value is not None:
                prefix_finish=schedule(points,travel,local["prefix"],83,sample_last=True)[2]
                owner.assertAlmostEqual(local["now"],prefix_finish,places=8)
                _,arrivals,finish=schedule(points,travel,value,83,sample_last=local["sample_end"])
                owner.assertEqual(value[:len(local["prefix"])],local["prefix"])
                owner.assertEqual(value[-1],local["end"])
                owner.assertLessEqual(finish,local["end_s"]+1e-8)
                records.append(dict(prefix=list(local["prefix"]),end=local["end"],route=value,
                    sample_end=local["sample_end"],deadline_s=local["end_s"],finish_s=finish,events=len(arrivals)))
            return trace
        previous=sys.gettrace()
        try:
            sys.settrace(trace);planner.plan(self.gp,1,83,400,{1})
        finally:sys.settrace(previous)
        self.assertTrue(records)
        self.assertTrue(any(r["finish_s"]<r["deadline_s"]-1 for r in records))
        self.assertTrue(any(len(set(r["route"][1:-1]))<len(r["route"][1:-1]) for r in records))
        EVIDENCE["recursive_calls"]=records

    def test_03_consecutive_nonconsecutive_and_previous_visits(self):
        points,travel=fixture(3);planner=GRGPlanner(points,travel,12,5,depth=3)
        for route in [[0,1,2,1,0],[0,1,1,0],[0,1,0],[0,0,0]]:
            q,a,f=self.compare(planner,route,11)
            self.assertEqual(len(q),len(route)-2)
            self.assertTrue(np.all(np.diff(a)>=5-1e-9))
            self.assertAlmostEqual(f,11+sum(travel[x,y] for x,y in zip(route,route[1:]))+5*len(q))
            EVIDENCE["events"].append(dict(route=route,arrivals_s=a,finish_s=f))
        a=planner.schedule([0,1,1,0],0)[1];self.assertAlmostEqual(a[1]-a[0],5)
        p=GRGPlanner(points[:2],travel[:2,:2],12,5).plan(self.gp,0,0,500,{1},blocked={0})
        self.assertGreater(len(p.arrivals_s),1);self.assertEqual(set(p.route[1:-1]),{1})

    def test_04_cache_depth_time_prefix_order_and_posterior(self):
        points,travel=fixture(2);disabled=uncached_plan()
        for version in [0,1]:
            for depth in [1,2,3]:
                planner=GRGPlanner(points,travel,12,5,depth)
                for current,elapsed in [(0,0),(1,100),(0,43200)]:
                    a=planner.plan(self.gp,current,elapsed,400,{1})
                    b=disabled(planner,self.gp,current,elapsed,400,{1})
                    self.assertEqual(a.route,b.route)
                    np.testing.assert_allclose(a.arrivals_s,b.arrivals_s,atol=1e-9)
                    self.assertAlmostEqual(a.finish_s,b.finish_s,places=8)
                    self.assertAlmostEqual(a.gain,b.gain,places=8)
                    self.assertEqual(a.gp_version,b.gp_version)
                    EVIDENCE["cache"].append(dict(depth=depth,current=current,elapsed_s=elapsed,
                                                  gp_version=version,route=a.route,matched=True))
            if version==0:self.gp.append(Measurement(*points[1],12.05,20.,"verification"))
        self.compare(planner,[0,1,2,1,0]);self.compare(planner,[0,2,1,1,0])

    def test_05_joint_mi_history_online_and_temporal_covariance(self):
        points,travel=fixture(2);planner=GRGPlanner(points,travel,12,5)
        q,_,_=self.compare(planner,[0,1,2,1,0])
        before=float(ReferenceMI(self.gp).scores(q));cov=self.gp.joint(q)[1]
        self.assertGreater(.5*np.log1p(np.diag(cov)/self.gp.noise).sum(),before)
        parameters=self.gp.parameters();self.gp.append(Measurement(*q[0],20.,"verification"))
        after=float(ReferenceMI(self.gp).scores(q));self.assertLess(after,before)
        self.compare(planner,[0,1,2,1,0]);self.assertEqual(len(self.gp.x),101)
        for name,value in parameters.items():np.testing.assert_array_equal(value,self.gp.parameters()[name])
        later=planner.score_routes([[0,1,2,1,0]],self.gp,43200)[0]
        self.assertGreater(abs(later-after),1e-4)

    def test_06_zero_distance_dwell_exact_budget_and_return(self):
        points,travel=fixture(1);planner=GRGPlanner(points,travel,12,5,depth=2)
        cost=travel[0,1]+10+travel[1,0]
        exact=planner.plan(self.gp,0,71,cost,(),blocked={0})
        self.assertEqual(exact.route,[0,1,1,0]);self.assertAlmostEqual(exact.finish_s,71+cost)
        tight=planner.plan(self.gp,0,71,cost-1e-5,(),blocked={0})
        self.assertEqual(len(tight.arrivals_s),1)
        back=travel[1,0];direct=planner.plan(self.gp,1,71,back,(),blocked={0})
        self.assertEqual(direct.route,[1,0]);self.assertEqual(direct.arrivals_s,[])
        self.assertEqual(planner.plan(self.gp,1,71,back-1e-5,(),blocked={0}).route,[])

    def test_07_candidates_and_one_top_level_failure_cycle(self):
        points,travel=fixture(2);failures=CandidateFailures();failures.mark_temporary(1,0,"timeout")
        planner=GRGPlanner(points,travel,12,5,depth=3);blocked=failures.blocked_for(1)
        first=planner.plan(self.gp,0,0,500,{1,2},blocked=blocked)
        self.assertNotIn(1,first.route[1:-1]);self.assertEqual(failures.blocked_for(1),{1})
        self.assertNotIn(1,failures.blocked_for(2));failures.mark_permanent(2,"static_no_path")
        nxt=planner.plan(self.gp,0,0,500,{1},blocked=failures.blocked_for(2)|{0})
        self.assertIn(1,nxt.route);self.assertNotIn(2,nxt.route)

    def test_08_obstacle_paths_and_previous_depth3_reproduction(self):
        config=json.loads((BASE/"config.json").read_text());grid=GridMap(config["map"])
        points,_=fixture(3);travel=travel_matrix(PythonNavigationSimulator(grid,.35),points)
        route=[0,3,1,2,3,0];planner=GRGPlanner(points,travel,12,5,depth=3)
        q,a,f=self.compare(planner,route);distance=0.
        for x,y in zip(route,route[1:]):
            path,length=grid.route(points[x],points[y]);self.assertGreater(len(path),0)
            self.assertTrue(all(grid.segment_free(u,v) for u,v in zip(path,path[1:])))
            self.assertAlmostEqual(float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()),length)
            distance+=length
        self.assertLessEqual(f,900);self.assertAlmostEqual(f,distance/.35+20)
        self.assertEqual(len(q),4);np.testing.assert_array_equal(q[0,:2],q[3,:2]);self.assertLess(q[0,2],q[3,2])
        current=planner.plan(self.gp,0,0,900,());self.compare(planner,current.route)
        EVIDENCE["reproduction"]=dict(original_route=route,arrivals_s=a,finish_s=f,dwell_s=20.,
            event_count=4,valid_revisit_route=True,current_route=current.route,
            interpretation="Two explicit node-3 events with distinct times and dwells. The shared recursive junction is passed once in the prefix, not concatenated twice.")


if __name__=="__main__":unittest.main(verbosity=2)
