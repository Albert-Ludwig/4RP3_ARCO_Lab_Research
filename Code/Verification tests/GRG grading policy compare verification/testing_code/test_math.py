"""Numerical invariants and strict recorded-acquisition tests; no ROS required."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from nav2_policy.gp import GaussianProcess
from nav2_policy.objectives import Objective,query_grid,priority_weights
from nav2_policy.acquisition import Acquisition
from nav2_policy.planner import make_planner
from nav2_policy.prepare import REPO,split_experiment
from nav2_policy.storage import save_csv,save_json


class MathTests(unittest.TestCase):
    def setUp(self):
        self.hp=dict(lengthscales_m_m_s=[2.,3.,100.],output_variance_c2=1.3,
                     noise_variance_c2=.04,mean_c=20.)
        self.gp=GaussianProcess([[0,0,-30],[1,2,-10]],[20.1,20.3],self.hp)
        self.q=query_grid([[0,0],[2,1],[4,2]],[0,100,200])
        self.s=np.array([[2,1,40],[3,1,80.]])
        self.w=priority_weights(self.q,dict(bounds_m=[1,5,0,3],multiplier=3))
    def test_posterior_matches_direct_conditioning(self):
        gp=self.gp;k=gp.kernel(gp.x,gp.x)+gp.noise*np.eye(len(gp.x));cross=gp.kernel(gp.x,self.q)
        mean=gp.mean+cross.T@np.linalg.solve(k,gp.y-gp.mean)
        cov=gp.kernel(self.q,self.q)-cross.T@np.linalg.solve(k,cross)
        np.testing.assert_allclose(gp.posterior(self.q)[0],mean,atol=1e-12)
        np.testing.assert_allclose(gp.covariance(self.q),cov,atol=1e-12)
        np.testing.assert_allclose(gp.posterior(self.q)[1],np.diag(cov),atol=1e-12)
    def test_mi_joint_logdet_and_redundancy(self):
        obj=Objective('mi',self.q,self.w);gp=self.gp
        expected=.5*np.linalg.slogdet(np.eye(2)+gp.covariance(self.s)/gp.noise)[1]
        self.assertAlmostEqual(obj.score(gp,self.s),expected,places=12)
        duplicated=np.repeat(self.s[:1],2,axis=0)
        self.assertLess(obj.score(gp,duplicated),2*obj.score(gp,self.s[:1]))
    def test_emse_exact_variance_reduction(self):
        obj=Objective('emse',self.q,self.w)
        before=self.gp.posterior(self.q)[1]
        after=GaussianProcess(self.gp.x,self.gp.y,self.hp)
        for x in self.s: after.append(x,20.)
        self.assertAlmostEqual(obj.score(self.gp,self.s),float(np.mean(before-after.posterior(self.q)[1])),places=12)
    def test_uniform_wivr_equals_emse(self):
        uniform=Objective('wivr',self.q,np.ones(len(self.q)))
        self.assertAlmostEqual(uniform.score(self.gp,self.s),Objective('emse',self.q,self.w).score(self.gp,self.s),places=12)
    def test_nonuniform_wivr_direct_formula(self):
        c=self.gp.covariance(self.s);cross=self.gp.covariance(self.s,self.q)
        reduction=np.sum(cross*np.linalg.solve(c+self.gp.noise*np.eye(2),cross),axis=0)
        self.assertAlmostEqual(Objective('wivr',self.q,self.w).score(self.gp,self.s),float(self.w@reduction/self.w.sum()),places=12)
    def test_full_temporal_domain_and_nonuniform_weights(self):
        self.assertEqual(set(self.q[:,2]),{0.,100.,200.})
        self.assertEqual(set(self.w),{1.,3.})
        with self.assertRaises(ValueError):priority_weights(self.q,dict(bounds_m=[-1,5,-1,5],multiplier=3))
    def test_cache_invalidated_after_update(self):
        obj=Objective('emse',self.q,self.w);a=obj.score(self.gp,self.s)
        self.assertEqual(obj.score(self.gp,self.s),a);self.assertEqual(obj.cache_hits,1)
        self.gp.append(self.s[0],20.);b=obj.score(self.gp,self.s)
        self.assertLess(b,a);self.assertEqual(obj.calls,2)
    def test_split_identity_disjoint(self):
        rows=[dict(record_id=str(i),x_m=0,y_m=0,t_s=i,temperature_c=20) for i in range(10)]
        a,h,e=split_experiment(rows,[[0,0]],.1)
        self.assertFalse({r['record_id'] for r in a}&{r['record_id'] for r in h})
        self.assertEqual(len(a)+len(h)+len(e),len(rows))
    def test_acquisition_no_interpolation_or_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'acquisition.csv'
            save_csv(path,[dict(record_id='a',x_m=1,y_m=1,t_s=5,temperature_c=21)])
            acq=Acquisition(path,.25,.1)
            self.assertIsNone(acq.acquire([1,1],4.))
            self.assertIsNone(acq.acquire([2,1],5.))
            self.assertEqual(acq.acquire([1,1],5.)['temperature_c'],21.)
            self.assertIsNone(acq.acquire([1,1],5.))
    def test_every_recursive_score_dispatches_selected_policy(self):
        class Available:
            def next_time(self,xy,arrival_s,deadline_s):return arrival_s
        points=np.array([[0,0],[1,0],[0,1]])
        travel=np.linalg.norm(points[:,None]-points[None,:],axis=2)*10
        source=REPO/'Code/Robot_close_loop_system/testing_code/baseline/grg.py'
        for policy in ('mi','emse','wivr'):
            obj=Objective(policy,self.q,self.w)
            planner=make_planner(source,points,travel,Available(),obj,
                                 dict(sensing_s=1,grg_depth=2,grg_time_splits=2,budget_s=100))
            # If recursion ever invokes the old MI-only implementation, this GP
            # deliberately has no covariance_many method and the test fails.
            result=planner.plan(self.gp,0,0,100,blocked=[0])
            self.assertGreater(obj.calls,1)
            events=planner.schedule(result.route,0)[0];events[:,2]*=3600
            self.assertAlmostEqual(result.gain,obj.score(self.gp,events),places=12)
            self.assertLessEqual(result.finish_s,100.)
    def test_unreachable_edges_remain_infeasible(self):
        class Available:
            def next_time(self,xy,arrival_s,deadline_s):return arrival_s
        planner=make_planner(REPO/'Code/Robot_close_loop_system/testing_code/baseline/grg.py',
            [[0,0],[1,0]],np.array([[0,np.inf],[np.inf,0]]),Available(),
            Objective('mi',self.q,self.w),dict(sensing_s=1,grg_depth=2,grg_time_splits=2,budget_s=100))
        self.assertEqual(planner.plan(self.gp,0,0,100,blocked=[0]).route,[0])


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(MathTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    save_json(Path(__file__).resolve().parents[1]/'result/math_tests.json',
              dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful()))
    raise SystemExit(0 if result.wasSuccessful() else 1)
