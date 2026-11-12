"""Timestamp-aligned recorded acquisition, including consecutive Home samples."""

import copy
import json
import unittest
import sys
import numpy as np

from data import BASE, ROOT
from replay import IntelSource, IntelPlanner
from gp import GPModel
from mission import MissionController
from navigation import GridMap, PythonNavigationSimulator

sys.path.append(str(ROOT / "testing_code/grg_correctness"))
from reference import ReferenceMI

EVIDENCE = {}


class ReplayChecks(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(record_id=i,sensor_id=16,x_m=1.5,y_m=2.,t_h=72+s/3600,
                       temperature_c=20.+i) for i,s in enumerate([30,60,90],1)]
        self.source=IntelSource(self.rows)
        self.planner=IntelPlanner(np.array([[1.5,2.]]),np.zeros((1,1)),72,5,source=self.source)

    def test_schedule_waits_for_real_record_and_charges_each_dwell(self):
        query,arrivals,finish=self.planner.schedule([0,0,0,0],0)
        np.testing.assert_allclose(arrivals,[30,60],atol=1e-8)
        np.testing.assert_allclose(query[:,2],[r["t_h"] for r in self.rows[:2]],atol=1e-12)
        self.assertAlmostEqual(finish,65,places=7)
        self.assertFalse(self.source.used)
        self.assertTrue(np.isinf(self.source.next_time([1.5,2.],73)))

    def test_next_record_matching_never_returns_future_value_early_or_reuses_record(self):
        self.assertAlmostEqual(self.source.next_time([1.5,2.],72),self.rows[0]["t_h"])
        self.assertFalse(self.source.measure([1.5,2.],72).valid())
        record=self.source.measure([1.5,2.],self.rows[0]["t_h"])
        self.assertTrue(record.valid())
        with self.assertRaises(AssertionError):
            self.source.measure([1.5,2.],self.rows[0]["t_h"])

    def test_full_loop_wait_clock_budget_and_only_executed_gp_updates(self):
        config=json.loads((BASE/"config.json").read_text())
        config["map"]["home_m"]=[1.5,2.]
        mission={**config["mission"],"start_h":72.,"budget_s":70.}
        gp=GPModel.load(ROOT/"10.18_result/baseline/intel_lab/forecast/initial_gp.pt")
        controller=MissionController(gp,self.planner,PythonNavigationSimulator(GridMap(config["map"]),.35),
                                     self.source,mission)
        summary=controller.run()
        self.assertEqual(summary["measurements"],2)
        self.assertEqual(len(gp.x),2002)
        self.assertEqual(gp.version,2)
        self.assertEqual(self.source.used,[1,2])
        self.assertAlmostEqual(summary["elapsed_s"],65,places=7)
        self.assertAlmostEqual(summary["measurement_wait_time_s"],55,places=7)
        self.assertEqual(summary["dwell_time_s"],10.)
        self.assertAlmostEqual(summary["mission_time_s"],65,places=7)
        self.assertTrue(summary["returned_home"])

    def test_recorded_timestamps_joint_mi_matches_independent_conditioning(self):
        gp=GPModel.load(ROOT/"10.18_result/baseline/intel_lab/forecast/initial_gp.pt")
        query,_,_=self.planner.schedule([0,0,0,0],0)
        expected=float(ReferenceMI(gp).scores(query))
        actual=float(self.planner.score_routes([[0,0,0,0]],gp,0)[0])
        self.assertAlmostEqual(actual,expected,places=8)
        EVIDENCE["recorded_time_mi_difference_nats"]=abs(actual-expected)


if __name__=="__main__":unittest.main(verbosity=2)
