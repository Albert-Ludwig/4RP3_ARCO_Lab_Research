"""Verify the chronological data split, recorded replay, and archived artifacts."""

import hashlib
import io
import json
from pathlib import Path
import unittest

from data import ROOT, WORKSPACE, PROCESSED, load_split
from experiment import save_json
import test_replay

RESULT=ROOT/"10.18_result/grg_correcness/current"


class DataChecks(unittest.TestCase):
    def test_exact_2000_distinct_real_training_records_and_chronological_split(self):
        train,test=load_split()
        self.assertEqual(len(train),2000)
        self.assertEqual(len({r["record_id"] for r in train}),2000)
        self.assertEqual([sum(day*24<=r["t_h"]<(day+1)*24 for r in train) for day in range(3)],
                         [667,667,666])
        self.assertTrue(all(0<=r["t_h"]<72 for r in train))
        self.assertTrue(all(72<=r["t_h"]<96 for r in test))
        self.assertFalse(set(r["record_id"] for r in train).intersection(r["record_id"] for r in test))
        self.assertTrue(all(0<=r["temperature_c"]<=50 for r in train+test))


def main():
    RESULT.mkdir(exist_ok=True,parents=True)
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(DataChecks),
                              unittest.defaultTestLoader.loadTestsFromModule(test_replay)])
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (RESULT/"intel_replay_tests.log").write_text(stream.getvalue())
    save_json(RESULT/"intel_replay_tests.json",dict(passed=result.wasSuccessful(),count=result.testsRun,
              failures=len(result.failures),errors=len(result.errors),evidence=test_replay.EVIDENCE))
    print(stream.getvalue())
    previous=Path('/private/tmp/gp_intel_archived_results.json')
    if previous.exists():
        before=json.loads(previous.read_text())
        changes=[]
        for relative,digest in before.items():
            if relative.endswith('10.18_test_report.ipynb'):continue
            path=WORKSPACE/relative
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:changes.append(relative)
        save_json(RESULT/"archived_results_preserved.json",dict(passed=not changes,
                    archived_file_count=len(before)-1,changed=changes,
                    note="Main Notebook now contains current Intel results only. Original synthetic cells and outputs are preserved in baseline/archived_notebooks and checked by the benchmark regression."))
        assert not changes,changes
    assert result.wasSuccessful()


if __name__=="__main__":main()
