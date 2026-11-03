"""Independent timestamp-boundary and no-future-label fixtures."""

import unittest
from strict_protocol import split_records, settings


class StrictChecks(unittest.TestCase):
    def rows(self):
        minutes = [0, 5, 10, 15, 15.01, 20, 30, 45, 60, 75]
        return [dict(sensor_id=1, record_id=i + 1, t_h=72 + minute / 60,
                     x_m=1., y_m=1., temperature_c=20.) for i, minute in enumerate(minutes)]

    def split(self, rows):
        return split_records(rows, {1}, 72., 900., [[0, 15], [15, 30], [30, 60]])

    def test_cutoff_belongs_to_reconstruction_not_forecast(self):
        acquired, reconstruction, future = self.split(self.rows())
        self.assertEqual([r["record_id"] for r in acquired], [1, 3])
        self.assertEqual([r["record_id"] for r in reconstruction], [4])
        self.assertTrue(all(r["t_h"] > 72.25 for group in future.values() for r in group))

    def test_forecast_bins_have_disjoint_boundaries(self):
        _, _, future = self.split(self.rows())
        self.assertEqual([r["record_id"] for r in future["forecast_0_15m"]], [6])
        self.assertEqual([r["record_id"] for r in future["forecast_15_30m"]], [8])
        self.assertEqual([r["record_id"] for r in future["forecast_30_60m"]], [10])

    def test_future_labels_cannot_change_acquisition_pool(self):
        original = self.rows()
        changed = [dict(r, temperature_c=-999 if r["t_h"] > 72.25 else r["temperature_c"]) for r in original]
        self.assertEqual(self.split(original)[0], self.split(changed)[0])

    def test_same_sensor_order_does_not_depend_on_input_order(self):
        self.assertEqual(self.split(self.rows()), self.split(list(reversed(self.rows()))))

    def test_requested_weight_orientation(self):
        config = settings()
        self.assertEqual(config["weights_mi_emse"]["fusion_40_60"], [.4, .6])
        self.assertEqual(config["weights_mi_emse"]["fusion_60_40"], [.6, .4])
        for pair in config["weights_mi_emse"].values():
            self.assertAlmostEqual(sum(pair), 1.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
