"""Deterministic regression tests: no camera, network, GUI or downloaded models."""
import os
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.distance_estimator import DistanceEstimator
from src.core.metric_depth import MetricDepthBackend



class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        with patch.object(DistanceEstimator, 'load_calibration'):
            self.estimator = DistanceEstimator(focal_length=500)

    def test_resolution_invariance(self):
        e = self.estimator
        box = (100, 100, 200, 300)
        self.assertAlmostEqual(e.estimate_distance('chair', box, (480, 640)),
                               e.estimate_distance('chair', tuple(v * 2 for v in box), (960, 1280)))
        self.assertEqual(e.compute_3d_coordinates(box, 2, 640, 480),
                         e.compute_3d_coordinates(tuple(v * 2 for v in box), 2, 1280, 960))

    def test_no_floor_assumption_for_raised_objects(self):
        e = self.estimator
        self.assertEqual(e.estimate_distance('bottle_cup', (100, 50, 150, 100), (480, 640)),
                         e.estimate_distance('bottle_cup', (100, 400, 150, 450), (480, 640)))

    def test_one_track_per_detection_and_expiration(self):
        e = self.estimator
        box = (100, 100, 200, 300)
        e.begin_frame(0)
        e._smooth_distance('chair', box, 3, 0)
        e.begin_frame(0.1)
        e._smooth_distance('chair', box, 3, 0.1)
        e._smooth_distance('chair', box, 1, 0.1)
        self.assertEqual(len(e.tracks), 2)
        e.begin_frame(2)
        self.assertEqual(len(e.tracks), 0)

    def test_approach_is_not_delayed(self):
        e = self.estimator
        box = (100, 100, 200, 300)
        e.begin_frame(0)
        e._smooth_distance('chair', box, 5, 0)
        e.begin_frame(0.1)
        self.assertEqual(e._smooth_distance('chair', box, 0.8, 0.1), 0.8)

    def test_metric_roi_rejects_invalid_and_mixed_depth(self):
        sample = MetricDepthBackend.object_distance
        box = (0, 0, 40, 40)
        self.assertEqual(sample(np.full((40, 40), 2.), box), 2.)
        self.assertIsNone(sample(np.full((40, 40), np.nan), box))
        mixed = np.ones((40, 40)); mixed[:, 20:] = 10
        self.assertIsNone(sample(mixed, box))

    def test_invalid_calibration(self):
        for value in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.estimator.update_focal_length(value, save=False)
            self.assertIsNone(self.estimator.calibrate_with_known_distance(value, 100, 1))

    def test_metric_source_is_explicit(self):
        obj = self.estimator.process_detection('chair', .9, (100, 100, 200, 300),
                                              (480, 640), metric_distance=.7)
        self.assertEqual(obj.risk_level, 'DANGER')
        self.assertEqual(obj.to_dict()['distance_method'], 'metric_depth')
        self.assertEqual(obj.distance_reliability, 'low')

if __name__ == '__main__':
    unittest.main()
