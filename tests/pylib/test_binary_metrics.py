import unittest

import numpy as np

from phenobase.pylib.binary_metrics import Metrics


class TestMetrics(unittest.TestCase):
    def test_classify_01(self):
        y_true = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        metrics = Metrics(y_true, y_pred)
        metrics.classify(thresh_lo=0.21, thresh_hi=0.8)
        # metrics.display_matrix()
        self.assertEqual(metrics.tp, 1.0)
        self.assertEqual(metrics.tn, 1.0)
        self.assertEqual(metrics.fp, 1.0)
        self.assertEqual(metrics.fn, 1.0)

    def test_classify_02(self):
        y_true = np.array([1.0, 0.0, 1.0])
        y_pred = np.array([0.9, 0.8, 0.7])
        metrics = Metrics(y_true, y_pred)
        metrics.classify()
        # metrics.display_matrix()
        self.assertEqual(metrics.tp, 2.0)
        self.assertEqual(metrics.tn, 0.0)
        self.assertEqual(metrics.fp, 1.0)
        self.assertEqual(metrics.fn, 0.0)

    def test_classify_03(self):
        y_true = np.array([0.0, 1.0, 0.0])
        y_pred = np.array([0.3, 0.2, 0.1])
        metrics = Metrics(y_true, y_pred)
        metrics.classify()
        # metrics.display_matrix()
        self.assertEqual(metrics.tp, 0.0)
        self.assertEqual(metrics.tn, 2.0)
        self.assertEqual(metrics.fp, 0.0)
        self.assertEqual(metrics.fn, 1.0)
