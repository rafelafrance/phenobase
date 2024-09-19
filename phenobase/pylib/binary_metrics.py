"""Wrap scikit-learn's metrics."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Metrics:
    y_true: np.ndarray
    y_pred: np.ndarray
    tp: np.float32 = 0.0
    tn: np.float32 = 0.0
    fp: np.float32 = 0.0
    fn: np.float32 = 0.0

    def display_matrix(self) -> None:
        print(
            f"tp = {self.tp:4.0f}    fn = {self.fn:4.0f}\n"
            f"fp = {self.fp:4.0f}    tn = {self.tn:4.0f}"
        )

    def classify(self, thresh_lo: float = 0.5, thresh_hi: float = 0.5) -> None:
        y: np.ndarray = np.stack((self.y_true, self.y_pred))
        self.tp = np.where((y[0, :] == 1.0) & (y[1, :] >= thresh_hi), 1.0, 0.0).sum()
        self.tn = np.where((y[0, :] == 0.0) & (y[1, :] < thresh_lo), 1.0, 0.0).sum()
        self.fn = np.where((y[0, :] == 1.0) & (y[1, :] < thresh_lo), 1.0, 0.0).sum()
        self.fp = np.where((y[0, :] == 0.0) & (y[1, :] >= thresh_hi), 1.0, 0.0).sum()

    @property
    def total(self) -> np.float32:
        return self.tp + self.tn + self.fp + self.fn

    @property
    def accuracy(self) -> np.float32:
        return (self.tp + self.tn) / self.total if self.total > 0.0 else 0.0

    @property
    def f1(self) -> np.float32:
        denominator = (2.0 * self.tp) + self.fp + self.fn
        return (2.0 * self.tp) / denominator if denominator > 0.0 else 0.0

    @property
    def precision(self) -> np.float32:
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator > 0.0 else 0.0

    @property
    def recall(self) -> np.float32:
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator > 0.0 else 0.0

    @property
    def sensitivity(self) -> np.float32:
        return self.precision

    @property
    def specificity(self) -> np.float32:
        denominator = self.tn + self.fp
        return self.tn / denominator if denominator > 0.0 else 0.0

    @property
    def tnr(self) -> np.float32:
        """True negative rate."""
        return self.specificity

    @property
    def tpr(self) -> np.float32:
        """True positive rate."""
        return self.precision

    @property
    def fpr(self) -> np.float32:
        """False positive rate."""
        return self.specificity

    @property
    def tss(self) -> np.float32:
        """True skills statistics, aka Youden's J statistic."""
        return self.tpr - self.fpr

    @property
    def balanced_accuracy(self) -> np.float32:
        return (self.tpr + self.tnr) / 2.0

    @property
    def ppv(self) -> np.float32:
        """Positive predictive value."""
        return self.precision

    @property
    def npv(self) -> np.float32:
        """Negative predictive value."""
        denominator = self.tn + self.fn
        return self.tn / denominator if denominator > 0.0 else 0.0

    @property
    def g_mean(self) -> np.float32:
        """Geometric mean."""
        return np.sqrt(self.sensitivity * self.specificity)
