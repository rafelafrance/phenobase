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
            f"fp = {self.fp:4.0f}    tn = {self.tn:4.0f}\n"
            f"total = {self.total:4.0f}"
        )

    def filter_y(self, thresh_lo: float = 0.5, thresh_hi: float = 0.5) -> None:
        y: np.ndarray = np.stack((self.y_true, self.y_pred))
        self.tp = np.where((y[0, :] == 1.0) & (y[1, :] >= thresh_hi), 1.0, 0.0).sum()
        self.tn = np.where((y[0, :] == 0.0) & (y[1, :] < thresh_lo), 1.0, 0.0).sum()
        self.fn = np.where((y[0, :] == 1.0) & (y[1, :] < thresh_lo), 1.0, 0.0).sum()
        self.fp = np.where((y[0, :] == 0.0) & (y[1, :] >= thresh_hi), 1.0, 0.0).sum()

    @property
    def total(self) -> np.float32:
        return self.tp + self.tn + self.fp + self.fn

    # -----------------------------------------------------------------------------
    @property
    def accuracy(self) -> np.float32:
        return (self.tp + self.tn) / self.total if self.total > 0.0 else 0.0

    @property
    def balanced_accuracy(self) -> np.float32:
        return (self.true_positive_rate + self.true_negative_rate) / 2.0

    # -----------------------------------------------------------------------------
    @property
    def true_positive_rate(self) -> np.float32:
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator > 0.0 else 0.0

    @property
    def sensitivity(self) -> np.float32:
        return self.true_positive_rate

    @property
    def recall(self) -> np.float32:
        return self.true_positive_rate

    # -----------------------------------------------------------------------------
    @property
    def true_negative_rate(self) -> np.float32:
        denominator = self.tn + self.fp
        return self.tn / denominator if denominator > 0.0 else 0.0

    @property
    def specificity(self) -> np.float32:
        return self.true_negative_rate

    # -----------------------------------------------------------------------------
    @property
    def positive_predictive_value(self) -> np.float32:
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator > 0.0 else 0.0

    @property
    def precision(self) -> np.float32:
        return self.positive_predictive_value

    # -----------------------------------------------------------------------------
    @property
    def negative_predictive_value(self) -> np.float32:
        denominator = self.tn + self.fn
        return self.tn / denominator if denominator > 0.0 else 0.0

    # -----------------------------------------------------------------------------
    @property
    def false_negative_rate(self) -> np.float32:
        denominator = self.fn + self.tp
        return self.fn / denominator if denominator > 0.0 else 0.0

    # -----------------------------------------------------------------------------
    @property
    def false_positive_rate(self) -> np.float32:
        denominator = self.tn + self.fp
        return self.fp / denominator if denominator > 0.0 else 0.0

    # -----------------------------------------------------------------------------

    @property
    def true_skills_statistics(self) -> np.float32:
        """AKA Youden's J statistic."""
        return self.sensitivity + self.specificity - 1.0

    # -----------------------------------------------------------------------------
    @property
    def geometric_mean(self) -> np.float32:
        return np.sqrt(self.sensitivity * self.specificity)

    # -----------------------------------------------------------------------------
    @property
    def f1(self) -> np.float32:
        denominator = (2.0 * self.tp) + self.fp + self.fn
        return (2.0 * self.tp) / denominator if denominator > 0.0 else 0.0

    def f_beta(self, beta: np.float32 = 1.0) -> np.float32:
        beta_sq = beta * beta
        numerator = (1 + beta_sq) * self.precision * self.recall
        return numerator / ((beta_sq * self.precision) + self.recall)
