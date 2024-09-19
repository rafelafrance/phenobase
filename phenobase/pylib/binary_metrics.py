"""Wrap scikit-learn's metrics."""
import math
from dataclasses import dataclass, field


@dataclass
class Metrics:
    tp: float = 0.0
    tn: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    y_true: list[float] = field(default_factory=list)
    y_pred: list[float] = field(default_factory=list)

    def display_matrix(self) -> None:
        print(
            f"tp = {self.tp:4.0f}    fn = {self.fn:4.0f}\n"
            f"fp = {self.fp:4.0f}    tn = {self.tn:4.0f}"
        )

    def add(self, true, pred) -> None:
        self.y_true.append(true)
        self.y_pred.append(pred)

    def remove_equivocal(
        self, threshold_lo: float = 0.5, threshold_hi: float = 0.5
    ) -> None:
        self.tp, self.tn, self.fn, self.fp = 0.0, 0.0, 0.0, 0.0
        for true, pred in zip(self.y_true, self.y_pred, strict=True):
            if threshold_lo <= pred < threshold_hi:
                continue
            self.tp += 1.0 if (true == 1.0 and pred >= threshold_hi) else 0.0
            self.tn += 1.0 if (true == 0.0 and pred < threshold_lo) else 0.0
            self.fn += 1.0 if (true == 1.0 and pred < threshold_lo) else 0.0
            self.fp += 1.0 if (true == 0.0 and pred >= threshold_hi) else 0.0

    @property
    def total(self) -> float:
        return self.tp + self.tn + self.fp + self.fn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total > 0.0 else 0.0

    @property
    def f1(self) -> float:
        denominator = (2.0 * self.tp) + self.fp + self.fn
        return (2.0 * self.tp) / denominator if denominator > 0.0 else 0.0

    @property
    def precision(self) -> float:
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator > 0.0 else 0.0

    @property
    def recall(self) -> float:
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator > 0.0 else 0.0

    @property
    def sensitivity(self) -> float:
        return self.precision

    @property
    def specificity(self) -> float:
        denominator = self.tn + self.fp
        return self.tn / denominator if denominator > 0.0 else 0.0

    @property
    def tnr(self) -> float:
        """True negative rate."""
        return self.specificity

    @property
    def tpr(self) -> float:
        """True positive rate."""
        return self.precision

    @property
    def fpr(self) -> float:
        """False positive rate."""
        return self.specificity

    @property
    def tss(self) -> float:
        """True skills statistics, aka Youden's J statistic."""
        return self.tpr - self.fpr

    @property
    def balanced_accuracy(self) -> float:
        return (self.tpr + self.tnr) / 2.0

    @property
    def ppv(self) -> float:
        """Positive predictive value."""
        return self.precision

    @property
    def npv(self) -> float:
        """Negative predictive value."""
        denominator = self.tn + self.fn
        return self.tn / denominator if denominator > 0.0 else 0.0

    @property
    def g_mean(self) -> float:
        """Geometric mean."""
        return math.sqrt(self.sensitivity * self.specificity)
