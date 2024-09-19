"""Wrap scikit-learn's metrics."""

from dataclasses import dataclass, field


@dataclass
class Metrics:
    tp: float = 0.0
    tn: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    y_true: list[float] = field(default_factory=list)
    y_pred: list[float] = field(default_factory=list)

    def display_matrix(self):
        print(
            f"tp = {self.tp:4.0f}    fn = {self.fn:4.0f}\n"
            f"fp = {self.fp:4.0f}    tn = {self.tn:4.0f}"
        )

    def add(self, true, pred):
        self.y_true.append(true)
        self.y_pred.append(pred)

    def remove_equivocal(self, threshold_lo: float = 0.5, threshold_hi: float = 0.5):
        self.tp, self.tn, self.fn, self.fp = 0.0, 0.0, 0.0, 0.0
        for true, pred in zip(self.y_true, self.y_pred, strict=True):
            if threshold_lo <= pred < threshold_hi:
                continue
            self.tp += 1.0 if (true == 1.0 and pred >= threshold_hi) else 0.0
            self.tn += 1.0 if (true == 0.0 and pred < threshold_lo) else 0.0
            self.fn += 1.0 if (true == 1.0 and pred < threshold_lo) else 0.0
            self.fp += 1.0 if (true == 0.0 and pred >= threshold_hi) else 0.0

    @property
    def total(self):
        return self.tp + self.tn + self.fp + self.fn

    @property
    def accuracy(self):
        return (self.tp + self.tn) / self.total if self.total > 0.0 else 0.0

    @property
    def f1(self):
        denominator = (2.0 * self.tp) + self.fp + self.fn
        return (2.0 * self.tp) / denominator if denominator > 0.0 else 0.0

    @property
    def precision(self):
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator > 0.0 else 0.0

    @property
    def recall(self):
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator > 0.0 else 0.0

    @property
    def sensitivity(self):
        return self.precision

    @property
    def specificity(self):
        denominator = self.tn + self.fp
        return self.tn / denominator if denominator > 0.0 else 0.0

    @property
    def tnr(self):
        """True negative rate."""
        return self.specificity

    @property
    def tpr(self):
        """True positive rate."""
        return self.precision

    @property
    def fpr(self):
        """False positive rate."""
        denominator = self.tn + self.fp
        return self.fp / denominator if denominator > 0.0 else 0.0

    @property
    def tss(self):
        """True skills statistics, aka Youden's J statistic."""
        return self.tpr - self.fpr

    @property
    def balanced_accuracy(self):
        return (self.tpr + self.tnr) / 2.0

    @property
    def ppv(self):
        """Positive predictive value."""
        return self.precision

    @property
    def npv(self):
        """Negative predictive value."""
        denominator = self.tn + self.fn
        return self.tn / denominator if denominator > 0.0 else 0.0
