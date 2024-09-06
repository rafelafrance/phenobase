"""Wrap scikit-learn's confusion metrics."""

from dataclasses import dataclass, field


@dataclass
class Metrics:
    tp: float = 0.0
    tn: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    y_true: list[float] = field(default_factory=list)
    y_pred: list[float] = field(default_factory=list)

    def display(self):
        print(
            f"tp = {self.tp:4d}    tn = {self.tn:4d}    "
            f"fp = {self.fp:4d}    fn = {self.fn:4d}"
        )

    def add(self, true, pred):
        self.y_true.append(true)
        self.y_pred.append(pred)

    def count(self):
        self.tp, self.tn, self.fn, self.fp = 0.0, 0.0, 0.0, 0.0
        for true, pred in zip(self.y_true, self.y_pred, strict=True):
            self.tp += 1.0 if (true == 1.0 and pred == 1.0) else 0.0
            self.tn += 1.0 if (true == 0.0 and pred == 0.0) else 0.0
            self.fn += 1.0 if (true == 1.0 and pred == 0.0) else 0.0
            self.fp += 1.0 if (true == 0.0 and pred == 1.0) else 0.0

    @property
    def total(self):
        return self.tp + self.tn + self.fp + self.fn

    @property
    def accuracy(self):
        return (self.tp + self.tn) / self.total

    @property
    def f1(self):
        return (2.0 * self.tp) / ((2.0 * self.tp) + self.fp + self.fn)

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn)

    @property
    def sensitivity(self):
        return self.precision

    @property
    def specificity(self):
        return self.tn / (self.tn / self.fp)

    @property
    def tnr(self):
        """True negative rate."""
        return self.specificity

    @property
    def tpr(self):
        """True positive rate."""
        return self.precision

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
        return self.tn / (self.tn + self.fn)
