"""Stage one: a cheap, deterministic classifier that decides what deserves a look.

A Random Forest is used here not because it wins benchmarks but because of four
properties that matter more in a triage pipeline than the last point of
accuracy:

1. it takes mixed tabular features without scaling;
2. it trains in seconds on a modest labelled set, which is all anyone has;
3. it is deterministic at inference, so the same event always scores the same;
4. it reports feature importances, so an analyst can be shown *why* something
   surfaced instead of being asked to trust a number.

Point four is the one that gets underrated.  An alert nobody can interrogate
gets ignored after the first week.

The threshold is chosen from the relative cost of a missed detection versus an
analyst's time — never from the default 0.5, which encodes the assumption that
both errors cost the same.  They do not.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve

from .features import FEATURE_NAMES, extract
from .schema import Event

__all__ = ["TriageModel", "Metrics", "time_split", "choose_threshold"]


def time_split(events: Sequence[Event], train_fraction: float = 0.7) -> tuple[list[Event], list[Event]]:
    """Split a time-ordered stream into earlier train and later test halves.

    A random split would leak: the same attack episode would land on both sides
    and the model would be scored on events whose neighbours it had memorised.
    Splitting on time reproduces the only situation that matters — being asked
    about traffic that has not happened yet.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    ordered = sorted(events, key=lambda e: e.ts)
    cut = int(len(ordered) * train_fraction)
    if cut == 0 or cut == len(ordered):
        raise ValueError("split produced an empty side; supply more events")
    return ordered[:cut], ordered[cut:]


@dataclass
class Metrics:
    """Evaluation at a chosen operating point, plus threshold-free PR-AUC."""

    threshold: float
    precision: float
    recall: float
    false_positive_rate: float
    average_precision: float
    n_events: int
    n_positive: int
    n_flagged: int
    alerts_per_1k_events: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "threshold": round(self.threshold, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "false_positive_rate": round(self.false_positive_rate, 6),
            "average_precision": round(self.average_precision, 4),
            "n_events": self.n_events,
            "n_positive": self.n_positive,
            "n_flagged": self.n_flagged,
            "alerts_per_1k_events": round(self.alerts_per_1k_events, 2),
        }

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def choose_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    cost_false_negative: float = 20.0,
    cost_false_positive: float = 1.0,
) -> float:
    """Pick the score cut-off that minimises expected cost.

    ``cost_false_negative`` is how much worse a missed detection is than an
    analyst spending a minute on a false alarm.  The default of 20 says a miss
    costs twenty wasted minutes, which is conservative for most environments and
    is exactly the number a team should argue about and then change.

    Returns 0.5 when the labels contain a single class, since no operating
    point can be estimated from that.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5

    candidates = np.unique(np.concatenate([scores, [0.0, 1.0]]))
    best_threshold = 0.5
    best_cost = float("inf")
    for threshold in candidates:
        predicted = scores >= threshold
        false_negatives = int(np.sum((y_true == 1) & ~predicted))
        false_positives = int(np.sum((y_true == 0) & predicted))
        cost = false_negatives * cost_false_negative + false_positives * cost_false_positive
        # `<=` rather than `<`: candidates are scanned in ascending order, so on
        # a tie the later — higher — threshold wins.  Equal cost for fewer
        # alerts is strictly better for the person reading them.
        if cost <= best_cost:
            best_cost = cost
            best_threshold = float(threshold)
    return best_threshold


@dataclass
class TriageModel:
    """Random Forest stage-one filter over normalised events."""

    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_leaf: int = 2
    random_state: int = 20260817
    #: Worker count passed to scikit-learn.  Deliberately left at ``None``
    #: (single process) rather than ``-1``.
    #:
    #: With ``n_jobs=-1`` the per-tree probabilities are accumulated in whatever
    #: order the workers finish, and floating-point addition is not associative,
    #: so the same seed produces score vectors that differ in the last bits.
    #: Measured on this package: four fits with an identical seed gave four
    #: distinct score vectors at ``-1`` and one at ``None``.  Those last bits
    #: matter, because an event sitting exactly on the threshold flips in and out
    #: of the alert set between runs.
    #:
    #: Reproducibility is one of the four reasons this pipeline uses a forest at
    #: all, so it wins over the wall-clock saving.  Set it to ``-1`` yourself if
    #: you are training on enough data to care and can live with the drift.
    n_jobs: int | None = None
    #: Operating point.  Leave as ``None`` to have :meth:`evaluate` derive one
    #: from the cost ratio on every call; set it explicitly to pin the model to
    #: a threshold calibrated elsewhere.  :meth:`evaluate` never writes to it.
    threshold: float | None = None
    #: The threshold the most recent :meth:`evaluate` actually used.  Purely
    #: informational — kept separate from :attr:`threshold` so that repeated
    #: calls with different cost ratios each derive their own operating point
    #: instead of inheriting the first one.
    last_threshold: float | None = field(default=None, repr=False, init=False)
    _model: RandomForestClassifier | None = field(default=None, repr=False, init=False)

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(self, events: Sequence[Event]) -> TriageModel:
        """Fit on a labelled, time-ordered stream."""
        events = list(events)
        X, y = extract(events)
        if X.shape[0] == 0:
            raise ValueError("cannot fit on an empty event stream")
        if len(np.unique(y)) < 2:
            raise ValueError(
                "training data contains a single class; the generator needs at "
                "least one attack episode inside the training window"
            )
        # class_weight='balanced' reweights the loss instead of resampling, so
        # no synthetic rows are invented and every real event is kept.
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self._model.fit(X, y)
        return self

    def score_events(self, events: Sequence[Event]) -> np.ndarray:
        """Return P(malicious) for each event, in input order."""
        if self._model is None:
            raise RuntimeError("model is not fitted; call fit() first")
        X, _ = extract(events)
        if X.shape[0] == 0:
            return np.zeros((0,))
        return self._model.predict_proba(X)[:, 1]

    def evaluate(
        self,
        events: Sequence[Event],
        cost_false_negative: float = 20.0,
        cost_false_positive: float = 1.0,
        threshold: float | None = None,
    ) -> Metrics:
        """Score *events* and report metrics at the chosen operating point.

        .. warning::
           When *threshold* is not supplied it is derived from the labels of
           *this* set, which is also the set the precision and recall below are
           measured on.  That makes those two figures optimistic; only
           ``average_precision``, which is threshold-free, is unaffected.  For
           an honest operating point, calibrate on a held-out slice of the
           training data and pass the result in as *threshold*.
        """
        events = list(events)
        _, y = extract(events)
        scores = self.score_events(events)

        if threshold is None:
            threshold = self.threshold
        if threshold is None:
            threshold = choose_threshold(y, scores, cost_false_negative, cost_false_positive)
        self.last_threshold = threshold

        predicted = scores >= threshold
        true_positives = int(np.sum((y == 1) & predicted))
        false_positives = int(np.sum((y == 0) & predicted))
        false_negatives = int(np.sum((y == 1) & ~predicted))
        true_negatives = int(np.sum((y == 0) & ~predicted))

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0
        fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) else 0.0
        ap = float(average_precision_score(y, scores)) if len(np.unique(y)) > 1 else 0.0

        return Metrics(
            threshold=float(threshold),
            precision=precision,
            recall=recall,
            false_positive_rate=fpr,
            average_precision=ap,
            n_events=len(y),
            n_positive=int(np.sum(y == 1)),
            n_flagged=int(np.sum(predicted)),
            alerts_per_1k_events=1000.0 * float(np.sum(predicted)) / len(y) if len(y) else 0.0,
        )

    def pr_curve(self, events: Sequence[Event]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(precision, recall, thresholds)`` for plotting."""
        _, y = extract(events)
        scores = self.score_events(events)
        return precision_recall_curve(y, scores)

    def top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Return the *n* most important features, descending.

        This is what gets shown next to an alert.  A detection an analyst can
        see the reasons for is one they will act on.
        """
        if self._model is None:
            raise RuntimeError("model is not fitted; call fit() first")
        importances = self._model.feature_importances_
        order = np.argsort(importances)[::-1][:n]
        return [(FEATURE_NAMES[i], float(importances[i])) for i in order]

    @staticmethod
    def explain_row(X: np.ndarray, index: int, n: int = 5) -> list[tuple[str, float]]:
        """Return the *n* most unusual features of row *index* in matrix *X*.

        This is a presentation aid, not a per-prediction attribution method: it
        shows which inputs were unusual relative to the rest of the batch, not
        how much each one moved the score.  For real attribution use a dedicated
        tool such as SHAP; this keeps the dependency list short and is honest
        about what it is.
        """
        if X.shape[0] == 0:
            return []
        if not 0 <= index < X.shape[0]:
            raise IndexError(f"index {index} is outside the event range")
        row = X[index]
        # Compare against the column median so that always-large features such
        # as log_bytes_out do not dominate the list.
        deviation = row - np.median(X, axis=0)
        order = np.argsort(np.abs(deviation))[::-1][:n]
        return [(FEATURE_NAMES[i], float(row[i])) for i in order]

    def explain_event(self, events: Sequence[Event], index: int, n: int = 5) -> list[tuple[str, float]]:
        """Convenience wrapper around :meth:`explain_row` that extracts first.

        Prefer :meth:`explain_row` in a loop: this re-runs feature extraction
        over the whole stream on every call.
        """
        X, _ = extract(events)
        return self.explain_row(X, index, n)
