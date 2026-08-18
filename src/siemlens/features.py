"""Streaming feature extraction.

Every feature here is computed from events *strictly earlier* than the one
being described.  That constraint is the whole point of the module: log data is
a time series, and a feature that peeks at later events will look excellent in
cross-validation and then fail in production, where the future is not
available.  For the same reason the extractor is a single forward pass with
mutable state rather than a set of pandas group-bys.

The feature set is small and deliberately interpretable.  Each one answers a
question an analyst would ask anyway:

* *rate* — how much is this account doing right now, compared to a moment ago?
* *rarity* — have we ever seen this account on this host, from this address,
  with this user agent?
* *temporal* — is this happening at an hour this account normally sleeps?
* *volume* — is the amount of data unusual for this account specifically?

Nothing here is clever.  Cleverness in feature engineering tends to produce
features nobody can explain to the person who has to action the alert.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .schema import ACTIONS, SOURCES, Event

__all__ = ["FEATURE_NAMES", "FeatureExtractor", "extract", "OFF_HOURS_START", "OFF_HOURS_END"]

#: Hours outside ``[OFF_HOURS_END, OFF_HOURS_START)`` count as off hours.
OFF_HOURS_START = 20
OFF_HOURS_END = 7

_WINDOW_60S = 60.0
_WINDOW_5M = 300.0
_WINDOW_1H = 3600.0

_BASE_FEATURES = (
    "hour_of_day",
    "is_off_hours",
    "is_weekend",
    "outcome_failure",
    "outcome_denied",
    "user_fail_5m",
    "user_events_5m",
    "user_events_60s",
    "ip_distinct_users_1h",
    "ip_events_1h",
    "first_seen_user_host",
    "first_seen_user_ip",
    "agent_rarity",
    "log_bytes_out",
    "bytes_z_for_user",
    "is_external_ip",
    "is_service_account",
)

#: Stable, ordered feature names.  The order matters: it is the column order of
#: the matrix returned by :func:`extract` and therefore the order in which
#: ``RandomForestClassifier.feature_importances_`` is reported.
FEATURE_NAMES: tuple[str, ...] = (
    _BASE_FEATURES
    + tuple(f"source_{s}" for s in SOURCES)
    + tuple(f"action_{a}" for a in ACTIONS)
)


@dataclass
class _Welford:
    """Online mean and variance, so per-user z-scores need no second pass."""

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.n - 1))

    def zscore(self, value: float) -> float:
        # Fewer than two observations gives no dispersion estimate; returning 0
        # is the honest answer, not a large score from a std of zero.
        if self.n < 2:
            return 0.0
        std = self.std
        if std <= 1e-9:
            return 0.0
        return (value - self.mean) / std


def _prune(window: deque, now: float, span: float) -> None:
    while window and now - window[0][0] > span:
        window.popleft()


class FeatureExtractor:
    """Stateful, single-pass feature extractor.

    Instantiate once per stream.  Call :meth:`transform_one` for each event in
    ascending timestamp order; the extractor computes the row first and only
    then folds the event into its own state, which is what keeps the current
    event out of its own rolling windows.
    """

    def __init__(self) -> None:
        self._user_window: dict[str, deque[tuple[float, int]]] = {}
        self._ip_window: dict[str, deque[tuple[float, str]]] = {}
        self._seen_user_host: set[tuple[str, str]] = set()
        self._seen_user_ip: set[tuple[str, str]] = set()
        self._agent_counts: Counter[str] = Counter()
        self._user_bytes: dict[str, _Welford] = {}
        self._total = 0
        self._last_ts: float | None = None

    def transform_one(self, event: Event) -> np.ndarray:
        """Return the feature row for *event* and absorb it into the state."""
        now = event.ts.timestamp()
        if self._last_ts is not None and now < self._last_ts - 1e-6:
            raise ValueError(
                "events must arrive in ascending timestamp order; "
                "sort the stream before extracting features"
            )
        self._last_ts = now

        user_window = self._user_window.setdefault(event.user, deque())
        ip_window = self._ip_window.setdefault(event.src_ip, deque())
        _prune(user_window, now, _WINDOW_5M)
        _prune(ip_window, now, _WINDOW_1H)

        user_fail_5m = sum(1 for ts, failed in user_window if failed)
        user_events_5m = len(user_window)
        user_events_60s = sum(1 for ts, _ in user_window if now - ts <= _WINDOW_60S)
        ip_events_1h = len(ip_window)
        ip_distinct_users_1h = len({u for _, u in ip_window})

        first_seen_user_host = 0.0 if (event.user, event.host) in self._seen_user_host else 1.0
        first_seen_user_ip = 0.0 if (event.user, event.src_ip) in self._seen_user_ip else 1.0

        agent_seen = self._agent_counts.get(event.user_agent, 0)
        agent_rarity = -math.log((agent_seen + 1.0) / (self._total + 2.0))

        stats = self._user_bytes.setdefault(event.user, _Welford())
        bytes_z = stats.zscore(float(event.bytes_out))

        hour = event.ts.hour + event.ts.minute / 60.0
        is_off_hours = 1.0 if (event.ts.hour >= OFF_HOURS_START or event.ts.hour < OFF_HOURS_END) else 0.0
        is_weekend = 1.0 if event.ts.weekday() >= 5 else 0.0

        row = [
            hour,
            is_off_hours,
            is_weekend,
            1.0 if event.outcome == "failure" else 0.0,
            1.0 if event.outcome == "denied" else 0.0,
            float(user_fail_5m),
            float(user_events_5m),
            float(user_events_60s),
            float(ip_distinct_users_1h),
            float(ip_events_1h),
            first_seen_user_host,
            first_seen_user_ip,
            agent_rarity,
            math.log1p(float(event.bytes_out)),
            bytes_z,
            0.0 if event.src_ip.startswith("10.") else 1.0,
            1.0 if event.user.startswith("svc-") else 0.0,
        ]
        row.extend(1.0 if event.source == s else 0.0 for s in SOURCES)
        row.extend(1.0 if event.action == a else 0.0 for a in ACTIONS)

        # --- state update happens strictly after the row is built ---
        user_window.append((now, 1 if event.outcome in ("failure", "denied") else 0))
        ip_window.append((now, event.user))
        self._seen_user_host.add((event.user, event.host))
        self._seen_user_ip.add((event.user, event.src_ip))
        self._agent_counts[event.user_agent] += 1
        stats.update(float(event.bytes_out))
        self._total += 1

        return np.asarray(row, dtype=np.float64)


def extract(events: Sequence[Event] | Iterable[Event]) -> tuple[np.ndarray, np.ndarray]:
    """Extract a feature matrix and label vector from a time-ordered stream.

    Returns ``(X, y)`` where ``X`` has shape ``(n_events, len(FEATURE_NAMES))``.
    The caller is responsible for passing events in ascending timestamp order;
    :func:`siemlens.schema.iter_sorted` does that.
    """
    extractor = FeatureExtractor()
    rows: list[np.ndarray] = []
    labels: list[int] = []
    for event in events:
        rows.append(extractor.transform_one(event))
        labels.append(event.label)

    if not rows:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros((0,), dtype=np.int64)

    return np.vstack(rows), np.asarray(labels, dtype=np.int64)
