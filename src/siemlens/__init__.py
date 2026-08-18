"""siemlens — two-stage security log triage.

A small, readable reference implementation of the pattern described in the
README: a cheap deterministic classifier decides what a human should look at,
and only then does an expensive language model explain what was found.

    >>> from siemlens import generate, TriageModel, time_split
    >>> events, episodes = generate(days=7, events_per_day=300, seed=1)
    >>> train, test = time_split(events)
    >>> model = TriageModel().fit(train)
    >>> metrics = model.evaluate(test)
    >>> metrics.recall > 0
    True

The two stages are deliberately separable. :mod:`siemlens.triage` has no
knowledge of language models, and :mod:`siemlens.interpret` never sees a
feature vector; they meet only at :class:`siemlens.interpret.Alert`.

Author: Akbar Tilek Uulu (Тилек Уулу Ахбар)
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Akbar Tilek Uulu"

from .features import FEATURE_NAMES, FeatureExtractor, extract
from .generate import AttackEpisode, generate
from .interpret import (
    Alert,
    Backend,
    OfflineBackend,
    Verdict,
    VerdictSchemaError,
    build_prompt,
    cluster_alerts,
    ground_verdict,
    interpret,
    parse_verdict,
    sanitise,
    scan_for_injection,
)
from .schema import Event, from_mapping, iter_sorted, read_jsonl, write_jsonl
from .triage import Metrics, TriageModel, choose_threshold, time_split

__all__ = [
    "__version__",
    "__author__",
    "Event",
    "from_mapping",
    "iter_sorted",
    "read_jsonl",
    "write_jsonl",
    "generate",
    "AttackEpisode",
    "FEATURE_NAMES",
    "FeatureExtractor",
    "extract",
    "TriageModel",
    "Metrics",
    "time_split",
    "choose_threshold",
    "Alert",
    "Verdict",
    "Backend",
    "OfflineBackend",
    "VerdictSchemaError",
    "build_prompt",
    "parse_verdict",
    "ground_verdict",
    "interpret",
    "cluster_alerts",
    "sanitise",
    "scan_for_injection",
]
