"""Tests for feature extraction.

The important ones here are the leakage tests. A feature that sees the event it
describes, or any event after it, will look excellent offline and fail in
production. These tests are what stop that from creeping back in.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from siemlens.features import FEATURE_NAMES, FeatureExtractor, extract
from siemlens.schema import Event

BASE = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


def make_event(**overrides) -> Event:
    params = {
        "ts": BASE,
        "source": "auth",
        "action": "login",
        "outcome": "success",
        "user": "user001",
        "src_ip": "10.0.0.5",
        "host": "host-01",
        "user_agent": "curl/8.4.0",
        "bytes_out": 1000,
        "label": 0,
    }
    params.update(overrides)
    return Event(**params)


def column(name: str) -> int:
    return FEATURE_NAMES.index(name)


# --------------------------------------------------------------------------
# shape and contract
# --------------------------------------------------------------------------


def test_matrix_width_matches_the_declared_feature_names():
    X, _ = extract([make_event()])
    assert X.shape == (1, len(FEATURE_NAMES))


def test_feature_names_are_unique():
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_empty_input_returns_an_empty_matrix_of_the_right_width():
    X, y = extract([])
    assert X.shape == (0, len(FEATURE_NAMES))
    assert y.shape == (0,)


def test_labels_are_carried_through_untouched():
    events = [make_event(label=0), make_event(ts=BASE + timedelta(seconds=1), label=1)]
    _, y = extract(events)
    assert list(y) == [0, 1]


def test_no_feature_is_named_after_the_label():
    # The label must never become an input; this catches an accidental rename.
    assert "label" not in FEATURE_NAMES


# --------------------------------------------------------------------------
# leakage
# --------------------------------------------------------------------------


def test_an_event_is_not_counted_in_its_own_window():
    X, _ = extract([make_event()])
    assert X[0, column("user_events_5m")] == 0.0
    assert X[0, column("user_events_60s")] == 0.0
    assert X[0, column("ip_events_1h")] == 0.0


def test_windows_count_only_earlier_events():
    events = [make_event(ts=BASE + timedelta(seconds=i)) for i in range(4)]
    X, _ = extract(events)
    assert list(X[:, column("user_events_5m")]) == [0.0, 1.0, 2.0, 3.0]


def test_first_seen_flags_are_set_once_and_then_cleared():
    events = [
        make_event(ts=BASE),
        make_event(ts=BASE + timedelta(seconds=1)),
    ]
    X, _ = extract(events)
    assert X[0, column("first_seen_user_host")] == 1.0
    assert X[1, column("first_seen_user_host")] == 0.0


def test_out_of_order_events_are_rejected():
    extractor = FeatureExtractor()
    extractor.transform_one(make_event(ts=BASE + timedelta(seconds=10)))
    with pytest.raises(ValueError, match="ascending timestamp order"):
        extractor.transform_one(make_event(ts=BASE))


def test_bytes_zscore_needs_two_observations_before_it_speaks():
    events = [make_event(ts=BASE + timedelta(seconds=i), bytes_out=1000) for i in range(2)]
    X, _ = extract(events)
    # First event: no history at all. Second: one prior sample, still no
    # dispersion estimate. Neither may claim an anomaly.
    assert X[0, column("bytes_z_for_user")] == 0.0
    assert X[1, column("bytes_z_for_user")] == 0.0


# --------------------------------------------------------------------------
# individual features behave as described
# --------------------------------------------------------------------------


def test_failures_are_counted_separately_from_successes():
    events = [
        make_event(ts=BASE, outcome="failure"),
        make_event(ts=BASE + timedelta(seconds=1), outcome="failure"),
        make_event(ts=BASE + timedelta(seconds=2), outcome="success"),
    ]
    X, _ = extract(events)
    assert X[2, column("user_fail_5m")] == 2.0
    assert X[2, column("outcome_failure")] == 0.0
    assert X[1, column("outcome_failure")] == 1.0


def test_denied_counts_towards_failures_in_the_window():
    events = [
        make_event(ts=BASE, source="network", action="connect", outcome="denied"),
        make_event(ts=BASE + timedelta(seconds=1)),
    ]
    X, _ = extract(events)
    assert X[1, column("user_fail_5m")] == 1.0
    assert X[0, column("outcome_denied")] == 1.0


def test_distinct_users_from_one_address_are_counted():
    events = [
        make_event(ts=BASE + timedelta(seconds=i), user=f"user{i:03d}", src_ip="203.0.113.1")
        for i in range(5)
    ]
    X, _ = extract(events)
    assert X[4, column("ip_distinct_users_1h")] == 4.0


def test_external_addresses_are_marked():
    internal, _ = extract([make_event(src_ip="10.1.2.3")])
    external, _ = extract([make_event(src_ip="203.0.113.9")])
    assert internal[0, column("is_external_ip")] == 0.0
    assert external[0, column("is_external_ip")] == 1.0


def test_off_hours_flag_follows_the_configured_boundaries():
    night, _ = extract([make_event(ts=BASE.replace(hour=3))])
    day, _ = extract([make_event(ts=BASE.replace(hour=14))])
    assert night[0, column("is_off_hours")] == 1.0
    assert day[0, column("is_off_hours")] == 0.0


def test_weekend_flag_is_set_on_saturday():
    saturday = datetime(2026, 6, 6, 12, 0, tzinfo=timezone.utc)
    assert saturday.weekday() == 5
    X, _ = extract([make_event(ts=saturday)])
    assert X[0, column("is_weekend")] == 1.0


def test_service_accounts_are_marked():
    X, _ = extract([make_event(user="svc-backup")])
    assert X[0, column("is_service_account")] == 1.0


def test_rare_user_agents_score_higher_than_common_ones():
    common = [make_event(ts=BASE + timedelta(seconds=i), user_agent="curl/8.4.0") for i in range(30)]
    rare = [make_event(ts=BASE + timedelta(seconds=31), user_agent="sqlmap/1.8")]
    X, _ = extract(common + rare)
    assert X[-1, column("agent_rarity")] > X[-2, column("agent_rarity")]


def test_window_expires_after_five_minutes():
    events = [
        make_event(ts=BASE),
        make_event(ts=BASE + timedelta(minutes=10)),
    ]
    X, _ = extract(events)
    assert X[1, column("user_events_5m")] == 0.0


def test_all_values_are_finite():
    events = [make_event(ts=BASE + timedelta(seconds=i), bytes_out=i * 1000) for i in range(20)]
    X, _ = extract(events)
    assert np.all(np.isfinite(X))
