"""Tests for stage one: the split, the threshold, and the model wrapper."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from siemlens.generate import generate
from siemlens.schema import Event
from siemlens.triage import TriageModel, choose_threshold, time_split

BASE = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


def make_event(offset: int, label: int = 0) -> Event:
    return Event(
        ts=BASE + timedelta(seconds=offset),
        source="auth",
        action="login",
        outcome="success",
        user="user001",
        src_ip="10.0.0.5",
        host="host-01",
        label=label,
    )


# --------------------------------------------------------------------------
# time split
# --------------------------------------------------------------------------


def test_time_split_puts_earlier_events_in_train():
    events = [make_event(i) for i in range(100)]
    train, test = time_split(events, 0.7)
    assert len(train) == 70
    assert len(test) == 30
    assert max(e.ts for e in train) <= min(e.ts for e in test)


def test_time_split_sorts_unordered_input():
    events = [make_event(i) for i in reversed(range(50))]
    train, test = time_split(events)
    assert max(e.ts for e in train) <= min(e.ts for e in test)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.5, 2.0])
def test_time_split_rejects_degenerate_fractions(fraction):
    with pytest.raises(ValueError):
        time_split([make_event(i) for i in range(10)], fraction)


def test_time_split_rejects_a_stream_too_short_to_split():
    with pytest.raises(ValueError, match="empty side"):
        time_split([make_event(0)], 0.7)


def test_generate_refuses_a_stream_with_no_attacks():
    with pytest.raises(ValueError, match="n_attacks"):
        generate(days=3, events_per_day=50, n_attacks=0)


def test_attacks_land_on_both_sides_of_the_split():
    # Episodes are spread deterministically across the window precisely so that
    # this holds for any seed; a random placement makes it a coin toss.
    events, _ = generate(days=10, events_per_day=200, n_attacks=10, seed=99)
    train, test = time_split(events)
    assert sum(e.label for e in train) > 0
    assert sum(e.label for e in test) > 0


# --------------------------------------------------------------------------
# threshold selection
# --------------------------------------------------------------------------


def test_threshold_falls_back_to_a_half_on_a_single_class():
    y = np.zeros(10, dtype=int)
    scores = np.linspace(0, 1, 10)
    assert choose_threshold(y, scores) == 0.5


def test_expensive_misses_push_the_threshold_down():
    y = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
    cautious = choose_threshold(y, scores, cost_false_negative=100.0, cost_false_positive=1.0)
    relaxed = choose_threshold(y, scores, cost_false_negative=1.0, cost_false_positive=100.0)
    assert cautious <= relaxed


def test_a_perfectly_separable_threshold_is_found():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    threshold = choose_threshold(y, scores)
    assert 0.2 < threshold <= 0.8


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


def test_unfitted_model_refuses_to_score():
    model = TriageModel()
    assert not model.is_fitted
    with pytest.raises(RuntimeError, match="not fitted"):
        model.score_events([make_event(0)])


def test_fitting_on_a_single_class_is_refused_with_a_useful_message():
    events = [make_event(i) for i in range(60)]
    with pytest.raises(ValueError, match="single class"):
        TriageModel().fit(events)


def test_fitting_on_an_empty_stream_is_refused():
    with pytest.raises(ValueError, match="empty event stream"):
        TriageModel().fit([])


def test_pipeline_finds_planted_attacks_in_held_out_traffic():
    events, _ = generate(days=10, events_per_day=250, n_attacks=12, seed=7)
    train, test = time_split(events)
    model = TriageModel(n_estimators=120, random_state=7).fit(train)
    metrics = model.evaluate(test)

    assert metrics.n_positive > 0, "the test window must contain attacks to be meaningful"
    assert metrics.recall > 0.5
    assert metrics.average_precision > 0.2
    assert 0.0 <= metrics.threshold <= 1.0


def test_alert_volume_stays_workable():
    events, _ = generate(days=10, events_per_day=250, n_attacks=12, seed=7)
    train, test = time_split(events)
    model = TriageModel(n_estimators=120, random_state=7).fit(train)
    metrics = model.evaluate(test)
    # The point of stage one is to reduce what a human reads. If it flags most
    # of the stream it has failed, whatever its recall says.
    assert metrics.n_flagged < 0.25 * metrics.n_events


def test_scores_are_probabilities():
    events, _ = generate(days=6, events_per_day=200, n_attacks=8, seed=11)
    train, test = time_split(events)
    model = TriageModel(n_estimators=60, random_state=11).fit(train)
    scores = model.score_events(test)
    assert scores.shape == (len(test),)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_scoring_is_deterministic():
    events, _ = generate(days=6, events_per_day=200, n_attacks=8, seed=11)
    train, test = time_split(events)
    model = TriageModel(n_estimators=60, random_state=11).fit(train)
    assert np.allclose(model.score_events(test), model.score_events(test))


def test_two_models_with_the_same_seed_agree_bit_for_bit():
    # Exact equality, not np.allclose. The whole reason this stage uses a forest
    # is that the same event scores the same every time; "close enough" hides
    # exactly the drift that makes an event on the threshold flip in and out of
    # the alert set between runs. This assertion fails if n_jobs is ever set
    # back to -1, because parallel workers accumulate the per-tree probabilities
    # in a non-deterministic order.
    events, _ = generate(days=6, events_per_day=200, n_attacks=8, seed=11)
    train, test = time_split(events)
    first = TriageModel(n_estimators=60, random_state=3).fit(train)
    second = TriageModel(n_estimators=60, random_state=3).fit(train)
    assert np.array_equal(first.score_events(test), second.score_events(test))


def test_the_demo_pipeline_is_reproducible_end_to_end():
    def run():
        events, _ = generate(days=5, events_per_day=200, n_attacks=6, seed=20260817)
        train, test = time_split(events)
        model = TriageModel(n_estimators=120, random_state=20260817).fit(train)
        metrics = model.evaluate(test)
        return metrics.threshold, metrics.n_flagged, model.score_events(test)

    first_threshold, first_flagged, first_scores = run()
    second_threshold, second_flagged, second_scores = run()

    assert first_threshold == second_threshold
    assert first_flagged == second_flagged
    assert np.array_equal(first_scores, second_scores)


def test_top_features_are_reported_in_descending_order():
    events, _ = generate(days=6, events_per_day=200, n_attacks=8, seed=11)
    train, _ = time_split(events)
    model = TriageModel(n_estimators=60, random_state=11).fit(train)
    top = model.top_features(8)
    assert len(top) == 8
    importances = [value for _, value in top]
    assert importances == sorted(importances, reverse=True)


def test_metrics_serialise_to_plain_json_types():
    events, _ = generate(days=6, events_per_day=200, n_attacks=8, seed=11)
    train, test = time_split(events)
    model = TriageModel(n_estimators=60, random_state=11).fit(train)
    data = model.evaluate(test).to_dict()
    assert set(data) >= {"threshold", "precision", "recall", "average_precision"}
    assert all(isinstance(v, (int, float)) for v in data.values())
