"""Tests for the normalised event schema."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from siemlens.schema import (
    ACTIONS,
    OUTCOMES,
    SOURCES,
    Event,
    from_mapping,
    iter_sorted,
    normalise_timestamp,
    read_jsonl,
    write_jsonl,
)

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
    }
    params.update(overrides)
    return Event(**params)


# --------------------------------------------------------------------------
# timestamps
# --------------------------------------------------------------------------


def test_naive_datetimes_are_assumed_to_be_utc():
    result = normalise_timestamp(datetime(2026, 6, 1, 12, 0))
    assert result.tzinfo is timezone.utc
    assert result.hour == 12


def test_offset_datetimes_are_converted_to_utc():
    result = normalise_timestamp("2026-06-01T15:00:00+03:00")
    assert result.hour == 12
    assert result.tzinfo is timezone.utc


def test_zulu_suffix_is_accepted():
    assert normalise_timestamp("2026-06-01T12:00:00Z") == BASE


def test_posix_timestamps_are_accepted():
    assert normalise_timestamp(BASE.timestamp()) == BASE


def test_unsupported_timestamp_types_are_rejected():
    with pytest.raises(TypeError):
        normalise_timestamp(["not", "a", "time"])


def test_event_normalises_its_timestamp_on_construction():
    event = make_event(ts="2026-06-01T15:00:00+03:00")
    assert event.ts == BASE


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("source", SOURCES)
def test_every_declared_source_is_accepted(source):
    assert make_event(source=source).source == source


@pytest.mark.parametrize("action", ACTIONS)
def test_every_declared_action_is_accepted(action):
    assert make_event(action=action).action == action


@pytest.mark.parametrize("outcome", OUTCOMES)
def test_every_declared_outcome_is_accepted(outcome):
    assert make_event(outcome=outcome).outcome == outcome


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError, match="unknown source"):
        make_event(source="carrier_pigeon")


def test_unknown_action_is_rejected():
    with pytest.raises(ValueError, match="unknown action"):
        make_event(action="telepathy")


def test_unknown_outcome_is_rejected():
    with pytest.raises(ValueError, match="unknown outcome"):
        make_event(outcome="maybe")


def test_negative_byte_counts_are_rejected():
    with pytest.raises(ValueError, match="bytes_out"):
        make_event(bytes_out=-1)


def test_labels_outside_zero_and_one_are_rejected():
    with pytest.raises(ValueError, match="label"):
        make_event(label=7)


def test_events_are_immutable():
    event = make_event()
    # dataclasses.FrozenInstanceError subclasses AttributeError.
    with pytest.raises(AttributeError):
        event.user = "someone-else"  # type: ignore[misc]


# --------------------------------------------------------------------------
# mapping and serialisation
# --------------------------------------------------------------------------


def test_unknown_keys_are_preserved_in_extra_rather_than_dropped():
    event = from_mapping(
        {
            "ts": BASE.isoformat(),
            "source": "auth",
            "action": "login",
            "outcome": "success",
            "user": "user001",
            "src_ip": "10.0.0.5",
            "host": "host-01",
            "vendor_event_id": "EVT-42",
        }
    )
    assert event.extra["vendor_event_id"] == "EVT-42"


def test_explicit_extra_and_unknown_keys_merge():
    event = from_mapping(
        {
            "ts": BASE.isoformat(),
            "source": "auth",
            "action": "login",
            "outcome": "success",
            "user": "u",
            "src_ip": "10.0.0.1",
            "host": "h",
            "extra": {"a": 1},
            "b": 2,
        }
    )
    assert event.extra == {"a": 1, "b": 2}


def test_to_dict_is_json_serialisable():
    payload = json.dumps(make_event().to_dict())
    assert "2026-06-01T12:00:00+00:00" in payload


def test_jsonl_round_trip_preserves_events(tmp_path):
    events = [make_event(ts=BASE.replace(second=i), user=f"user{i}") for i in range(5)]
    path = tmp_path / "events.jsonl"
    written = write_jsonl(events, str(path))
    assert written == 5

    restored = read_jsonl(str(path))
    assert [e.user for e in restored] == [e.user for e in events]
    assert [e.ts for e in restored] == [e.ts for e in events]


def test_read_jsonl_reports_the_offending_line(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text(
        json.dumps(make_event().to_dict()) + "\n" + json.dumps({"ts": BASE.isoformat()}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=":2:"):
        read_jsonl(str(path))


def test_read_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "sparse.jsonl"
    path.write_text("\n" + json.dumps(make_event().to_dict()) + "\n\n", encoding="utf-8")
    assert len(read_jsonl(str(path))) == 1


def test_iter_sorted_orders_by_timestamp():
    events = [make_event(ts=BASE.replace(second=s)) for s in (5, 1, 3)]
    seconds = [e.ts.second for e in iter_sorted(events)]
    assert seconds == [1, 3, 5]


def test_key_groups_events_by_actor_and_action():
    assert make_event().key == make_event(bytes_out=999).key
    assert make_event().key != make_event(user="other").key
