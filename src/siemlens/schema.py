"""Normalised security event schema.

Everything upstream of the models — collection, parsing, vendor quirks — is
expected to produce :class:`Event` objects.  Keeping one schema between
collection and analysis is what makes the rest of the pipeline possible: the
feature extractor and the interpreter both address fields by name, and neither
has to know which product emitted the record.

The schema is deliberately small.  It covers the fields that carry signal for
triage (who, from where, to what, with what result, how much data) and nothing
else.  Vendor-specific extras belong in :attr:`Event.extra`, which is never fed
to a model and never interpolated into a prompt.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "Event",
    "SOURCES",
    "ACTIONS",
    "OUTCOMES",
    "normalise_timestamp",
    "from_mapping",
    "read_jsonl",
    "write_jsonl",
]

#: Log sources the generator and feature extractor understand.
SOURCES = ("auth", "network", "host", "app")

#: Actions, grouped loosely by source.  Kept as a flat tuple so that the
#: categorical encoder has a stable, closed vocabulary.
ACTIONS = (
    "login",
    "logout",
    "privilege_escalation",
    "file_read",
    "file_write",
    "process_exec",
    "connect",
    "dns_query",
    "http_request",
    "config_change",
)

#: Result of the action.  ``denied`` is distinct from ``failure``: the former is
#: a policy decision, the latter is a failed attempt.
OUTCOMES = ("success", "failure", "denied")


def normalise_timestamp(value: Any) -> datetime:
    """Coerce *value* to a timezone-aware UTC :class:`~datetime.datetime`.

    Accepts ``datetime`` objects (naive ones are assumed to be UTC), ISO 8601
    strings including the ``Z`` suffix, and POSIX timestamps as int or float.

    Normalising time at the edge matters more than it looks: once events from
    several sources share a clock, every rolling-window feature downstream
    becomes meaningful.  Mixed local times silently corrupt them instead.
    """
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    else:
        raise TypeError(f"unsupported timestamp type: {type(value)!r}")

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class Event:
    """A single normalised security event.

    Parameters mirror the schema described in the module docstring.  ``label``
    is only populated for generated or hand-labelled data and is never visible
    to :mod:`siemlens.features`; it exists so that evaluation has ground truth.
    """

    ts: datetime
    source: str
    action: str
    outcome: str
    user: str
    src_ip: str
    host: str
    user_agent: str = ""
    bytes_out: int = 0
    label: int = 0
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ts", normalise_timestamp(self.ts))
        if self.source not in SOURCES:
            raise ValueError(f"unknown source {self.source!r}; expected one of {SOURCES}")
        if self.action not in ACTIONS:
            raise ValueError(f"unknown action {self.action!r}; expected one of {ACTIONS}")
        if self.outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {self.outcome!r}; expected one of {OUTCOMES}")
        if self.bytes_out < 0:
            raise ValueError("bytes_out must not be negative")
        if self.label not in (0, 1):
            raise ValueError("label must be 0 or 1")

    @property
    def key(self) -> str:
        """Stable fingerprint used for caching interpretation results."""
        return f"{self.source}|{self.action}|{self.outcome}|{self.user}|{self.host}"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ts"] = self.ts.isoformat()
        data["extra"] = dict(self.extra)
        return data


def from_mapping(row: Mapping[str, Any]) -> Event:
    """Build an :class:`Event` from a plain mapping, ignoring unknown keys.

    Unknown keys are not dropped silently — they are moved into ``extra`` so
    that nothing from the original record is lost, while keeping the typed
    fields closed.
    """
    known = {
        "ts",
        "source",
        "action",
        "outcome",
        "user",
        "src_ip",
        "host",
        "user_agent",
        "bytes_out",
        "label",
    }
    kwargs = {k: row[k] for k in known if k in row}
    extra = dict(row.get("extra") or {})
    for k, v in row.items():
        if k not in known and k != "extra":
            extra[k] = v
    if extra:
        kwargs["extra"] = extra
    return Event(**kwargs)  # type: ignore[arg-type]


def read_jsonl(path: str) -> list[Event]:
    """Read newline-delimited JSON into a timestamp-sorted list of events."""
    events: list[Event] = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(from_mapping(json.loads(line)))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    events.sort(key=lambda e: e.ts)
    return events


def write_jsonl(events: Iterable[Event], path: str) -> int:
    """Write events as newline-delimited JSON.  Returns the number written."""
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def iter_sorted(events: Iterable[Event]) -> Iterator[Event]:
    """Yield events in ascending timestamp order.

    Feature extraction depends on this: every rolling-window feature is
    computed from strictly earlier events, and out-of-order input would leak
    the future into the past.
    """
    yield from sorted(events, key=lambda e: e.ts)
