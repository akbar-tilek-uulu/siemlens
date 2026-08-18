"""Deterministic synthetic log generator.

The hardest part of any detection experiment is getting labelled data.  Real
logs are confidential, and public corpora are either tiny or stale, so this
module produces a stream that is *structurally* realistic — diurnal rhythm,
long-tailed user activity, a handful of noisy service accounts — and injects a
small number of labelled attack episodes into it.

It is not a substitute for real telemetry, and results measured here should
never be quoted as detection rates on production data.  What it is good for:
running the whole pipeline end to end, checking that features behave, and
comparing model changes against a fixed seed.

Every draw goes through one seeded :class:`random.Random`, so a given seed
always produces the same stream.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from .schema import Event

__all__ = ["generate", "AttackEpisode"]

_USER_AGENTS = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "curl/8.4.0",
    "python-requests/2.31.0",
)

_RARE_AGENTS = (
    "sqlmap/1.8",
    "Go-http-client/2.0",
    "WinHttp.WinHttpRequest.5.1",
)


class AttackEpisode:
    """Bookkeeping for one injected attack, used to report what was planted."""

    __slots__ = ("kind", "user", "src_ip", "start", "count")

    def __init__(self, kind: str, user: str, src_ip: str, start: datetime, count: int) -> None:
        self.kind = kind
        self.user = user
        self.src_ip = src_ip
        self.start = start
        self.count = count

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"AttackEpisode(kind={self.kind!r}, user={self.user!r}, "
            f"src_ip={self.src_ip!r}, start={self.start.isoformat()}, count={self.count})"
        )


def _internal_ip(rng: random.Random) -> str:
    return f"10.{rng.randint(0, 15)}.{rng.randint(0, 255)}.{rng.randint(2, 254)}"


def _external_ip(rng: random.Random) -> str:
    return f"{rng.randint(45, 210)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"


def _business_hours_offset(rng: random.Random) -> float:
    """Draw an hour-of-day weighted towards a working day.

    Two overlapping normals centred on the morning and afternoon peaks, with a
    thin uniform floor so that night-time activity exists but is rare.  Without
    that floor, every off-hours event would be an attack and the hour feature
    would trivially separate the classes.
    """
    roll = rng.random()
    if roll < 0.06:
        return rng.uniform(0.0, 24.0)
    centre = 10.5 if rng.random() < 0.5 else 15.5
    hour = rng.gauss(centre, 2.0)
    return min(max(hour, 0.0), 23.999)


def _benign_event(rng: random.Random, day: datetime, users: list[str], hosts: list[str]) -> Event:
    user = rng.choice(users)
    # Service accounts are chattier and use non-browser agents.
    is_service = user.startswith("svc-")
    hour = rng.uniform(0, 24) if is_service else _business_hours_offset(rng)
    ts = day + timedelta(hours=hour)

    source = rng.choices(("auth", "network", "host", "app"), weights=(3, 4, 2, 3))[0]
    if source == "auth":
        action = rng.choices(("login", "logout"), weights=(4, 1))[0]
        outcome = rng.choices(("success", "failure"), weights=(19, 1))[0]
        bytes_out = 0
    elif source == "network":
        action = rng.choices(("connect", "dns_query", "http_request"), weights=(2, 3, 5))[0]
        outcome = rng.choices(("success", "denied"), weights=(24, 1))[0]
        bytes_out = int(abs(rng.gauss(4_000, 3_000)))
    elif source == "host":
        action = rng.choices(("process_exec", "file_read", "file_write"), weights=(4, 3, 2))[0]
        outcome = "success"
        bytes_out = int(abs(rng.gauss(1_500, 1_200)))
    else:
        action = rng.choices(("http_request", "file_read", "config_change"), weights=(6, 3, 1))[0]
        outcome = rng.choices(("success", "denied"), weights=(29, 1))[0]
        bytes_out = int(abs(rng.gauss(6_000, 5_000)))

    # Service accounts always use the same non-browser agent; human accounts draw
    # from the browsers plus curl, so that "not a browser" alone is not a giveaway.
    agent = "python-requests/2.31.0" if is_service else rng.choice(_USER_AGENTS[:4])

    return Event(
        ts=ts,
        source=source,
        action=action,
        outcome=outcome,
        user=user,
        src_ip=_internal_ip(rng),
        host=rng.choice(hosts),
        user_agent=agent,
        bytes_out=bytes_out,
        label=0,
    )


def _brute_force(rng: random.Random, start: datetime, target: str, hosts: list[str]) -> list[Event]:
    """Many failed logins from one external address, then one success."""
    src_ip = _external_ip(rng)
    host = rng.choice(hosts)
    agent = rng.choice(_RARE_AGENTS)
    attempts = rng.randint(18, 40)
    events = []
    offset = 0.0
    for _ in range(attempts):
        offset += rng.uniform(0.6, 3.0)
        events.append(
            Event(
                ts=start + timedelta(seconds=offset),
                source="auth",
                action="login",
                outcome="failure",
                user=target,
                src_ip=src_ip,
                host=host,
                user_agent=agent,
                bytes_out=0,
                label=1,
            )
        )
    events.append(
        Event(
            ts=events[-1].ts + timedelta(seconds=rng.uniform(1.0, 4.0)),
            source="auth",
            action="login",
            outcome="success",
            user=target,
            src_ip=src_ip,
            host=host,
            user_agent=agent,
            bytes_out=0,
            label=1,
        )
    )
    return events


def _credential_stuffing(rng: random.Random, start: datetime, users: list[str], hosts: list[str]) -> list[Event]:
    """One address trying many distinct accounts once each."""
    src_ip = _external_ip(rng)
    agent = rng.choice(_RARE_AGENTS)
    targets = rng.sample(users, k=min(len(users), rng.randint(12, 25)))
    events = []
    offset = 0.0
    for user in targets:
        offset += rng.uniform(1.0, 5.0)
        events.append(
            Event(
                ts=start + timedelta(seconds=offset),
                source="auth",
                action="login",
                outcome=rng.choices(("failure", "denied"), weights=(9, 1))[0],
                user=user,
                src_ip=src_ip,
                host=rng.choice(hosts),
                user_agent=agent,
                bytes_out=0,
                label=1,
            )
        )
    return events


def _exfiltration(rng: random.Random, start: datetime, user: str, hosts: list[str]) -> list[Event]:
    """A burst of unusually large outbound transfers, off hours."""
    src_ip = _internal_ip(rng)
    host = rng.choice(hosts)
    events = []
    count = rng.randint(10, 22)
    offset = 0.0
    for _ in range(count):
        offset += rng.uniform(2.0, 9.0)
        events.append(
            Event(
                ts=start + timedelta(seconds=offset),
                source="network",
                action="connect",
                outcome="success",
                user=user,
                src_ip=src_ip,
                host=host,
                user_agent="Go-http-client/2.0",
                bytes_out=int(abs(rng.gauss(9_000_000, 2_500_000))),
                label=1,
            )
        )
    return events


def _privilege_abuse(rng: random.Random, start: datetime, user: str, hosts: list[str]) -> list[Event]:
    """Escalation followed by a run of configuration changes on one host.

    The host is drawn at random rather than chosen for being unfamiliar to the
    account, so this episode is carried by the action types and the unusual
    user agent, not by the first-seen features.
    """
    src_ip = _internal_ip(rng)
    host = rng.choice(hosts)
    events = [
        Event(
            ts=start,
            source="host",
            action="privilege_escalation",
            outcome="success",
            user=user,
            src_ip=src_ip,
            host=host,
            user_agent="WinHttp.WinHttpRequest.5.1",
            bytes_out=0,
            label=1,
        )
    ]
    offset = 0.0
    for _ in range(rng.randint(4, 10)):
        offset += rng.uniform(3.0, 15.0)
        events.append(
            Event(
                ts=start + timedelta(seconds=offset),
                source="app",
                action="config_change",
                outcome="success",
                user=user,
                src_ip=src_ip,
                host=host,
                user_agent="WinHttp.WinHttpRequest.5.1",
                bytes_out=int(abs(rng.gauss(2_000, 900))),
                label=1,
            )
        )
    return events


def generate(
    days: int = 14,
    events_per_day: int = 600,
    n_users: int = 40,
    n_hosts: int = 12,
    n_attacks: int = 14,
    seed: int = 20260817,
    start: datetime | None = None,
) -> tuple[list[Event], list[AttackEpisode]]:
    """Generate a labelled event stream.

    Returns the events sorted by timestamp together with the list of attack
    episodes that were planted, so that a caller can report exactly what the
    detector was asked to find.

    With the default parameters this is 6 840 benign events — weekends are
    thinned to 35 per cent, so it is not simply ``days * events_per_day`` — plus
    a few hundred malicious ones.  The resulting imbalance of roughly one in
    twenty is optimistic against production, where it is orders of magnitude
    worse, but it keeps the demo fast.
    """
    if days < 1:
        raise ValueError("days must be >= 1")
    if events_per_day < 1:
        raise ValueError("events_per_day must be >= 1")
    if n_users < 5:
        raise ValueError("n_users must be >= 5")
    if n_attacks < 1:
        raise ValueError(
            "n_attacks must be >= 1; a stream with no positives cannot train a classifier"
        )

    rng = random.Random(seed)
    origin = start or datetime(2026, 6, 1, tzinfo=timezone.utc)

    users = [f"user{i:03d}" for i in range(n_users - 3)] + ["svc-backup", "svc-deploy", "svc-scan"]
    hosts = [f"host-{i:02d}" for i in range(n_hosts)]

    events: list[Event] = []
    for day_index in range(days):
        day = origin + timedelta(days=day_index)
        # Weekends are quieter; this keeps rate features from being flat.
        weight = 0.35 if day.weekday() >= 5 else 1.0
        for _ in range(int(events_per_day * weight)):
            events.append(_benign_event(rng, day, users, hosts))

    episodes: list[AttackEpisode] = []
    kinds = ("brute_force", "credential_stuffing", "exfiltration", "privilege_abuse")
    for index in range(n_attacks):
        kind = rng.choice(kinds)
        # Episodes are spread deterministically across the window rather than
        # placed at random.  Random placement can put every attack on one side
        # of a later time-based split, which leaves the model with a single
        # class to train on, or the evaluation with no positives to find.  That
        # failure is rare, silent and seed-dependent — exactly the kind of
        # flakiness that is not worth carrying.
        day = origin + timedelta(days=(index * days) // n_attacks)
        # Attacks skew towards off hours but are not exclusive to them.
        hour = rng.uniform(0, 6) if rng.random() < 0.6 else rng.uniform(0, 24)
        at = day + timedelta(hours=hour)
        target = rng.choice([u for u in users if not u.startswith("svc-")])

        if kind == "brute_force":
            burst = _brute_force(rng, at, target, hosts)
        elif kind == "credential_stuffing":
            burst = _credential_stuffing(rng, at, users, hosts)
        elif kind == "exfiltration":
            burst = _exfiltration(rng, at, target, hosts)
        else:
            burst = _privilege_abuse(rng, at, target, hosts)

        events.extend(burst)
        # Credential stuffing sprays many accounts from one address, so there is
        # no single target user to report; the source address is the actor.
        actor = "(multiple)" if kind == "credential_stuffing" else burst[0].user
        episodes.append(AttackEpisode(kind, actor, burst[0].src_ip, at, len(burst)))

    events.sort(key=lambda e: e.ts)
    return events, episodes
