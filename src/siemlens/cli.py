"""Command line interface.

``python -m siemlens demo`` runs the whole pipeline on synthetic data and
prints a report: what was planted, what the classifier caught, at what cost in
false alarms, and what stage two made of the top alerts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from . import __version__
from .features import extract
from .generate import generate
from .interpret import OfflineBackend, cluster_alerts, interpret
from .schema import read_jsonl, write_jsonl
from .triage import TriageModel, time_split


def _cmd_generate(args: argparse.Namespace) -> int:
    events, episodes = generate(
        days=args.days,
        events_per_day=args.events_per_day,
        n_attacks=args.attacks,
        seed=args.seed,
    )
    written = write_jsonl(events, args.out)
    print(f"wrote {written} events to {args.out}")
    print(f"planted {len(episodes)} attack episodes:")
    for episode in episodes:
        print(
            f"  {episode.start.isoformat()}  {episode.kind:<20} "
            f"user={episode.user:<12} src_ip={episode.src_ip:<16} events={episode.count}"
        )
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    if args.data:
        events = read_jsonl(args.data)
        episodes = []
    else:
        events, episodes = generate(
            days=args.days,
            events_per_day=args.events_per_day,
            n_attacks=args.attacks,
            seed=args.seed,
        )

    train, test = time_split(events, args.train_fraction)
    model = TriageModel(random_state=args.seed).fit(train)
    metrics = model.evaluate(
        test,
        cost_false_negative=args.cost_fn,
        cost_false_positive=args.cost_fp,
    )

    print(f"events: {len(events)}  train: {len(train)}  test: {len(test)}")
    if episodes:
        print(f"planted episodes: {len(episodes)}")
    print()
    print("stage one — Random Forest")
    print(metrics)
    print()
    print("most important features")
    for name, importance in model.top_features(10):
        print(f"  {importance:6.4f}  {name}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(metrics.to_dict(), fh, indent=2)
        print(f"\nmetrics written to {args.json}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    events, episodes = generate(
        days=args.days,
        events_per_day=args.events_per_day,
        n_attacks=args.attacks,
        seed=args.seed,
    )
    train, test = time_split(events, args.train_fraction)

    model = TriageModel(random_state=args.seed).fit(train)
    metrics = model.evaluate(test, cost_false_negative=args.cost_fn, cost_false_positive=args.cost_fp)
    scores = model.score_events(test)

    print("=" * 72)
    print(f"siemlens {__version__} — two-stage triage demo")
    print("=" * 72)
    print(f"\ngenerated {len(events)} events over {args.days} days, {len(episodes)} attack episodes")
    print(f"train {len(train)} events -> test {len(test)} events (time-based split)\n")

    print("--- stage one: Random Forest ---")
    print(metrics)
    print("\ntop features")
    for name, importance in model.top_features(8):
        print(f"  {importance:6.4f}  {name}")

    alerts = cluster_alerts(test, scores, metrics.threshold)
    print(f"\n{metrics.n_flagged} flagged events clustered into {len(alerts)} alerts")
    print(f"(that is what stage two actually pays for: {len(alerts)} calls, not {metrics.n_flagged})")

    backend = OfflineBackend()
    print("\n--- stage two: interpretation ---")
    print("backend: OfflineBackend (deterministic rules, no network, no API key)\n")

    # Extract once and look rows up by object identity: cluster_alerts hands
    # back the same Event objects, and re-extracting per alert would repeat a
    # full pass over the stream for every call.
    X, _ = extract(test)
    position = {id(event): i for i, event in enumerate(test)}

    for index, alert in enumerate(alerts[: args.top], start=1):
        # Explain the highest-scoring event in the cluster, not the earliest.
        # Alert sorts its events by time, and the first event of a burst has
        # empty rolling windows by construction — describing it would print
        # features that have nothing to do with why the cluster scored highly.
        rows = [position[id(e)] for e in alert.events if id(e) in position]
        row_index = max(rows, key=lambda i: scores[i]) if rows else 0
        alert.reason_features = TriageModel.explain_row(X, row_index, n=4)
        verdict = interpret(alert, backend)
        print(f"[{index}] score {alert.score:.3f}  events {len(alert.events)}  "
              f"user {alert.events[0].user}  src_ip {alert.events[0].src_ip}")
        print(f"    taxonomy   {verdict.taxonomy}  (confidence {verdict.confidence:.2f})")
        print(f"    summary    {verdict.summary}")
        print(f"    entities   {', '.join(verdict.entities) or '-'}")
        if verdict.dropped_entities:
            print(f"    DROPPED    {', '.join(verdict.dropped_entities)}  <- not present in source events")
        if verdict.injection_notes:
            for note in verdict.injection_notes:
                print(f"    INJECTION  {note}")
        for step in verdict.next_steps:
            print(f"    next       {step}")
        print()

    print("Reminder: these numbers come from synthetic data. They say the pipeline")
    print("works end to end. They say nothing about detection rates on real traffic.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="siemlens",
        description="Two-stage security log triage: Random Forest filters, an LLM explains.",
    )
    parser.add_argument("--version", action="version", version=f"siemlens {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--days", type=int, default=14, help="days of traffic to synthesise")
    common.add_argument("--events-per-day", type=int, default=600, dest="events_per_day")
    common.add_argument("--attacks", type=int, default=14, help="attack episodes to plant")
    common.add_argument("--seed", type=int, default=20260817)

    p_gen = sub.add_parser("generate", parents=[common], help="write synthetic events to JSONL")
    p_gen.add_argument("--out", default="events.jsonl")
    p_gen.set_defaults(func=_cmd_generate)

    p_eval = sub.add_parser("evaluate", parents=[common], help="train and score stage one")
    p_eval.add_argument("--data", help="JSONL file to use instead of synthetic data")
    p_eval.add_argument("--train-fraction", type=float, default=0.7, dest="train_fraction")
    p_eval.add_argument("--cost-fn", type=float, default=20.0, dest="cost_fn")
    p_eval.add_argument("--cost-fp", type=float, default=1.0, dest="cost_fp")
    p_eval.add_argument("--json", help="also write metrics to this JSON file")
    p_eval.set_defaults(func=_cmd_evaluate)

    p_demo = sub.add_parser("demo", parents=[common], help="run the whole pipeline and report")
    p_demo.add_argument("--train-fraction", type=float, default=0.7, dest="train_fraction")
    p_demo.add_argument("--cost-fn", type=float, default=20.0, dest="cost_fn")
    p_demo.add_argument("--cost-fp", type=float, default=1.0, dest="cost_fp")
    p_demo.add_argument("--top", type=int, default=5, help="alerts to interpret")
    p_demo.set_defaults(func=_cmd_demo)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
