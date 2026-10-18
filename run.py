#!/usr/bin/env python3
"""
Command line entry point.

    python run.py doctor                 # check setup; prints names and statuses, never secrets
    python run.py dry-run                # forecast the unscored bot testing area, print, submit nothing
    python run.py dry-run --fake-llm     # same, with offline stand-in answers (no provider key, no cost)
    python run.py submit-test --confirm  # REAL posts, but only to the unscored bot testing area
    python run.py run --live             # the timer's command; also needs armed = true in config.toml
    python run.py status                 # what the local ledger holds

Secrets are read only from ./.env (see env.example); the ambient environment is scrubbed first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

# Must run before anything imports litellm / forecasting_tools.
from bot import envguard  # noqa: E402

EXIT_OK, EXIT_USAGE, EXIT_BLOCKED = 0, 2, 3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Metaculus FutureEval forecasting bot")
    parser.add_argument("--config", default=str(PROJECT_DIR / "config.toml"))
    parser.add_argument("--env-file", default=str(PROJECT_DIR / ".env"))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check configuration, token and connectivity")
    sub.add_parser("status", help="show the local ledger")

    dry = sub.add_parser("dry-run", help="forecast the bot testing area and print; submits nothing")
    dry.add_argument("--fake-llm", action="store_true", help="offline stand-in answers; no key, no cost")
    dry.add_argument("--limit", type=int, default=None, help="at most this many questions")
    dry.add_argument("--verbose", action="store_true", help="also print the full comment text")

    test = sub.add_parser("submit-test", help="post real forecasts to the unscored bot testing area")
    test.add_argument("--confirm", action="store_true", help="required: yes, post to the testing area")
    test.add_argument("--limit", type=int, default=1)
    test.add_argument("--verbose", action="store_true")

    run = sub.add_parser("run", help="one pass over the scored tournaments")
    run.add_argument("--live", action="store_true", help="required to submit (with armed = true)")
    run.add_argument("--only", default=None, help="process just this tournament name from config.toml")
    run.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = envguard.apply(args.env_file)

    from bot.config import load_config
    from bot.logsetup import setup_logging

    config = load_config(args.config)
    setup_logging(config.log_dir if args.command in ("run", "submit-test", "dry-run") else None)
    envguard.apply(args.env_file)  # again, in case an import pulled something back in

    import logging

    logger = logging.getLogger("run")
    logger.info(report.describe())
    for warning in report.warnings:
        logger.warning(warning)
    logger.info(
        "ensemble: aggregate=%s model_failover=%d min_successful_fraction=%g; binary: extremize=%g clip=[%g, %g]; %s",
        config.ensemble.aggregate,
        config.ensemble.model_failover,
        config.ensemble.min_successful_fraction,
        config.calibration.binary_extremize,
        config.calibration.binary_min,
        config.calibration.binary_max,
        "; ".join(
            f"{name}: search={','.join(m.removeprefix('openrouter/') for m in profile.search_models)}"
            for name, profile in config.profiles.items()
        ),
    )

    if args.command == "doctor":
        from bot.doctor import doctor

        return doctor(config, report)
    if args.command == "status":
        return _status(config)

    if args.command == "run" and not args.live:
        print(
            "Refusing: `run` submits to scored tournaments and needs --live (and armed = true in config.toml).\n"
            "To see forecasts without submitting, use `dry-run` (it uses the unscored bot testing area;\n"
            "previewing forecasts on open tournament questions and then changing the bot is against the rules).",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if args.command == "submit-test" and not args.confirm:
        print("Refusing: submit-test posts real forecasts and comments to the bot testing area; add --confirm.", file=sys.stderr)
        return EXIT_USAGE
    if not report.has("METACULUS_TOKEN"):
        print("METACULUS_TOKEN is not set in .env; nothing to do.", file=sys.stderr)
        return EXIT_BLOCKED

    from forecasting_tools import MetaculusClient

    from bot.fake import fake_responder
    from bot.ledger import Ledger
    from bot.llm import LlmFactory
    from bot.lock import AlreadyRunning, RunLock
    from bot.runner import NotArmed, RunOptions, Runner, openrouter_remaining_usd
    from bot.spend import SpendMeter

    fake = bool(getattr(args, "fake_llm", False))
    if args.command == "run":
        tournaments = tuple(t for t in config.tournaments if args.only in (None, t.name))
        if not tournaments:
            print(f"no tournament named {args.only!r} in config.toml", file=sys.stderr)
            return EXIT_USAGE
        options = RunOptions(live=True, tournaments=tournaments, limit=args.limit)
    elif args.command == "submit-test":
        options = RunOptions(live=True, testing_submit=True, tournaments=(config.testing,), limit=args.limit, verbose=args.verbose)
    else:
        options = RunOptions(live=False, tournaments=(config.testing,), limit=args.limit, verbose=args.verbose)

    from bot.guard import start_watchdog

    # Backstop: whatever happens, this process ends and frees the lock for the next timer tick.
    start_watchdog(config.limits.run_time_budget_s + config.limits.question_timeout_s + 300)

    try:
        with RunLock(config.state_dir / "run.lock"):
            ledger = Ledger(config.state_dir / "ledger.sqlite3")
            try:
                meter = None if fake else SpendMeter(config.limits, config.prices, ledger)
                factory = LlmFactory(config, meter, responder=fake_responder if fake else None)
                runner = Runner(
                    config=config,
                    ledger=ledger,
                    client=MetaculusClient(),
                    factory=factory,
                    meter=meter,
                    options=options,
                    credit_check=openrouter_remaining_usd,
                )
                if fake:
                    print("NOTE: --fake-llm: the numbers below come from an offline stand-in, not from any model.")
                summary = runner.run()
            finally:
                ledger.close()
    except AlreadyRunning:
        logger.info("another run is still going; exiting")
        return EXIT_OK
    except NotArmed as error:
        print(f"Refusing: {error}", file=sys.stderr)
        return EXIT_USAGE

    line = json.dumps(summary.as_dict(), sort_keys=True)
    logger.info("summary %s", line)
    if args.command != "run":
        print("\nsummary: " + line)
    return EXIT_BLOCKED if summary.stopped_reason and "missing in .env" in summary.stopped_reason else EXIT_OK


def _status(config) -> int:
    from bot.ledger import Ledger

    path = config.state_dir / "ledger.sqlite3"
    if not path.exists():
        print("no ledger yet (the bot has not run)")
        return EXIT_OK
    ledger = Ledger(path)
    try:
        print(f"armed: {config.armed}")
        print(f"questions by status: {ledger.counts_by_status() or 'none'}")
        print(f"spend today: ${ledger.spend_on():.2f}   spend total: ${ledger.spend_total():.2f}   season cap: ${config.limits.max_season_usd:.2f}")
        for status in ("submitting", "failed", "gave_up"):
            for row in ledger.in_state(status)[:20]:
                print(f"  {status}: q{row['question_id']} attempts={row['attempts']} {(row['last_error'] or '')[:120]}")
    finally:
        ledger.close()
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
