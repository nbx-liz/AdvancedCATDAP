from __future__ import annotations

import argparse
from typing import Sequence

from advanced_catdap.runtime.gui import run_desktop_mode, run_web_mode
from advanced_catdap.runtime.worker import run_worker_mode


def _build_legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AdvancedCATDAP desktop entrypoint")
    parser.add_argument("--worker", action="store_true", help="Run one analysis job and exit")
    parser.add_argument("--job-id")
    parser.add_argument("--dataset-id")
    parser.add_argument("--params")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--db-path", default="data/jobs.db")
    return parser


def _build_subcommand_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AdvancedCATDAP desktop entrypoint")
    sub = parser.add_subparsers(dest="mode")

    desktop = sub.add_parser("desktop")
    desktop.add_argument("--data-dir", default="data")
    desktop.add_argument("--db-path", default="data/jobs.db")

    # Legacy alias kept for compatibility with existing wrappers/tests.
    gui = sub.add_parser("gui")
    gui.add_argument("--data-dir", default="data")
    gui.add_argument("--db-path", default="data/jobs.db")

    web = sub.add_parser("web")
    web.add_argument("--data-dir", default="data")
    web.add_argument("--db-path", default="data/jobs.db")

    worker = sub.add_parser("worker")
    worker.add_argument("--job-id", required=True)
    worker.add_argument("--dataset-id", required=True)
    worker.add_argument("--params", required=True)
    worker.add_argument("--data-dir", default="data")
    worker.add_argument("--db-path", default="data/jobs.db")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    # Internal canonical mode: subcommands.
    if argv and len(argv) > 0 and argv[0] in {"desktop", "web", "gui", "worker"}:
        args = _build_subcommand_parser().parse_args(argv)
        if not args.mode:
            args.mode = "desktop"
        if args.mode == "gui":
            args.mode = "desktop"
        return args

    # Backward-compatible mode: --worker flag.
    parser = _build_legacy_parser()
    args = parser.parse_args(argv)
    args.mode = "worker" if args.worker else "desktop"

    if args.mode == "worker":
        missing = [
            name
            for name in ("job_id", "dataset_id", "params")
            if getattr(args, name) in (None, "")
        ]
        if missing:
            parser.error(f"--worker requires: {', '.join('--' + m.replace('_', '-') for m in missing)}")

    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "worker":
        return run_worker_mode(args)
    if args.mode == "web":
        return run_web_mode()
    return run_desktop_mode()
