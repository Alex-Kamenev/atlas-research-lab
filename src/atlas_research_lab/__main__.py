"""Generate a local synthetic research report; no game or data-import hooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import render_markdown, run_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare research baselines on synthetic learners.")
    parser.add_argument("--output", type=Path, required=True, help="output directory for report.json and report.md")
    parser.add_argument("--learners", type=int, default=32)
    parser.add_argument("--attempts", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args(argv)
    destination = args.output.expanduser()
    for name in ("report.json", "report.md"):
        target = destination / name
        if target.exists() or target.is_symlink():
            parser.error("output report already exists; choose a new output directory")
    if destination.exists() and not destination.is_dir():
        parser.error("output must be a directory")
    try:
        report = run_experiment(learners=args.learners, attempts=args.attempts,
                                seed=args.seed, bootstrap=args.bootstrap)
    except ValueError as error:
        parser.error(str(error))
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "report.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with (destination / "report.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(render_markdown(report))
    print("Wrote aggregate synthetic report.json and report.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
