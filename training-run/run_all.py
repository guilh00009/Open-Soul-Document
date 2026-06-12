#!/usr/bin/env python3
"""Launch all training tracks: Open Soul self-reflection + agentic capabilities.

Usage:
  cd training-run
  python generate_dataset.py
  python generate_agentic_dataset.py
  LAUNCH_TRAINING=1 python run_all.py
  LAUNCH_TRAINING=1 python run_all.py --tracks opensoul
  LAUNCH_TRAINING=1 python run_all.py --tracks agentic
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch multi-track Castform training")
    parser.add_argument(
        "--tracks",
        default="opensoul,agentic",
        help="Comma-separated: opensoul, agentic",
    )
    args = parser.parse_args()
    tracks = {t.strip() for t in args.tracks.split(",") if t.strip()}

    env = os.environ.copy()
    if "LAUNCH_TRAINING" not in env:
        env["LAUNCH_TRAINING"] = "1"

    scripts = []
    if "opensoul" in tracks:
        scripts.append("run.py")
    if "agentic" in tracks:
        scripts.append("run_agentic.py")

    if not scripts:
        print("No tracks selected.", file=sys.stderr)
        raise SystemExit(1)

    for script in scripts:
        print(f"\n{'='*60}\nLaunching {script}\n{'='*60}")
        rc = subprocess.call([sys.executable, script], env=env)
        if rc != 0:
            raise SystemExit(rc)


if __name__ == "__main__":
    main()
