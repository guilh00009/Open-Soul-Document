#!/usr/bin/env python3
"""List models and training hyperparameters available on Castform.

Uses the public launch-args endpoint (no API key required).
Run this before starting a training job to choose a base model.
"""

from __future__ import annotations

import httpx

from benchmax import config

API_URL = f"{config.platform_url()}/v1/train/launch-args"


def fetch_launch_args() -> list[dict]:
    response = httpx.get(
        API_URL,
        headers={"Accept": "application/json"},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["args"]


def main() -> None:
    args = fetch_launch_args()
    model_spec = next((a for a in args if a["name"] == "model"), None)

    print("=" * 60)
    print("Castform trainable models")
    print("=" * 60)

    if model_spec and model_spec.get("enum"):
        default = model_spec.get("default", "")
        for model_id in model_spec["enum"]:
            marker = " (default)" if model_id == default else ""
            pool = "gpu4" if "4B" in model_id else "gpu8"
            print(f"  • {model_id}{marker}")
            print(f"      pool: {pool}")
        print()
        print(model_spec.get("description", ""))
    else:
        print("  (model enum not returned — check platform docs)")

    print()
    print("=" * 60)
    print("Other launch parameters")
    print("=" * 60)
    skip = {"env_cls_path", "env_metadata_path", "train_dataset_path", "eval_dataset_path", "model"}
    for spec in args:
        if spec["name"] in skip:
            continue
        default = spec.get("default")
        default_str = f", default={default!r}" if default is not None else ""
        print(f"  {spec['name']} ({spec['type']}{default_str})")
        print(f"      {spec['description']}")

    print()
    print("To launch training after setup:")
    print("  1. Set LAUNCH_TRAINING = True in run.py")
    print("  2. python run.py")


if __name__ == "__main__":
    main()
