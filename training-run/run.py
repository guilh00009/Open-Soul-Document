# ---------- Launcher (env lives in opensoul_env.py for pickle-safe bundling) ----------
import asyncio
import dataclasses
import json
import os

import friction
import opensoul_doc_blob
import opensoul_env
import rewards
import training_utils
from opensoul_env import OpenSoulSelfReflectionEnv

RUN_NAME = "opensoul-v5-self-reflection-v2"
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3.5-4B")

TRAIN_MODELS: list[tuple[str, str]] = [
    ("Qwen/Qwen3.5-4B", f"{RUN_NAME}-4b"),
    ("Qwen/Qwen3.5-35B-A3B", f"{RUN_NAME}-35b"),
]

LAUNCH_TRAINING = os.environ.get("LAUNCH_TRAINING", "0").lower() in ("1", "true", "yes")

pip_dependencies = ["openai"]
local_modules = [
    opensoul_env,
    opensoul_doc_blob,
    rewards,
    friction,
    training_utils,
]

train_data = [json.loads(line) for line in open("train_dataset.jsonl")]
eval_data = [json.loads(line) for line in open("eval_dataset.jsonl")]


def preview_setup() -> None:
    from list_models import fetch_launch_args

    print(f"Run name:       {RUN_NAME}")
    print(f"Base model:     {BASE_MODEL}")
    print(f"Train rows:     {len(train_data)}")
    print(f"Eval rows:      {len(eval_data)}")
    print(f"Launch flag:    {LAUNCH_TRAINING}")
    print(f"Judge model:    {opensoul_env.DEFAULT_JUDGE_MODEL}")
    print(f"Adaptive rubrics: {opensoul_env.USE_ADAPTIVE_RUBRICS}")
    print(f"Diversity scale:  {opensoul_env.USE_DIVERSITY_SCALING}")
    print(f"Holistic ranking: {opensoul_env.USE_HOLISTIC_RANKING}")
    print(f"DEPO gating:      {opensoul_env.USE_DEPO}")
    if LAUNCH_TRAINING:
        print("Models to train:")
        for model_id, run_label in TRAIN_MODELS:
            print(f"  • {model_id} → {run_label}")
    print(f"Static rubrics: {len(rewards.GROUP_RUBRICS)}")
    print(f"Full doc sections: {sorted(opensoul_env.FULL_DOCUMENT_SECTIONS)}")
    slices = opensoul_doc_blob.build_slices()
    print(f"Section slices: {len(slices)} (largest: {max(len(v) for v in slices.values()):,} chars)")
    print()
    args = fetch_launch_args()
    model_spec = next((a for a in args if a["name"] == "model"), None)
    if model_spec and model_spec.get("enum"):
        print("Trainable models:")
        for model_id in model_spec["enum"]:
            default = " (default)" if model_id == model_spec.get("default") else ""
            print(f"  • {model_id}{default}")


if __name__ == "__main__":
    if not LAUNCH_TRAINING:
        preview_setup()
        print(
            "\nTraining not launched (LAUNCH_TRAINING=false).\n"
            "Set LAUNCH_TRAINING=1 when ready."
        )
        raise SystemExit(0)

    from benchmax.platform import (
        TrainerClient,
        upload_training_run,
        validate_env,
        ensure_session,
    )

    ensure_session()

    from preflight import run_preflight

    asyncio.run(run_preflight("opensoul"))

    models_to_train = TRAIN_MODELS
    if os.environ.get("TRAIN_MODELS"):
        models_to_train = [
            (m.strip(), f"{RUN_NAME}-{m.split('/')[-1].lower()}")
            for m in os.environ["TRAIN_MODELS"].split(",")
            if m.strip()
        ]
    elif os.environ.get("BASE_MODEL"):
        models_to_train = [(os.environ["BASE_MODEL"], RUN_NAME)]

    skip_remote = os.environ.get("SKIP_REMOTE_VALIDATION", "0").lower() in (
        "1",
        "true",
        "yes",
    )

    if not validate_env(
        env_class=OpenSoulSelfReflectionEnv,
        env_args={},
        train_dataset=train_data,
        eval_dataset=eval_data,
        local=skip_remote,
        pip_dependencies=pip_dependencies,
        local_modules=local_modules,
    ):
        raise SystemExit("Validation failed — fix the env or dataset before launching.")

    uploaded = upload_training_run(
        env_class=OpenSoulSelfReflectionEnv,
        train_dataset=train_data,
        eval_dataset=eval_data,
        run_name=RUN_NAME,
        constructor_args={},
        pip_dependencies=pip_dependencies,
        local_modules=local_modules,
    )

    trainer = TrainerClient()
    launched: list[tuple[str, str]] = []
    for model_id, run_label in models_to_train:
        run_id = trainer.launch_training_run(
            training_run_type="simple",
            name=run_label,
            launcher_args={
                "model": model_id,
                "max_rollout_len": 12000,
                "max_turns": 3,
                "group_size": 9,
                "learning_rate": 1e-5,
                "num_epochs": 5,
                "lora_rank": 128,
                "lora_alpha": 256,
            },
            **dataclasses.asdict(uploaded),
        )
        launched.append((model_id, run_id))
        print(f"Launched {model_id}: https://app.castform.com/train/{run_id}")

    print("\nAll training runs:")
    for model_id, run_id in launched:
        print(f"  {model_id} → https://app.castform.com/train/{run_id}")
