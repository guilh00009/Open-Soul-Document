# Code capabilities training launcher (RLVR / verifiable unit tests)
import dataclasses
import json
import os

import code_env
import code_rewards
import code_sandbox
import training_utils
from code_env import CodeCapabilitiesEnv

RUN_NAME = "code-capabilities-v1"
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3.5-4B")

TRAIN_MODELS: list[tuple[str, str]] = [
    ("Qwen/Qwen3.5-4B", f"{RUN_NAME}-4b"),
    ("Qwen/Qwen3.5-35B-A3B", f"{RUN_NAME}-35b"),
]

LAUNCH_TRAINING = os.environ.get("LAUNCH_TRAINING", "0").lower() in ("1", "true", "yes")

pip_dependencies: list[str] = []
local_modules = [code_env, code_rewards, code_sandbox, training_utils]

train_data = [json.loads(line) for line in open("code_train_dataset.jsonl")]
eval_data = [json.loads(line) for line in open("code_eval_dataset.jsonl")]


def preview_setup() -> None:
    from list_models import fetch_launch_args

    print(f"Run name:       {RUN_NAME}")
    print(f"Train rows:     {len(train_data)}")
    print(f"Eval rows:      {len(eval_data)}")
    print(f"Max turns:      {CodeCapabilitiesEnv.recommended_max_turns}")
    print(f"DEPO gating:    {code_env.USE_DEPO}")
    print(f"Pass-rate RL:   {code_rewards.USE_PASS_RATE_REWARD}")
    print(f"Launch:         {LAUNCH_TRAINING}")
    cats: dict[str, int] = {}
    for row in train_data:
        cats[row.get("category", "?")] = cats.get(row.get("category", "?"), 0) + 1
    print(f"Categories:     {cats}")
    print()
    args = fetch_launch_args()
    spec = next((a for a in args if a["name"] == "model"), None)
    if spec and spec.get("enum"):
        for m in spec["enum"]:
            print(f"  • {m}")


if __name__ == "__main__":
    if not LAUNCH_TRAINING:
        preview_setup()
        print("\nTraining not launched. Set LAUNCH_TRAINING=1 to launch.")
        raise SystemExit(0)

    from benchmax.platform import (
        TrainerClient,
        upload_training_run,
        validate_env,
        ensure_session,
    )

    ensure_session()

    import asyncio
    from preflight import run_preflight

    asyncio.run(run_preflight("code"))

    models = TRAIN_MODELS
    if os.environ.get("TRAIN_MODELS"):
        models = [
            (m.strip(), f"{RUN_NAME}-{m.split('/')[-1].lower()}")
            for m in os.environ["TRAIN_MODELS"].split(",") if m.strip()
        ]

    if not validate_env(
        env_class=CodeCapabilitiesEnv,
        env_args={},
        train_dataset=train_data,
        eval_dataset=eval_data,
        local=False,
        pip_dependencies=pip_dependencies,
        local_modules=local_modules,
    ):
        raise SystemExit("Code env validation failed.")

    uploaded = upload_training_run(
        env_class=CodeCapabilitiesEnv,
        train_dataset=train_data,
        eval_dataset=eval_data,
        run_name=RUN_NAME,
        constructor_args={},
        pip_dependencies=pip_dependencies,
        local_modules=local_modules,
    )

    trainer = TrainerClient()
    for model_id, label in models:
        run_id = trainer.launch_training_run(
            training_run_type="simple",
            name=label,
            launcher_args={
                "model": model_id,
                "max_rollout_len": 8192,
                "max_turns": 1,
                "group_size": 12,
                "learning_rate": 5e-6,
                "num_epochs": 4,
                "lora_rank": 128,
                "lora_alpha": 256,
            },
            **dataclasses.asdict(uploaded),
        )
        print(f"Launched {model_id}: https://app.castform.com/train/{run_id}")
