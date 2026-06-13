# Agentic capabilities training launcher (Hermes / OpenClaw patterns)
import dataclasses
import json
import os

import agentic_env
import agentic_rewards
import agentic_tools
import agentic_workspace
import rewards
import tool_call_helpers
import training_utils
from agentic_env import AgenticCapabilitiesEnv

RUN_NAME = "agentic-capabilities-v1"
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3.5-4B")

TRAIN_MODELS: list[tuple[str, str]] = [
    ("Qwen/Qwen3.5-4B", f"{RUN_NAME}-4b"),
    ("Qwen/Qwen3.5-35B-A3B", f"{RUN_NAME}-35b"),
]

LAUNCH_TRAINING = os.environ.get("LAUNCH_TRAINING", "0").lower() in ("1", "true", "yes")

pip_dependencies = ["openai"]
local_modules = [
    agentic_env,
    agentic_rewards,
    agentic_tools,
    agentic_workspace,
    rewards,
    tool_call_helpers,
    training_utils,
]

train_data = [json.loads(line) for line in open("agentic_train_dataset.jsonl")]
eval_data = [json.loads(line) for line in open("agentic_eval_dataset.jsonl")]


def preview_setup() -> None:
    from list_models import fetch_launch_args

    print(f"Run name:     {RUN_NAME}")
    print(f"Train rows:   {len(train_data)}")
    print(f"Eval rows:    {len(eval_data)}")
    print(f"Tools:        {len(agentic_tools.TOOL_DEFINITIONS)}")
    print(f"Max turns:    {AgenticCapabilitiesEnv.recommended_max_turns}")
    print(f"Launch:       {LAUNCH_TRAINING}")
    print(f"DEPO gating:  {agentic_env.USE_DEPO}")
    cats = {}
    for row in train_data:
        cats[row.get("category", "?")] = cats.get(row.get("category", "?"), 0) + 1
    print(f"Categories:   {cats}")
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

    asyncio.run(run_preflight("agentic"))

    models = TRAIN_MODELS
    if os.environ.get("TRAIN_MODELS"):
        models = [
            (m.strip(), f"{RUN_NAME}-{m.split('/')[-1].lower()}")
            for m in os.environ["TRAIN_MODELS"].split(",") if m.strip()
        ]

    if not validate_env(
        env_class=AgenticCapabilitiesEnv,
        env_args={},
        train_dataset=train_data,
        eval_dataset=eval_data,
        local=False,
        pip_dependencies=pip_dependencies,
        local_modules=local_modules,
    ):
        raise SystemExit("Agentic validation failed.")

    uploaded = upload_training_run(
        env_class=AgenticCapabilitiesEnv,
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
                "max_rollout_len": 16000,
                "max_turns": 8,
                "group_size": 9,
                "learning_rate": 1e-5,
                "num_epochs": 5,
                "lora_rank": 128,
                "lora_alpha": 256,
            },
            **dataclasses.asdict(uploaded),
        )
        print(f"Launched {model_id}: https://app.castform.com/train/{run_id}")
