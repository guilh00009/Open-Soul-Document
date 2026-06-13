# ---------- Setup ----------
import dataclasses
import json
import os
from pathlib import Path
from typing import Any

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.reward_helpers import extract_completion_text
from benchmax.envs.types import ToolDefinition
from benchmax.platform import (
    TrainerClient,
    ensure_session,
    upload_training_run,
    validate_env,
)

import fidelity
import system_prompt
from fidelity import score_fidelity
from system_prompt import AGENT_SYSTEM_PROMPT

RUN_NAME = "fable5-traces-35b"
BASE_MODEL = "Qwen/Qwen3.5-35B-A3B"  # largest model available on Castform
DATA_DIR = Path(__file__).parent


# ---------- Environment Definition ----------
class FableTracesEnv(BaseEnv):
    """Distill Claude Fable 5 agent behavior from Glint-Research/Fable-5-traces."""

    recommended_max_turns = 1

    system_prompt = AGENT_SYSTEM_PROMPT

    @classmethod
    def dataset_preprocess(cls, example: Any, **kwargs):
        return super().dataset_preprocess(example, **kwargs)

    async def compute_reward(self, rollout_id, messages, task=None, **kwargs):
        text = extract_completion_text(messages)
        ground_truth = (task or {}).get("ground_truth", "")
        fidelity = score_fidelity(text, ground_truth)
        return {"fidelity": fidelity}

    async def list_tools(self) -> list[ToolDefinition]:
        return []

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args) -> Any:
        return ""


# ---------- Bundle Args ----------
pip_dependencies = ["datasets"]


# ---------- Dataset ----------
def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}. Run `python prepare_dataset.py` first."
        )
    return [json.loads(line) for line in path.open()]


train_data = _load_jsonl(DATA_DIR / "train_dataset.jsonl")
eval_data = _load_jsonl(DATA_DIR / "eval_dataset.jsonl")


def main() -> None:
    api_key = os.environ.get("PLATFORM_API_KEY")
    if api_key:
        ensure_session()
    else:
        ensure_session()

    print(f"Run: {RUN_NAME}")
    print(f"Model: {BASE_MODEL}")
    print(f"Train examples: {len(train_data)}, Eval examples: {len(eval_data)}")

    if not validate_env(
        env_class=FableTracesEnv,
        env_args={},
        train_dataset=train_data,
        eval_dataset=eval_data,
        local=False,
        pip_dependencies=pip_dependencies,
        local_modules=[fidelity, system_prompt],
        api_key=api_key,
        max_turns=1,
        remote_examples=2,
    ):
        raise SystemExit("Validation failed — fix the env or dataset before launching.")

    uploaded = upload_training_run(
        env_class=FableTracesEnv,
        train_dataset=train_data,
        eval_dataset=eval_data,
        run_name=RUN_NAME,
        constructor_args={},
        pip_dependencies=pip_dependencies,
        local_modules=[fidelity, system_prompt],
        api_key=api_key,
    )

    trainer = TrainerClient(api_key=api_key) if api_key else TrainerClient()
    run_id = trainer.launch_training_run(
        training_run_type="simple",
        name=RUN_NAME,
        launcher_args={
            "model": BASE_MODEL,
            "num_epochs": 5,
            "max_rollout_len": 8192,
            "max_turns": 1,
            "group_size": 8,
        },
        **dataclasses.asdict(uploaded),
    )
    print(f"Training run: https://app.castform.com/train/{run_id}")


if __name__ == "__main__":
    main()
