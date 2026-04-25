"""
train_grpo_openenv.py — GRPO + LoRA training for PrivilegeDesk
                        (direct WorldState, OpenEnv notebook-style pipeline)

Mirrors the official OpenEnv TRL Wordle GRPO notebook
(competitor_research/notebook.ipynb) as closely as possible. Differences vs.
the notebook are intentional and limited to:

  1. Environment is called in-process via WorldState (no HTTP sync client).
  2. PrivilegeDesk is a multi-task environment — the notebook trains on a single
     game (Wordle). We sample one of the 5 core IAM tasks per episode using
     the stratified "phase 3" mix (access_decision, emergency_breakglass,
     jit_escalation, access_review, separation_of_duties_audit).
     `multi_agent_oversight` is excluded — it is adversarial/phase-4 territory
     and kept for a separate run.
  3. Prompt & completion lengths are larger than Wordle:
       - Wordle prompts are short, outputs are 1 word (~8 tokens).
       - PrivilegeDesk observations are nested JSON (tools, pending requests,
         objectives, sub-agent lists) → ~4–8 KB of user content.
       - Outputs are <think>…</think> + one JSON tool call → up to ~1024 tokens.
     We therefore set max_prompt_length=4096, max_completion_length=1024.
     Everything else (lr, grad_accum, num_generations, vllm colocate,
     trackio, push_to_hub) matches the notebook.

Usage
─────
  python training/train_grpo_openenv.py
  python training/train_grpo_openenv.py --model-id Qwen/Qwen3-1.7B --dataset-size 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from env.world_state import WorldState  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("grpo_openenv")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Task registry + stratified mix (notebook "cell-1" analogue)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskMeta:
    task_id: str
    max_steps: int
    optimal_steps: int
    difficulty_level: int = 2


TASK_REGISTRY: Dict[str, TaskMeta] = {
    "access_decision":            TaskMeta("access_decision",             5, 4,  2),
    "emergency_breakglass":       TaskMeta("emergency_breakglass",       10, 7,  2),
    "jit_escalation":             TaskMeta("jit_escalation",             15, 10, 2),
    "access_review":              TaskMeta("access_review",              25, 15, 1),
    "separation_of_duties_audit": TaskMeta("separation_of_duties_audit", 25, 15, 1),
}

# Stratified "phase 3" mix — all 5 IAM tasks, skewed toward easier ones.
TASK_WEIGHTS: Dict[str, float] = {
    "access_decision":            0.20,
    "emergency_breakglass":       0.20,
    "jit_escalation":             0.25,
    "access_review":              0.20,
    "separation_of_duties_audit": 0.15,
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. System prompt
# ─────────────────────────────────────────────────────────────────────────────

system_prompt = textwrap.dedent("""
You are an enterprise IAM (Identity & Access Management) specialist agent.
You operate inside the PrivilegeDesk environment, which simulates a corporate
zero-standing-privilege access control system.

Your job is to use the available tools to complete the assigned IAM task.
You MUST reason inside <think>...</think> tags first, then emit EXACTLY ONE JSON object.
Output NOTHING else — no extra text, no second JSON block, no repetition.

<think>
Brief reasoning about what to investigate or decide next.
</think>
{
  "tool_name": "<tool_name>",
  "arguments": { "<key>": "<value>", ... }
}

Rules:
- Only call tools listed in available_tools in the observation
- No text outside the <think> block and the single JSON object
- ONE tool call per response — never output two JSON objects in one reply
- Do NOT repeat a tool call you already made with the same arguments
- For access.decide: use "approve" or "deny" for the decision field
- For entitlement.revoke: provide the entitlement_id
- For review.submit / sod.submit_report: call when you have finished all revocations
- For access.grant / access.deny: call only after all approvals are in
- When done, the environment will signal done=true — do not continue after that

All context you need is in the observation JSON.
""").strip()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Helper functions (notebook "cell-10" analogue)
# ─────────────────────────────────────────────────────────────────────────────

def make_user_prompt(observation: Dict[str, Any], history: List[str]) -> str:
    """Build the user turn from the current WorldState observation + history."""
    metadata = observation.get("tool_metadata", {})
    tool_names = observation.get("available_tools", [])
    enriched_tools = [
        f"name: {n} | desc: {metadata[n]['desc']} | args: "
        + (", ".join(f"{k}: {v}" for k, v in metadata[n]["args"].items()) or "no args")
        if n in metadata else n
        for n in tool_names
    ]

    obs_summary: Dict[str, Any] = {
        "task_goal":        observation.get("task_goal"),
        "step":             observation.get("step"),
        "max_steps":        observation.get("max_steps"),
        "available_tools":  enriched_tools,
        "last_tool_result": observation.get("tool_result"),
        "objectives":       observation.get("objectives", []),
        "pending_requests": observation.get("pending_requests", {}),
    }

    history_section = (
        "PREVIOUS TOOL CALLS AND RESULTS:\n" + "\n".join(history[-6:])
        if history else "[No tool calls yet]"
    )

    return (
        f"Current observation:\n{json.dumps(obs_summary, indent=2)}\n\n"
        f"{history_section}\n\n"
        "What is your next tool call? Respond with <think>...</think> then JSON."
    )


def parse_action(text: str, available_tools: List[str]) -> Dict[str, Any]:
    """Extract the tool call from the model output. Returns a format score for shaping."""
    import re
    has_think = "<think>" in text and "</think>" in text

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            blob = json.loads(text[start:end + 1])
            if isinstance(blob, dict) and "tool_name" in blob:
                return {
                    "tool_name": blob["tool_name"],
                    "arguments": blob.get("arguments", {}),
                    "_format_score": 1.0 if has_think else 0.7,
                }
        except json.JSONDecodeError:
            pass

    m = re.search(r'\{[^{}]*"tool_name"\s*:\s*"([^"]+)"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            blob = json.loads(m.group())
            return {"tool_name": blob["tool_name"],
                    "arguments": blob.get("arguments", {}),
                    "_format_score": 0.4}
        except json.JSONDecodeError:
            return {"tool_name": m.group(1), "arguments": {}, "_format_score": 0.4}

    for t in available_tools:
        if t in text:
            return {"tool_name": t, "arguments": {}, "_format_score": 0.1}

    return {"tool_name": "__UNPARSEABLE__", "arguments": {}, "_format_score": 0.0}


print("Helper functions defined.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Rollout function (notebook "cell-12" analogue)
# ─────────────────────────────────────────────────────────────────────────────

def rollout_once(
    trainer: GRPOTrainer,
    tokenizer,
    dataset_prompt: str,
    system_prompt: str,
    max_turns: int,
    task_id: str,
    seed: int,
    difficulty_level: int = 2,
) -> Dict[str, Any]:
    """Execute one full PrivilegeDesk episode using generate_rollout_completions."""
    world = WorldState()
    observation = world.reset(seed=seed, task_id=task_id, difficulty_level=difficulty_level)

    prompt_ids:     List[int]   = []
    completion_ids: List[int]   = []
    logprobs:       List        = []
    step_rewards:   List[float] = []
    history:        List[str]   = []
    episode_score:  float       = 0.0
    done:           bool        = False
    steps_taken:    int         = 0
    think_steps:    int         = 0

    MAX_TOK_ACCUM = 4096

    for _turn in range(max_turns):
        if done or len(completion_ids) >= MAX_TOK_ACCUM:
            break

        user_prompt = make_user_prompt(observation, history)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=True,
        )

        log.debug(
            "\n[task=%s seed=%d step=%d] ── PROMPT ──────────────────────────\n%s\n────────────────────────────────────",
            task_id, seed, _turn + 1, prompt_text,
        )

        rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]
        prompt_ids.extend(rollout_outputs["prompt_ids"])
        completion_ids.extend(rollout_outputs["completion_ids"])
        logprobs.extend(rollout_outputs.get("logprobs") or [])
        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            rollout_outputs["completion_ids"], skip_special_tokens=False
        )

        log.debug(
            "\n[task=%s seed=%d step=%d] ── COMPLETION ──────────────────────\n%s\n────────────────────────────────────",
            task_id, seed, _turn + 1, completion_text,
        )

        if "<think>" in completion_text and "</think>" in completion_text:
            think_steps += 1
        steps_taken += 1

        action = parse_action(completion_text, observation.get("available_tools", []))
        fmt_score = action.pop("_format_score", 1.0)

        if action["tool_name"] == "__UNPARSEABLE__":
            step_rewards.append(-0.10)
            history.append(f"Step {_turn + 1}: [UNPARSEABLE]")
            continue

        observation, step_reward, terminated, truncated, info = world.step(action)
        done = terminated or truncated
        if "tool_result" in info:
            observation["tool_result"] = info["tool_result"]

        fmt_penalty = (1.0 - fmt_score) * -0.10
        if done and info.get("episode_score") is not None:
            episode_score = float(info["episode_score"])
            step_rewards.append(episode_score + fmt_penalty)
        else:
            step_rewards.append(float(step_reward) + fmt_penalty)

        tool_res = observation.get("tool_result") or {}
        output = json.dumps(tool_res.get("result", {}))
        if len(output) > 2000:
            output = output[:2000] + "... (truncated)"
        entry = (
            f"Step {_turn + 1}: {action.get('tool_name')} "
            f"{json.dumps(action.get('arguments', {}))}\n  Output: {output}"
        )
        if tool_res.get("status") == "error":
            entry += "\n  Status: error"
        history.append(entry)

    # Fallback grader pass if max_turns exhausted without a terminal signal
    if not done and episode_score == 0.0:
        fallback = float(world.compute_episode_score().get("score", 0.0))
        if fallback > 0.0:
            episode_score = fallback
            if step_rewards:
                step_rewards[-1] = episode_score

    mean_step_reward = sum(step_rewards) / max(len(step_rewards), 1)
    format_rate = think_steps / max(steps_taken, 1)
    meta = TASK_REGISTRY[task_id]

    return {
        "prompt_ids":       prompt_ids,
        "completion_ids":   completion_ids,
        "logprobs":         logprobs,
        "episode_score":    episode_score,
        "mean_step_reward": mean_step_reward,
        "steps_taken":      steps_taken,
        "optimal_steps":    meta.optimal_steps,
        "format_rate":      format_rate,
    }


def rollout_func(prompts, trainer=None):
    """Called by GRPOTrainer once per training batch."""
    episode_prompt_ids: List[List[int]]   = []
    episode_completion_ids: List[List[int]] = []
    episode_logprobs: List[List[float]]   = []
    episode_scores: List[float]           = []
    step_reward_means: List[float]        = []
    steps_taken_list: List[int]           = []
    optimal_steps_list: List[int]         = []
    format_rates: List[float]             = []

    task_ids = list(TASK_WEIGHTS.keys())
    task_probs = list(TASK_WEIGHTS.values())

    for i, prompt_text in enumerate(prompts):
        task_id = random.choices(task_ids, weights=task_probs, k=1)[0]
        meta = TASK_REGISTRY[task_id]
        seed = random.randint(0, 1_000_000)

        log.info("Episode %d | task=%s | seed=%d", i + 1, task_id, seed)

        episode = rollout_once(
            trainer=trainer,
            tokenizer=tokenizer,
            dataset_prompt=prompt_text,
            system_prompt=system_prompt,
            max_turns=meta.max_steps,
            task_id=task_id,
            seed=seed,
            difficulty_level=meta.difficulty_level,
        )

        episode_prompt_ids.append(episode["prompt_ids"])
        episode_completion_ids.append(episode["completion_ids"])
        episode_logprobs.append(episode["logprobs"])
        episode_scores.append(episode["episode_score"])
        step_reward_means.append(episode["mean_step_reward"])
        steps_taken_list.append(episode["steps_taken"])
        optimal_steps_list.append(episode["optimal_steps"])
        format_rates.append(episode["format_rate"])

        log.info(
            "  → score=%.3f | mean_step=%.3f | steps=%d | fmt=%.0f%%",
            episode["episode_score"], episode["mean_step_reward"],
            episode["steps_taken"], episode["format_rate"] * 100,
        )

    return {
        "prompt_ids":       episode_prompt_ids,
        "completion_ids":   episode_completion_ids,
        "logprobs":         episode_logprobs,
        "episode_score":    episode_scores,
        "mean_step_reward": step_reward_means,
        "steps_taken":      steps_taken_list,
        "optimal_steps":    optimal_steps_list,
        "format_rate":      format_rates,
    }


print("Rollout functions defined.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Reward functions (notebook "cell-14" analogue — 4 signals)
# ─────────────────────────────────────────────────────────────────────────────

def reward_episode_score(completions, **kwargs):
    rewards = kwargs.get("episode_score")
    return [float(r) for r in rewards] if rewards else [0.0] * len(completions)


def reward_step_efficiency(completions, **kwargs):
    rewards = kwargs.get("mean_step_reward")
    return [float(r) for r in rewards] if rewards else [0.0] * len(completions)


def reward_format(completions, **kwargs):
    rates = kwargs.get("format_rate") or [0.0] * len(completions)
    return [0.10 if float(r) >= 0.5 else 0.0 for r in rates]


def reward_efficiency(completions, **kwargs):
    steps_list = kwargs.get("steps_taken")   or [0]   * len(completions)
    optimal    = kwargs.get("optimal_steps") or [10]  * len(completions)
    scores     = kwargs.get("episode_score") or [0.0] * len(completions)
    rewards = []
    for steps, h_star, score in zip(steps_list, optimal, scores):
        if float(score) <= 0.0:
            rewards.append(0.0); continue
        steps, h_star = int(steps), int(h_star)
        if steps <= h_star:
            rewards.append(0.10)
        else:
            rewards.append(round(0.10 * (0.85 ** (steps - h_star)), 4))
    return rewards


print("Reward functions: episode_score, step_efficiency, format, efficiency")


# ─────────────────────────────────────────────────────────────────────────────
# 6. CLI + main (notebook cells 6, 16, 18, 20, 22, 25 inline)
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GRPO training for PrivilegeDesk, OpenEnv-notebook style",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-id",     default="Qwen/Qwen3-1.7B")
    p.add_argument("--dataset-size", type=int, default=1000)
    p.add_argument("--output-dir",   default="./outputs/grpo_curriculum")
    p.add_argument("--push-to-hub",  action="store_true")
    p.add_argument("--debug",        action="store_true", help="Log full prompt and completion text for each step")
    p.add_argument("--report-to",    default="trackio", choices=["trackio", "tensorboard", "none"])
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        log.debug("Debug logging enabled — full prompt/completion will be printed each step.")

    # ── 2. Init model and tokenizer ─────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log.info("Model: %s", args.model_id)

    # ── 7. Create dataset ───────────────────────────────────────────────────
    dataset = Dataset.from_dict({
        "prompt": ["Resolve this IAM privilege management task."] * args.dataset_size
    })
    log.info("Dataset: %d prompts", len(dataset))

    # ── 8. Configure GRPO ───────────────────────────────────────────────────
    #
    # Matching the notebook, with one PrivilegeDesk-specific override:
    #
    #   max_completion_length    8 → 1024   (<think> + JSON tool call)
    #
    # max_prompt_length was removed in newer TRL versions; prompt length is
    # handled by the tokenizer / vLLM context window instead.
    #
    # Everything else — lr, grad_accum, num_generations, vLLM colocate,
    # gradient checkpointing, push_to_hub — is identical to the notebook.
    grpo_config = GRPOConfig(
        num_train_epochs=1,
        learning_rate=5e-6,
        gradient_accumulation_steps=64,
        per_device_train_batch_size=1,
        warmup_steps=20,
        num_generations=2,
        max_completion_length=1024,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        vllm_max_model_len=8192,
        output_dir=args.output_dir,
        report_to=args.report_to,
        trackio_space_id=args.output_dir if args.report_to == "trackio" else None,
        logging_steps=1,
        save_steps=10,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        push_to_hub=args.push_to_hub,
    )
    log.info("Output: %s | vLLM mode: colocate", args.output_dir)

    # ── 9. Create trainer ───────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=args.model_id,
        model_init_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "eager",
        },
        processing_class=tokenizer,
        reward_funcs=[
            reward_episode_score,
            reward_step_efficiency,
            reward_format,
            reward_efficiency,
        ],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
    )

    # GPU snapshot
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        total = round(gpu.total_memory / 1024**3, 3)
        log.info("GPU: %s — %s GB total, %s GB reserved", gpu.name, total, start_mem)

    # ── Train ───────────────────────────────────────────────────────────────
    trainer_stats = trainer.train()

    if torch.cuda.is_available():
        used = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        log.info(
            "Training time: %.1f min | Peak memory: %s GB",
            trainer_stats.metrics["train_runtime"] / 60, used,
        )

    # ── 10. Save and (optionally) push ─────────────────────────────────────
    trainer.save_model(args.output_dir)
    if args.push_to_hub:
        trainer.push_to_hub()
        log.info("Model saved to %s and pushed to Hub.", args.output_dir)
    else:
        log.info("Model saved to %s.", args.output_dir)
