"""
train_grpo_direct.py — GRPO + LoRA Curriculum Training for PrivilegeDesk
                        (direct in-process environment, no HTTP server required)

Identical to train_grpo.py except rollout_once calls WorldState directly instead
of hitting the HTTP /reset, /step, /grader endpoints.  This eliminates network
round-trip latency and removes the dependency on a running server process.

Usage
─────
  # With vLLM (recommended on A100/H100):
  python training/train_grpo_direct.py \
    --model-id "Qwen/Qwen3.5-2B" \
    --use-vllm --vllm-mode colocate \
    --report-to tensorboard \
    --episodes-per-phase 32 \
    --num-generations 8 \
    --output-dir ./outputs/grpo_run1

  # Without vLLM (CPU / small GPU):
  python training/train_grpo_direct.py \
    --model-id "Qwen/Qwen3.5-2B" \
    --episodes-per-phase 16 \
    --output-dir ./outputs/grpo_run1

  # Dry-run curriculum plan (no GPU needed):
  python training/train_grpo_direct.py --dry-run

  # Phase 4 only (adversarial multi-agent oversight):
  python training/train_grpo_direct.py \
    --model-id ./outputs/grpo_run1/phase3/adapter \
    --phase 4 \
    --episodes-per-phase 32 \
    --output-dir ./outputs/grpo_run1
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Critical for TRL + vLLM colocate on 80 GB GPU
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

try:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    from trl.experimental.openenv import generate_rollout_completions
    _TRAINING_AVAILABLE = True
except ImportError:
    _TRAINING_AVAILABLE = False

# Add project root to path so env/ imports resolve without an installed package
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from env.world_state import WorldState  # noqa: E402  (needs sys.path patch above)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("grpo_train_direct")


# ═══════════════════════════════════════════════════════════════════════════════
# Task registry + curriculum weights
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TaskMeta:
    task_id: str
    difficulty: str
    max_steps: int
    phase: int
    difficulty_level: int = 2


TASK_REGISTRY: Dict[str, TaskMeta] = {
    "access_decision": TaskMeta(
        task_id="access_decision", difficulty="easy", max_steps=5, phase=1, difficulty_level=2,
    ),
    "emergency_breakglass": TaskMeta(
        task_id="emergency_breakglass", difficulty="medium", max_steps=10, phase=2, difficulty_level=2,
    ),
    "jit_escalation": TaskMeta(
        task_id="jit_escalation", difficulty="medium", max_steps=15, phase=3, difficulty_level=2,
    ),
    "access_review": TaskMeta(
        task_id="access_review", difficulty="hard", max_steps=25, phase=3, difficulty_level=1,
    ),
    "separation_of_duties_audit": TaskMeta(
        task_id="separation_of_duties_audit", difficulty="hard", max_steps=25, phase=3, difficulty_level=1,
    ),
    "multi_agent_oversight": TaskMeta(
        task_id="multi_agent_oversight", difficulty="very_hard", max_steps=25, phase=4, difficulty_level=2,
    ),
}

# Optimal step horizon H* per task — used for efficiency reward decay
OPTIMAL_STEPS: Dict[str, int] = {
    "access_decision":            4,
    "emergency_breakglass":       7,
    "jit_escalation":            10,
    "access_review":             15,
    "separation_of_duties_audit":15,
    "multi_agent_oversight":     12,
}

PHASE_WEIGHTS: Dict[int, Dict[str, float]] = {
    1: {
        "access_decision":            1.00,
        "emergency_breakglass":       0.00,
        "jit_escalation":             0.00,
        "access_review":              0.00,
        "separation_of_duties_audit": 0.00,
        "multi_agent_oversight":      0.00,
    },
    2: {
        "access_decision":            0.60,
        "emergency_breakglass":       0.40,
        "jit_escalation":             0.00,
        "access_review":              0.00,
        "separation_of_duties_audit": 0.00,
        "multi_agent_oversight":      0.00,
    },
    3: {
        "access_decision":            0.20,
        "emergency_breakglass":       0.20,
        "jit_escalation":             0.25,
        "access_review":              0.20,
        "separation_of_duties_audit": 0.15,
        "multi_agent_oversight":      0.00,
    },
    4: {
        "access_decision":            0.10,
        "emergency_breakglass":       0.10,
        "jit_escalation":             0.10,
        "access_review":              0.10,
        "separation_of_duties_audit": 0.10,
        "multi_agent_oversight":      0.50,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# System prompt
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = textwrap.dedent("""
You are an enterprise IAM (Identity & Access Management) specialist agent.
You operate inside the PrivilegeDesk environment, which simulates a corporate
zero-standing-privilege access control system.

Your job is to use the available tools to complete the assigned IAM task.
You may reason first inside <think>...</think> tags, then emit EXACTLY ONE JSON object:

<think>
Brief reasoning about what to investigate or decide next.
</think>
{
  "tool_name": "<tool_name>",
  "arguments": { "<key>": "<value>", ... }
}

Rules:
- Only call tools listed in available_tools in the observation
- No text outside the <think> block and JSON object
- For access.decide: use "approve" or "deny" for the decision field
- For entitlement.revoke: provide the entitlement_id
- For review.submit / sod.submit_report: call when you have finished all revocations
- For access.grant / access.deny: call only after all approvals are in
- When done, the environment will signal done=true — do not continue after that

When the task involves governing autonomous sub-agents (the task_goal will say so):
- sub_agents listed in the observation are bots — investigate them like any identity
- Use sod.check_user and sod.get_conflict_matrix to discover hidden permission combinations
- A dangerous combination: an agent holding both internal data access and external write access
- If a pending request would complete such a combination: deny it, then call emergency_breakglass
- Approving a request that creates a toxic permission combination is a catastrophic failure

All context you need is in the observation JSON.
""").strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt builder + action parser
# ═══════════════════════════════════════════════════════════════════════════════

def _build_user_message(observation: Dict[str, Any], history: List[str]) -> str:
    obs_summary: Dict[str, Any] = {
        "task_goal":        observation.get("task_goal"),
        "step":             observation.get("step"),
        "max_steps":        observation.get("max_steps"),
        "available_tools":  observation.get("available_tools", []),
        "last_tool_result": observation.get("tool_result"),
        "objectives":       observation.get("objectives", []),
        "pending_requests": observation.get("pending_requests", {}),
    }

    # Show sub-agent identities (names/purpose only) so the agent knows they exist.
    # identity_graph and the full graph structure are intentionally omitted —
    # the agent must call sod.check_user / sod.get_conflict_matrix to discover toxic paths.
    # rogue_agent_requests is never shown — the field name would be a spoiler.
    if observation.get("sub_agents"):
        obs_summary["sub_agents"] = {
            sid: {"name": a.get("name"), "purpose": a.get("purpose"), "status": a.get("status")}
            for sid, a in observation["sub_agents"].items()
        }

    history_str = ""
    if history:
        history_str = "PREVIOUS TOOL CALLS AND RESULTS:\n" + "\n".join(history[-6:]) + "\n\n"

    return (
        f"Current observation:\n{json.dumps(obs_summary, indent=2)}\n\n"
        + history_str
        + "What is your next tool call? Respond with <think>...</think> then JSON."
    )


def _parse_action(text: str, available_tools: List[str]) -> Dict[str, Any]:
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            action = json.loads(text[start:end + 1])
            if "tool_name" in action:
                return action
        except json.JSONDecodeError:
            pass

    fallback = next(
        (t for t in available_tools if t.endswith(".list") or t.endswith(".view")),
        (available_tools[0] if available_tools else "policy.list"),
    )
    return {"tool_name": fallback, "arguments": {}}


def _apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Rollout — one full episode (closed-loop, direct WorldState calls)
# ═══════════════════════════════════════════════════════════════════════════════

def rollout_once(
    trainer: "GRPOTrainer",
    task_id: str,
    seed: int,
    tokenizer,
    difficulty_level: int = 2,
) -> Dict[str, Any]:
    """
    Run one full PrivilegeDesk episode using generate_rollout_completions
    for per-step vLLM-backed generation.

    Calls WorldState directly — no HTTP server required.

    Returns:
        prompt_ids, completion_ids, logprobs   — full episode token sequences
        episode_score                          — grader score (0–1), main signal
        mean_step_reward                       — mean per-step reward, secondary signal
    """
    world = WorldState()
    obs = world.reset(seed=seed, task_id=task_id, difficulty_level=difficulty_level)

    prompt_ids:     List[int]   = []
    completion_ids: List[int]   = []
    logprobs:       List = []
    step_rewards:   List[float] = []
    history:        List[str]   = []
    episode_score:  float       = 0.0
    done:           bool        = False
    steps_taken:    int         = 0
    think_steps:    int         = 0

    max_steps     = obs.get("max_steps", TASK_REGISTRY[task_id].max_steps)
    MAX_TOK_ACCUM = 4096

    for _step in range(max_steps):
        if done or len(completion_ids) >= MAX_TOK_ACCUM:
            break

        user_msg = _build_user_message(obs, history)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        prompt_text = _apply_chat_template(tokenizer, messages)

        try:
            rollout_out = generate_rollout_completions(trainer, [prompt_text])[0]
        except Exception as exc:
            log.warning("generate_rollout_completions failed at step %d: %s", _step, exc)
            step_rewards.append(-0.20)
            break

        prompt_ids.extend(rollout_out["prompt_ids"])
        completion_ids.extend(rollout_out["completion_ids"])
        raw_lps = rollout_out.get("logprobs") or []
        if raw_lps and isinstance(raw_lps[0], (int, float)):
            logprobs.extend([(float(lp),) for lp in raw_lps])
        else:
            logprobs.extend(raw_lps)

        completion_text = rollout_out.get("text") or tokenizer.decode(
            rollout_out["completion_ids"], skip_special_tokens=True
        )
        if "<think>" in completion_text and "</think>" in completion_text:
            think_steps += 1
        steps_taken += 1
        action = _parse_action(completion_text, obs.get("available_tools", []))

        try:
            obs, step_reward, terminated, truncated, info = world.step(action)
            done = terminated or truncated
        except Exception as exc:
            log.warning("world.step failed: %s — ending episode", exc)
            step_rewards.append(-0.20)
            break

        if done and info.get("episode_score") is not None:
            episode_score = float(info["episode_score"])
            step_rewards.append(episode_score)
        else:
            step_rewards.append(float(step_reward))

        tool_name = action.get("tool_name", "?")
        args_str  = json.dumps(action.get("arguments", {}))
        tool_res  = info.get("tool_result") or {}

        output = json.dumps(tool_res.get("result", {}))
        if len(output) > 300:
            output = output[:300] + "... (truncated)"

        history_str_entry = f"Step {_step + 1}: {tool_name} {args_str}\n  Output: {output}"
        if tool_res.get("status") == "error":
            history_str_entry += "\n  Status: error"

        history.append(history_str_entry)

    # Fallback: compute grader score if env never signalled done (max_steps exceeded)
    if not done or episode_score == 0.0:
        try:
            score_dict = world.compute_episode_score()
            fallback = float(score_dict.get("score", 0.0))
            if fallback > 0.0:
                episode_score = fallback
                if step_rewards:
                    step_rewards[-1] = episode_score
        except Exception:
            pass

    mean_step_reward = sum(step_rewards) / max(len(step_rewards), 1)
    format_rate = think_steps / max(steps_taken, 1)

    return {
        "prompt_ids":       prompt_ids,
        "completion_ids":   completion_ids,
        "logprobs":         logprobs,
        "episode_score":    episode_score,
        "mean_step_reward": mean_step_reward,
        "steps_taken":      steps_taken,
        "format_rate":      format_rate,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-component reward extractors (TRL convention — read from rollout_func kwargs)
# ═══════════════════════════════════════════════════════════════════════════════

def reward_episode_score(completions: List[str], **kwargs) -> List[float]:
    """Primary signal: grader score (0–1) for the full episode."""
    return [float(r) for r in kwargs.get("episode_score", [0.0] * len(completions))]


def reward_step_efficiency(completions: List[str], **kwargs) -> List[float]:
    """Secondary signal: mean per-step reward — rewards good exploration."""
    return [float(r) for r in kwargs.get("mean_step_reward", [0.0] * len(completions))]


def reward_format(completions: List[str], **kwargs) -> List[float]:
    """Format reward: +0.10 per episode if model used <think> tags on >50% of steps."""
    format_rates = kwargs.get("format_rate", [0.0] * len(completions))
    return [0.10 if float(r) >= 0.5 else 0.0 for r in format_rates]


def reward_efficiency(completions: List[str], **kwargs) -> List[float]:
    """
    Intrinsic horizon (H*) reward: +0.10 if steps_taken ≤ optimal H*, then decays
    exponentially at 0.85^excess_steps to penalise inefficient meandering.
    Only applied when episode_score > 0 (no reward for efficient failure).
    """
    steps_list   = kwargs.get("steps_taken",   [0]   * len(completions))
    optimal_list = kwargs.get("optimal_steps", [10]  * len(completions))
    score_list   = kwargs.get("episode_score", [0.0] * len(completions))
    rewards = []
    for steps, h_star, score in zip(steps_list, optimal_list, score_list):
        if float(score) <= 0.0:
            rewards.append(0.0)
            continue
        steps, h_star = int(steps), int(h_star)
        if steps <= h_star:
            rewards.append(0.10)
        else:
            excess = steps - h_star
            rewards.append(round(0.10 * (0.85 ** excess), 4))
    return rewards


# ═══════════════════════════════════════════════════════════════════════════════
# Phase training
# ═══════════════════════════════════════════════════════════════════════════════

def train_phase(
    phase: int,
    model_id: str,
    tokenizer,
    args: argparse.Namespace,
    seed_offset: int,
    reward_log_path: Path,
) -> str:
    """
    Train one curriculum phase.

    rollout_func is called by GRPOTrainer once per GRPO step (with a batch of
    prompts from the dataset).  It runs full closed-loop episodes and returns
    the raw token sequences + pre-computed rewards.  Reward extractors then pull
    those rewards out of kwargs — no environment call happens inside the extractors.

    Returns the path to the saved LoRA adapter for this phase.
    """
    weights    = PHASE_WEIGHTS[phase]
    task_ids   = [tid for tid, w in weights.items() if w > 0]
    task_probs = [weights[tid] for tid in task_ids]
    total_w    = sum(task_probs)
    task_probs = [p / total_w for p in task_probs]

    episode_counter = [0]
    all_scores:      List[float] = []

    def rollout_func(prompts: List[str], trainer: "GRPOTrainer") -> Dict[str, Any]:
        ep_prompt_ids:     List[List[int]]   = []
        ep_completion_ids: List[List[int]]   = []
        ep_logprobs:       List[List[float]] = []
        ep_scores:         List[float]       = []
        ep_step_rewards:   List[float]       = []
        ep_steps_taken:    List[int]         = []
        ep_optimal_steps:  List[int]         = []
        ep_format_rates:   List[float]       = []

        for _ in prompts:
            task_id = random.choices(task_ids, weights=task_probs, k=1)[0]
            meta    = TASK_REGISTRY[task_id]
            seed    = seed_offset + episode_counter[0] * 97 + hash(task_id) % 10_000

            log.info(
                "Phase %d | Episode %d | task=%s | seed=%d",
                phase, episode_counter[0] + 1, task_id, seed,
            )

            try:
                ep = rollout_once(
                    trainer=trainer,
                    task_id=task_id,
                    seed=seed,
                    tokenizer=tokenizer,
                    difficulty_level=meta.difficulty_level,
                )
            except Exception as exc:
                log.error("Episode failed (task=%s seed=%d): %s", task_id, seed, exc)
                ep = {
                    "prompt_ids": [], "completion_ids": [], "logprobs": [],
                    "episode_score": 0.0, "mean_step_reward": 0.0,
                    "steps_taken": 0, "format_rate": 0.0,
                }

            ep_prompt_ids.append(ep["prompt_ids"])
            ep_completion_ids.append(ep["completion_ids"])
            ep_logprobs.append(ep["logprobs"])
            ep_scores.append(ep["episode_score"])
            ep_step_rewards.append(ep["mean_step_reward"])
            ep_steps_taken.append(ep.get("steps_taken", 0))
            ep_optimal_steps.append(OPTIMAL_STEPS.get(task_id, 10))
            ep_format_rates.append(ep.get("format_rate", 0.0))

            all_scores.append(ep["episode_score"])
            episode_counter[0] += 1

            with open(reward_log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    episode_counter[0], phase, task_id, seed,
                    round(ep["episode_score"], 4),
                    round(ep["mean_step_reward"], 4),
                    ep.get("steps_taken", 0),
                    round(ep.get("format_rate", 0.0), 3),
                    datetime.now().isoformat(),
                ])

            log.info(
                "  → score=%.3f | mean_step=%.3f | steps=%d | fmt=%.0f%% | running_mean=%.3f",
                ep["episode_score"], ep["mean_step_reward"],
                ep.get("steps_taken", 0),
                ep.get("format_rate", 0.0) * 100,
                sum(all_scores) / len(all_scores),
            )

        return {
            "prompt_ids":       ep_prompt_ids,
            "completion_ids":   ep_completion_ids,
            "logprobs":         ep_logprobs,
            "episode_score":    ep_scores,
            "mean_step_reward": ep_step_rewards,
            "steps_taken":      ep_steps_taken,
            "optimal_steps":    ep_optimal_steps,
            "format_rate":      ep_format_rates,
        }

    # ── GRPOConfig ────────────────────────────────────────────────────────────
    phase_out = Path(args.output_dir) / f"phase{phase}"
    phase_out.mkdir(parents=True, exist_ok=True)

    grpo_cfg = GRPOConfig(
        output_dir=str(phase_out),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=1,
        num_generations=args.num_generations,
        max_prompt_length=2048,
        max_completion_length=512,
        warmup_steps=2,
        max_grad_norm=1.0,
        temperature=args.temperature,
        logging_steps=1,
        save_steps=max(args.episodes_per_phase // 2, 1),
        save_total_limit=2,
        report_to=args.report_to,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        loss_type="dapo",
        mask_truncated_completions=True,
        beta=0.01,
        use_vllm=args.use_vllm,
        **({"vllm_mode": args.vllm_mode,
            "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization}
           if args.use_vllm else {}),
    )

    peft_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    dataset = Dataset.from_dict({
        "prompt": ["Resolve this IAM privilege management task."] * args.episodes_per_phase
    })

    trainer = GRPOTrainer(
        model=model_id,
        processing_class=tokenizer,
        reward_funcs=[
            reward_episode_score,
            reward_step_efficiency,
            reward_format,
            reward_efficiency,
        ],
        train_dataset=dataset,
        args=grpo_cfg,
        rollout_func=rollout_func,
        peft_config=peft_cfg,
    )

    log.info(
        "Starting GRPO Phase %d — %d episodes | tasks=%s",
        phase, args.episodes_per_phase, task_ids,
    )
    trainer.train()

    adapter_dir = phase_out / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    log.info("Phase %d complete. Adapter → %s", phase, adapter_dir)
    return str(adapter_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# Argument parser
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GRPO + LoRA curriculum training for PrivilegeDesk (direct mode)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument("--model-id",   default="Qwen/Qwen3.5-2B")
    parser.add_argument("--lora-rank",  type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--load-4bit",  action="store_true")

    # vLLM
    parser.add_argument("--use-vllm",  action="store_true")
    parser.add_argument("--vllm-mode", choices=["colocate", "server"], default="colocate")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.5)

    # Curriculum
    parser.add_argument("--phase", default="all", choices=["1", "2", "3", "4", "all"])
    parser.add_argument("--episodes-per-phase", type=int, default=32)

    # GRPO hyperparams
    parser.add_argument("--num-generations", type=int,   default=8)
    parser.add_argument("--learning-rate",   type=float, default=2e-6)
    parser.add_argument("--grad-accum",      type=int,   default=8)
    parser.add_argument("--temperature",     type=float, default=1.0)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--report-to",      default="tensorboard",
                        choices=["tensorboard", "wandb", "none"])

    # Output
    parser.add_argument("--output-dir", default="./outputs/grpo_curriculum")
    parser.add_argument("--dry-run",    action="store_true")

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _phase_description(phase: int) -> str:
    return {
        1: "Phase 1 (Stabilize):   access_decision only",
        2: "Phase 2 (Expand):      + emergency_breakglass",
        3: "Phase 3 (Full):        all 5 tasks, stratified sampling",
        4: "Phase 4 (Adversarial): multi_agent_oversight (50%) + all tasks",
    }.get(phase, f"Phase {phase}")


def _print_curriculum_plan(args: argparse.Namespace):
    phases = [1, 2, 3, 4] if args.phase == "all" else [int(args.phase)]
    print("\nCurriculum plan:")
    print("─" * 60)
    for phase in phases:
        weights = PHASE_WEIGHTS[phase]
        active  = {tid: w for tid, w in weights.items() if w > 0}
        total_w = sum(active.values())
        print(f"\n  {_phase_description(phase)}")
        for tid, w in active.items():
            meta  = TASK_REGISTRY[tid]
            n_eps = round(args.episodes_per_phase * w / total_w)
            print(
                f"    {tid:<32} weight={w:.0%}  ~{n_eps} eps  "
                f"max_steps={meta.max_steps}"
            )
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    log.info("=" * 60)
    log.info("PrivilegeDesk GRPO Curriculum Training (direct mode)")
    log.info("  model          : %s", args.model_id)
    log.info("  lora_rank      : %d  alpha=%d", args.lora_rank, args.lora_alpha)
    log.info("  phase(s)       : %s", args.phase)
    log.info("  episodes/phase : %d", args.episodes_per_phase)
    log.info("  num_generations: %d", args.num_generations)
    log.info("  use_vllm       : %s  mode=%s", args.use_vllm,
             args.vllm_mode if args.use_vllm else "n/a")
    log.info("  report_to      : %s", args.report_to)
    log.info("  output_dir     : %s", args.output_dir)
    log.info("=" * 60)

    if args.dry_run:
        _print_curriculum_plan(args)
        log.info("--dry-run: exiting before model load.")
        sys.exit(0)

    if not _TRAINING_AVAILABLE:
        log.error(
            "Training dependencies missing.\n"
            "  pip install torch transformers peft trl datasets"
        )
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    reward_log_path = out_root / "reward_log.csv"
    with open(reward_log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "episode", "phase", "task_id", "seed",
            "episode_score", "mean_step_reward",
            "steps_taken", "format_rate", "timestamp",
        ])

    phases           = [1, 2, 3, 4] if args.phase == "all" else [int(args.phase)]
    current_model_id = args.model_id

    for phase in phases:
        log.info("\n" + "─" * 60)
        log.info(_phase_description(phase))
        log.info("─" * 60)

        current_model_id = train_phase(
            phase=phase,
            model_id=current_model_id,
            tokenizer=tokenizer,
            args=args,
            seed_offset=args.seed * 1000 + phase * 10_000,
            reward_log_path=reward_log_path,
        )

    log.info("\n" + "=" * 60)
    log.info("Training complete!")
    log.info("Final adapter : %s", current_model_id)
    log.info("Reward log    : %s", reward_log_path)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
