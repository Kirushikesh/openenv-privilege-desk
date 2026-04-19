"""
train_grpo.py — GRPO + LoRA Curriculum Training for PrivilegeDesk

Follows the kube-sre-gym pattern (hackathon winner):
  • rollout_func handles multi-step episodes with real env feedback (closed-loop)
  • generate_rollout_completions drives per-step vLLM generation inside the loop
  • reward_funcs are lightweight kwargs extractors — rewards are pre-computed in rollout
  • DAPO improvements: asymmetric clipping, mask_truncated_completions, light KL penalty

Curriculum phases
─────────────────
  Phase 1 – Stabilize : access_decision only
  Phase 2 – Expand    : + emergency_breakglass
  Phase 3 – Full      : all 5 tasks, stratified sampling

Each phase trains a separate GRPOTrainer, carrying the LoRA adapter forward.

Usage
─────
  # With vLLM (recommended on A100/H100):
  python training/train_grpo.py \
    --model-id "Qwen/Qwen3.5-2B" \
    --env-url  "http://localhost:8000" \
    --use-vllm --vllm-mode colocate \
    --report-to tensorboard \
    --episodes-per-phase 32 \
    --num-generations 8 \
    --output-dir ./outputs/grpo_run1

  # Without vLLM (CPU / small GPU):
  python training/train_grpo.py \
    --model-id "Qwen/Qwen3.5-2B" \
    --env-url  "http://localhost:8000" \
    --episodes-per-phase 16 \
    --output-dir ./outputs/grpo_run1

  # Dry-run curriculum plan (no GPU needed):
  python training/train_grpo.py --dry-run
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

import requests

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("grpo_train")


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
}

PHASE_WEIGHTS: Dict[int, Dict[str, float]] = {
    1: {
        "access_decision":            1.00,
        "emergency_breakglass":       0.00,
        "jit_escalation":             0.00,
        "access_review":              0.00,
        "separation_of_duties_audit": 0.00,
    },
    2: {
        "access_decision":            0.60,
        "emergency_breakglass":       0.40,
        "jit_escalation":             0.00,
        "access_review":              0.00,
        "separation_of_duties_audit": 0.00,
    },
    3: {
        "access_decision":            0.20,
        "emergency_breakglass":       0.20,
        "jit_escalation":             0.25,
        "access_review":              0.20,
        "separation_of_duties_audit": 0.15,
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
You must respond with EXACTLY ONE JSON object per turn:

{
  "tool_name": "<tool_name>",
  "arguments": { "<key>": "<value>", ... }
}

Rules:
- Only call tools listed in available_tools in the observation
- Do not add any text outside the JSON object
- For access.decide: use "approve" or "deny" for the decision field
- For entitlement.revoke: provide the entitlement_id
- For review.submit / sod.submit_report: call when you have finished all revocations
- For access.grant / access.deny: call only after all approvals are in
- When done, the environment will signal done=true — do not continue after that

All context you need is in the observation JSON.
""").strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt builder + action parser
# ═══════════════════════════════════════════════════════════════════════════════

def _build_user_message(observation: Dict[str, Any], history: List[str]) -> str:
    obs_summary = {
        "task_id":               observation.get("task_id"),
        "task_goal":             observation.get("task_goal"),
        "step":                  observation.get("step"),
        "max_steps":             observation.get("max_steps"),
        "available_tools":       observation.get("available_tools", []),
        "pending_requests":      observation.get("pending_requests", {}),
        "policies":              observation.get("policies", {}),
        "incidents":             observation.get("incidents", {}),
        "conflict_matrix":       observation.get("conflict_matrix", {}),
        "approval_chains":       observation.get("approval_chains", {}),
        "entitlements": {
            eid: {k: v for k, v in e.items()
                  if k in ("role", "user_id", "resource_id",
                           "days_since_use", "expires_at", "source")}
            for eid, e in list(observation.get("entitlements", {}).items())[:15]
        },
        "last_tool_result":      observation.get("tool_result"),
        "review_target_user_id": observation.get("review_target_user_id"),
        "objectives":            observation.get("objectives", []),
    }
    history_str = ""
    if history:
        history_str = "Recent history:\n" + "\n".join(history[-6:]) + "\n\n"

    return (
        f"Current observation:\n{json.dumps(obs_summary, indent=2)}\n\n"
        + history_str
        + "What is your next tool call? Respond with JSON only."
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
# Rollout — one full episode (closed-loop, env feedback at every step)
# ═══════════════════════════════════════════════════════════════════════════════

def rollout_once(
    trainer: "GRPOTrainer",
    env_url: str,
    task_id: str,
    seed: int,
    tokenizer,
    difficulty_level: int = 2,
) -> Dict[str, Any]:
    """
    Run one full PrivilegeDesk episode using generate_rollout_completions
    for per-step vLLM-backed generation.

    Token sequences are accumulated across turns — same pattern as kube-sre-gym:
    GRPO assigns the episode-level reward to the full token sequence.

    Returns:
        prompt_ids, completion_ids, logprobs   — full episode token sequences
        episode_score                          — grader score (0–1), main signal
        mean_step_reward                       — mean per-step reward, secondary signal
    """
    resp = requests.post(
        f"{env_url}/reset",
        json={"task_id": task_id, "seed": seed, "difficulty_level": difficulty_level},
        timeout=30,
    )
    resp.raise_for_status()
    obs = resp.json().get("observation", resp.json())

    prompt_ids:     List[int]   = []
    completion_ids: List[int]   = []
    logprobs:       List = []   # List[Tuple[float,...]] — wrapped for TRL compatibility
    step_rewards:   List[float] = []
    history:        List[str]   = []
    episode_score:  float       = 0.0
    done:           bool        = False

    max_steps     = obs.get("max_steps", TASK_REGISTRY[task_id].max_steps)
    MAX_TOK_ACCUM = 4096  # prevent CUDA OOM on long episodes

    for _step in range(max_steps):
        if done or len(completion_ids) >= MAX_TOK_ACCUM:
            break

        user_msg = _build_user_message(obs, history)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]
        prompt_text = _apply_chat_template(tokenizer, messages)

        # generate_rollout_completions delegates to vLLM (colocate/server) or
        # falls back to standard HF generate when use_vllm=False.
        try:
            rollout_out = generate_rollout_completions(trainer, [prompt_text])[0]
        except Exception as exc:
            log.warning("generate_rollout_completions failed at step %d: %s", _step, exc)
            step_rewards.append(-0.20)
            break

        prompt_ids.extend(rollout_out["prompt_ids"])
        completion_ids.extend(rollout_out["completion_ids"])
        # TRL expects logprobs as List[Tuple[float,...]] so lp[0] yields the float.
        # generate_rollout_completions may return plain floats — wrap if needed.
        raw_lps = rollout_out.get("logprobs") or []
        if raw_lps and isinstance(raw_lps[0], (int, float)):
            logprobs.extend([(float(lp),) for lp in raw_lps])
        else:
            logprobs.extend(raw_lps)

        completion_text = rollout_out.get("text") or tokenizer.decode(
            rollout_out["completion_ids"], skip_special_tokens=True
        )
        action = _parse_action(completion_text, obs.get("available_tools", []))

        try:
            step_resp = requests.post(
                f"{env_url}/step", json={"action": action}, timeout=30,
            )
            step_resp.raise_for_status()
            step_data = step_resp.json()
        except Exception as exc:
            log.warning("Step request failed: %s — ending episode", exc)
            step_rewards.append(-0.20)
            break

        obs          = step_data.get("observation", {})
        step_reward  = float(step_data.get("reward", 0.0) or 0.0)
        done         = step_data.get("done", False)
        metadata     = step_data.get("metadata", {})

        if done and metadata.get("episode_score") is not None:
            episode_score = float(metadata["episode_score"])
            step_rewards.append(episode_score)   # terminal step gets grader score
        else:
            step_rewards.append(step_reward)

        tool_name = action.get("tool_name", "?")
        status    = (obs.get("tool_result") or {}).get("status", "?")
        history.append(f"Step {_step + 1}: {tool_name} → {status}")

    # Fallback: fetch grader score if env never set done=True (max_steps exceeded)
    if not done or episode_score == 0.0:
        try:
            gr = requests.post(f"{env_url}/grader", json={}, timeout=10)
            fallback = float((gr.json() if gr.ok else {}).get("score", 0.0))
            if fallback > 0.0:
                episode_score = fallback
                if step_rewards:
                    step_rewards[-1] = episode_score
        except Exception:
            pass

    mean_step_reward = sum(step_rewards) / max(len(step_rewards), 1)

    return {
        "prompt_ids":       prompt_ids,
        "completion_ids":   completion_ids,
        "logprobs":         logprobs,
        "episode_score":    episode_score,
        "mean_step_reward": mean_step_reward,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Reward extractors (TRL convention — read from rollout_func kwargs)
# ═══════════════════════════════════════════════════════════════════════════════

def reward_episode_score(completions: List[str], **kwargs) -> List[float]:
    """Primary signal: grader score (0–1) for the full episode."""
    return [float(r) for r in kwargs.get("episode_score", [0.0] * len(completions))]


def reward_step_efficiency(completions: List[str], **kwargs) -> List[float]:
    """Secondary signal: mean per-step reward — rewards good exploration."""
    return [float(r) for r in kwargs.get("mean_step_reward", [0.0] * len(completions))]


# ═══════════════════════════════════════════════════════════════════════════════
# Phase training
# ═══════════════════════════════════════════════════════════════════════════════

def train_phase(
    phase: int,
    model_id: str,
    tokenizer,
    env_url: str,
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

    Returns the path to the saved LoRA adapter for this phase (used as model_id
    for the next phase to continue training from).
    """
    weights    = PHASE_WEIGHTS[phase]
    task_ids   = [tid for tid, w in weights.items() if w > 0]
    task_probs = [weights[tid] for tid in task_ids]
    total_w    = sum(task_probs)
    task_probs = [p / total_w for p in task_probs]

    episode_counter = [0]
    all_scores:      List[float] = []

    # ── rollout_func: called by GRPOTrainer with one prompt per episode slot ──
    def rollout_func(prompts: List[str], trainer: "GRPOTrainer") -> Dict[str, Any]:
        ep_prompt_ids:     List[List[int]]   = []
        ep_completion_ids: List[List[int]]   = []
        ep_logprobs:       List[List[float]] = []
        ep_scores:         List[float]       = []
        ep_step_rewards:   List[float]       = []

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
                    env_url=env_url,
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
                }

            ep_prompt_ids.append(ep["prompt_ids"])
            ep_completion_ids.append(ep["completion_ids"])
            ep_logprobs.append(ep["logprobs"])
            ep_scores.append(ep["episode_score"])
            ep_step_rewards.append(ep["mean_step_reward"])

            all_scores.append(ep["episode_score"])
            episode_counter[0] += 1

            with open(reward_log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    episode_counter[0], phase, task_id, seed,
                    round(ep["episode_score"], 4),
                    round(ep["mean_step_reward"], 4),
                    datetime.now().isoformat(),
                ])

            log.info(
                "  → episode_score=%.3f | mean_step=%.3f | running_mean=%.3f",
                ep["episode_score"], ep["mean_step_reward"],
                sum(all_scores) / len(all_scores),
            )

        return {
            "prompt_ids":       ep_prompt_ids,
            "completion_ids":   ep_completion_ids,
            "logprobs":         ep_logprobs,
            "episode_score":    ep_scores,
            "mean_step_reward": ep_step_rewards,
        }

    # ── GRPOConfig ────────────────────────────────────────────────────────────
    phase_out = Path(args.output_dir) / f"phase{phase}"
    phase_out.mkdir(parents=True, exist_ok=True)

    grpo_cfg = GRPOConfig(
        output_dir=str(phase_out),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        generation_batch_size=args.num_generations,
        num_generations=args.num_generations,
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
        # DAPO improvements (same as kube-sre-gym)
        loss_type="dapo",              # asymmetric clipping + dynamic sampling
        mask_truncated_completions=True,  # drop token-capped episodes from loss
        beta=0.01,                     # light KL — allow the model to diverge from base
        # vLLM (optional)
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

    # Dummy dataset — each entry triggers one episode in rollout_func.
    # The prompt text is ignored; rollout_func samples tasks from PHASE_WEIGHTS.
    dataset = Dataset.from_dict({
        "prompt": ["Resolve this IAM privilege management task."] * args.episodes_per_phase
    })

    trainer = GRPOTrainer(
        model=model_id,
        processing_class=tokenizer,
        reward_funcs=[reward_episode_score, reward_step_efficiency],
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
        description="GRPO + LoRA curriculum training for PrivilegeDesk",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument("--model-id",   default="Qwen/Qwen3.5-2B")
    parser.add_argument("--lora-rank",  type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--load-4bit",  action="store_true",
                        help="Load model in 4-bit (BitsAndBytes) for low VRAM")

    # Environment
    parser.add_argument("--env-url", default=os.getenv("ENV_URL", "http://localhost:8000"))

    # vLLM
    parser.add_argument("--use-vllm",  action="store_true",
                        help="Enable vLLM for fast inference during rollouts (recommended on H100)")
    parser.add_argument("--vllm-mode", choices=["colocate", "server"], default="colocate",
                        help="vLLM mode: colocate (1 GPU) or server (separate process)")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.5,
                        help="Fraction of GPU memory for vLLM (0.0–1.0)")

    # Curriculum
    parser.add_argument("--phase", default="all", choices=["1", "2", "3", "all"])
    parser.add_argument("--episodes-per-phase", type=int, default=32)

    # GRPO hyperparams
    parser.add_argument("--num-generations", type=int,   default=8,
                        help="G for GRPO — completions per prompt for group advantage")
    parser.add_argument("--learning-rate",   type=float, default=2e-6)
    parser.add_argument("--grad-accum",      type=int,   default=8)
    parser.add_argument("--temperature",     type=float, default=1.0,
                        help="T=1.0 is optimal for GRPO exploration diversity")
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--report-to",      default="tensorboard",
                        choices=["tensorboard", "wandb", "none"],
                        help="Logging backend for tensorboard / wandb / none")

    # Output
    parser.add_argument("--output-dir", default="./outputs/grpo_curriculum")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print curriculum plan and exit (no GPU needed)")

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _phase_description(phase: int) -> str:
    return {
        1: "Phase 1 (Stabilize): access_decision only",
        2: "Phase 2 (Expand): + emergency_breakglass",
        3: "Phase 3 (Full): all 5 tasks, stratified sampling",
    }.get(phase, f"Phase {phase}")


def _print_curriculum_plan(args: argparse.Namespace):
    phases = [1, 2, 3] if args.phase == "all" else [int(args.phase)]
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
    log.info("PrivilegeDesk GRPO Curriculum Training")
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

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Reward log ────────────────────────────────────────────────────────────
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    reward_log_path = out_root / "reward_log.csv"
    with open(reward_log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "episode", "phase", "task_id", "seed",
            "episode_score", "mean_step_reward", "timestamp",
        ])

    # ── Curriculum loop ───────────────────────────────────────────────────────
    phases           = [1, 2, 3] if args.phase == "all" else [int(args.phase)]
    current_model_id = args.model_id   # updated each phase to load the saved adapter

    for phase in phases:
        log.info("\n" + "─" * 60)
        log.info(_phase_description(phase))
        log.info("─" * 60)

        current_model_id = train_phase(
            phase=phase,
            model_id=current_model_id,
            tokenizer=tokenizer,
            env_url=args.env_url,
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
