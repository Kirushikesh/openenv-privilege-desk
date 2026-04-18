"""
train_grpo.py — GRPO + LoRA Curriculum Training for PrivilegeDesk

Trains a single LLM-based agent across all 5 privilege management tasks using
Group Relative Policy Optimization (GRPO) with curriculum learning.

Curriculum phases
─────────────────
  Phase 1 – Stabilize   : access_decision only          (easy,  5 steps)
  Phase 2 – Expand      : + emergency_breakglass         (medium, 10 steps)
  Phase 3 – Full        : + jit_escalation, access_review, separation_of_duties_audit
                          Stratified sampling — easy tasks carry more weight early;
                          harder tasks get more samples only after the model has
                          mastered simpler ones.

Design notes
────────────
• counter-argument to the original AI recommendation:
    - Tasks are ordered by *actual* difficulty (max_steps + grading complexity),
      NOT by the vague label in the template.  access_review is hard (25 steps,
      precision+recall grading), so it enters the curriculum *after* jit_escalation.
    - "Harder tasks get more samples" is inverted early on.  Stratified sampling
      starts heavily skewed toward easy tasks and shifts as training matures.
    - Task 4 (emergency_breakglass) is structurally different from Task 1 —
      it adds incident verification, security flagging, and a multi-step chain —
      so it enters Phase 2 on its own before we pile on even harder tasks.

Usage
─────
  python training/train_grpo.py \
    --model-id "Qwen/Qwen2.5-3B-Instruct" \
    --env-url  "http://localhost:8000" \
    --phase all \
    --episodes-per-phase 32 \
    --num-generations 8 \
    --lora-rank 16 \
    --output-dir ./outputs/grpo_run1

Environment variables
─────────────────────
  HF_TOKEN      Hugging Face / API auth token (required for model download)
  ENV_URL       Override for --env-url
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# ── Optional heavy deps (imported only when training, not for --dry-run) ──────
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import GRPOConfig, GRPOTrainer
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
# Task registry — ground truth for curriculum ordering
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TaskMeta:
    task_id: str
    difficulty: str          # "easy" | "medium" | "hard"
    max_steps: int           # from task_templates.py
    phase: int               # curriculum phase this task enters (1, 2, or 3)
    difficulty_level: int = 2  # environment difficulty_level for training (1–3)


TASK_REGISTRY: Dict[str, TaskMeta] = {
    "access_decision": TaskMeta(
        task_id="access_decision",
        difficulty="easy",
        max_steps=5,
        phase=1,
        difficulty_level=2,  # medium world complexity for training
    ),
    "emergency_breakglass": TaskMeta(
        task_id="emergency_breakglass",
        difficulty="medium",
        max_steps=10,
        phase=2,
        difficulty_level=2,
    ),
    "jit_escalation": TaskMeta(
        task_id="jit_escalation",
        difficulty="medium",
        max_steps=15,
        phase=3,
        difficulty_level=2,
    ),
    "access_review": TaskMeta(
        task_id="access_review",
        difficulty="hard",
        max_steps=25,
        phase=3,
        difficulty_level=1,  # start at easy world complexity for hard tasks
    ),
    "separation_of_duties_audit": TaskMeta(
        task_id="separation_of_duties_audit",
        difficulty="hard",
        max_steps=25,
        phase=3,
        difficulty_level=1,
    ),
}

# Sampling weights per phase — index 0=phase1, 1=phase2, 2=phase3
# Values are relative weights; tasks not in the current phase have weight 0.
# Key insight: easy tasks dominate early; weights shift toward harder tasks as
# the curriculum matures.  This is the *stratified sampling* the doc referred to.
PHASE_WEIGHTS: Dict[int, Dict[str, float]] = {
    1: {   # Phase 1: only access_decision
        "access_decision":              1.00,
        "emergency_breakglass":         0.00,
        "jit_escalation":               0.00,
        "access_review":                0.00,
        "separation_of_duties_audit":   0.00,
    },
    2: {   # Phase 2: easy dominates, breakglass added
        "access_decision":              0.60,
        "emergency_breakglass":         0.40,
        "jit_escalation":               0.00,
        "access_review":                0.00,
        "separation_of_duties_audit":   0.00,
    },
    3: {   # Phase 3: all tasks, harder gets proportionally more budget now
        "access_decision":              0.20,
        "emergency_breakglass":         0.20,
        "jit_escalation":               0.25,
        "access_review":                0.20,
        "separation_of_duties_audit":   0.15,  # most steps → fewer episodes needed
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# System prompt (same structure as inference.py)
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
# Rollout helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_user_message(observation: Dict[str, Any], history: List[str]) -> str:
    """Build a compact prompt from the current observation and recent history."""
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
        # Truncate large entitlements dict to keep prompt compact
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


def rollout_once(
    env_url: str,
    task_id: str,
    seed: int,
    tokenizer,
    model,
    difficulty_level: int = 2,
    temperature: float = 0.8,
    max_new_tokens: int = 256,
    device: str = "cuda",
) -> Tuple[List[str], List[str], List[float]]:
    """
    Run one full episode and collect (prompts, completions, rewards).

    Reward design (verified against grader.py + world_state.py):
      - Steps 1 … N-1 : raw per-step reward from the aggregator  (range −0.40 to +0.35)
      - Step N (terminal): episode_score from the grader  (range 0.0–1.0)
        The terminal step's regular reward is *replaced* by the graded episode score,
        matching exactly how kube-sre-gym assigns reward — the environment judge's score
        is the training signal, not an additive bonus on top of the step reward.
        No multiplier is applied.  The grader score (0–1) is already on a compatible
        scale with the per-step rewards (−0.40–+0.35) and provides plenty of variance
        for GRPO advantage computation.

    Returns three parallel lists:
        prompts     – one formatted prompt string per step
        completions – one model completion string per step
        rewards     – one float reward per step
    """
    import torch  # guard for --dry-run import mode

    # Reset environment
    resp = requests.post(
        f"{env_url}/reset",
        json={"task_id": task_id, "seed": seed, "difficulty_level": difficulty_level},
        timeout=30,
    )
    resp.raise_for_status()
    obs = resp.json().get("observation", resp.json())
    done = False

    prompts: List[str] = []
    completions: List[str] = []
    rewards: List[float] = []
    history: List[str] = []

    max_steps = obs.get("max_steps", TASK_REGISTRY[task_id].max_steps)

    for _step in range(max_steps):
        if done:
            break

        # Build prompt
        user_msg = _build_user_message(obs, history)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]

        # Tokenize with chat template
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        completion_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        # Parse action from completion
        action = _parse_action(completion_text, obs.get("available_tools", []))

        # Step environment
        try:
            step_resp = requests.post(
                f"{env_url}/step",
                json={"action": action},
                timeout=30,
            )
            step_resp.raise_for_status()
            step_data = step_resp.json()
        except Exception as exc:
            log.warning("Step request failed: %s — terminating episode", exc)
            rewards.append(-0.20)  # penalty equivalent to _ERROR_PENALTY
            prompts.append(text)
            completions.append(completion_text)
            break

        obs = step_data.get("observation", {})
        step_reward = float(step_data.get("reward", 0.0) or 0.0)
        done = step_data.get("done", False)
        # The environment already computes & returns episode_score in metadata
        # on the terminal step (world_state.py lines 131-133).  Extract it here
        # so we never need a separate /grader call.
        metadata = step_data.get("metadata", {})
        episode_score_from_env = metadata.get("episode_score")  # None on non-terminal steps

        tool_result = obs.get("tool_result", {})
        status = (tool_result or {}).get("status", "?")
        observations_preview = ((tool_result or {}).get("observations") or [""])[0][:60]

        # ── Reward assignment ──────────────────────────────────────────────────
        # Non-terminal steps: use the raw per-step reward (−0.40 to +0.35).
        # Terminal step: REPLACE with the episode graded score (0.0–1.0).
        #   Why replace, not add?
        #   - The graded score is the ground-truth signal for this whole episode.
        #   - Adding it on top of a step reward obscures the signal and inflates
        #     the last-step reward disproportionately vs all earlier steps.
        #   - kube-sre-gym and OpenENV-Hackathon do NOT apply any multiplier;
        #     they use the environment's reward directly at every step.
        if done and episode_score_from_env is not None:
            assigned_reward = float(episode_score_from_env)
        else:
            assigned_reward = step_reward

        prompts.append(text)
        completions.append(completion_text)
        rewards.append(assigned_reward)
        history.append(
            f"Step {_step + 1}: {action.get('tool_name', '?')} → {status}: {observations_preview}"
        )

    # ── Fallback: if episode ended without done=True (e.g. max_steps hit without
    #    the env setting terminated), fetch the final grade from /grader.
    if rewards and episode_score_from_env is None:
        try:
            grade_resp = requests.post(f"{env_url}/grader", json={}, timeout=10)
            fallback_score = float((grade_resp.json() if grade_resp.ok else {}).get("score", 0.0))
            rewards[-1] = fallback_score  # replace last step reward, same as above
            log.debug("Used /grader fallback: episode_score=%.3f", fallback_score)
        except Exception:
            pass  # leave last step reward as-is

    return prompts, completions, rewards


def _parse_action(text: str, available_tools: List[str]) -> Dict[str, Any]:
    """Best-effort JSON parse; falls back to a safe no-op tool."""
    # Try to extract the first JSON object
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            action = json.loads(text[start:end + 1])
            if "tool_name" in action:
                return action
        except json.JSONDecodeError:
            pass

    # Fallback: pick the first available read-only tool
    fallback = next(
        (t for t in available_tools if t.endswith(".list") or t.endswith(".view")),
        (available_tools[0] if available_tools else "policy.list"),
    )
    return {"tool_name": fallback, "arguments": {}}


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_phase_dataset(
    env_url: str,
    phase: int,
    num_episodes: int,
    tokenizer,
    model,
    seed_offset: int = 0,
    temperature: float = 0.8,
    device: str = "cuda",
) -> Tuple[List[str], List[str], List[float]]:
    """
    Collect `num_episodes` rollouts for the given curriculum phase.

    Episodes are sampled from tasks according to PHASE_WEIGHTS[phase].
    Returns flat parallel lists of (prompt, completion, reward).
    """
    weights = PHASE_WEIGHTS[phase]
    task_ids  = [tid for tid, w in weights.items() if w > 0]
    task_probs = [weights[tid] for tid in task_ids]
    # Normalize
    total = sum(task_probs)
    task_probs = [p / total for p in task_probs]

    all_prompts: List[str] = []
    all_completions: List[str] = []
    all_rewards: List[float] = []

    for ep_idx in range(num_episodes):
        # Sample task according to curriculum weights
        task_id = random.choices(task_ids, weights=task_probs, k=1)[0]
        meta = TASK_REGISTRY[task_id]
        seed = seed_offset + ep_idx * 97 + hash(task_id) % 10_000

        log.info(
            "Phase %d | Episode %d/%d | task=%s | seed=%d",
            phase, ep_idx + 1, num_episodes, task_id, seed,
        )

        try:
            prompts, completions, rewards = rollout_once(
                env_url=env_url,
                task_id=task_id,
                seed=seed,
                tokenizer=tokenizer,
                model=model,
                difficulty_level=meta.difficulty_level,
                temperature=temperature,
                device=device,
            )
        except Exception as exc:
            log.error("Episode failed (task=%s, seed=%d): %s", task_id, seed, exc)
            continue

        all_prompts.extend(prompts)
        all_completions.extend(completions)
        all_rewards.extend(rewards)

        log.info(
            "  → steps=%d | rewards=[min=%.2f mean=%.2f max=%.2f]",
            len(rewards),
            min(rewards) if rewards else 0,
            sum(rewards) / max(len(rewards), 1),
            max(rewards) if rewards else 0,
        )

    return all_prompts, all_completions, all_rewards


# ═══════════════════════════════════════════════════════════════════════════════
# Model + LoRA setup
# ═══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(
    model_id: str,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    device: str = "cuda",
    load_in_4bit: bool = False,
):
    """Load a causal LM with LoRA adapters for GRPO fine-tuning."""
    if not _TRAINING_AVAILABLE:
        raise RuntimeError(
            "Training dependencies not installed. "
            "Run: pip install torch transformers peft trl"
        )

    log.info("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Loading model: %s  (4bit=%s)", model_id, load_in_4bit)

    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    # Wrap with LoRA
    resolved_targets = target_modules or [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=resolved_targets,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# GRPO training step
# ═══════════════════════════════════════════════════════════════════════════════

def run_grpo_phase(
    model,
    tokenizer,
    prompts: List[str],
    completions: List[str],
    rewards: List[float],
    output_dir: str,
    phase: int,
    learning_rate: float = 2e-6,
    num_generations: int = 8,
    per_device_batch: int = 2,
    gradient_accumulation: int = 4,
    max_prompt_length: int = 2048,
    max_completion_length: int = 256,
):
    """
    Run one GRPO training pass over the collected rollout data.

    GRPO computes group-relative advantages:
        A_i = (r_i - mean(r_group)) / (std(r_group) + ε)

    This requires `num_generations` completions per prompt to compute meaningful
    group statistics.  Here we treat each unique prompt as a "group" and each
    step's completion as one member.
    """
    if not prompts:
        log.warning("Phase %d: no rollout data collected, skipping GRPO step.", phase)
        return

    phase_output = Path(output_dir) / f"phase{phase}"
    phase_output.mkdir(parents=True, exist_ok=True)

    # Build a dataset-compatible dict for TRL GRPOTrainer
    # GRPOTrainer expects: {"prompt": str, "completion": str, "reward": float}
    dataset_records = [
        {"prompt": p, "completion": c, "reward": float(r)}
        for p, c, r in zip(prompts, completions, rewards)
    ]

    # Save rollout data for analysis
    rollout_path = phase_output / "rollouts.jsonl"
    with rollout_path.open("w") as f:
        for rec in dataset_records:
            f.write(json.dumps(rec) + "\n")
    log.info("Saved %d rollout records → %s", len(dataset_records), rollout_path)

    # TRL GRPOTrainer configuration
    grpo_cfg = GRPOConfig(
        output_dir=str(phase_output),
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=gradient_accumulation,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        logging_steps=1,
        save_steps=50,
        warmup_ratio=0.05,
        report_to="none",
        remove_unused_columns=False,
    )

    # Reward function for GRPOTrainer — looks up pre-collected rewards
    reward_lookup = {(p, c): r for p, c, r in zip(prompts, completions, rewards)}

    def reward_fn(prompts_batch, completions_batch, **kwargs) -> List[float]:
        return [
            reward_lookup.get((p, c), 0.0)
            for p, c in zip(prompts_batch, completions_batch)
        ]

    try:
        from datasets import Dataset
        hf_dataset = Dataset.from_list([{"prompt": p} for p in prompts])
    except ImportError:
        log.error("'datasets' package not found. Install: pip install datasets")
        return

    trainer = GRPOTrainer(
        model=model,
        args=grpo_cfg,
        train_dataset=hf_dataset,
        tokenizer=tokenizer,
        reward_funcs=reward_fn,
    )

    log.info("Starting GRPO training for Phase %d (%d samples)...", phase, len(prompts))
    trainer.train()

    # Save LoRA adapter after each phase
    adapter_dir = phase_output / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    log.info("Saved Phase %d adapter → %s", phase, adapter_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation helper
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_phase(
    env_url: str,
    phase: int,
    tokenizer,
    model,
    eval_episodes: int = 5,
    seed_offset: int = 900_000,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Run deterministic evaluation episodes on each task available at this phase.
    Returns per-task mean episode scores.
    """
    weights = PHASE_WEIGHTS[phase]
    active_tasks = [tid for tid, w in weights.items() if w > 0]
    results: Dict[str, List[float]] = {tid: [] for tid in active_tasks}

    for task_id in active_tasks:
        meta = TASK_REGISTRY[task_id]
        for ep_i in range(eval_episodes):
            seed = seed_offset + ep_i * 13 + hash(task_id) % 5_000
            try:
                _, _, rewards = rollout_once(
                    env_url=env_url,
                    task_id=task_id,
                    seed=seed,
                    tokenizer=tokenizer,
                    model=model,
                    difficulty_level=meta.difficulty_level,
                    temperature=0.0,   # greedy for eval
                    device=device,
                )
                # rewards[-1] IS the episode_score (grader output, 0–1) for
                # terminal steps, per the design in rollout_once().
                episode_score = float(rewards[-1]) if rewards else 0.0
                results[task_id].append(episode_score)
            except Exception as exc:
                log.warning("Eval episode failed (task=%s): %s", task_id, exc)

    summary = {
        tid: round(sum(scores) / max(len(scores), 1), 4)
        for tid, scores in results.items()
    }
    log.info("Phase %d eval: %s", phase, summary)
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Main — full curriculum training loop
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GRPO + LoRA curriculum training for PrivilegeDesk",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument("--model-id",    default="Qwen/Qwen2.5-3B-Instruct",
                        help="HF model ID or local path")
    parser.add_argument("--lora-rank",   type=int, default=16,
                        help="LoRA rank r")
    parser.add_argument("--lora-alpha",  type=int, default=32,
                        help="LoRA scaling alpha")
    parser.add_argument("--load-4bit",   action="store_true",
                        help="Load model in 4-bit (BitsAndBytes) for low VRAM")

    # Environment
    parser.add_argument("--env-url",     default=os.getenv("ENV_URL", "http://localhost:8000"),
                        help="PrivilegeDesk server URL")

    # Curriculum
    parser.add_argument("--phase",       default="all",
                        choices=["1", "2", "3", "all"],
                        help="Which curriculum phase(s) to run")
    parser.add_argument("--episodes-per-phase", type=int, default=32,
                        help="Number of rollout episodes per curriculum phase")
    parser.add_argument("--eval-episodes",      type=int, default=5,
                        help="Eval episodes per task after each phase")

    # GRPO hyperparams
    parser.add_argument("--num-generations",    type=int,   default=8,
                        help="GRPO: number of completions per prompt for group advantage")
    parser.add_argument("--learning-rate",      type=float, default=2e-6,
                        help="AdamW learning rate")
    parser.add_argument("--per-device-batch",   type=int,   default=2)
    parser.add_argument("--grad-accum",         type=int,   default=4)
    parser.add_argument("--temperature",        type=float, default=0.8,
                        help="Sampling temperature during rollout")
    parser.add_argument("--seed",               type=int,   default=42,
                        help="Random seed base")

    # Output
    parser.add_argument("--output-dir",  default="./outputs/grpo_curriculum",
                        help="Root directory for adapters and logs")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print config and exit without training (no GPU needed)")

    args = parser.parse_args()

    # ── Seed
    random.seed(args.seed)

    log.info("=" * 60)
    log.info("PrivilegeDesk GRPO Curriculum Training")
    log.info("  model          : %s", args.model_id)
    log.info("  lora_rank      : %d  alpha=%d  4bit=%s", args.lora_rank, args.lora_alpha, args.load_4bit)
    log.info("  phase(s)       : %s", args.phase)
    log.info("  episodes/phase : %d", args.episodes_per_phase)
    log.info("  num_generations: %d", args.num_generations)
    log.info("  output_dir     : %s", args.output_dir)
    log.info("=" * 60)

    if args.dry_run:
        log.info("--dry-run: exiting before model load.")
        _print_curriculum_plan(args)
        sys.exit(0)

    if not _TRAINING_AVAILABLE:
        log.error(
            "Training dependencies missing. Install with:\n"
            "  pip install torch transformers peft trl datasets"
        )
        sys.exit(1)

    # ── Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Using device: %s", device)
    if device == "cpu":
        log.warning("No CUDA GPU detected — training will be extremely slow.")

    # ── Load model
    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model_id,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        load_in_4bit=args.load_4bit,
        device=device,
    )

    # ── Determine which phases to run
    phases = [1, 2, 3] if args.phase == "all" else [int(args.phase)]

    eval_summary: Dict[int, Dict[str, float]] = {}
    seed_offset_base = args.seed * 1000

    for phase in phases:
        log.info("\n" + "─" * 60)
        log.info("CURRICULUM PHASE %d", phase)
        log.info(_phase_description(phase))
        log.info("─" * 60)

        # Collect rollouts for this phase
        prompts, completions, rewards = build_phase_dataset(
            env_url=args.env_url,
            phase=phase,
            num_episodes=args.episodes_per_phase,
            tokenizer=tokenizer,
            model=model,
            seed_offset=seed_offset_base + phase * 10_000,
            temperature=args.temperature,
            device=device,
        )

        log.info(
            "Phase %d rollouts: %d steps | reward [min=%.3f mean=%.3f max=%.3f]",
            phase, len(rewards),
            min(rewards) if rewards else 0,
            sum(rewards) / max(len(rewards), 1),
            max(rewards) if rewards else 0,
        )

        # GRPO training step
        run_grpo_phase(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            completions=completions,
            rewards=rewards,
            output_dir=args.output_dir,
            phase=phase,
            learning_rate=args.learning_rate,
            num_generations=args.num_generations,
            per_device_batch=args.per_device_batch,
            gradient_accumulation=args.grad_accum,
        )

        # Post-phase evaluation
        log.info("Evaluating after Phase %d...", phase)
        eval_summary[phase] = evaluate_phase(
            env_url=args.env_url,
            phase=phase,
            tokenizer=tokenizer,
            model=model,
            eval_episodes=args.eval_episodes,
            seed_offset=seed_offset_base + 900_000,
            device=device,
        )

    # ── Save final model
    final_out = Path(args.output_dir) / "final"
    final_out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_out))
    tokenizer.save_pretrained(str(final_out))

    # Save eval summary
    summary_path = final_out / "eval_summary.json"
    summary_path.write_text(json.dumps(eval_summary, indent=2))

    log.info("\n" + "=" * 60)
    log.info("Training complete!")
    log.info("Final model → %s", final_out)
    log.info("Eval summary:")
    for ph, scores in eval_summary.items():
        log.info("  Phase %d: %s", ph, scores)
    log.info("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _phase_description(phase: int) -> str:
    descs = {
        1: "Phase 1 (Stabilize): access_decision only — confirm GRPO converges",
        2: "Phase 2 (Expand): + emergency_breakglass — adds incident verification",
        3: "Phase 3 (Full): all 5 tasks — stratified sampling favors harder tasks more",
    }
    return descs.get(phase, f"Phase {phase}")


def _print_curriculum_plan(args):
    """Print a human-readable curriculum plan (for --dry-run)."""
    phases = [1, 2, 3] if args.phase == "all" else [int(args.phase)]
    print("\nCurriculum Plan:")
    print("─" * 60)
    for phase in phases:
        weights = PHASE_WEIGHTS[phase]
        active = {tid: w for tid, w in weights.items() if w > 0}
        total_w = sum(active.values())
        print(f"\n  Phase {phase}: {_phase_description(phase)}")
        for tid, w in active.items():
            meta = TASK_REGISTRY[tid]
            n_eps = round(args.episodes_per_phase * w / total_w)
            print(
                f"    {tid:<32} weight={w:.0%}  ~{n_eps} episodes  "
                f"(max_steps={meta.max_steps}  difficulty={meta.difficulty})"
            )
    print()


if __name__ == "__main__":
    main()
