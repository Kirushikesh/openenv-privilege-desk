"""
evals.py — GPT-4o baseline evaluation across all 6 PrivilegeDesk tasks

Uses LangChain ChatOpenAI so the same prompt infrastructure (SYSTEM_PROMPT,
_build_user_message, _parse_action) mirrors train_grpo.py exactly — this gives
a clean "before fine-tuning" baseline to contrast against trained-model curves.

Metrics collected per episode:
  episode_score     — /grader score (0.0–1.0), primary signal
  mean_step_reward  — mean per-step reward from /step
  format_rate       — fraction of steps where model used <think>...</think>
  steps_taken       — actual steps vs H* optimal

Usage:
  export OPENAI_API_KEY=sk-...
  python evals.py --env-url http://localhost:8000

  # Fewer seeds, specific tasks:
  python evals.py --seeds 1 --tasks access_decision multi_agent_oversight

  # Save results to custom path:
  python evals.py --output-dir ./outputs/evals

Dependencies (add to project or install separately):
  pip install langchain-openai langchain-core
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evals")


# ═══════════════════════════════════════════════════════════════════════════════
# Task registry — mirrors TASK_REGISTRY + OPTIMAL_STEPS in train_grpo.py
# ═══════════════════════════════════════════════════════════════════════════════

ALL_TASKS: List[Dict[str, Any]] = [
    {"task_id": "access_decision",            "max_steps": 5,  "optimal_steps": 4,  "difficulty_level": 2},
    {"task_id": "emergency_breakglass",        "max_steps": 10, "optimal_steps": 7,  "difficulty_level": 2},
    {"task_id": "jit_escalation",             "max_steps": 15, "optimal_steps": 10, "difficulty_level": 2},
    {"task_id": "access_review",              "max_steps": 25, "optimal_steps": 15, "difficulty_level": 1},
    {"task_id": "separation_of_duties_audit", "max_steps": 25, "optimal_steps": 15, "difficulty_level": 1},
    {"task_id": "multi_agent_oversight",      "max_steps": 25, "optimal_steps": 12, "difficulty_level": 2},
]

TASK_MAP: Dict[str, Dict[str, Any]] = {t["task_id"]: t for t in ALL_TASKS}


# ═══════════════════════════════════════════════════════════════════════════════
# System prompt — identical to train_grpo.py
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
# Prompt builder — identical to train_grpo.py
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

    # Show sub-agent identities (names/purpose only) — not identity_graph.
    # Agent must call sod.check_user / sod.get_conflict_matrix to find toxic paths.
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


# ═══════════════════════════════════════════════════════════════════════════════
# Episode runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_episode(
    llm: "ChatOpenAI",
    env_url: str,
    task_id: str,
    seed: int,
    difficulty_level: int,
    max_steps: int,
) -> Dict[str, Any]:
    """Run one full episode and return metrics."""
    try:
        resp = requests.post(
            f"{env_url}/reset",
            json={"task_id": task_id, "seed": seed, "difficulty_level": difficulty_level},
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as exc:
        log.error("Reset failed (task=%s seed=%d): %s", task_id, seed, exc)
        return {"task_id": task_id, "seed": seed, "error": str(exc), "episode_score": 0.0}

    obs = resp.json().get("observation", resp.json())

    step_rewards:  List[float] = []
    history:       List[str]   = []
    episode_score: float       = 0.0
    done:          bool        = False
    steps_taken:   int         = 0
    think_steps:   int         = 0

    for _step in range(max_steps):
        if done:
            break

        user_msg = _build_user_message(obs, history)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]

        try:
            response = llm.invoke(messages)
            completion_text = response.content
        except Exception as exc:
            log.warning("LLM call failed at step %d: %s", _step, exc)
            step_rewards.append(-0.20)
            break

        if "<think>" in completion_text and "</think>" in completion_text:
            think_steps += 1
        steps_taken += 1

        action = _parse_action(completion_text, obs.get("available_tools", []))
        log.debug("  step=%d  tool=%s  args=%s", _step + 1, action.get("tool_name"), action.get("arguments"))

        try:
            step_resp = requests.post(f"{env_url}/step", json={"action": action}, timeout=30)
            step_resp.raise_for_status()
            step_data = step_resp.json()
        except Exception as exc:
            log.warning("Step failed at step %d: %s", _step, exc)
            step_rewards.append(-0.20)
            break

        obs         = step_data.get("observation", {})
        step_reward = float(step_data.get("reward", 0.0) or 0.0)
        done        = step_data.get("done", False)
        metadata    = step_data.get("metadata", {})

        if done and metadata.get("episode_score") is not None:
            episode_score = float(metadata["episode_score"])
            step_rewards.append(episode_score)
        else:
            step_rewards.append(step_reward)

        tool_name = action.get("tool_name", "?")
        args_str  = json.dumps(action.get("arguments", {}))
        tool_res  = obs.get("tool_result") or {}
        output    = json.dumps(tool_res.get("result", {}))
        if len(output) > 300:
            output = output[:300] + "... (truncated)"

        entry = f"Step {_step + 1}: {tool_name} {args_str}\n  Output: {output}"
        if tool_res.get("status") == "error":
            entry += "\n  Status: error"
        history.append(entry)

    # Fallback: call /grader if env never set done=True
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
    format_rate      = think_steps / max(steps_taken, 1)

    return {
        "task_id":          task_id,
        "seed":             seed,
        "episode_score":    round(episode_score, 4),
        "mean_step_reward": round(mean_step_reward, 4),
        "steps_taken":      steps_taken,
        "format_rate":      round(format_rate, 4),
        "think_steps":      think_steps,
        "error":            None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Reporting helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _avg(vals: List[float]) -> float:
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def print_summary(results: List[Dict[str, Any]], optimal_steps: Dict[str, int]) -> None:
    """Print a per-task summary table to stdout."""
    from collections import defaultdict

    by_task: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        if not r.get("error"):
            by_task[r["task_id"]].append(r)

    col = "{:<32} {:>8} {:>10} {:>8} {:>8} {:>6}"
    header = col.format("Task", "Score", "StepRew", "Steps", "H*", "Think%")
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for task in ALL_TASKS:
        tid   = task["task_id"]
        h_star = optimal_steps.get(tid, task["optimal_steps"])
        rows  = by_task.get(tid, [])
        if not rows:
            print(col.format(tid, "n/a", "n/a", "n/a", str(h_star), "n/a"))
            continue

        avg_score  = _avg([r["episode_score"]    for r in rows])
        avg_step   = _avg([r["mean_step_reward"] for r in rows])
        avg_steps  = _avg([r["steps_taken"]      for r in rows])
        avg_think  = _avg([r["format_rate"]      for r in rows])

        print(col.format(
            tid,
            f"{avg_score:.3f}",
            f"{avg_step:.3f}",
            f"{avg_steps:.1f}",
            str(h_star),
            f"{avg_think*100:.0f}%",
        ))

    print("=" * len(header) + "\n")


def save_csv(results: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id", "seed", "episode_score", "mean_step_reward",
        "steps_taken", "format_rate", "think_steps", "error",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    log.info("Results saved → %s", path)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:

    parser = argparse.ArgumentParser(description="GPT-4o baseline eval on all 6 PrivilegeDesk tasks")
    parser.add_argument("--env-url",    default="http://localhost:8000",         help="PrivilegeDesk server URL")
    parser.add_argument("--model",      default="gpt-4o",                        help="OpenAI model name")
    parser.add_argument("--seeds",      type=int, default=3,                     help="Number of random seeds per task")
    parser.add_argument("--seed-start", type=int, default=1,                     help="Starting seed value")
    parser.add_argument("--tasks",      nargs="*", default=None,                 help="Subset of task_ids to eval (default: all 6)")
    parser.add_argument("--output-dir", default="./outputs/evals",               help="Directory for CSV results")
    parser.add_argument("--temperature", type=float, default=0.0,                help="LLM temperature (0.0 = deterministic)")
    parser.add_argument("--verbose",    action="store_true",                     help="Show per-step debug logs")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("evals").setLevel(logging.DEBUG)

    # Filter tasks
    task_filter = set(args.tasks) if args.tasks else None
    tasks_to_run = [t for t in ALL_TASKS if task_filter is None or t["task_id"] in task_filter]

    if not tasks_to_run:
        raise SystemExit(f"No matching tasks found. Available: {[t['task_id'] for t in ALL_TASKS]}")

    log.info("Model     : %s (temperature=%.1f)", args.model, args.temperature)
    log.info("Env URL   : %s", args.env_url)
    log.info("Tasks     : %s", [t["task_id"] for t in tasks_to_run])
    log.info("Seeds     : %d (%d–%d)", args.seeds, args.seed_start, args.seed_start + args.seeds - 1)

    # Check server health
    try:
        health = requests.get(f"{args.env_url}/health", timeout=5)
        health.raise_for_status()
        log.info("Server    : healthy ✓")
    except Exception as exc:
        raise SystemExit(f"Server not reachable at {args.env_url}: {exc}")

    llm = init_chat_model(args.model).with_retry(stop_after_attempt=6)

    results: List[Dict[str, Any]] = []
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    total = len(tasks_to_run) * len(seeds)
    done_count = 0

    for task in tasks_to_run:
        task_id         = task["task_id"]
        max_steps       = task["max_steps"]
        difficulty_level = task["difficulty_level"]

        for seed in seeds:
            done_count += 1
            log.info("[%d/%d] task=%-32s seed=%d", done_count, total, task_id, seed)

            result = run_episode(
                llm=llm,
                env_url=args.env_url,
                task_id=task_id,
                seed=seed,
                difficulty_level=difficulty_level,
                max_steps=max_steps,
            )
            results.append(result)

            if result.get("error"):
                log.warning("  ERROR: %s", result["error"])
            else:
                log.info(
                    "  score=%.3f  step_rew=%.3f  steps=%d/%d  think=%.0f%%",
                    result["episode_score"],
                    result["mean_step_reward"],
                    result["steps_taken"],
                    max_steps,
                    result["format_rate"] * 100,
                )

    # Summary table
    optimal_map = {t["task_id"]: t["optimal_steps"] for t in ALL_TASKS}
    print_summary(results, optimal_map)

    # Save CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = Path(args.output_dir) / f"gpt4o_baseline_{timestamp}.csv"
    save_csv(results, csv_path)


if __name__ == "__main__":
    main()
