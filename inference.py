"""
inference.py — PrivilegeDesk OpenEnv Agent

Runs an LLM agent through all 3 privilege management tasks and emits structured stdout logs.

Required environment variables:
    API_BASE_URL   LLM API endpoint (OpenAI-compatible)
    MODEL_NAME     Model identifier
    HF_TOKEN       HuggingFace / API key

Stdout format (must not deviate):
    [START] task=<task> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""
import argparse
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN     = os.getenv("HF_TOKEN")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:8000")

BENCHMARK    = "privilege_desk"
MAX_STEPS    = 20
TEMPERATURE  = 0.0

TASK_IDS = ["access_decision", "jit_escalation", "access_review"]

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an enterprise IAM (Identity & Access Management) specialist agent.
You operate inside the PrivilegeDesk environment, which simulates a corporate
zero-standing-privilege access control system.

Your job is to use the available tools to complete the assigned task.
You must respond with EXACTLY ONE JSON object per turn, with this structure:

{
  "tool_name": "<tool_name>",
  "arguments": { "<key>": "<value>", ... }
}

Rules:
- Only call tools listed in available_tools
- Do not add explanations outside the JSON
- For access.decide: use "approve" or "deny" for the decision field
- For entitlement.revoke: provide entitlement_id
- For review.submit: call it when you have finished revoking all risky entitlements
- When done, the environment will signal done=true

Available context is in the observation JSON you receive.
""").strip()


# ── Logging helpers (judge-parsed format) ────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    """[START] line — emitted exactly once at episode begin."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    """[STEP] line — emitted immediately after each env.step() returns."""
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """[END] line — always emitted (even on exception) via finally block."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM call ─────────────────────────────────────────────────────────────────

def call_llm(client: OpenAI, observation: Dict[str, Any], history: List[str]) -> Dict[str, Any]:
    """Ask the LLM for the next action given the current observation."""

    # Build compact observation summary for the prompt
    obs_summary = {
        "task_id":          observation.get("task_id"),
        "task_goal":        observation.get("task_goal"),
        "step":             observation.get("step"),
        "max_steps":        observation.get("max_steps"),
        "available_tools":  observation.get("available_tools", []),
        "pending_requests": observation.get("pending_requests", {}),
        "policies":         observation.get("policies", {}),
        "entitlements":     {eid: {k: v for k, v in e.items()
                                   if k in ("role", "user_id", "resource_id",
                                            "days_since_use", "expires_at")}
                             for eid, e in list(observation.get("entitlements", {}).items())[:10]},
        "last_tool_result": observation.get("tool_result"),
        "objectives":       observation.get("objectives", []),
        "review_target_user_id": observation.get("review_target_user_id"),
    }

    user_msg = (
        f"Current observation:\n{json.dumps(obs_summary, indent=2)}\n\n"
        + (f"History:\n" + "\n".join(history[-4:]) + "\n\n" if history else "")
        + "What is your next tool call? Respond with JSON only."
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        text = completion.choices[0].message.content or "{}"
        action = json.loads(text)
        if "tool_name" not in action:
            raise ValueError("No tool_name in response")
        return action
    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", file=sys.stderr, flush=True)
        return {"tool_name": "policy.list", "arguments": {}}


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(client: OpenAI, task_id: str) -> None:
    """Run one full episode for a task, emitting [START]/[STEP]/[END] logs."""
    import requests

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_error: Optional[str] = None

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset
        try:
            resp = requests.post(
                f"{ENV_URL}/reset",
                json={"task_id": task_id, "seed": 42},
                timeout=30,
            )
            resp.raise_for_status()
            reset_data = resp.json()
        except Exception as e:
            last_error = str(e)
            log_end(success=False, steps=0, score=0.0, rewards=[])
            return

        observation = reset_data.get("observation", reset_data)
        done = False

        # Episode loop
        while not done and steps_taken < MAX_STEPS:
            steps_taken += 1

            # Get action from LLM
            action = call_llm(client, observation, history)
            tool = action.get("tool_name", "unknown")
            args = action.get("arguments", {})
            action_str = f"{tool}({json.dumps(args)[:40]})"

            # Execute step
            try:
                step_resp = requests.post(
                    f"{ENV_URL}/step",
                    json={"action": action},
                    timeout=30,
                )
                step_resp.raise_for_status()
                step_data = step_resp.json()
                last_error = None
            except Exception as e:
                last_error = str(e)
                reward = 0.0
                done = True
                rewards.append(reward)
                log_step(step=steps_taken, action=action_str, reward=reward, done=done, error=last_error)
                break

            observation = step_data.get("observation", {})
            reward = step_data.get("reward", 0.0) or 0.0
            done = step_data.get("done", False)
            tool_result = observation.get("tool_result", {})
            status = (tool_result or {}).get("status", "?")
            obs_lines = (tool_result or {}).get("observations", [])

            rewards.append(reward)
            obs_preview = obs_lines[0][:60] if obs_lines else ""
            history.append(f"Step {steps_taken}: {tool} → {status}: {obs_preview}")

            log_step(step=steps_taken, action=action_str, reward=reward, done=done, error=last_error)

        # Get final grade from environment's grader endpoint
        try:
            grade_resp = requests.post(f"{ENV_URL}/grader", json={}, timeout=10)
            grade_data = grade_resp.json() if grade_resp.ok else {}
            score = grade_data.get("score", 0.0)
        except Exception:
            # Fallback: compute mean of step rewards
            score = sum(rewards) / len(rewards) if rewards else 0.0

        # Clamp score strictly to (0.01, 0.99) as required by judge
        score = max(0.01, min(score, 0.99))
        success = score > 0.333  # above random baseline

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", file=sys.stderr, flush=True)
        last_error = str(exc)

    finally:
        # [END] MUST always be emitted
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PrivilegeDesk baseline inference")
    parser.add_argument("--task", choices=TASK_IDS, help="Run a specific task only")
    parser.add_argument("--all", action="store_true", help="Run all 3 tasks (default)")
    parser.add_argument("--url", default=ENV_URL, help="Environment base URL")
    args = parser.parse_args()

    global ENV_URL
    ENV_URL = args.url

    # Check for TASK_NAME environment variable (judge may set this)
    target_task = os.getenv("TASK_NAME")
    if target_task:
        # Map task names to task_ids
        if "access_decision" in target_task or "easy" in target_task:
            args.task = "access_decision"
        elif "jit_escalation" in target_task or "medium" in target_task:
            args.task = "jit_escalation"
        elif "access_review" in target_task or "hard" in target_task:
            args.task = "access_review"

    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN environment variable not set", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    tasks = [args.task] if args.task else TASK_IDS

    for task_id in tasks:
        run_episode(client, task_id)


if __name__ == "__main__":
    main()