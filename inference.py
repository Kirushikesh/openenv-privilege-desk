"""
inference.py — Baseline agent for PrivilegeDesk.

Mandatory fields (set via environment variables):
    API_BASE_URL   LLM API endpoint (OpenAI-compatible)
    MODEL_NAME     Model identifier
    HF_TOKEN       HuggingFace / API key

Usage:
    python inference.py
    python inference.py --task access_decision
    python inference.py --all  # runs all 3 tasks
"""
import argparse
import json
import os
import sys
import textwrap
from typing import Any, Dict, List

from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "hf_placeholder"
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:8000")

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
        print(f"  [LLM error] {exc} — falling back to policy.list")
        return {"tool_name": "policy.list", "arguments": {}}


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(client: OpenAI, task_id: str) -> Dict[str, Any]:
    """Run one full episode for a task against the local environment."""
    import requests

    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print('='*60)

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
        print(f"  ERROR: Could not connect to environment at {ENV_URL}: {e}")
        return {"task_id": task_id, "error": str(e), "score": 0.0}

    observation = reset_data.get("observation", reset_data)
    print(f"  Goal: {observation.get('task_goal', '')[:100]}...")

    history: List[str] = []
    total_reward = 0.0
    done = False
    step = 0

    while not done and step < MAX_STEPS:
        step += 1

        # Get action from LLM
        action = call_llm(client, observation, history)
        tool = action.get("tool_name")
        args = action.get("arguments", {})
        print(f"  Step {step:2d}: {tool}({json.dumps(args)[:60]})")

        # Execute step
        try:
            step_resp = requests.post(
                f"{ENV_URL}/step",
                json={"action": action},
                timeout=30,
            )
            step_resp.raise_for_status()
            step_data  = step_resp.json()
        except Exception as e:
            print(f"    ERROR in /step: {e}")
            break

        observation  = step_data.get("observation", {})
        reward       = step_data.get("reward", 0.0) or 0.0
        done         = step_data.get("done", False)
        tool_result  = observation.get("tool_result", {})
        status       = (tool_result or {}).get("status", "?")
        obs_lines    = (tool_result or {}).get("observations", [])

        total_reward += reward
        obs_preview   = obs_lines[0][:80] if obs_lines else ""
        history.append(f"Step {step}: {tool} → {status}: {obs_preview}")

        print(f"          → status={status}, step_reward={reward:+.3f}, done={done}")
        if obs_preview:
            print(f"             {obs_preview}")

    # Get final grade
    try:
        grade_resp = requests.post(f"{ENV_URL}/grader", json={}, timeout=10)
        grade_data = grade_resp.json() if grade_resp.ok else {}
    except Exception:
        grade_data = {}

    episode_score = grade_data.get("score", 0.0)
    print(f"\n  ✓ Episode complete | steps={step} | step_reward={total_reward:.3f} | score={episode_score:.3f}")
    if grade_data.get("breakdown"):
        print(f"  Breakdown: {json.dumps(grade_data['breakdown'], indent=4)}")

    return {
        "task_id":       task_id,
        "steps":         step,
        "total_reward":  round(total_reward, 4),
        "episode_score": episode_score,
        "breakdown":     grade_data.get("breakdown", {}),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PrivilegeDesk baseline inference")
    parser.add_argument("--task",  choices=TASK_IDS, help="Run a specific task only")
    parser.add_argument("--all",   action="store_true", help="Run all 3 tasks (default)")
    parser.add_argument("--url",   default=ENV_URL, help="Environment base URL")
    args = parser.parse_args()

    global ENV_URL
    ENV_URL = args.url

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    tasks = [args.task] if args.task else TASK_IDS

    print(f"\nPrivilegeDesk Baseline Inference")
    print(f"  Model:  {MODEL_NAME}")
    print(f"  Env:    {ENV_URL}")
    print(f"  Tasks:  {tasks}")

    results = []
    for task_id in tasks:
        result = run_episode(client, task_id)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    for r in results:
        score = r.get("episode_score", 0.0)
        bar   = "█" * int(score * 20)
        print(f"  {r['task_id']:25s} score={score:.3f}  {bar}")

    avg = sum(r.get("episode_score", 0.0) for r in results) / len(results)
    print(f"\n  Average Score: {avg:.3f}")
    print('='*60)

    # Output JSON for automated evaluation
    output = {"results": results, "average_score": round(avg, 4)}
    print(json.dumps(output, indent=2))

    return 0 if avg > 0.0 else 1


if __name__ == "__main__":
    sys.exit(main())
