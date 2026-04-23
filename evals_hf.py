"""
evals_hf.py — HuggingFace model baseline evaluation across all 6 PrivilegeDesk tasks

Mirrors evals.py exactly, but replaces ChatOpenAI with a custom LangChain
SimpleChatModel wrapper that loads a HuggingFace model locally — the same way
train_grpo_direct.py does it.  Intended to capture the *pre-fine-tuning* baseline.

Usage:
  # Full eval, all 6 tasks, 3 seeds each:
  python evals_hf.py --model-id "Qwen/Qwen3-2B-Instruct"

  # 4-bit quantisation (low VRAM):
  python evals_hf.py --model-id "Qwen/Qwen3-2B-Instruct" --load-4bit

  # Specific tasks / seeds:
  python evals_hf.py --model-id "./outputs/grpo_run1/phase1/adapter" \
                     --seeds 1 --tasks access_decision emergency_breakglass

  # Custom output dir + verbose step logs:
  python evals_hf.py --model-id "Qwen/Qwen3-2B-Instruct" \
                     --output-dir ./outputs/hf_baseline --verbose

Dependencies:
  pip install torch transformers peft langchain-core requests
  # optional (4-bit quant): pip install bitsandbytes
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import requests
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

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
log = logging.getLogger("evals_hf")


# ═══════════════════════════════════════════════════════════════════════════════
# Task registry — mirrors TASK_REGISTRY + OPTIMAL_STEPS in train_grpo_direct.py
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
# System prompt — identical to train_grpo_direct.py / evals.py
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = textwrap.dedent("""
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
- Do NOT repeat a tool call you already made with the same arguments — check PREVIOUS TOOL CALLS before deciding
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
# HuggingFace chat model wrapper (LangChain SimpleChatModel)
# ═══════════════════════════════════════════════════════════════════════════════

class HFChatModel(SimpleChatModel):
    """
    Wraps a locally-loaded HuggingFace causal-LM as a LangChain chat model.

    Model + tokenizer are held as private (non-Pydantic) attributes so they
    aren't serialised.  Everything else (generation hyper-params, model_id
    string) is a normal Pydantic field.
    """

    model_id: str
    temperature: float = 0.3
    max_new_tokens: int = 512

    # Private — not part of Pydantic schema
    _model: Any = PrivateAttr(default=None)
    _tokenizer: Any = PrivateAttr(default=None)

    def load(self, load_4bit: bool = False) -> "HFChatModel":
        """Load model + tokenizer from model_id.  Call once after construction."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("Loading tokenizer from %s …", self.model_id)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._tokenizer = tokenizer

        log.info("Loading model from %s (4bit=%s) …", self.model_id, load_4bit)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16 if not load_4bit else None,
            device_map="auto",
            load_in_4bit=load_4bit,
            trust_remote_code=True,
        )
        model.eval()
        self._model = model
        log.info("Model loaded.  Device map: %s", self._model.hf_device_map if hasattr(self._model, "hf_device_map") else "auto")
        return self

    # ── LangChain interface ──────────────────────────────────────────────────

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> str:
        import torch

        # Convert LangChain messages → HF chat-template dicts
        conversation: List[Dict[str, str]] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                conversation.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                conversation.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                conversation.append({"role": "assistant", "content": msg.content})

        # Apply chat template — same try/except as train_grpo_direct.py
        try:
            prompt_text = self._tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt_text = self._tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(
            self._model.device
        )
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self._tokenizer.eos_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )
        if self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature)
        else:
            gen_kwargs["do_sample"] = False

        if stop:
            # Convert stop strings to stop token ids where possible
            stop_ids = []
            for s in stop:
                ids = self._tokenizer.encode(s, add_special_tokens=False)
                if ids:
                    stop_ids.extend(ids)
            if stop_ids:
                gen_kwargs["eos_token_id"] = stop_ids

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        # Decode only the newly generated tokens
        new_tokens = output_ids[0][input_len:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    @property
    def _llm_type(self) -> str:
        return "hf-chat-model"


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt builder — identical to train_grpo_direct.py / evals.py
# ═══════════════════════════════════════════════════════════════════════════════

def _build_user_message(observation: Dict[str, Any], history: List[str]) -> str:
    metadata   = observation.get("tool_metadata", {})
    tool_names = observation.get("available_tools", [])
    enriched_tools = [
        f"name: {name} | desc: {metadata[name]['desc']} | args: "
        + (", ".join(f"{k}: {v}" for k, v in metadata[name]["args"].items()) or "no args")
        if name in metadata else name
        for name in tool_names
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
    """Extract action from model output with graceful fallbacks. Never returns None.

    _format_score reflects output quality:
      1.0 — valid JSON + <think> tags
      0.7 — valid JSON, no <think>
      0.4 — partial JSON (tool_name via regex)
      0.1 — tool name found in raw text
      0.0 — completely unparseable
    """
    import re
    has_think = "<think>" in text and "</think>" in text

    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
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
            return {
                "tool_name": blob["tool_name"],
                "arguments": blob.get("arguments", {}),
                "_format_score": 0.4,
            }
        except json.JSONDecodeError:
            return {"tool_name": m.group(1), "arguments": {}, "_format_score": 0.4}

    m2 = re.search(r'"?tool_name"?\s*[=:]\s*"?(\w+\.\w+)"?', text)
    if m2 and m2.group(1) in available_tools:
        return {"tool_name": m2.group(1), "arguments": {}, "_format_score": 0.1}

    for t in available_tools:
        if t in text:
            return {"tool_name": t, "arguments": {}, "_format_score": 0.1}

    return {"tool_name": "__UNPARSEABLE__", "arguments": {}, "_format_score": 0.0}


# ═══════════════════════════════════════════════════════════════════════════════
# Episode runner — identical logic to evals.py
# ═══════════════════════════════════════════════════════════════════════════════

def run_episode(
    llm: HFChatModel,
    env_url: str,
    task_id: str,
    seed: int,
    difficulty_level: int,
    max_steps: int,
    debug_dir: Optional[Path] = None,
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

    if debug_dir is not None:
        try:
            state_resp = requests.get(f"{env_url}/full_state", timeout=10)
            if state_resp.ok:
                debug_dir.mkdir(parents=True, exist_ok=True)
                (debug_dir / f"{task_id}_seed{seed}.json").write_text(
                    json.dumps(state_resp.json(), indent=2)
                )
        except Exception as exc:
            log.warning("  Failed to save debug state: %s", exc)

    step_rewards:     List[float] = []
    history:          List[str]   = []
    episode_score:    float       = 0.0
    done:             bool        = False
    steps_taken:      int         = 0
    think_steps:      int         = 0
    llm_responses_md: List[str]   = []

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

        llm_responses_md.append(f"## Step {_step}\n{completion_text}\n")

        has_think = "<think>" in completion_text and "</think>" in completion_text
        if has_think:
            think_steps += 1
        steps_taken += 1

        action    = _parse_action(completion_text, obs.get("available_tools", []))
        fmt_score = action.pop("_format_score", 1.0)

        if action["tool_name"] == "__UNPARSEABLE__":
            log.debug("  step=%d  UNPARSEABLE — penalty -0.10", _step + 1)
            step_rewards.append(-0.10)
            history.append(f"Step {_step + 1}: [UNPARSEABLE — could not extract any tool call]")
            continue

        if fmt_score < 1.0:
            log.debug("  step=%d  partial parse (fmt=%.1f)  tool=%s", _step + 1, fmt_score, action.get("tool_name"))
        else:
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
        info        = step_data.get("info", {})
        step_reward = float(step_data.get("reward", 0.0) or 0.0)
        done        = step_data.get("done", False)

        if "tool_result" in info:
            obs["tool_result"] = info["tool_result"]

        fmt_penalty = (1.0 - fmt_score) * -0.10

        if done and info.get("episode_score") is not None:
            episode_score = float(info["episode_score"])
            step_rewards.append(episode_score + fmt_penalty)
        else:
            step_rewards.append(step_reward + fmt_penalty)

        tool_name = action.get("tool_name", "?")
        args_str  = json.dumps(action.get("arguments", {}))
        tool_res  = obs.get("tool_result") or {}
        output    = json.dumps(tool_res.get("result", {}))
        if len(output) > 2000:
            output = output[:2000] + "... (truncated)"

        entry = f"Step {_step + 1}: {tool_name} {args_str}\n  Output: {output}"
        if tool_res.get("status") == "error":
            entry += "\n  Status: error"
        history.append(entry)

    # Fallback grader call if env never set done=True
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

    if debug_dir is not None:
        try:
            md_path = debug_dir / f"{task_id}_seed{seed}_llm_responses.md"
            md_path.write_text("\n".join(llm_responses_md))
        except Exception as exc:
            log.warning("  Failed to save LLM responses: %s", exc)

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
# Reporting helpers — identical to evals.py
# ═══════════════════════════════════════════════════════════════════════════════

def _avg(vals: List[float]) -> float:
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def print_summary(results: List[Dict[str, Any]], optimal_steps: Dict[str, int]) -> None:
    from collections import defaultdict
    by_task: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        if not r.get("error"):
            by_task[r["task_id"]].append(r)

    col    = "{:<32} {:>8} {:>10} {:>8} {:>8} {:>6}"
    header = col.format("Task", "Score", "StepRew", "Steps", "H*", "Think%")
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for task in ALL_TASKS:
        tid    = task["task_id"]
        h_star = optimal_steps.get(tid, task["optimal_steps"])
        rows   = by_task.get(tid, [])
        if not rows:
            print(col.format(tid, "n/a", "n/a", "n/a", str(h_star), "n/a"))
            continue

        print(col.format(
            tid,
            f"{_avg([r['episode_score']    for r in rows]):.3f}",
            f"{_avg([r['mean_step_reward'] for r in rows]):.3f}",
            f"{_avg([r['steps_taken']      for r in rows]):.1f}",
            str(h_star),
            f"{_avg([r['format_rate']      for r in rows]) * 100:.0f}%",
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
    parser = argparse.ArgumentParser(
        description="HuggingFace model baseline eval on all 6 PrivilegeDesk tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-id",      default="Qwen/Qwen3-2B-Instruct",   help="HuggingFace model ID or local path")
    parser.add_argument("--load-4bit",     action="store_true",                 help="Load model in 4-bit (bitsandbytes)")
    parser.add_argument("--env-url",       default="http://localhost:8000",     help="PrivilegeDesk server URL")
    parser.add_argument("--seeds",         type=int, default=3,                 help="Number of random seeds per task")
    parser.add_argument("--seed-start",    type=int, default=1,                 help="Starting seed value")
    parser.add_argument("--tasks",         nargs="*", default=None,             help="Subset of task_ids to eval (default: all 6)")
    parser.add_argument("--output-dir",    default="./outputs/hf_evals",        help="Directory for CSV results")
    parser.add_argument("--temperature",   type=float, default=0.3,             help="Generation temperature (0 = greedy)")
    parser.add_argument("--max-new-tokens",type=int,   default=512,             help="Max tokens per model response")
    parser.add_argument("--verbose",       action="store_true",                 help="Show per-step debug logs")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("evals_hf").setLevel(logging.DEBUG)

    task_filter  = set(args.tasks) if args.tasks else None
    tasks_to_run = [t for t in ALL_TASKS if task_filter is None or t["task_id"] in task_filter]

    if not tasks_to_run:
        raise SystemExit(f"No matching tasks. Available: {[t['task_id'] for t in ALL_TASKS]}")

    log.info("Model      : %s (4bit=%s, temp=%.1f)", args.model_id, args.load_4bit, args.temperature)
    log.info("Env URL    : %s", args.env_url)
    log.info("Tasks      : %s", [t["task_id"] for t in tasks_to_run])
    log.info("Seeds      : %d (%d–%d)", args.seeds, args.seed_start, args.seed_start + args.seeds - 1)

    # Check server health before loading the (large) model
    try:
        health = requests.get(f"{args.env_url}/health", timeout=5)
        health.raise_for_status()
        log.info("Server     : healthy ✓")
    except Exception as exc:
        raise SystemExit(f"Server not reachable at {args.env_url}: {exc}")

    # Build and load the HF model wrapper
    llm = HFChatModel(
        model_id=args.model_id,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    ).load(load_4bit=args.load_4bit)

    results:    List[Dict[str, Any]] = []
    seeds       = list(range(args.seed_start, args.seed_start + args.seeds))
    total       = len(tasks_to_run) * len(seeds)
    done_count  = 0

    for task in tasks_to_run:
        task_id          = task["task_id"]
        max_steps        = task["max_steps"]
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
                debug_dir=Path(args.output_dir) / "debug",
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

    optimal_map = {t["task_id"]: t["optimal_steps"] for t in ALL_TASKS}
    print_summary(results, optimal_map)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = args.model_id.replace("/", "_").replace(".", "-")
    csv_path  = Path(args.output_dir) / f"hf_baseline_{model_slug}_{timestamp}.csv"
    save_csv(results, csv_path)


if __name__ == "__main__":
    main()
