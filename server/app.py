"""
FastAPI application for the PrivilegeDesk Environment.

Endpoints (OpenEnv standard):
    POST /reset     - start a new episode
    POST /step      — execute a tool call
    GET  /state     — current episode state
    GET  /schema    — action/observation JSON schemas
    WS   /ws        — WebSocket for persistent sessions

Additional endpoints (hackathon required):
    GET  /tasks     — list the 3 tasks with action schemas
    POST /grader    — return episode score breakdown (0.0–1.0)
    POST /baseline  — run baseline agent and return scores

NOTE on HTTP statefulness
--------------------------
OpenEnv's create_app() HTTP /reset and /step handlers are intentionally
stateless — each request creates a new throwaway env instance. State is
only preserved over WebSocket.

We remove those stateless routes and replace them with our own singleton-
backed versions so that curl-style HTTP testing works correctly. WebSocket
(/ws) is untouched and still creates proper per-session environments.
"""
from typing import Any, Dict, Optional

from fastapi import HTTPException
from pydantic import BaseModel
from starlette.routing import Route

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Install with: uv sync") from e

try:
    from ..models import PrivilegeDeskAction, PrivilegeDeskObservation
    from .privilege_desk_environment import PrivilegeDeskEnvironment
except (ImportError, ModuleNotFoundError):
    from models import PrivilegeDeskAction, PrivilegeDeskObservation
    from server.privilege_desk_environment import PrivilegeDeskEnvironment

from env.world_state import WorldState
from reward.grader import grade
from pipeline.task_templates import TASK_TEMPLATES


# ── Module-level singleton — shared across /reset, /step, /grader ─────────────
# Mimics what OpenEnv's WebSocket session manager does per-connection, but
# over plain HTTP for easy curl / script testing.

_world: WorldState = WorldState()


# ── Build base app from OpenEnv (registers /ws, /schema, /health, /state) ────

app = create_app(
    PrivilegeDeskEnvironment,
    PrivilegeDeskAction,
    PrivilegeDeskObservation,
    env_name="privilege_desk",
    max_concurrent_envs=1,
)

# Remove the stateless /reset and /step routes that create_app registered.
# FastAPI uses first-match routing, so if we don't remove them our overrides
# (registered below) would never be reached.
_OVERRIDE_PATHS = {"/reset", "/step"}
app.routes[:] = [
    r for r in app.routes
    if not (
        isinstance(r, Route)
        and r.path in _OVERRIDE_PATHS
        and "POST" in (r.methods or set())
    )
]


# ── Pydantic request models ───────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "access_decision"
    seed: Optional[int] = None
    difficulty_level: int = 1


class StepRequest(BaseModel):
    action: Dict[str, Any]  # {"tool_name": "...", "arguments": {...}}


# ── /reset ────────────────────────────────────────────────────────────────────

@app.post("/reset")
def reset_episode(body: ResetRequest = None) -> Dict[str, Any]:
    """Reset to a new episode and return the initial observation.

    The world state is stored in a module-level singleton so that subsequent
    /step and /grader calls operate on the same episode.
    """
    global _world
    req = body or ResetRequest()

    _world = WorldState()
    obs = _world.reset(
        seed=req.seed,
        task_id=req.task_id,
        difficulty_level=req.difficulty_level,
    )

    return {
        "observation": obs,
        "reward": 0.0,
        "done": False,
        "info": {
            "task_id": req.task_id,
            "seed": req.seed,
            "episode": "started",
        },
    }


# ── /step ─────────────────────────────────────────────────────────────────────

@app.post("/step")
def step_episode(body: StepRequest) -> Dict[str, Any]:
    """Execute one tool call on the current episode.

    body.action must be a dict with:
        tool_name (str)   — e.g. "request.view", "access.decide"
        arguments (dict)  — tool-specific kwargs (can be empty {})
    """
    global _world

    if _world._router is None:
        raise HTTPException(
            status_code=409,
            detail="No active episode. Call POST /reset first.",
        )

    obs, reward, terminated, truncated, info = _world.step(body.action)
    done = terminated or truncated

    # Clamp step reward strictly to (0.01, 0.99) — Phase 2 requirement
    clamped_reward = min(max(round(reward, 4), 0.10), 0.90)

    return {
        "observation": obs,
        "reward": clamped_reward,
        "done": done,
        "info": info,
    }


# ── /grader ───────────────────────────────────────────────────────────────────

@app.post("/grader")
def grade_episode(body: Dict[str, Any] = None) -> Dict[str, Any]:
    """Return the grading breakdown for the current live episode.

    Reads the world state already mutated by /reset + /step calls and
    scores it — no re-execution, no fake episode.
    """
    global _world

    if _world._router is None or not _world._raw:
        raise HTTPException(
            status_code=409,
            detail="No active episode. Call POST /reset first, then POST /step.",
        )

    score = grade(_world._raw)
    # Ensure score is strictly in (0, 1) — never exactly 0.0 or 1.0
    raw_score = score.get("score", 0.10)
    final_score = max(0.10, min(0.90, raw_score))
    return {
        "task_id": _world._raw.get("task_id", "unknown"),
        "score": round(final_score, 4),
        "breakdown": score.get("breakdown", {}),
        "weights": score.get("weights", {}),
        "details": score.get("details", {}),
        "steps_taken": _world.step_count,
        "episode_done": _world.done,
    }


# ── /tasks ────────────────────────────────────────────────────────────────────

@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    """List all available tasks with their schemas."""
    tasks = []
    for task_id, template in TASK_TEMPLATES.items():
        tasks.append({
            "task_id": task_id,
            "difficulty": template["difficulty"],
            "task_goal": template["task_goal"],
            "max_steps": template["max_steps"],
            "available_tools": template["available_tools"],
            "grading_weights": template["grading_weights"],
            "action_schema": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "enum": template["available_tools"],
                        "description": "Tool to call",
                    },
                    "arguments": {
                        "type": "object",
                        "description": "Tool-specific arguments",
                    },
                },
                "required": ["tool_name"],
            },
        })
    return {"tasks": tasks, "count": len(tasks)}


# ── /baseline ─────────────────────────────────────────────────────────────────

@app.post("/baseline")
def run_baseline() -> Dict[str, Any]:
    """Run a naive baseline agent on all 3 tasks and return scores."""
    results = []

    for task_id in TASK_TEMPLATES:
        ws = WorldState()
        ws.reset(seed=42, task_id=task_id)
        total_reward = 0.0
        steps = 0

        if task_id == "access_decision":
            _, r, _, _, _ = ws.step({
                "tool_name": "access.decide",
                "arguments": {"decision": "approve", "role": "viewer",
                              "ttl_hours": 4, "justification_category": "operational"},
            })
            total_reward += r
            steps = 1
        elif task_id == "access_review":
            _, r, _, _, _ = ws.step({
                "tool_name": "review.submit",
                "arguments": {"summary": "baseline review"},
            })
            total_reward += r
            steps = 1
        elif task_id == "jit_escalation":
            req_id = next(iter(ws._raw.get("pending_requests", {})), None)
            if req_id:
                _, r, _, _, _ = ws.step({
                    "tool_name": "access.grant",
                    "arguments": {"request_id": req_id},
                })
                total_reward += r
                steps = 1

        score_info = grade(ws._raw)
        # Ensure score is strictly in (0, 1) — never exactly 0.0 or 1.0
        raw_score = score_info.get("score", 0.10)
        episode_score = max(0.10, min(0.90, raw_score))
        results.append({
            "task_id": task_id,
            "steps": steps,
            "total_step_reward": round(total_reward, 4),
            "episode_score": round(episode_score, 4),
        })

    avg_score = sum(r["episode_score"] for r in results) / len(results) if results else 0.0

    return {
        "baseline_agent": "naive_terminal_tool",
        "results": results,
        "average_episode_score": round(avg_score, 4),
        "note": "Naive baseline — hits the terminal tool immediately with default args.",
    }


# ── Server entry point ────────────────────────────────────────────────────────

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
