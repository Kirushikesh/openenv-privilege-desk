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
"""
from typing import Any, Dict

from fastapi import HTTPException
from fastapi.responses import JSONResponse

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

from pipeline.task_templates import TASK_TEMPLATES

# ── Base app from OpenEnv ─────────────────────────────────────────────────────

app = create_app(
    PrivilegeDeskEnvironment,
    PrivilegeDeskAction,
    PrivilegeDeskObservation,
    env_name="privilege_desk",
    max_concurrent_envs=4,
)

# Shared env instance for HTTP endpoints (not WebSocket — those are isolated)
_http_env = PrivilegeDeskEnvironment()


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


# ── /grader ───────────────────────────────────────────────────────────────────

@app.post("/grader")
def grade_episode(body: Dict[str, Any] = None) -> Dict[str, Any]:
    """Return the episode score breakdown for the most recent completed episode.

    Optionally accepts { "task_id": "..." } to filter, but returns the
    last episode score by default.
    """
    score = _http_env.get_episode_score()
    if not score:
        raise HTTPException(
            status_code=409,
            detail="No completed episode found. Run /reset then /step until done=True.",
        )
    return {
        "task_id": score.get("task_id"),
        "score": score.get("score", 0.0),
        "breakdown": score.get("breakdown", {}),
        "weights": score.get("weights", {}),
        "details": score.get("details", {}),
    }


# ── /baseline ─────────────────────────────────────────────────────────────────

@app.post("/baseline")
def run_baseline() -> Dict[str, Any]:
    """Run a simple baseline agent on all 3 tasks and return average scores."""
    results = []

    for task_id in TASK_TEMPLATES:
        env = PrivilegeDeskEnvironment()
        obs = env.reset(seed=42, task_id=task_id)
        total_reward = 0.0
        done = False
        steps = 0

        # Naive baseline: just call the first available tool repeatedly until done
        available = obs.available_tools
        first_tool = available[0] if available else "policy.list"

        while not done and steps < 5:
            action = PrivilegeDeskAction(tool_name=first_tool, arguments={})
            obs = env.step(action)
            total_reward += (obs.reward or 0.0)
            done = obs.done
            steps += 1

        # Get final score
        score_info = env.get_episode_score()
        results.append({
            "task_id": task_id,
            "steps": steps,
            "total_step_reward": round(total_reward, 4),
            "episode_score": score_info.get("score", 0.0) if score_info else 0.0,
        })

    avg_score = sum(r["episode_score"] for r in results) / len(results) if results else 0.0

    return {
        "baseline_agent": "naive_first_tool",
        "results": results,
        "average_episode_score": round(avg_score, 4),
        "note": "This is a naive baseline. A well-prompted LLM agent should score significantly higher.",
    }


# ── Server entry point ────────────────────────────────────────────────────────

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
