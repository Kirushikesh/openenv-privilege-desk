"""
WorldState — the central stateful object for one PrivilegeDesk episode.

Wraps the raw world state dict and provides the Gymnasium-like API:
  - reset(seed, task_id)       → initial visible observation
  - step(action_dict)          → (observation, reward, terminated, truncated, info)
  - visible_state()            → agent-facing partial view
  - full_state()               → complete internal state (for grading)
"""
import copy
import random
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from .action_router import ActionRouter
from .tools import get_available_tools


class WorldState:
    """Interactive world state for one PrivilegeDesk episode."""

    MAX_STEPS = 25
    CONSECUTIVE_ERROR_LIMIT = 10

    def __init__(self, max_steps: int = None):
        self._raw: Dict[str, Any] = {}
        self._router: Optional[ActionRouter] = None
        self._step_count: int = 0
        self._terminated: bool = False
        self._truncated: bool = False
        self._episode_reward: float = 0.0
        self._reward_agg = None

        if max_steps is not None:
            self.MAX_STEPS = max_steps

    def _ensure_reward_aggregator(self):
        if self._reward_agg is None:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from reward.aggregator import RewardAggregator
            self._reward_agg = RewardAggregator()
        return self._reward_agg

    def _ensure_generator(self):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from pipeline.episode_generator import EpisodeGenerator
        return EpisodeGenerator()

    # ── Gymnasium-like API ────────────────────────────────────────────────────

    def reset(
        self,
        seed: int = None,
        task_id: str = "access_decision",
        difficulty_level: int = 1,
        world_state: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Reset to a new episode."""
        if world_state is not None:
            self._raw = copy.deepcopy(world_state)
        else:
            gen = self._ensure_generator()
            self._raw = gen.generate(
                task_id=task_id,
                difficulty_level=difficulty_level,
                seed=seed or random.randint(0, 999_999),
            )

        self.MAX_STEPS = self._raw.get("max_steps", self.MAX_STEPS)
        self._router = ActionRouter(self._raw)

        agg = self._ensure_reward_aggregator()
        agg.reset(task_id=task_id)

        self._step_count = 0
        self._terminated = False
        self._truncated = False
        self._episode_reward = 0.0

        return self.visible_state()

    def step(
        self, action_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one agent action."""
        if self._router is None:
            # reset() was never called — return a clear error instead of crashing
            return (
                {"task_id": "", "task_goal": "", "step": 0, "max_steps": 0,
                 "available_tools": [], "notifications": [
                     {"level": "error", "message": "Call /reset before /step"}
                 ]},
                0.0, True, True,
                {"error": "Environment not initialised. Call /reset first."},
            )
        if self._terminated or self._truncated:
            return (self.visible_state(), 0.0, self._terminated, self._truncated,
                    {"error": "Episode already ended. Call reset()."})

        self._step_count += 1

        # Dispatch
        if self._router is None:
            if self._raw:
                from .action_router import ActionRouter
                self._router = ActionRouter(self._raw)
            else:
                return (self.visible_state(), 0.0, False, False,
                        {"error": "Environment not initialized. Call reset() first."})

        tool_result = self._router.dispatch(action_dict)

        # Per-step reward
        step_reward = self._compute_step_reward(action_dict, tool_result)
        self._episode_reward += step_reward

        # Termination checks
        self._check_termination(tool_result)

        observation = self.visible_state()

        info = {
            "step": self._step_count,
            "tool_result": tool_result,
            "step_reward": step_reward,
            "episode_reward": self._episode_reward,
        }

        if self._terminated or self._truncated:
            score_dict = self.compute_episode_score()
            info["episode_score"] = float(score_dict.get("score", 0.10))

        return observation, step_reward, self._terminated, self._truncated, info

    def visible_state(self) -> Dict[str, Any]:
        """Return agent-facing partial view (no hidden_state)."""
        raw = self._raw
        obs = {
            "task_id": raw.get("task_id", ""),
            "task_goal": raw.get("task_goal", ""),
            "step": self._step_count,
            "max_steps": self.MAX_STEPS,
            "current_time": raw.get("current_time", ""),
            "available_tools": raw.get("available_tools", []),
            # Org
            "users": {uid: {k: v for k, v in u.items()
                           if not k.startswith("_") and k != "status"}
                      for uid, u in raw.get("users", {}).items()},
            "org_graph": raw.get("org_graph", {}),
            "resources": raw.get("resources", {}),
            "policies": raw.get("policies", {}),
            "groups": raw.get("groups", {}),
            # Entitlements (strip hidden _is_risky fields)
            "entitlements": {eid: {k: v for k, v in e.items() if not k.startswith("_")}
                             for eid, e in raw.get("entitlements", {}).items()
                             if e.get("status") != "revoked"},
            "pending_requests": raw.get("pending_requests", {}),
            "approval_chains": raw.get("approval_chains", {}),
            "workflows": {wid: {k: v for k, v in wf.items() if not k.startswith("_")}
                          for wid, wf in raw.get("workflows", {}).items()},
            "incidents": raw.get("incidents", {}),
            "conflict_matrix": raw.get("conflict_matrix", {}),
            "compensating_controls": raw.get("compensating_controls", {}),
            # Last action results
            "audit_log": raw.get("audit_log", [])[-5:],   # last 5 actions
            "notifications": [],
        }

        # For access review: show which user to review
        if raw.get("review_target_user_id"):
            obs["review_target_user_id"] = raw["review_target_user_id"]

        # Subgoal descriptions (not their status — agent must earn them)
        obs["objectives"] = [
            {"id": sg["id"], "description": sg["description"]}
            for sg in raw.get("subgoals", [])
        ]

        return obs

    def full_state(self) -> Dict[str, Any]:
        """Complete internal state including hidden_state (for grading)."""
        return copy.deepcopy(self._raw)

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_step_reward(self, action_dict: Dict, tool_result: Dict) -> float:
        agg = self._ensure_reward_aggregator()
        return agg.step_reward(
            step=self._step_count,
            action=action_dict,
            tool_result=tool_result,
            world_state=self._raw,
        )

    def compute_episode_score(self) -> Dict[str, Any]:
        """Compute full grading breakdown at episode end."""
        agg = self._ensure_reward_aggregator()
        return agg.episode_score(self._raw)

    # ── Termination ───────────────────────────────────────────────────────────

    def _check_termination(self, tool_result: Dict):
        # Tool explicitly signalled termination
        if self._raw.get("_terminated"):
            self._terminated = True
            return

        # Max steps
        if self._step_count >= self.MAX_STEPS:
            self._truncated = True
            return

        # Too many consecutive errors
        audit = self._raw.get("audit_log", [])
        if len(audit) >= self.CONSECUTIVE_ERROR_LIMIT:
            last_n = [e.get("status") for e in audit[-self.CONSECUTIVE_ERROR_LIMIT:]]
            if all(s == "error" for s in last_n):
                self._truncated = True

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def done(self) -> bool:
        return self._terminated or self._truncated

    @property
    def episode_reward(self) -> float:
        return self._episode_reward
