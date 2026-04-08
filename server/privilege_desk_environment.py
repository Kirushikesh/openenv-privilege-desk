"""
PrivilegeDesk Environment — OpenEnv-compatible wrapper.

Exposes the WorldState over the standard OpenEnv Environment interface.
"""
from typing import Any, ClassVar, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import PrivilegeDeskAction, PrivilegeDeskObservation
except ImportError:
    from models import PrivilegeDeskAction, PrivilegeDeskObservation

from env.world_state import WorldState


class PrivilegeDeskEnvironment(Environment[PrivilegeDeskAction, PrivilegeDeskObservation, State]):
    """
    Zero-Standing-Privilege Ops Environment for training AI agents on IAM tasks.

    The agent is dropped into a synthetic enterprise and must handle:
      Task 1 (easy):   Access Decision — approve/deny a single access request
      Task 2 (medium): JIT Escalation — route through approval chains, set TTL
      Task 3 (hard):   Access Review — audit entitlements, revoke risky grants

    Each episode is procedurally generated from a seed, ensuring no two
    episodes are identical while remaining deterministically gradable.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # Class-level reference to the most recently active WorldState.
    # Updated on every reset() and step() so /grader can read the real
    # live state without framework hacks.
    _active_world: ClassVar[Optional["WorldState"]] = None

    def __init__(self):
        super().__init__()
        self._world = WorldState()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._last_episode_score: Optional[float] = None

    # ── OpenEnv API ───────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> PrivilegeDeskObservation:
        """Reset to a new episode.

        Kwargs:
            task_id (str):          "access_decision" | "jit_escalation" | "access_review"
            difficulty_level (int): 1-3 (scales entity counts)
        """
        task_id = kwargs.get("task_id", "access_decision")
        difficulty_level = kwargs.get("difficulty_level", 1)

        obs_dict = self._world.reset(
            seed=seed,
            task_id=task_id,
            difficulty_level=difficulty_level,
        )

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._last_episode_score = None

        # Update class-level reference so /grader can always find this world
        PrivilegeDeskEnvironment._active_world = self._world

        return self._build_observation(obs_dict, reward=0.0, done=False, tool_result=None)

    def step(
        self,
        action: PrivilegeDeskAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> PrivilegeDeskObservation:
        """Execute one tool call."""
        action_dict = {
            "tool_name": action.tool_name,
            "arguments": action.arguments,
        }

        obs_dict, step_reward, terminated, truncated, info = self._world.step(action_dict)
        self._state.step_count = self._world.step_count

        # Keep class reference current so /grader always sees the latest state
        PrivilegeDeskEnvironment._active_world = self._world

        done = terminated or truncated
        if done:
            self._last_episode_score = info.get("episode_score")

        return self._build_observation(
            obs_dict,
            reward=step_reward,
            done=done,
            tool_result=info.get("tool_result"),
            metadata={
                "step": info.get("step"),
                "terminated": terminated,
                "truncated": truncated,
                "step_reward": info.get("step_reward", 0.0),
                "episode_reward": info.get("episode_reward", 0.0),
                "episode_score": info.get("episode_score"),
            },
        )

    @property
    def state(self) -> State:
        return self._state

    def get_episode_score(self) -> Optional[float]:
        """Return the last episode's graded score."""
        if self._last_episode_score is not None:
            return self._last_episode_score
        # If episode is still running, compute current score
        score_dict = self._world.compute_episode_score()
        return float(score_dict.get("score", 0.10))

    def get_metadata(self):
        from openenv.core.env_server.interfaces import EnvironmentMetadata
        return EnvironmentMetadata(
            name="PrivilegeDesk",
            description=(
                "Zero-Standing-Privilege Ops environment where an AI agent handles "
                "enterprise access requests, JIT privilege escalation, approval routing, "
                "and periodic access reviews with audit-based revocation."
            ),
            version="1.0.0",
        )

    def close(self) -> None:
        pass

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_observation(
        self, obs_dict: dict, reward: float, done: bool,
        tool_result: Optional[dict], metadata: dict = None
    ) -> PrivilegeDeskObservation:
        return PrivilegeDeskObservation(
            # Core fields
            task_id=obs_dict.get("task_id", ""),
            task_goal=obs_dict.get("task_goal", ""),
            step=obs_dict.get("step", 0),
            max_steps=obs_dict.get("max_steps", 25),
            current_time=obs_dict.get("current_time", ""),
            available_tools=obs_dict.get("available_tools", []),
            # Org
            users=obs_dict.get("users", {}),
            org_graph=obs_dict.get("org_graph", {}),
            resources=obs_dict.get("resources", {}),
            policies=obs_dict.get("policies", {}),
            groups=obs_dict.get("groups", {}),
            # Access state
            entitlements=obs_dict.get("entitlements", {}),
            pending_requests=obs_dict.get("pending_requests", {}),
            approval_chains=obs_dict.get("approval_chains", {}),
            workflows=obs_dict.get("workflows", {}),
            # Objectives
            objectives=obs_dict.get("objectives", []),
            audit_log=obs_dict.get("audit_log", []),
            notifications=obs_dict.get("notifications", []),
            review_target_user_id=obs_dict.get("review_target_user_id"),
            # OpenEnv base fields
            done=done,
            reward=reward,
            tool_result=tool_result,
            metadata=metadata or {},
        )
