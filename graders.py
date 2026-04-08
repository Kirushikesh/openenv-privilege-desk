"""
graders.py — Root-level grader entry points for the PrivilegeDesk environment.

This file exists at the repo root so the OpenEnv judge can import grader
functions directly via the `grader:` field in openenv.yaml WITHOUT needing
the package to be pip-installed first.

Each public function accepts a trajectory/world_state dict and returns a
plain float strictly in (0.01, 0.99) as required by the hackathon judge.
"""
import os
import sys

# Make the package importable from the repo root (Docker / openenv validate)
_root = os.path.dirname(__file__)
if _root not in sys.path:
    sys.path.insert(0, _root)

try:
    from reward.grader import (
        grade_access_decision as _grade_access,
        grade_jit_escalation  as _grade_jit,
        grade_access_review   as _grade_review,
        _clamp,
    )
except ImportError:
    # Fallback for running directly from repo root
    sys.path.insert(0, os.path.join(_root, "reward"))
    from grader import (  # type: ignore[no-redef]
        grade_access_decision as _grade_access,
        grade_jit_escalation  as _grade_jit,
        grade_access_review   as _grade_review,
        _clamp,
    )


def _float_score(grade_fn, world_state: dict) -> float:
    """Call a grader function and extract a clamped float score."""
    try:
        result = grade_fn(world_state or {})
        if isinstance(result, dict):
            return _clamp(float(result.get("score", 0.10)))
        return _clamp(float(result))
    except Exception:
        return 0.10  # safe floor on any error


def access_decision_grader(trajectory: dict = None) -> float:
    """Grader for task: access_decision. Returns float in (0.01, 0.99)."""
    trajectory = trajectory or {}
    # Extract world_state if it exists in trajectory, otherwise use trajectory itself
    world_state = trajectory.get("world_state", trajectory)
    return _float_score(_grade_access, world_state)


def jit_escalation_grader(trajectory: dict = None) -> float:
    """Grader for task: jit_escalation. Returns float in (0.01, 0.99)."""
    trajectory = trajectory or {}
    # Extract world_state if it exists in trajectory, otherwise use trajectory itself
    world_state = trajectory.get("world_state", trajectory)
    return _float_score(_grade_jit, world_state)


def access_review_grader(trajectory: dict = None) -> float:
    """Grader for task: access_review. Returns float in (0.01, 0.99)."""
    trajectory = trajectory or {}
    # Extract world_state if it exists in trajectory, otherwise use trajectory itself
    world_state = trajectory.get("world_state", trajectory)
    return _float_score(_grade_review, world_state)


__all__ = [
    "access_decision_grader",
    "jit_escalation_grader",
    "access_review_grader",
]
