"""
Reward Aggregator — computes per-step and episode-level rewards.
"""
from typing import Any, Dict

from .grader import grade


# Per-tool step reward signals
STEP_REWARD_MAP = {
    "request.view":         +0.03,
    "request.list":         +0.02,
    "policy.lookup":        +0.05,
    "policy.list":          +0.02,
    "org.get_user":         +0.02,
    "org.get_manager":      +0.04,
    "org.list_users":       +0.01,
    "entitlement.list":     +0.03,
    "entitlement.inspect":  +0.04,
    "entitlement.revoke":   +0.00,  # graded at episode end
    "audit.query":          +0.04,
    "group.resolve":        +0.04,
    "workflow.check_active":+0.04,
    "approval.route":       +0.00,  # partial reward via state check
    "approval.check_status":+0.01,
    "access.decide":        +0.00,  # graded at episode end
    "access.grant":         +0.00,  # graded at episode end
    "access.set_ttl":       +0.02,
    "review.submit":        +0.00,  # graded at episode end
}

ERROR_PENALTY = -0.02
REDUNDANT_PENALTY = -0.01
WRONG_APPROVER_PENALTY = -0.03


class RewardAggregator:
    """Combines per-step incremental rewards with episode-level grading."""

    def __init__(self):
        self._task_id = "access_decision"

    def reset(self, task_id: str = "access_decision"):
        self._task_id = task_id

    def step_reward(
        self,
        step: int,
        action: Dict[str, Any],
        tool_result: Dict[str, Any],
        world_state: Dict[str, Any],
    ) -> float:
        tool_name = action.get("tool_name", "")
        status = tool_result.get("status", "error")

        # Error penalty
        if status == "error":
            return ERROR_PENALTY

        # Redundancy penalty — same tool with same args as previous step
        audit = world_state.get("audit_log", [])
        if len(audit) >= 2:
            prev = audit[-2]
            if (prev.get("tool_name") == tool_name and
                    prev.get("arguments") == action.get("arguments", {})):
                return REDUNDANT_PENALTY

        # Base reward from map
        base = STEP_REWARD_MAP.get(tool_name, 0.0)

        # Bonus for correct approval routing
        if tool_name == "approval.route":
            routed = world_state.get("completion_state", {}).get("approvals_routed", [])
            if routed and routed[-1].get("correct"):
                base = +0.05
            else:
                base = WRONG_APPROVER_PENALTY

        # Bonus for identifying a risky entitlement correctly
        if tool_name == "entitlement.inspect":
            eid = action.get("arguments", {}).get("entitlement_id", "")
            risky_set = set(world_state.get("hidden_state", {}).get("risky_entitlement_ids", []))
            if eid in risky_set:
                base += 0.05   # found a genuinely risky entitlement

        # Bonus for correct revocation
        if tool_name == "entitlement.revoke":
            eid = action.get("arguments", {}).get("entitlement_id", "")
            risky_set = set(world_state.get("hidden_state", {}).get("risky_entitlement_ids", []))
            critical_set = set(world_state.get("hidden_state", {}).get("workflow_critical_entitlements", []))
            if eid in risky_set:
                base = +0.08   # correct revocation
            elif eid in critical_set:
                base = -0.10  # revoked something critical!
            else:
                base = -0.03  # unnecessary revocation

        return round(base, 4)

    def episode_score(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Compute the final episode grade (0.0–1.0) and return full breakdown."""
        result = grade(world_state)
        return {
            "task_id": world_state.get("task_id"),
            "score": result.get("score", 0.0),
            "breakdown": result.get("breakdown", {}),
            "weights": result.get("weights", {}),
            "details": result.get("details", {}),
        }
