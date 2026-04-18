"""
Reward Aggregator — computes per-step and episode-level rewards.

Step reward range: [-0.50, +0.40]
  Negative: errors, redundant calls, wrong routing, revoking critical entitlements
  Positive: exploration, correct investigation, correct decisions
"""
from typing import Any, Dict

from .grader import grade


# Per-tool base step rewards — exploration and investigation signal
STEP_REWARD_MAP = {
    # Low-value listing/enumeration — cheap but necessary
    "request.list":            0.05,
    "policy.list":             0.05,
    "org.list_users":          0.05,
    "approval.check_status":   0.05,

    # Medium-value retrieval — meaningful investigation
    "request.view":            0.10,
    "org.get_user":            0.10,
    "org.get_manager":         0.10,
    "entitlement.list":        0.10,
    "audit.query":             0.10,
    "group.resolve":           0.10,
    "workflow.check_active":   0.10,
    "sod.get_conflict_matrix": 0.10,
    "audit.flag":              0.10,

    # High-value investigation — correct analysis path
    "policy.lookup":                    0.15,
    "entitlement.inspect":              0.15,
    "sod.check_user":                   0.15,
    "sod.get_compensating_controls":    0.15,
    "incident.verify":                  0.15,

    # Action tools — base rate, overridden by context
    "approval.route":   0.05,   # overridden: +0.30 correct, -0.20 wrong
    "ticket.attach":    0.15,
    "entitlement.revoke": 0.05, # overridden: +0.35 correct, -0.40 critical

    # Access decision / grant tools
    "access.decide":  0.20,
    "access.grant":   0.20,
    "access.set_ttl": 0.10,

    # Terminal submission tools
    "review.submit":     0.20,
    "sod.submit_report": 0.20,
}

_ERROR_PENALTY   = -0.20   # tool call returned error
_REDUNDANT_PENALTY = -0.10 # repeated identical call
_WRONG_APPROVER  = -0.20   # routed to wrong approver
_CORRECT_APPROVER = 0.30   # routed to correct approver
_CORRECT_REVOKE  = 0.35    # revoked a genuinely risky entitlement
_CRITICAL_REVOKE = -0.40   # revoked a workflow-critical entitlement (mistake)
_WRONG_REVOKE    = -0.10   # revoked an entitlement that wasn't risky at all
_RISKY_INSPECT_BONUS = 0.10  # inspecting a risky entitlement (discovery signal)


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

        if status == "error":
            return _ERROR_PENALTY

        # Redundancy check — same tool + same args as previous step
        audit = world_state.get("audit_log", [])
        if len(audit) >= 2:
            prev = audit[-2]
            if (prev.get("tool_name") == tool_name and
                    prev.get("arguments") == action.get("arguments", {})):
                return _REDUNDANT_PENALTY

        base = STEP_REWARD_MAP.get(tool_name, 0.05)

        # Approval routing: strong signal in both directions
        if tool_name == "approval.route":
            routed = world_state.get("completion_state", {}).get("approvals_routed", [])
            if routed and routed[-1].get("correct"):
                base = _CORRECT_APPROVER
            else:
                base = _WRONG_APPROVER

        # Entitlement inspection: bonus for discovering a genuinely risky entitlement
        if tool_name == "entitlement.inspect":
            eid = action.get("arguments", {}).get("entitlement_id", "")
            risky_set = set(world_state.get("hidden_state", {}).get("risky_entitlement_ids", []))
            if eid in risky_set:
                base += _RISKY_INSPECT_BONUS

        # Revocation: large positive for correct, large negative for mistakes
        if tool_name == "entitlement.revoke":
            eid = action.get("arguments", {}).get("entitlement_id", "")
            risky_set = set(world_state.get("hidden_state", {}).get("risky_entitlement_ids", []))
            critical_set = set(world_state.get("hidden_state", {}).get("workflow_critical_entitlements", []))
            if eid in risky_set:
                base = _CORRECT_REVOKE
            elif eid in critical_set:
                base = _CRITICAL_REVOKE
            else:
                base = _WRONG_REVOKE

        return round(base, 4)

    def episode_score(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Compute the final episode grade (0.0–1.0) and return full breakdown."""
        result = grade(world_state)
        return {
            "task_id": world_state.get("task_id"),
            "score": result.get("score", 0.1),
            "breakdown": result.get("breakdown", {}),
            "weights": result.get("weights", {}),
            "details": result.get("details", {}),
        }
