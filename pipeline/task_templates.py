"""
Task templates — defines the three task families for PrivilegeDesk.

Each task template specifies:
  - task_id, difficulty, natural-language goal
  - max_steps, required entity counts
  - subgoals with per-subgoal partial rewards
"""
from typing import Dict, Any

TASK_TEMPLATES: Dict[str, Dict[str, Any]] = {

    # ── Task 1: Access Decision (Easy) ───────────────────────────────────────
    "access_decision": {
        "task_id": "access_decision",
        "difficulty": "easy",
        "task_goal": (
            "A new access request has arrived. Review the pending request, "
            "inspect the requester's current entitlements and the applicable "
            "policy, then make a decision: approve or deny the request. "
            "If approving, select the correct role and set an appropriate TTL. "
            "Use the access.decide tool to submit your decision."
        ),
        "max_steps": 5,
        "available_tools": [
            "policy.lookup", "policy.list",
            "org.get_user", "entitlement.list",
            "request.view", "access.decide",
        ],
        "required_entities": {
            "users": (5, 8),
            "resources": (3, 5),
            "policies": (3, 6),
            "pending_requests": (1, 1),
            "entitlements": (5, 12),
        },
        "subgoals": [
            {"id": "sg_view_request",    "description": "View the pending access request",              "reward": 0.05},
            {"id": "sg_lookup_policy",   "description": "Look up the applicable policy",                "reward": 0.10},
            {"id": "sg_check_existing",  "description": "Check requester's existing entitlements",      "reward": 0.05},
            {"id": "sg_decide",          "description": "Submit the access decision (approve/deny)",    "reward": 0.80},
        ],
        "grading_weights": {
            "correct_decision":  0.40,   # approve vs deny
            "correct_role":      0.25,   # right role level selected
            "correct_ttl":       0.20,   # TTL within policy bounds
            "correct_justification": 0.15, # right reason category
        },
    },

    # ── Task 2: JIT Escalation (Medium) ──────────────────────────────────────
    "jit_escalation": {
        "task_id": "jit_escalation",
        "difficulty": "medium",
        "task_goal": (
            "An engineer needs urgent temporary elevated access to a production "
            "system. Process this JIT (just-in-time) escalation request: "
            "look up the applicable policy, identify all required approvers in "
            "the correct order, route the request through the approval chain, "
            "attach the incident ticket as justification, set the correct TTL, "
            "and activate the temporary grant once approved. "
            "If any approval is denied or policy violated, deny the escalation."
        ),
        "max_steps": 15,
        "available_tools": [
            "policy.lookup", "policy.list",
            "org.get_user", "org.get_manager", "org.list_users",
            "request.view", "request.list",
            "approval.route", "approval.check_status",
            "ticket.attach",
            "access.grant", "access.deny", "access.set_ttl",
            "entitlement.list",
        ],
        "required_entities": {
            "users": (6, 10),
            "resources": (4, 8),
            "policies": (4, 8),
            "pending_requests": (1, 2),
            "entitlements": (8, 18),
            "approval_chains": (2, 4),
        },
        "subgoals": [
            {"id": "sg_view_request",       "description": "View the escalation request",                     "reward": 0.03},
            {"id": "sg_lookup_policy",      "description": "Look up the applicable escalation policy",        "reward": 0.07},
            {"id": "sg_find_approvers",     "description": "Identify all required approvers",                 "reward": 0.15},
            {"id": "sg_route_approvals",    "description": "Route to approvers in the correct order",         "reward": 0.15},
            {"id": "sg_attach_ticket",      "description": "Attach incident/ticket justification",            "reward": 0.10},
            {"id": "sg_set_ttl",            "description": "Set the correct TTL on the escalation",           "reward": 0.10},
            {"id": "sg_activate_grant",     "description": "Activate the grant (or deny if conditions fail)", "reward": 0.40},
        ],
        "grading_weights": {
            "correct_approvers":     0.20,
            "correct_routing_order": 0.15,
            "ticket_attached":       0.15,
            "correct_role":          0.15,
            "correct_ttl":           0.15,
            "correct_final_decision":0.20,
        },
    },

    # ── Task 3: Access Review (Hard) ─────────────────────────────────────────
    "access_review": {
        "task_id": "access_review",
        "difficulty": "hard",
        "task_goal": (
            "Conduct an access review for user {review_target_user_id}. Inspect all of "
            "their current entitlements (including those inherited through "
            "groups), cross-reference with the audit log to identify unused or "
            "stale access, flag over-privileged grants, and revoke the minimum "
            "set of risky entitlements — while preserving access that is needed "
            "for active workflows. Submit your review when complete."
        ),
        "max_steps": 25,
        "available_tools": [
            "policy.lookup", "policy.list",
            "org.get_user", "org.list_users",
            "entitlement.list", "entitlement.inspect", "entitlement.revoke",
            "audit.query",
            "group.resolve",
            "workflow.check_active",
            "review.submit",
        ],
        "required_entities": {
            "users": (8, 15),
            "resources": (6, 12),
            "policies": (5, 10),
            "entitlements": (12, 25),
            "groups": (2, 5),
            "workflows": (2, 4),
            "audit_entries": (10, 20),
        },
        "subgoals": [
            {"id": "sg_list_entitlements",  "description": "List the user's entitlements",                "reward": 0.05},
            {"id": "sg_resolve_groups",     "description": "Resolve group-inherited entitlements",        "reward": 0.10},
            {"id": "sg_query_audit",        "description": "Query audit log for usage data",               "reward": 0.10},
            {"id": "sg_identify_risky",     "description": "Correctly identify risky entitlements",       "reward": 0.15},
            {"id": "sg_check_workflows",    "description": "Verify active workflows before revoking",     "reward": 0.10},
            {"id": "sg_revoke_correctly",   "description": "Revoke risky entitlements without breakage",  "reward": 0.20},
            {"id": "sg_submit_review",      "description": "Submit the completed access review",          "reward": 0.30},
        ],
        "grading_weights": {
            "precision":              0.30,  # revoked entitlements that were actually risky
            "recall":                 0.30,  # risky entitlements that were caught
            "workflow_preservation":  0.20,  # no active workflows broken
            "policy_compliance":      0.10,  # remaining access is policy-compliant
            "review_submitted":       0.10,  # did agent submit?
        },
    },
}


def get_task(task_id: str) -> Dict[str, Any]:
    if task_id not in TASK_TEMPLATES:
        raise ValueError(f"Unknown task_id '{task_id}'. Available: {list(TASK_TEMPLATES)}")
    return TASK_TEMPLATES[task_id]


def list_tasks():
    return list(TASK_TEMPLATES.keys())
