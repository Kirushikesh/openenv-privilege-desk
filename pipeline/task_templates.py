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

    # ── Task 4: Emergency Break-Glass (Medium) ───────────────────────────────
    "emergency_breakglass": {
        "task_id": "emergency_breakglass",
        "difficulty": "medium",
        "task_goal": (
            "A production incident has been reported. An on-call engineer needs "
            "emergency elevated access via the break-glass procedure. Verify the "
            "incident is active and valid, check the engineer's current access, "
            "look up the break-glass policy for the affected resource, attach the "
            "incident ticket, flag the security team, set the correct TTL, and "
            "activate the emergency grant. If the incident is invalid or break-glass "
            "is not allowed for the resource, do NOT grant access."
        ),
        "max_steps": 10,
        "available_tools": [
            "incident.verify",
            "policy.lookup", "policy.list",
            "org.get_user",
            "entitlement.list",
            "ticket.attach",
            "access.grant", "access.set_ttl", "access.deny",
            "audit.flag",
        ],
        "required_entities": {
            "users": (5, 10),
            "resources": (3, 8),
            "policies": (3, 8),
            "pending_requests": (1, 1),
            "entitlements": (5, 12),
            "incidents": (1, 2),
        },
        "subgoals": [
            {"id": "sg_verify_incident",  "description": "Verify the incident is active and valid",            "reward": 0.10},
            {"id": "sg_check_engineer",   "description": "Check the on-call engineer's clearance",             "reward": 0.05},
            {"id": "sg_lookup_policy",    "description": "Look up the break-glass policy for the resource",    "reward": 0.10},
            {"id": "sg_attach_ticket",    "description": "Attach the incident ticket to the grant record",     "reward": 0.10},
            {"id": "sg_flag_security",    "description": "Flag the security team about the emergency override","reward": 0.15},
            {"id": "sg_set_ttl",          "description": "Set correct TTL (within break-glass limit)",         "reward": 0.10},
            {"id": "sg_activate_grant",   "description": "Activate the emergency access grant",                "reward": 0.40},
        ],
        "grading_weights": {
            "incident_valid":      0.15,
            "correct_role":        0.15,
            "correct_ttl":         0.20,
            "ticket_attached":     0.15,
            "security_flagged":    0.15,
            "correct_final_grant": 0.20,
        },
    },

    # ── Task 5: Separation of Duties Audit (Hard) ───────────────────────────
    "separation_of_duties_audit": {
        "task_id": "separation_of_duties_audit",
        "difficulty": "hard",
        "task_goal": (
            "Conduct a Separation of Duties (SoD) audit across the organization. "
            "Retrieve the SoD conflict matrix to understand which role combinations "
            "are forbidden. Check each user for SoD violations (conflicting entitlement "
            "pairs). For each potential violation, check whether an active compensating "
            "control exists before flagging it. Revoke the minimum set of entitlements "
            "needed to resolve unmitigated violations, while preserving access backed by "
            "active compensating controls. Submit your audit report when complete."
        ),
        "max_steps": 25,
        "available_tools": [
            "org.list_users", "org.get_user",
            "entitlement.list", "entitlement.inspect", "entitlement.revoke",
            "sod.get_conflict_matrix", "sod.check_user",
            "sod.get_compensating_controls", "sod.submit_report",
        ],
        "required_entities": {
            "users": (6, 15),
            "resources": (4, 10),
            "policies": (3, 8),
            "entitlements": (10, 25),
            "sod_conflicts": (3, 6),
            "sod_violations": (2, 6),
            "compensating_controls": (1, 3),
        },
        "subgoals": [
            {"id": "sg_get_conflicts",    "description": "Retrieve the SoD conflict matrix",                   "reward": 0.05},
            {"id": "sg_list_users",       "description": "List users to audit",                                "reward": 0.05},
            {"id": "sg_check_violations", "description": "Check users for SoD violations",                    "reward": 0.20},
            {"id": "sg_check_controls",   "description": "Check compensating controls on potential violations","reward": 0.10},
            {"id": "sg_revoke_correctly", "description": "Revoke entitlements for unmitigated violations",     "reward": 0.20},
            {"id": "sg_submit_report",    "description": "Submit the completed SoD audit report",             "reward": 0.40},
        ],
        "grading_weights": {
            "violations_found":        0.30,
            "false_positives":         0.15,
            "correct_revocations":     0.25,
            "compensating_recognized": 0.10,
            "report_submitted":        0.20,
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
