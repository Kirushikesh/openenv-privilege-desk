"""
Task-specific graders for PrivilegeDesk.

Each grader receives the full world_state (including hidden_state) and
returns a score breakdown dict with a 'score' key in (0.0, 1.0) — STRICTLY.

Phase 2 hackathon requirement: scores must be in open interval (0, 1).
All graders clamp final output to [0.01, 0.99].
"""
from typing import Any, Dict

# Score bounds: strictly between 0 and 1 (hackathon Phase 2 requirement)
_MIN = 0.10
_MAX = 0.90


def _clamp(score: float) -> float:
    """Ensure score is strictly in (0, 1) — never exactly 0.0 or 1.0."""
    return min(max(round(score, 4), 0.10), 0.90)


def _clamp_breakdown(scores: dict) -> dict:
    """Clamp every individual sub-score to strictly (0, 1)."""
    return {k: _clamp(v) for k, v in scores.items()}


# ── Task 1: Access Decision ───────────────────────────────────────────────────

def grade_access_decision(world_state: Dict[str, Any]) -> Dict[str, Any]:
    """Grade the agent's access decision against the hidden ground truth."""
    hidden = world_state.get("hidden_state", {})
    correct_decisions = hidden.get("correct_decisions", {})
    requests = world_state.get("pending_requests", {})

    weights = {
        "correct_decision":      0.40,
        "correct_role":          0.25,
        "correct_ttl":           0.20,
        "correct_justification": 0.15,
    }

    # Baseline scores: agent gets partial credit for attempting the task
    # even if it didn't submit a full decision — avoids a flat 0.0
    scores = {
        "correct_decision":      0.0,
        "correct_role":          0.0,
        "correct_ttl":           0.0,
        "correct_justification": 0.0,
    }
    details = {}

    # Find the request that was decided
    decided_req = next(
        (r for r in requests.values() if r.get("status") in ("approved", "denied")),
        None
    )

    if not decided_req:
        # No decision submitted — partial credit for viewing (policy_compliance baseline)
        details["error"] = "No decision was submitted"
        scores = _clamp_breakdown(scores)
        total = _clamp(sum(scores[k] * weights[k] for k in weights))
        return {"score": total, "breakdown": scores,
                "weights": weights, "details": details}

    req_id = decided_req["request_id"]
    agent_decision = decided_req.get("_agent_decision", {})
    correct = correct_decisions.get(req_id, {})

    if not correct:
        details["error"] = "No correct decision found for this request"
        scores = _clamp_breakdown(scores)
        total = _clamp(sum(scores[k] * weights[k] for k in weights))
        return {"score": total, "breakdown": scores,
                "weights": weights, "details": details}

    # 1. Correct approve/deny
    agent_choice = agent_decision.get("decision", "")
    correct_choice = "approve" if correct.get("should_approve") else "deny"
    if agent_choice == correct_choice:
        scores["correct_decision"] = 1.0
    details["decision"] = {"agent": agent_choice, "correct": correct_choice}

    # 2. Correct role (only matters if approving)
    if correct.get("should_approve") and agent_choice == "approve":
        agent_role = agent_decision.get("role", "")
        correct_role = correct.get("correct_role", "")
        from pipeline.episode_generator import ROLE_RANK
        if agent_role == correct_role:
            scores["correct_role"] = 1.0
        elif ROLE_RANK.get(agent_role, 99) < ROLE_RANK.get(correct_role, 0):
            scores["correct_role"] = 0.5   # lower than allowed — partial credit
        details["role"] = {"agent": agent_role, "correct": correct_role}

    # 3. Correct TTL (within ±2h of correct value)
    if agent_choice == "approve":
        agent_ttl = agent_decision.get("ttl_hours") or 0
        correct_ttl = correct.get("correct_ttl_hours", 0)
        if agent_ttl == correct_ttl:
            scores["correct_ttl"] = 1.0
        elif abs(agent_ttl - correct_ttl) <= 2:
            scores["correct_ttl"] = 0.5
        details["ttl"] = {"agent": agent_ttl, "correct": correct_ttl}

    # 4. Correct justification category
    agent_just = agent_decision.get("justification_category", "")
    correct_just = correct.get("correct_justification_category", "")
    if agent_just == correct_just:
        scores["correct_justification"] = 1.0
    details["justification"] = {"agent": agent_just, "correct": correct_just}

    scores = _clamp_breakdown(scores)
    total = _clamp(sum(scores[k] * weights[k] for k in weights))
    return {"score": total, "breakdown": scores, "weights": weights, "details": details}


# ── Task 2: JIT Escalation ────────────────────────────────────────────────────

def grade_jit_escalation(world_state: Dict[str, Any]) -> Dict[str, Any]:
    """Grade the JIT escalation workflow."""
    hidden = world_state.get("hidden_state", {})
    correct_chains = hidden.get("correct_approval_chains", {})
    requests = world_state.get("pending_requests", {})
    completion = world_state.get("completion_state", {})

    weights = {
        "correct_approvers":      0.20,
        "correct_routing_order":  0.15,
        "ticket_attached":        0.15,
        "correct_role":           0.15,
        "correct_ttl":            0.15,
        "correct_final_decision": 0.20,
    }

    scores = {k: 0.0 for k in weights}
    details = {}

    # Find the processed request (prefer decided ones, fall back to any)
    req = next(
        (r for r in requests.values() if r.get("status") in ("approved", "denied")),
        next(iter(requests.values()), None)
    )

    if not req:
        details["error"] = "No request found"
        scores = _clamp_breakdown(scores)
        return {"score": _clamp(0.0), "breakdown": scores,
                "weights": weights, "details": details}

    req_id = req["request_id"]
    correct_chain = correct_chains.get(req_id, [])
    routed = completion.get("approvals_routed", [])
    routed_ids = [r["approver_id"] for r in routed if r.get("request_id") == req_id]

    # 1. Correct approvers identified (set match)
    if set(routed_ids) >= set(correct_chain):
        scores["correct_approvers"] = 1.0
    elif set(routed_ids) & set(correct_chain):
        scores["correct_approvers"] = len(set(routed_ids) & set(correct_chain)) / len(set(correct_chain))
    details["approvers"] = {"agent_routed": routed_ids, "correct": correct_chain}

    # 2. Correct routing order
    if routed_ids[:len(correct_chain)] == correct_chain:
        scores["correct_routing_order"] = 1.0
    elif len(routed_ids) > 0 and correct_chain and routed_ids[0] == correct_chain[0]:
        scores["correct_routing_order"] = 0.5
    details["routing_order"] = {"agent": routed_ids, "correct": correct_chain}

    # 3. Ticket attached (non-empty ticket_id pre-populated in each request)
    # Gives 0.5 always (the ticket exists from generation) — partial baseline credit
    # Agent gets full 1.0 only if they explicitly referenced the ticket in their routing
    ticket_referenced = bool(completion.get("ticket_referenced"))
    ticket_exists = bool(req.get("ticket_id"))
    if ticket_referenced:
        scores["ticket_attached"] = 1.0
    elif ticket_exists:
        scores["ticket_attached"] = 0.5  # ticket exists but agent didn't reference it
    details["ticket"] = {"ticket_id": req.get("ticket_id"), "referenced": ticket_referenced}

    # 4. Correct role
    from pipeline.episode_generator import ROLE_RANK
    correct_decisions = hidden.get("correct_decisions", {})
    correct_d = correct_decisions.get(req_id, {})
    correct_role = correct_d.get("correct_role", "viewer")
    requested_role = req.get("requested_role", "")
    if ROLE_RANK.get(requested_role, 99) <= ROLE_RANK.get(correct_role, 0):
        scores["correct_role"] = 1.0
    details["role"] = {"requested": requested_role, "max_allowed": correct_role}

    # 5. Correct TTL
    agent_ttl = req.get("_agent_ttl")
    correct_ttl = correct_d.get("correct_ttl_hours", 4)
    if agent_ttl is not None:
        if agent_ttl == correct_ttl:
            scores["correct_ttl"] = 1.0
        elif abs(agent_ttl - correct_ttl) <= 2:
            scores["correct_ttl"] = 0.5
    details["ttl"] = {"agent": agent_ttl, "correct": correct_ttl}

    # 6. Final decision
    grant_activated = completion.get("grant_activated", False)
    should_approve = correct_d.get("should_approve", True)
    if grant_activated and should_approve:
        scores["correct_final_decision"] = 1.0
    elif not grant_activated and not should_approve:
        scores["correct_final_decision"] = 1.0
    details["final_decision"] = {
        "grant_activated": grant_activated, "should_approve": should_approve}

    scores = _clamp_breakdown(scores)
    total = _clamp(sum(scores[k] * weights[k] for k in weights))
    return {"score": total, "breakdown": scores, "weights": weights, "details": details}


# ── Task 3: Access Review ─────────────────────────────────────────────────────

def grade_access_review(world_state: Dict[str, Any]) -> Dict[str, Any]:
    """Grade the access review using precision/recall against ground truth."""
    hidden = world_state.get("hidden_state", {})
    risky_set = set(hidden.get("minimum_revocation_set", []))
    workflow_critical = set(hidden.get("workflow_critical_entitlements", []))

    completion = world_state.get("completion_state", {})
    revoked_set = set(completion.get("entitlements_revoked", []))
    review_submitted = completion.get("review_submitted", False)

    weights = {
        "precision":             0.30,
        "recall":                0.30,
        "workflow_preservation": 0.20,
        "policy_compliance":     0.10,
        "review_submitted":      0.10,
    }

    scores = {k: 0.0 for k in weights}
    details = {}

    # 1. Precision: of revoked entitlements, what fraction were actually risky?
    if revoked_set:
        true_positives = revoked_set & risky_set
        precision = len(true_positives) / len(revoked_set)
        scores["precision"] = round(precision, 4)
        details["precision"] = {
            "revoked": len(revoked_set),
            "correctly_revoked": len(true_positives),
            "false_positives": len(revoked_set - risky_set),
        }
    # else: precision stays 0.0 (agent revoked nothing — no penalty for un-attempted)

    # 2. Recall: of risky entitlements, what fraction did the agent catch?
    if risky_set:
        true_positives = revoked_set & risky_set
        recall = len(true_positives) / len(risky_set)
        scores["recall"] = round(recall, 4)
        details["recall"] = {
            "risky_total": len(risky_set),
            "caught": len(true_positives),
            "missed": len(risky_set - revoked_set),
        }

    # 3. Workflow preservation
    # When agent revokes nothing: broken=empty → preservation would naively be 1.0.
    # Fix: only award full preservation credit if the review was actually submitted.
    broken = revoked_set & workflow_critical
    if not broken and review_submitted:
        scores["workflow_preservation"] = 0.80  # submitted with no breakage — almost full
    elif not broken and not review_submitted:
        scores["workflow_preservation"] = 0.30  # did nothing — not much credit
    else:
        # Partial penalty per broken workflow entitlement
        scores["workflow_preservation"] = max(0.05, 1.0 - (len(broken) * 0.25))
    details["workflow_preservation"] = {
        "broken_entitlements": list(broken),
        "critical_count": len(workflow_critical),
    }

    # 4. Policy compliance
    entitlements = world_state.get("entitlements", {})
    policies = world_state.get("policies", {})
    from pipeline.episode_generator import ROLE_RANK
    violations = 0
    remaining = {eid: e for eid, e in entitlements.items()
                 if e.get("status") != "revoked" and eid not in revoked_set}
    checked = 0
    for e in remaining.values():
        matching_policy = next(
            (p for p in policies.values() if p["resource_id"] == e["resource_id"]),
            None
        )
        if matching_policy:
            checked += 1
            max_role = matching_policy.get("max_role", "owner")
            if ROLE_RANK.get(e["role"], 0) > ROLE_RANK.get(max_role, 3):
                violations += 1
    if checked > 0:
        # Cap at 0.90 so a clean baseline never hits exactly 1.0
        scores["policy_compliance"] = min(0.90, max(0.0, 1.0 - (violations / checked)))
    else:
        scores["policy_compliance"] = 0.50  # no data to check — neutral
    details["policy_compliance"] = {"violations": violations, "checked": checked}

    # 5. Review submitted
    if review_submitted:
        scores["review_submitted"] = 1.0
    details["review_submitted"] = review_submitted

    scores = _clamp_breakdown(scores)
    total = _clamp(sum(scores[k] * weights[k] for k in weights))
    return {"score": total, "breakdown": scores, "weights": weights, "details": details}


# ── Dispatcher ────────────────────────────────────────────────────────────────

GRADERS = {
    "access_decision": grade_access_decision,
    "jit_escalation":  grade_jit_escalation,
    "access_review":   grade_access_review,
}


def grade(world_state: Dict[str, Any]) -> Dict[str, Any]:
    task_id = world_state.get("task_id", "access_decision")
    grader = GRADERS.get(task_id)
    if not grader:
        return {"score": _clamp(0.0), "error": f"No grader for task_id '{task_id}'"}
    return grader(world_state)
