"""
Task-specific graders for PrivilegeDesk.

Each grader receives the full world_state (including hidden_state) and
returns a score breakdown dict with a 'score' key in [0.0, 1.0].

No clamping is applied here — raw weighted sums of [0, 1] sub-scores
naturally stay in [0.0, 1.0]. The root graders.py applies the Round 1
hackathon clamp (0.01, 0.99) at the judge boundary.
"""
from typing import Any, Dict


def _score(value: float) -> float:
    return round(value, 4)


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
        total = _score(sum(scores[k] * weights[k] for k in weights))
        return {"score": total, "breakdown": scores,
                "weights": weights, "details": details}

    req_id = decided_req["request_id"]
    agent_decision = decided_req.get("_agent_decision", {})
    correct = correct_decisions.get(req_id, {})

    if not correct:
        details["error"] = "No correct decision found for this request"
        total = _score(sum(scores[k] * weights[k] for k in weights))
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

    total = _score(sum(scores[k] * weights[k] for k in weights))
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
        return {"score": 0.0, "breakdown": scores,
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
    grant_denied = completion.get("grant_denied", False)
    should_approve = correct_d.get("should_approve", True)

    if grant_activated and should_approve:
        # Correctly activated an approved grant
        scores["correct_final_decision"] = 1.0
    elif grant_denied and not should_approve:
        # Correctly denied an invalid escalation
        scores["correct_final_decision"] = 1.0
    elif not grant_activated and not grant_denied and not should_approve:
        # Correct outcome (not granted) but via timeout instead of explicit deny
        scores["correct_final_decision"] = 0.5
    else:
        # Wrong decision (e.g. granted when should have denied, or vice versa)
        scores["correct_final_decision"] = 0.0

    details["final_decision"] = {
        "grant_activated": grant_activated,
        "grant_denied": grant_denied,
        "should_approve": should_approve
    }

    total = _score(sum(scores[k] * weights[k] for k in weights))
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

    total = _score(sum(scores[k] * weights[k] for k in weights))
    return {"score": total, "breakdown": scores, "weights": weights, "details": details}


# ── Task 4: Emergency Break-Glass ────────────────────────────────────────────

def grade_emergency_breakglass(world_state: Dict[str, Any]) -> Dict[str, Any]:
    """Grade the break-glass procedure against hidden ground truth."""
    hidden = world_state.get("hidden_state", {})
    correct_bg = hidden.get("correct_breakglass", {})
    completion = world_state.get("completion_state", {})
    requests = world_state.get("pending_requests", {})

    weights = {
        "incident_valid":      0.15,
        "correct_role":        0.15,
        "correct_ttl":         0.20,
        "ticket_attached":     0.15,
        "security_flagged":    0.15,
        "correct_final_grant": 0.20,
    }
    scores = {k: 0.0 for k in weights}
    details = {}

    # Find the breakglass request
    req = next(
        (r for r in requests.values() if r.get("_breakglass")),
        next(iter(requests.values()), None),
    )
    if not req:
        details["error"] = "No breakglass request found"
        return {"score": 0.0, "breakdown": scores,
                "weights": weights, "details": details}

    incident_is_valid = correct_bg.get("incident_is_valid", True)
    breakglass_allowed = correct_bg.get("breakglass_allowed", True)
    should_grant = incident_is_valid and breakglass_allowed

    # 1. Incident verified (agent called incident.verify)
    incident_verified = bool(completion.get("incident_verified"))
    scores["incident_valid"] = 1.0 if incident_verified else 0.0
    details["incident_valid"] = {"verified": incident_verified, "is_valid": incident_is_valid}

    # 2. Correct role — agent-set role (via access.grant role param) or request's role
    from pipeline.episode_generator import ROLE_RANK
    agent_role = req.get("_agent_role") or req.get("requested_role", "")
    correct_role = correct_bg.get("correct_role", "editor")
    if agent_role == correct_role:
        scores["correct_role"] = 1.0
    elif ROLE_RANK.get(agent_role, 99) < ROLE_RANK.get(correct_role, 0):
        scores["correct_role"] = 0.5  # under-privileged — partial
    details["role"] = {"agent": agent_role, "correct": correct_role}

    # 3. Correct TTL (exact = 1.0, within ±1h = 0.5)
    agent_ttl = req.get("_agent_ttl")
    correct_ttl = correct_bg.get("correct_ttl_hours", 2)
    if agent_ttl is not None:
        diff = abs(int(agent_ttl) - int(correct_ttl))
        if diff == 0:
            scores["correct_ttl"] = 1.0
        elif diff <= 1:
            scores["correct_ttl"] = 0.5
    details["ttl"] = {"agent": agent_ttl, "correct": correct_ttl}

    # 4. Ticket attached
    ticket_referenced = bool(completion.get("ticket_referenced"))
    scores["ticket_attached"] = 1.0 if ticket_referenced else 0.3
    details["ticket"] = {"referenced": ticket_referenced}

    # 5. Security team flagged
    security_flagged = bool(completion.get("security_flagged"))
    scores["security_flagged"] = 1.0 if security_flagged else 0.0
    details["security_flagged"] = security_flagged

    # 6. Correct final grant/deny decision
    grant_activated = completion.get("grant_activated", False)
    grant_denied = completion.get("grant_denied", False)

    if grant_activated and should_grant:
        scores["correct_final_grant"] = 1.0
    elif grant_denied and not should_grant:
        scores["correct_final_grant"] = 1.0
    elif not grant_activated and not grant_denied and not should_grant:
        scores["correct_final_grant"] = 0.5  # timed out without wrong grant
    else:
        scores["correct_final_grant"] = 0.0
    details["final_grant"] = {
        "grant_activated": grant_activated,
        "grant_denied": grant_denied,
        "should_grant": should_grant,
    }

    total = _score(sum(scores[k] * weights[k] for k in weights))
    return {"score": total, "breakdown": scores, "weights": weights, "details": details}


# ── Task 5: Separation of Duties Audit ───────────────────────────────────────

def grade_separation_of_duties_audit(world_state: Dict[str, Any]) -> Dict[str, Any]:
    """Grade the SoD audit: precision/recall on violations, revocations, report."""
    hidden = world_state.get("hidden_state", {})
    true_violations = hidden.get("sod_true_violations", [])
    all_violations = hidden.get("sod_all_violations", [])

    completion = world_state.get("completion_state", {})
    identified = completion.get("sod_violations_identified", [])  # [{user_id, conflict_id}]
    revoked_set = set(completion.get("entitlements_revoked", []))
    report_submitted = completion.get("sod_report_submitted", False)

    weights = {
        "violations_found":        0.30,
        "false_positives":         0.15,
        "correct_revocations":     0.25,
        "compensating_recognized": 0.10,
        "report_submitted":        0.20,
    }
    scores = {k: 0.0 for k in weights}
    details = {}

    true_viol_set = {(v["user_id"], v["conflict_id"]) for v in true_violations}
    all_viol_set = {(v["user_id"], v["conflict_id"]) for v in all_violations}
    identified_set = {(i["user_id"], i["conflict_id"]) for i in identified}
    controlled_viol_set = all_viol_set - true_viol_set

    # All entitlement IDs belonging to true violations (valid revocation targets)
    true_viol_ent_ids = {
        eid
        for v in true_violations
        for eid in (v.get("entitlement_id_a", ""), v.get("entitlement_id_b", ""))
        if eid
    }
    # All entitlement IDs belonging to controlled violations (should NOT be revoked)
    controlled_ent_ids = {
        eid
        for v in all_violations
        if (v["user_id"], v["conflict_id"]) in controlled_viol_set
        for eid in (v.get("entitlement_id_a", ""), v.get("entitlement_id_b", ""))
        if eid
    }

    # 1. violations_found: recall of true violations (did agent call sod.check_user on them?)
    if true_viol_set:
        found = identified_set & true_viol_set
        scores["violations_found"] = len(found) / len(true_viol_set)
        details["violations_found"] = {
            "true_total": len(true_viol_set),
            "found": len(found),
            "missed": len(true_viol_set - found),
        }
    else:
        scores["violations_found"] = 1.0
        details["violations_found"] = {"note": "no true violations in episode"}

    # 2. false_positives: penalize revoking entitlements that should NOT be revoked
    # (belong to controlled violations or are not part of any violation)
    wrongly_revoked = revoked_set & controlled_ent_ids
    non_violation_revoked = revoked_set - true_viol_ent_ids - controlled_ent_ids
    total_wrong = len(wrongly_revoked) + len(non_violation_revoked)
    if revoked_set:
        fp_rate = total_wrong / len(revoked_set)
        scores["false_positives"] = 1.0 - fp_rate
    else:
        scores["false_positives"] = 0.5  # agent revoked nothing — neutral
    details["false_positives"] = {
        "revoked_total": len(revoked_set),
        "wrongly_revoked": total_wrong,
    }

    # 3. correct_revocations: for each true violation, was at least one entitlement revoked?
    if true_violations:
        resolved = sum(
            1 for v in true_violations
            if v.get("entitlement_id_a") in revoked_set or v.get("entitlement_id_b") in revoked_set
        )
        scores["correct_revocations"] = resolved / len(true_violations)
        details["correct_revocations"] = {
            "true_violations": len(true_violations),
            "resolved": resolved,
        }
    else:
        scores["correct_revocations"] = 1.0
        details["correct_revocations"] = {"note": "no true violations to resolve"}

    # 4. compensating_recognized: agent checked controls for violations it found
    controls_checked = completion.get("sod_controls_checked", [])
    checked_set = {(c["user_id"], c["conflict_id"]) for c in controls_checked}
    if identified_set:
        checked_fraction = len(identified_set & checked_set) / len(identified_set)
        scores["compensating_recognized"] = checked_fraction
    else:
        scores["compensating_recognized"] = 0.5  # nothing found — neutral
    details["compensating_recognized"] = {
        "violations_identified": len(identified_set),
        "controls_checked": len(identified_set & checked_set),
    }

    # 5. report submitted
    scores["report_submitted"] = 1.0 if report_submitted else 0.0
    details["report_submitted"] = report_submitted

    total = _score(sum(scores[k] * weights[k] for k in weights))
    return {"score": total, "breakdown": scores, "weights": weights, "details": details}


# ── Dispatcher ────────────────────────────────────────────────────────────────

GRADERS = {
    "access_decision":           grade_access_decision,
    "jit_escalation":            grade_jit_escalation,
    "access_review":             grade_access_review,
    "emergency_breakglass":      grade_emergency_breakglass,
    "separation_of_duties_audit": grade_separation_of_duties_audit,
}


def grade(world_state: Dict[str, Any]) -> Dict[str, Any]:
    task_id = world_state.get("task_id", "access_decision")
    grader = GRADERS.get(task_id)
    if not grader:
        return {"score": 0.0, "error": f"No grader for task_id '{task_id}'"}
    return grader(world_state)
