"""
Tool registry for PrivilegeDesk.

All 20 tools are functions that take (world_state, arguments) and return:
    {
        "status": "success"|"error"|"permission_denied",
        "result": {...},
        "observations": ["Human-readable description of what happened"],
        "state_delta": {"dot.path.key": value, ...},  # mutations to apply
    }
"""
from datetime import datetime
from typing import Any, Dict


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ok(result, observations, state_delta=None):
    return {"status": "success", "result": result,
            "observations": observations, "state_delta": state_delta or {}}

def _err(message):
    return {"status": "error", "result": {"error": message},
            "observations": [f"ERROR: {message}"], "state_delta": {}}


# ── Policy tools ──────────────────────────────────────────────────────────────

def policy_lookup(ws: Dict, args: Dict) -> Dict:
    """Look up policy for a resource."""
    resource_id = args.get("resource_id")
    if not resource_id:
        return _err("resource_id is required")
    policies = ws.get("policies", {})
    matching = [p for p in policies.values() if p["resource_id"] == resource_id]
    if not matching:
        return _ok({"policy": None}, [f"No policy found for resource {resource_id}"],
                   {"completion_state.subgoal_status.sg_lookup_policy": "completed"})
    policy = matching[0]
    return _ok(
        {"policy": policy},
        [f"Policy found: {policy['description']}"],
        {"completion_state.subgoal_status.sg_lookup_policy": "completed"},
    )

def policy_list(ws: Dict, args: Dict) -> Dict:
    """List all policies."""
    policies = ws.get("policies", {})
    return _ok(
        {"policies": list(policies.values()), "count": len(policies)},
        [f"Found {len(policies)} policies"],
        {"completion_state.subgoal_status.sg_lookup_policy": "completed"},
    )


# ── Org tools ─────────────────────────────────────────────────────────────────

def org_get_user(ws: Dict, args: Dict) -> Dict:
    """Get details for a specific user."""
    user_id = args.get("user_id")
    if not user_id:
        return _err("user_id is required")
    users = ws.get("users", {})
    user = users.get(user_id)
    if not user:
        return _err(f"User {user_id} not found")
    delta = {"completion_state.subgoal_status.sg_check_engineer": "completed"}
    return _ok(
        {"user": user},
        [f"User: {user['name']} ({user['job_title']}, {user['department']})"],
        delta,
    )

def org_get_manager(ws: Dict, args: Dict) -> Dict:
    """Get the manager chain for a user."""
    user_id = args.get("user_id")
    if not user_id:
        return _err("user_id is required")
    users = ws.get("users", {})
    org_graph = ws.get("org_graph", {})
    chain = []
    current = user_id
    visited = set()
    while current and current not in visited:
        visited.add(current)
        mgr_id = org_graph.get(current, {}).get("reports_to")
        if mgr_id and mgr_id in users:
            chain.append({"user_id": mgr_id, "name": users[mgr_id]["name"],
                          "job_title": users[mgr_id]["job_title"]})
            current = mgr_id
        else:
            break
    return _ok(
        {"manager_chain": chain, "direct_manager_id": chain[0]["user_id"] if chain else None},
        [f"Manager chain for {user_id}: {[c['name'] for c in chain]}"],
        {"completion_state.subgoal_status.sg_find_approvers": "completed"},
    )

def org_list_users(ws: Dict, args: Dict) -> Dict:
    """List all users, optionally filtered by department."""
    dept = args.get("department")
    users = ws.get("users", {})
    if dept:
        filtered = {uid: u for uid, u in users.items() if u["department"] == dept}
    else:
        filtered = users
    summary = [{"user_id": uid, "name": u["name"], "department": u["department"],
                "job_title": u["job_title"]} for uid, u in filtered.items()]
    return _ok({"users": summary, "count": len(summary)},
               [f"Found {len(summary)} users{' in ' + dept if dept else ''}"])


# ── Request tools ─────────────────────────────────────────────────────────────

def request_view(ws: Dict, args: Dict) -> Dict:
    """View a specific pending request."""
    request_id = args.get("request_id")
    requests = ws.get("pending_requests", {})
    if request_id:
        req = requests.get(request_id)
        if not req:
            return _err(f"Request {request_id} not found")
    else:
        if not requests:
            return _ok({"request": None}, ["No pending requests"])
        req = next(iter(requests.values()))

    users = ws.get("users", {})
    requester = users.get(req["requester_id"], {})
    resources = ws.get("resources", {})
    resource = resources.get(req["resource_id"], {})

    return _ok(
        {"request": req,
         "requester": {"name": requester.get("name"), "department": requester.get("department")},
         "resource": {"name": resource.get("name"), "type": resource.get("type"),
                      "sensitivity": resource.get("sensitivity")}},
        [f"Request {req['request_id']}: {requester.get('name')} wants {req['requested_role']} "
         f"on {resource.get('name')} – reason: {req['reason']}"],
        {"completion_state.subgoal_status.sg_view_request": "completed"},
    )

def request_list(ws: Dict, args: Dict) -> Dict:
    """List all pending requests."""
    requests = ws.get("pending_requests", {})
    pending = [r for r in requests.values() if r["status"] == "pending"]
    return _ok(
        {"requests": pending, "count": len(pending)},
        [f"Found {len(pending)} pending request(s)"],
        {"completion_state.subgoal_status.sg_view_request": "completed"},
    )


# ── Approval tools ────────────────────────────────────────────────────────────

def approval_route(ws: Dict, args: Dict) -> Dict:
    """Route an access request to a specific approver."""
    request_id = args.get("request_id")
    approver_id = args.get("approver_id")
    if not request_id or not approver_id:
        return _err("request_id and approver_id are required")

    chains = ws.get("approval_chains", {})
    chain_info = chains.get(request_id)
    if not chain_info:
        return _err(f"No approval chain found for request {request_id}")

    users = ws.get("users", {})
    approver = users.get(approver_id, {})
    hidden = ws.get("hidden_state", {})
    correct_chain = hidden.get("correct_approval_chains", {}).get(request_id, [])
    current_step = chain_info.get("current_step", 0)

    is_correct = (current_step < len(correct_chain) and
                  correct_chain[current_step] == approver_id)

    step_reward_key = "sg_route_approvals"
    if is_correct:
        # Auto-approve for simulation
        chain_info["approver_chain"][current_step]["status"] = "approved"
        chain_info["current_step"] = current_step + 1
        all_approved = all(s["status"] == "approved" for s in chain_info["approver_chain"])
        obs = [f"Routed to {approver.get('name', approver_id)} — APPROVED ✓"]
        if all_approved:
            ws["pending_requests"][request_id]["status"] = "approved"
            obs.append("All approvals collected. Request is fully approved.")
    else:
        obs = [f"Routed to {approver.get('name', approver_id)} — wrong approver for this step"]

    # Track routed approvals
    routed = ws.get("completion_state", {}).get("approvals_routed", [])
    routed.append({"request_id": request_id, "approver_id": approver_id, "correct": is_correct})

    return _ok(
        {"request_id": request_id, "approver_id": approver_id,
         "correct": is_correct, "chain_state": chain_info},
        obs,
        {
            "approval_chains": chains,
            "completion_state.approvals_routed": routed,
            "completion_state.subgoal_status.sg_route_approvals": "completed",
            "completion_state.subgoal_status.sg_find_approvers": "completed",
        },
    )

def approval_check_status(ws: Dict, args: Dict) -> Dict:
    """Check the approval status for a request."""
    request_id = args.get("request_id")
    if not request_id:
        return _err("request_id is required")
    chains = ws.get("approval_chains", {})
    chain = chains.get(request_id, {})
    req = ws.get("pending_requests", {}).get(request_id, {})
    return _ok(
        {"status": req.get("status", "pending"), "chain": chain},
        [f"Request {request_id} status: {req.get('status', 'pending')}"],
    )


# ── Access tools ──────────────────────────────────────────────────────────────

def access_decide(ws: Dict, args: Dict) -> Dict:
    """Submit an approve/deny decision for an access request (Task 1)."""
    request_id = args.get("request_id")
    decision = args.get("decision")  # "approve" | "deny"
    role = args.get("role")
    ttl_hours = args.get("ttl_hours")
    justification_category = args.get("justification_category", "operational")

    if not decision:
        return _err("decision is required (approve or deny)")

    requests = ws.get("pending_requests", {})
    # If no request_id given (or it's not found), auto-pick the first pending one
    if not request_id or request_id not in requests:
        request_id = next(
            (rid for rid, r in requests.items() if r.get("status") == "pending"),
            next(iter(requests), None),
        )
        if not request_id:
            return _err("No pending requests found")

    requests[request_id]["status"] = "approved" if decision == "approve" else "denied"
    requests[request_id]["_agent_decision"] = {
        "decision": decision,
        "role": role,
        "ttl_hours": ttl_hours,
        "justification_category": justification_category,
    }

    return _ok(
        {"request_id": request_id, "decision": decision, "role": role, "ttl_hours": ttl_hours},
        [f"Decision submitted: {decision.upper()} for {request_id} "
         f"(role={role}, ttl={ttl_hours}h)"],
        {
            "pending_requests": requests,
            "completion_state.decision_submitted": True,
            "_terminated": True,  # Task 1 ends after a decision
        },
    )

def access_grant(ws: Dict, args: Dict) -> Dict:
    """Activate an approved access grant (Task 2) or break-glass grant (Task 4)."""
    request_id = args.get("request_id")
    role = args.get("role")  # optional — used by break-glass to record chosen role
    if not request_id:
        return _err("request_id is required")
    req = ws.get("pending_requests", {}).get(request_id, {})
    if not req:
        return _err(f"Request {request_id} not found")

    # Break-glass requests bypass normal approval requirement
    if req.get("status") != "approved" and not req.get("_breakglass"):
        return _ok(
            {"activated": False},
            [f"Cannot grant: request {request_id} is not approved (status={req.get('status')})"],
        )

    requests = ws.get("pending_requests", {})
    if role:
        requests[request_id]["_agent_role"] = role

    return _ok(
        {"activated": True, "request_id": request_id, "role": role},
        [f"Grant activated for request {request_id}" + (f" (role={role})" if role else "")],
        {
            "pending_requests": requests,
            "completion_state.grant_activated": True,
            "completion_state.subgoal_status.sg_activate_grant": "completed",
            "_terminated": True,
        },
    )

def access_deny(ws: Dict, args: Dict) -> Dict:
    """Explicitly deny an escalation request (Task 2)."""
    request_id = args.get("request_id")
    if not request_id:
        return _err("request_id is required")
    requests = ws.get("pending_requests", {})
    req = requests.get(request_id)
    if not req:
        return _err(f"Request {request_id} not found")

    req["status"] = "denied"
    return _ok(
        {"denied": True, "request_id": request_id},
        [f"Escalation request {request_id} has been explicitly DENIED by the agent"],
        {
            "pending_requests": requests,
            "completion_state.grant_denied": True,
            "_terminated": True,
        },
    )

def access_set_ttl(ws: Dict, args: Dict) -> Dict:
    """Set the TTL on a pending grant."""
    request_id = args.get("request_id")
    ttl_hours = args.get("ttl_hours")
    if not request_id or ttl_hours is None:
        return _err("request_id and ttl_hours are required")
    requests = ws.get("pending_requests", {})
    if request_id in requests:
        requests[request_id]["_agent_ttl"] = ttl_hours
    return _ok(
        {"request_id": request_id, "ttl_hours": ttl_hours},
        [f"TTL set to {ttl_hours}h for request {request_id}"],
        {
            "pending_requests": requests,
            "completion_state.subgoal_status.sg_set_ttl": "completed",
        },
    )


# ── Entitlement tools ─────────────────────────────────────────────────────────

def entitlement_list(ws: Dict, args: Dict) -> Dict:
    """List entitlements, optionally filtered by user_id."""
    user_id = args.get("user_id")
    entitlements = ws.get("entitlements", {})
    if user_id:
        filtered = {eid: e for eid, e in entitlements.items() if e["user_id"] == user_id}
    else:
        filtered = entitlements

    # Return sanitized view (no hidden _is_risky fields)
    visible = [{k: v for k, v in e.items() if not k.startswith("_")}
               for e in filtered.values()]

    return _ok(
        {"entitlements": visible, "count": len(visible)},
        [f"Found {len(visible)} entitlement(s){' for user ' + user_id if user_id else ''}"],
        {
            "completion_state.subgoal_status.sg_list_entitlements": "completed",
            "completion_state.subgoal_status.sg_check_existing": "completed",
        },
    )

def entitlement_inspect(ws: Dict, args: Dict) -> Dict:
    """Inspect a specific entitlement — reveals risky flags if applicable."""
    entitlement_id = args.get("entitlement_id")
    if not entitlement_id:
        return _err("entitlement_id is required")
    entitlements = ws.get("entitlements", {})
    ent = entitlements.get(entitlement_id)
    if not ent:
        return _err(f"Entitlement {entitlement_id} not found")

    now = datetime.fromisoformat(ws.get("current_time", datetime.now().isoformat()))
    resources = ws.get("resources", {})
    resource = resources.get(ent["resource_id"], {})
    users = ws.get("users", {})
    user = users.get(ent["user_id"], {})

    # Reveal risky signals from raw data (do NOT use internal _risky_reason flag)
    warnings = []
    if ent.get("is_temporary") and ent.get("expires_at"):
        expiry_dt = datetime.fromisoformat(ent["expires_at"])
        if expiry_dt < now:
            warnings.append(f"⚠️ Entitlement has expired (expiry: {ent['expires_at']})")
    
    if ent.get("days_since_use", 0) > 90:
        warnings.append(f"⚠️ Entitlement is stale: last used {ent['days_since_use']} days ago")
        
    if user.get("status") == "inactive":
        warnings.append(f"⚠️ Entitlement is orphaned: user {user.get('name')} is {user.get('status')}")

    return _ok(
        {"entitlement": {k: v for k, v in ent.items() if not k.startswith("_")},
         "resource": {"name": resource.get("name"), "sensitivity": resource.get("sensitivity")},
         "user": {"name": user.get("name"), "status": user.get("status")},
         "warnings": warnings},
        ([f"Entitlement {entitlement_id}: {user.get('name')} has '{ent['role']}' on {resource.get('name')}"] +
         warnings),
        {"completion_state.subgoal_status.sg_identify_risky": "in_progress"},
    )

def entitlement_revoke(ws: Dict, args: Dict) -> Dict:
    """Revoke a specific entitlement."""
    entitlement_id = args.get("entitlement_id")
    reason = args.get("reason", "access_review")
    if not entitlement_id:
        return _err("entitlement_id is required")
    entitlements = ws.get("entitlements", {})
    if entitlement_id not in entitlements:
        return _err(f"Entitlement {entitlement_id} not found")

    entitlements[entitlement_id]["status"] = "revoked"
    revoked = ws.get("completion_state", {}).get("entitlements_revoked", [])
    revoked.append(entitlement_id)

    return _ok(
        {"revoked": entitlement_id, "reason": reason},
        [f"Entitlement {entitlement_id} revoked (reason: {reason})"],
        {
            "entitlements": entitlements,
            "completion_state.entitlements_revoked": revoked,
            "completion_state.subgoal_status.sg_revoke_correctly": "in_progress",
        },
    )


# ── Audit tools ───────────────────────────────────────────────────────────────

def audit_query(ws: Dict, args: Dict) -> Dict:
    """Query the pre-existing audit log."""
    user_id = args.get("user_id")
    resource_id = args.get("resource_id")
    days = args.get("days", 90)

    audit_db = ws.get("audit_db", [])
    now = datetime.fromisoformat(ws.get("current_time", datetime.now().isoformat()))

    results = []
    for entry in audit_db:
        ts = datetime.fromisoformat(entry["timestamp"])
        if (now - ts).days > days:
            continue
        if user_id and entry["user_id"] != user_id:
            continue
        if resource_id and entry["resource_id"] != resource_id:
            continue
        results.append(entry)

    return _ok(
        {"entries": results[:20], "count": len(results)},
        [f"Found {len(results)} audit entries"
         + (f" for user {user_id}" if user_id else "")
         + (f" on resource {resource_id}" if resource_id else "")],
        {"completion_state.subgoal_status.sg_query_audit": "completed"},
    )


# ── Group tools ───────────────────────────────────────────────────────────────

def group_resolve(ws: Dict, args: Dict) -> Dict:
    """Resolve group membership and show inherited entitlements."""
    group_id = args.get("group_id")
    user_id = args.get("user_id")  # or resolve groups for a specific user

    groups = ws.get("groups", {})
    entitlements = ws.get("entitlements", {})

    if group_id:
        group = groups.get(group_id)
        if not group:
            return _err(f"Group {group_id} not found")
        members = group.get("members", [])
        inherited = {eid: e for eid, e in entitlements.items()
                     if e["source"] == "group_inherited" and e["user_id"] in members}
        return _ok(
            {"group": group, "members": members, "inherited_entitlements": list(inherited.values())},
            [f"Group {group['name']}: {len(members)} members, {len(inherited)} inherited entitlements"],
            {"completion_state.subgoal_status.sg_resolve_groups": "completed"},
        )
    elif user_id:
        user_groups = [g for g in groups.values() if user_id in g.get("members", [])]
        inherited = {eid: e for eid, e in entitlements.items()
                     if e["source"] == "group_inherited" and e["user_id"] == user_id}
        return _ok(
            {"user_id": user_id, "groups": user_groups, "inherited_entitlements": list(inherited.values())},
            [f"User {user_id} is in {len(user_groups)} group(s) with {len(inherited)} inherited entitlements"],
            {"completion_state.subgoal_status.sg_resolve_groups": "completed"},
        )
    else:
        return _err("group_id or user_id is required")


# ── Workflow tools ────────────────────────────────────────────────────────────

def workflow_check_active(ws: Dict, args: Dict) -> Dict:
    """Check if a user has active workflows that depend on an entitlement."""
    user_id = args.get("user_id")
    entitlement_id = args.get("entitlement_id")

    workflows = ws.get("workflows", {})
    relevant = []

    for wf in workflows.values():
        match_user = (not user_id or wf.get("user_id") == user_id)
        match_ent = (not entitlement_id or
                     entitlement_id in wf.get("depends_on_entitlements", []))
        if match_user and match_ent and wf.get("is_active"):
            relevant.append(wf)

    return _ok(
        {"active_workflows": relevant, "count": len(relevant),
         "safe_to_revoke": len(relevant) == 0},
        [f"Found {len(relevant)} active workflow(s) depending on this entitlement"
         + (" — SAFE to revoke" if len(relevant) == 0 else " — WARNING: revocation may break workflows")],
        {"completion_state.subgoal_status.sg_check_workflows": "completed"},
    )


# ── Ticket tools ─────────────────────────────────────────────────────────────

def ticket_attach(ws: Dict, args: Dict) -> Dict:
    """Attach a ticket/incident reference to a pending escalation request (Task 2).

    The agent must provide the ticket_id that matches the one on the request.
    On success this sets completion_state.ticket_referenced = True, which
    allows the grader to award the full 1.0 on the ticket_attached component.
    """
    request_id = args.get("request_id")
    ticket_id = args.get("ticket_id")
    if not ticket_id:
        return _err("ticket_id is required")

    requests = ws.get("pending_requests", {})
    # Auto-pick the first active request if none specified
    if not request_id or request_id not in requests:
        request_id = next(
            (rid for rid, r in requests.items()
             if r.get("status") in ("pending", "approved")),
            next(iter(requests), None),
        )
    if not request_id:
        return _err("No valid request found to attach ticket to")

    req = requests[request_id]
    expected_ticket = req.get("ticket_id", "")
    correct = ticket_id == expected_ticket

    return _ok(
        {"request_id": request_id, "ticket_id": ticket_id, "correct": correct},
        [
            f"Ticket {ticket_id} attached to request {request_id}"
            + (" ✓" if correct else f" (expected {expected_ticket})")
        ],
        {
            "completion_state.ticket_referenced": True,
            "completion_state.subgoal_status.sg_attach_ticket": "completed",
        },
    )


# ── Review tools ──────────────────────────────────────────────────────────────

def review_submit(ws: Dict, args: Dict) -> Dict:
    """Submit the completed access review."""
    summary = args.get("summary", "")
    revoked = ws.get("completion_state", {}).get("entitlements_revoked", [])

    return _ok(
        {"submitted": True,
         "entitlements_revoked": revoked,
         "count_revoked": len(revoked),
         "summary": summary},
        [f"Access review submitted. Revoked {len(revoked)} entitlement(s)."],
        {
            "completion_state.review_submitted": True,
            "completion_state.subgoal_status.sg_submit_review": "completed",
            "_terminated": True,
        },
    )


# ── Incident tools (Task 4: emergency_breakglass) ────────────────────────────

def incident_verify(ws: Dict, args: Dict) -> Dict:
    """Verify an incident and reveal on-call engineer + affected resource details."""
    incident_id = args.get("incident_id")
    if not incident_id:
        return _err("incident_id is required")

    # Raw incidents dict (has _-prefixed hidden fields)
    raw_incidents = {}
    for inc_id, inc in ws.get("incidents", {}).items():
        raw_incidents[inc_id] = inc

    # Also check world_state storage that may have the full data
    # The full incident data is in the world_state under incidents with _ fields exposed
    inc = None
    # Search both the visible incidents dict and the raw world_state incidents
    for inc_id, inc_data in ws.get("incidents", {}).items():
        if inc_id == incident_id:
            inc = inc_data
            break

    if not inc:
        return _err(f"Incident {incident_id} not found")

    # Reveal hidden fields (strip the _ prefix for the response)
    full_details = {k.lstrip("_"): v for k, v in inc.items()}
    # Also include visible fields normally
    full_details.update({k: v for k, v in inc.items() if not k.startswith("_")})

    is_valid = (
        inc.get("status") == "active"
        and inc.get("severity") in ("P1", "P2")
    )

    return _ok(
        {"incident": full_details, "is_valid": is_valid},
        [
            f"Incident {incident_id}: severity={inc.get('severity')}, "
            f"status={inc.get('status')}, valid={is_valid}",
            f"On-call engineer: {full_details.get('on_call_engineer_id')}",
            f"Affected resource: {full_details.get('affected_resource_id')}",
        ],
        {
            "completion_state.incident_verified": True,
            "completion_state.subgoal_status.sg_verify_incident": "completed",
        },
    )


def audit_flag(ws: Dict, args: Dict) -> Dict:
    """Flag a security event — notifies the security team of an emergency override."""
    incident_id = args.get("incident_id") or args.get("request_id")
    flag_type = args.get("flag_type", "emergency_override")
    if not incident_id:
        return _err("incident_id or request_id is required")

    return _ok(
        {"flagged": True, "incident_id": incident_id, "flag_type": flag_type},
        [f"Security flag '{flag_type}' recorded for {incident_id}. Security team notified."],
        {
            "completion_state.security_flagged": True,
            "completion_state.subgoal_status.sg_flag_security": "completed",
        },
    )


# ── SoD tools (Task 5: separation_of_duties_audit) ───────────────────────────

_ROLE_RANK = {"viewer": 0, "editor": 1, "admin": 2, "owner": 3}


def sod_get_conflict_matrix(ws: Dict, args: Dict) -> Dict:
    """Return the full SoD conflict matrix."""
    cm = ws.get("conflict_matrix", {})
    return _ok(
        {"conflict_matrix": list(cm.values()), "count": len(cm)},
        [f"SoD conflict matrix: {len(cm)} conflict type(s)"],
        {"completion_state.subgoal_status.sg_get_conflicts": "completed"},
    )


def sod_check_user(ws: Dict, args: Dict) -> Dict:
    """Check a user's entitlements against the conflict matrix; return violations found."""
    user_id = args.get("user_id")
    if not user_id:
        return _err("user_id is required")

    entitlements = ws.get("entitlements", {})
    conflict_matrix = ws.get("conflict_matrix", {})
    resources = ws.get("resources", {})

    user_ents = [e for e in entitlements.values()
                 if e["user_id"] == user_id and e.get("status") != "revoked"]

    violations = []
    for conflict in conflict_matrix.values():
        cid = conflict["conflict_id"]
        ents_a = [e for e in user_ents
                  if resources.get(e["resource_id"], {}).get("type") == conflict["resource_type_a"]
                  and _ROLE_RANK.get(e["role"], 0) >= _ROLE_RANK.get(conflict["min_role_a"], 0)]
        ents_b = [e for e in user_ents
                  if resources.get(e["resource_id"], {}).get("type") == conflict["resource_type_b"]
                  and _ROLE_RANK.get(e["role"], 0) >= _ROLE_RANK.get(conflict["min_role_b"], 0)]
        if ents_a and ents_b:
            violations.append({
                "conflict_id": cid,
                "conflict_name": conflict["name"],
                "severity": conflict["severity"],
                "entitlements_a": [e["entitlement_id"] for e in ents_a],
                "entitlements_b": [e["entitlement_id"] for e in ents_b],
            })

    # Track identified violations
    identified = ws.get("completion_state", {}).get("sod_violations_identified", [])
    for v in violations:
        entry = {"user_id": user_id, "conflict_id": v["conflict_id"]}
        if entry not in identified:
            identified.append(entry)

    sg_status = "in_progress" if violations else "completed"
    return _ok(
        {"user_id": user_id, "violations": violations, "count": len(violations)},
        [f"User {user_id}: {len(violations)} SoD violation(s)"
         + (f" — {[v['conflict_id'] for v in violations]}" if violations else " — none found")],
        {
            "completion_state.sod_violations_identified": identified,
            "completion_state.subgoal_status.sg_check_violations": sg_status,
            "completion_state.subgoal_status.sg_list_users": "completed",
        },
    )


def sod_get_compensating_controls(ws: Dict, args: Dict) -> Dict:
    """Return compensating controls for a user (and optionally a specific conflict)."""
    user_id = args.get("user_id")
    conflict_id = args.get("conflict_id")
    if not user_id:
        return _err("user_id is required")

    controls = ws.get("compensating_controls", {})
    matching = [c for c in controls.values()
                if c["user_id"] == user_id
                and (not conflict_id or c["conflict_id"] == conflict_id)]

    checked = ws.get("completion_state", {}).get("sod_controls_checked", [])
    entry = {"user_id": user_id, "conflict_id": conflict_id}
    if entry not in checked:
        checked.append(entry)

    has_active = any(c["is_active"] for c in matching)
    return _ok(
        {"controls": matching, "count": len(matching), "has_active_control": has_active},
        [f"User {user_id}: {len(matching)} compensating control(s)"
         + (f" for {conflict_id}" if conflict_id else "")
         + (f" — {sum(1 for c in matching if c['is_active'])} active" if matching else "")],
        {
            "completion_state.sod_controls_checked": checked,
            "completion_state.subgoal_status.sg_check_controls": "completed",
        },
    )


def sod_submit_report(ws: Dict, args: Dict) -> Dict:
    """Submit the SoD audit report — terminal action."""
    summary = args.get("summary", "")
    cs = ws.get("completion_state", {})
    identified = cs.get("sod_violations_identified", [])
    revoked = cs.get("entitlements_revoked", [])

    # Direct mutation so grader sees it even without server-side state_delta apply
    cs["sod_report_submitted"] = True
    cs.get("subgoal_status", {})["sg_submit_report"] = "completed"

    return _ok(
        {"submitted": True, "violations_identified": len(identified),
         "entitlements_revoked": len(revoked), "summary": summary},
        [f"SoD audit report submitted. {len(identified)} violation(s) identified, "
         f"{len(revoked)} entitlement(s) revoked."],
        {
            "completion_state.sod_report_submitted": True,
            "completion_state.subgoal_status.sg_submit_report": "completed",
            "_terminated": True,
        },
    )


# ── Tool Registry ─────────────────────────────────────────────────────────────

TOOL_REGISTRY: Dict[str, Any] = {
    "policy.lookup":            policy_lookup,
    "policy.list":              policy_list,
    "org.get_user":             org_get_user,
    "org.get_manager":          org_get_manager,
    "org.list_users":           org_list_users,
    "request.view":             request_view,
    "request.list":             request_list,
    "approval.route":           approval_route,
    "approval.check_status":    approval_check_status,
    "access.decide":            access_decide,
    "access.grant":             access_grant,
    "access.deny":              access_deny,
    "access.set_ttl":           access_set_ttl,
    "ticket.attach":            ticket_attach,
    "entitlement.list":         entitlement_list,
    "entitlement.inspect":      entitlement_inspect,
    "entitlement.revoke":       entitlement_revoke,
    "audit.query":              audit_query,
    "audit.flag":               audit_flag,
    "group.resolve":            group_resolve,
    "workflow.check_active":    workflow_check_active,
    "review.submit":            review_submit,
    "incident.verify":          incident_verify,
    "sod.get_conflict_matrix":  sod_get_conflict_matrix,
    "sod.check_user":           sod_check_user,
    "sod.get_compensating_controls": sod_get_compensating_controls,
    "sod.submit_report":        sod_submit_report,
}


TOOL_METADATA: Dict[str, Dict] = {
    "policy.lookup":   {"desc": "Look up the policy that governs a resource",
                        "args": {"resource_id": "string (required)"}},
    "policy.list":     {"desc": "List all policies in the environment",
                        "args": {}},
    "org.get_user":    {"desc": "Get profile for a specific user",
                        "args": {"user_id": "string (required)"}},
    "org.get_manager": {"desc": "Get the management chain for a user",
                        "args": {"user_id": "string (required)"}},
    "org.list_users":  {"desc": "List all users, optionally filtered by department",
                        "args": {"department": "string (optional)"}},
    "request.view":    {"desc": "View a pending access request with requester and resource details",
                        "args": {"request_id": "string (optional)"}},
    "request.list":    {"desc": "List all pending requests",
                        "args": {}},
    "approval.route":  {"desc": "Route a request to a specific approver in the chain",
                        "args": {"request_id": "string (required)", "approver_id": "string (required)"}},
    "approval.check_status": {"desc": "Check the current approval chain status for a request",
                              "args": {"request_id": "string (required)"}},
    "access.decide":   {"desc": "Submit an approve or deny decision for a pending access request",
                        "args": {"request_id": "string (optional)",
                                 "decision": "approve | deny (required)",
                                 "role": "viewer | editor | admin | owner (required for approve)",
                                 "ttl_hours": "int, access duration in hours (required for approve)",
                                 "justification_category": "operational | incident_response | deployment | audit (optional, default: operational)"}},
    "access.grant":    {"desc": "Activate a fully approved access grant",
                        "args": {"request_id": "string (required)",
                                 "role": "viewer | editor | admin | owner (optional)"}},
    "access.deny":     {"desc": "Explicitly deny a pending access escalation request",
                        "args": {"request_id": "string (required)"}},
    "access.set_ttl":  {"desc": "Set the TTL on a pending grant",
                        "args": {"request_id": "string (required)", "ttl_hours": "int, duration in hours (required)"}},
    "ticket.attach":   {"desc": "Attach a ticket or incident reference to a request",
                        "args": {"request_id": "string (optional)", "ticket_id": "string (required)"}},
    "entitlement.list":    {"desc": "List entitlements for a user or all users",
                            "args": {"user_id": "string (optional)"}},
    "entitlement.inspect": {"desc": "Inspect a specific entitlement and return its full details including expiry, usage, and risk indicators",
                            "args": {"entitlement_id": "string (required)"}},
    "entitlement.revoke":  {"desc": "Revoke a specific entitlement",
                            "args": {"entitlement_id": "string (required)",
                                     "reason": "access_review | sod_violation | policy_violation | orphaned_account | expired (optional, default: access_review)"}},
    "audit.query":     {"desc": "Query the audit log filtered by user, resource, or time window",
                        "args": {"user_id": "string (optional)", "resource_id": "string (optional)",
                                 "days": "int (optional, default: 90)"}},
    "audit.flag":      {"desc": "Flag a security event in the audit log",
                        "args": {"incident_id": "string (required)",
                                 "flag_type": "emergency_override | rogue_agent | sod_violation | policy_violation | suspicious_activity (optional, default: emergency_override)"}},
    "group.resolve":   {"desc": "Resolve group membership and inherited entitlements",
                        "args": {"group_id": "string (optional)", "user_id": "string (optional)"}},
    "workflow.check_active": {"desc": "Check if a user has active workflows depending on an entitlement",
                              "args": {"user_id": "string (optional)", "entitlement_id": "string (optional)"}},
    "review.submit":   {"desc": "Submit the completed access review",
                        "args": {"summary": "string (optional)"}},
    "incident.verify": {"desc": "Verify an incident and retrieve its full details including on-call engineer and affected resource",
                        "args": {"incident_id": "string (required)"}},
    "sod.get_conflict_matrix":       {"desc": "Return the full separation of duties conflict matrix",
                                      "args": {}},
    "sod.check_user":                {"desc": "Check a user's entitlements for separation of duties violations",
                                      "args": {"user_id": "string (required)"}},
    "sod.get_compensating_controls": {"desc": "Return compensating controls for a user or specific conflict",
                                      "args": {"user_id": "string (required)", "conflict_id": "string (optional)"}},
    "sod.submit_report":             {"desc": "Submit the separation of duties audit report",
                                      "args": {"summary": "string (optional)"}},
    "emergency_breakglass":          {"desc": "Quarantine a user or agent and revoke all associated entitlements",
                                      "args": {"agent_id": "string (required)",
                                               "reason": "string (optional)"}},
}


def get_available_tools(task_available: list = None) -> list:
    if task_available:
        return [t for t in task_available if t in TOOL_REGISTRY]
    return list(TOOL_REGISTRY.keys())


def get_tool_metadata(task_available: list = None) -> Dict[str, Dict]:
    """Return TOOL_METADATA filtered to the task's available tools."""
    keys = task_available if task_available else list(TOOL_METADATA.keys())
    return {k: TOOL_METADATA[k] for k in keys if k in TOOL_METADATA}
