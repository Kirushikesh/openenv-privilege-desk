"""
PrivilegeDesk — Custom Gradio UI

A professional, dark-themed UI for the PrivilegeDesk IAM training environment.
Provides three modes: Interactive Playground, Live AI Demo, and Task Reference.
"""
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
import gradio as gr

# ── Config ────────────────────────────────────────────────────────────────────

BASE_URL = os.getenv("PRIVILEGE_DESK_URL", "http://127.0.0.1:8000")

TASK_INFO = {
    "access_decision": {
        "name": "Access Decision",
        "difficulty": "Easy",
        "diff_class": "easy",
        "emoji": "🔑",
        "desc": (
            "Review pending access requests and make approve/deny decisions "
            "with correct role assignment and TTL bounds per policy."
        ),
        "max_steps": 5,
        "tools": [
            "policy.lookup", "policy.list", "org.get_user",
            "entitlement.list", "request.view", "access.decide",
        ],
        "weights": {
            "Decision (approve/deny)": "40%",
            "Correct Role": "25%",
            "Correct TTL": "20%",
            "Justification": "15%",
        },
    },
    "jit_escalation": {
        "name": "JIT Escalation",
        "difficulty": "Medium",
        "diff_class": "medium",
        "emoji": "⚡",
        "desc": (
            "Process just-in-time privilege escalation requests: route through "
            "the ordered approval chain, attach an incident ticket, set TTL, "
            "and activate the temporary grant."
        ),
        "max_steps": 15,
        "tools": [
            "policy.lookup", "policy.list", "org.get_user", "org.get_manager",
            "org.list_users", "request.view", "request.list",
            "approval.route", "approval.check_status", "ticket.attach",
            "access.grant", "access.deny", "access.set_ttl", "entitlement.list",
        ],
        "weights": {
            "Correct Approvers": "20%",
            "Routing Order": "15%",
            "Ticket Attached": "15%",
            "Correct Role": "15%",
            "Correct TTL": "15%",
            "Final Decision": "20%",
        },
    },
    "emergency_breakglass": {
        "name": "Emergency Breakglass",
        "difficulty": "Medium",
        "diff_class": "medium",
        "emoji": "🚨",
        "desc": (
            "Handle emergency break-glass access for production incidents: "
            "verify the incident, attach the ticket, flag the security team, "
            "set TTL, and activate emergency access."
        ),
        "max_steps": 10,
        "tools": [
            "incident.verify", "policy.lookup", "policy.list", "org.get_user",
            "entitlement.list", "ticket.attach", "access.grant",
            "access.set_ttl", "access.deny", "audit.flag",
        ],
        "weights": {
            "Incident Valid": "15%",
            "Correct Role": "15%",
            "Correct TTL": "20%",
            "Ticket Attached": "15%",
            "Security Flagged": "15%",
            "Final Grant": "20%",
        },
    },
    "access_review": {
        "name": "Access Review",
        "difficulty": "Hard",
        "diff_class": "hard",
        "emoji": "🔍",
        "desc": (
            "Audit a user's entitlements (including group-inherited), "
            "identify stale/over-privileged access, and revoke the minimum "
            "necessary — without breaking active workflows."
        ),
        "max_steps": 25,
        "tools": [
            "policy.lookup", "policy.list", "org.get_user", "org.list_users",
            "entitlement.list", "entitlement.inspect", "entitlement.revoke",
            "audit.query", "group.resolve", "workflow.check_active", "review.submit",
        ],
        "weights": {
            "Precision": "30%",
            "Recall": "30%",
            "Workflow Preservation": "20%",
            "Policy Compliance": "10%",
            "Review Submitted": "10%",
        },
    },
    "separation_of_duties_audit": {
        "name": "SoD Audit",
        "difficulty": "Hard",
        "diff_class": "hard",
        "emoji": "⚖️",
        "desc": (
            "Conduct an organization-wide Separation of Duties audit: "
            "find forbidden entitlement pairs, check compensating controls, "
            "revoke unmitigated violations, and file the audit report."
        ),
        "max_steps": 25,
        "tools": [
            "org.list_users", "org.get_user",
            "entitlement.list", "entitlement.inspect", "entitlement.revoke",
            "sod.get_conflict_matrix", "sod.check_user",
            "sod.get_compensating_controls", "sod.submit_report",
        ],
        "weights": {
            "Violations Found": "30%",
            "False Positives": "15%",
            "Correct Revocations": "25%",
            "Controls Recognized": "10%",
            "Report Submitted": "20%",
        },
    },
}

# ── CSS ───────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* === PrivilegeDesk Dark Theme === */

.gradio-container {
    background: #080d1a !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    max-width: 1300px !important;
    margin: 0 auto !important;
}

/* Tab styling */
.tab-nav button {
    background: transparent !important;
    border: none !important;
    color: #64748b !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 10px 16px !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s !important;
}
.tab-nav button.selected {
    color: #60a5fa !important;
    border-bottom-color: #3b82f6 !important;
}
.tab-nav {
    background: #0f172a !important;
    border-bottom: 1px solid #1e293b !important;
}

/* Input overrides */
input, textarea, select {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
input:focus, textarea:focus {
    border-color: #3b82f6 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px #3b82f622 !important;
}

/* Button overrides */
button.primary {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}
button.primary:hover {
    background: linear-gradient(135deg, #2563eb, #60a5fa) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px #3b82f644 !important;
}
button.secondary {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
}
button.secondary:hover {
    border-color: #475569 !important;
    color: #e2e8f0 !important;
}

/* Label styling */
label span, .label-wrap span {
    color: #94a3b8 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* Chatbot */
.chatbot .message-wrap {
    background: #0f172a !important;
}
.chatbot .message.user {
    background: #1e3a5f !important;
    border: 1px solid #1e40af44 !important;
    border-radius: 12px !important;
    color: #93c5fd !important;
}
.chatbot .message.bot {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}

/* Dropdown */
.dropdown-arrow { color: #64748b !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #475569; }

/* Badge classes */
.badge-easy { background: #064e3b; color: #34d399; border: 1px solid #065f4655; }
.badge-medium { background: #451a03; color: #fb923c; border: 1px solid #92400e55; }
.badge-hard { background: #450a0a; color: #f87171; border: 1px solid #991b1b55; }

/* Code block */
.code-block {
    background: #020617 !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    color: #7dd3fc !important;
    font-family: 'Monaco', 'Menlo', monospace !important;
}
"""


# ── API helpers ───────────────────────────────────────────────────────────────

def api_call(endpoint: str, method: str = "GET", data: dict = None) -> dict:
    try:
        url = f"{BASE_URL}{endpoint}"
        if method == "POST":
            r = requests.post(url, json=data or {}, timeout=30)
        else:
            r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        return {"error": "Cannot connect to server. Is it running on port 8000?"}
    except requests.Timeout:
        return {"error": "Server timeout — please try again"}
    except Exception as e:
        return {"error": str(e)}


# ── Visual helpers ────────────────────────────────────────────────────────────

def _reward_color(r: float) -> str:
    if r >= 0.6:
        return "#34d399"
    if r >= 0.3:
        return "#fb923c"
    return "#f87171"


def _reward_bar_html(r: float) -> str:
    pct = int(r * 100)
    color = _reward_color(r)
    emoji = "🟢" if r >= 0.6 else ("🟡" if r >= 0.3 else "🔴")
    return f"""
    <div style="display:flex;align-items:center;gap:10px;padding:6px 0">
        <span style="font-size:16px">{emoji}</span>
        <div style="flex:1;height:8px;background:#1e293b;border-radius:4px;overflow:hidden">
            <div style="width:{pct}%;height:100%;background:{color};border-radius:4px;
                        transition:width 0.6s cubic-bezier(0.4,0,0.2,1)"></div>
        </div>
        <span style="color:{color};font-weight:700;font-size:14px;min-width:50px;text-align:right">{r:.3f}</span>
    </div>
    """


def _score_html(score: float) -> str:
    color = _reward_color(score)
    grade = "A" if score > 0.8 else ("B" if score > 0.6 else ("C" if score > 0.4 else "D"))
    bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
    return f"""
    <div style="background:linear-gradient(135deg,#0d1b2a,#1a2744);border:1px solid {color}44;
                border-radius:16px;padding:28px;text-align:center">
        <div style="font-size:72px;font-weight:900;color:{color};line-height:1;
                    text-shadow:0 0 30px {color}66;margin-bottom:8px">{grade}</div>
        <div style="font-size:28px;color:{color};font-weight:700;margin-bottom:10px">{score:.4f}</div>
        <div style="font-family:monospace;color:{color}aa;font-size:15px;
                    letter-spacing:4px;margin-bottom:16px">[{bar}]</div>
        <div style="color:#64748b;font-size:12px;text-transform:uppercase;
                    letter-spacing:0.1em">Episode Score</div>
    </div>
    """


def _breakdown_html(breakdown: dict) -> str:
    if not breakdown:
        return "<p style='color:#64748b;font-size:13px'>No breakdown available yet</p>"
    rows = []
    for k, v in breakdown.items():
        try:
            val = float(v)
        except (TypeError, ValueError):
            continue
        pct = int(val * 100)
        color = _reward_color(val)
        label = k.replace("_", " ").title()
        rows.append(f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
            <div style="width:160px;color:#94a3b8;font-size:12px;flex-shrink:0">{label}</div>
            <div style="flex:1;height:6px;background:#0f172a;border-radius:3px;overflow:hidden">
                <div style="width:{pct}%;height:100%;background:{color};border-radius:3px"></div>
            </div>
            <div style="color:{color};font-size:12px;width:40px;text-align:right;font-weight:600">{val:.3f}</div>
        </div>
        """)
    return '<div style="padding:4px 0">' + "".join(rows) + "</div>"


def _format_goal_html(obs: dict) -> str:
    task_id = obs.get("task_id", "")
    goal = obs.get("task_goal", "No goal specified")
    info = TASK_INFO.get(task_id, {})
    diff = info.get("difficulty", "")
    diff_class = info.get("diff_class", "easy")
    emoji = info.get("emoji", "📋")
    max_steps = obs.get("max_steps", 25)
    step = obs.get("step", 0)
    tools = obs.get("available_tools", [])
    step_pct = int((step / max_steps) * 100) if max_steps else 0

    tools_html = "".join(
        f'<span style="display:inline-block;background:#0f172a;border:1px solid #334155;'
        f'border-radius:4px;padding:2px 7px;font-size:11px;font-family:monospace;'
        f'color:#7dd3fc;margin:2px">{t}</span>'
        for t in tools[:8]
    )

    return f"""
    <div style="background:linear-gradient(135deg,#0d1b2a,#132038);border:1px solid #1e40af33;
                border-radius:16px;padding:22px;margin-bottom:12px">
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px">
            <span style="font-size:36px;filter:drop-shadow(0 0 10px #3b82f6aa)">{emoji}</span>
            <div style="flex:1">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
                    <span style="color:#e2e8f0;font-weight:700;font-size:18px">
                        {task_id.replace("_", " ").title()}
                    </span>
                    <span class="badge badge-{diff_class}" style="display:inline-block;
                        padding:2px 10px;border-radius:20px;font-size:10px;font-weight:700;
                        letter-spacing:0.1em;text-transform:uppercase">{diff}</span>
                </div>
                <div style="color:#64748b;font-size:12px">
                    Step {step} of {max_steps}
                </div>
            </div>
            <div style="background:#0f172a;border:1px solid #334155;border-radius:8px;
                        padding:8px 16px;text-align:center">
                <div style="color:#64748b;font-size:10px;text-transform:uppercase;
                            letter-spacing:0.08em">Max Steps</div>
                <div style="color:#60a5fa;font-size:24px;font-weight:800;line-height:1.2">{max_steps}</div>
            </div>
        </div>

        <!-- Progress bar -->
        <div style="margin-bottom:14px">
            <div style="height:4px;background:#1e293b;border-radius:2px;overflow:hidden">
                <div style="width:{step_pct}%;height:100%;
                            background:linear-gradient(90deg,#3b82f6,#06b6d4);border-radius:2px;
                            transition:width 0.5s ease"></div>
            </div>
        </div>

        <!-- Goal text -->
        <div style="color:#94a3b8;font-size:13px;line-height:1.75;
                    border-top:1px solid #1e293b;padding-top:14px;margin-bottom:12px">
            {goal}
        </div>

        <!-- Tools -->
        <div>
            <div style="color:#475569;font-size:10px;text-transform:uppercase;
                        letter-spacing:0.1em;margin-bottom:6px">Available Tools</div>
            <div>{tools_html}</div>
        </div>
    </div>
    """


def _format_obs_panels(obs: dict) -> str:
    parts = []

    # Pending requests
    pending = obs.get("pending_requests", {})
    if pending:
        parts.append("""
        <div style="color:#475569;font-size:10px;text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:8px;margin-top:4px">
            Pending Requests
        </div>
        """)
        for req_id, req in list(pending.items())[:3]:
            requester = req.get("user_id", "?")
            resource = req.get("resource_id", "?")
            role = req.get("requested_role", "?")
            reason = (req.get("justification", "") or "")[:60]
            parts.append(f"""
            <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;
                        padding:12px;margin-bottom:8px">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                    <code style="color:#60a5fa;font-size:11px">{req_id}</code>
                    <span style="background:#1d4ed855;color:#93c5fd;padding:1px 8px;
                                 border-radius:10px;font-size:10px;font-weight:700">PENDING</span>
                </div>
                <div style="color:#cbd5e1;font-size:12px;line-height:1.6">
                    👤 <strong style="color:#a5f3fc">{requester}</strong>
                    → 🗄️ <strong style="color:#a5f3fc">{resource}</strong><br>
                    Role requested: <code style="color:#a78bfa">{role}</code><br>
                    {f'<span style="color:#64748b">Reason: {reason}</span>' if reason else ""}
                </div>
            </div>
            """)

    # Last tool result
    tool_result = obs.get("tool_result") or {}
    if tool_result:
        status = tool_result.get("status", "unknown")
        obs_lines = tool_result.get("observations", [])
        status_color = "#34d399" if status == "success" else "#f87171"
        status_icon = "✅" if status == "success" else "❌"
        parts.append(f"""
        <div style="background:#0f172a;border-left:3px solid {status_color};
                    border-radius:0 8px 8px 0;padding:12px;margin-bottom:10px">
            <div style="color:{status_color};font-size:11px;font-weight:700;
                        text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px">
                {status_icon} Tool Result: {status}
            </div>
            {"".join(f'<div style="color:#94a3b8;font-size:12px;margin-top:2px;line-height:1.5">• {line}</div>' for line in obs_lines[:5])}
        </div>
        """)

    # Objectives
    objectives = obs.get("objectives", [])
    if objectives:
        parts.append("""
        <div style="color:#475569;font-size:10px;text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:8px">Objectives</div>
        """)
        for obj in objectives:
            done = obj.get("completed", False)
            desc = obj.get("description", "")
            icon = "✅" if done else "⭕"
            text_color = "#34d399" if done else "#64748b"
            parts.append(f"""
            <div style="display:flex;gap:8px;align-items:flex-start;padding:5px 0;
                        border-bottom:1px solid #1e293b">
                <span style="font-size:13px;margin-top:1px">{icon}</span>
                <span style="color:{text_color};font-size:12px;line-height:1.5">{desc}</span>
            </div>
            """)

    # Audit log
    audit = obs.get("audit_log", [])
    if audit:
        parts.append("""
        <div style="color:#475569;font-size:10px;text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:8px;margin-top:12px">Recent Audit Log</div>
        """)
        for entry in list(reversed(audit))[:3]:
            action = entry.get("action", entry.get("tool_name", "?"))
            ts = str(entry.get("timestamp", ""))[:19]
            parts.append(f"""
            <div style="font-family:monospace;font-size:11px;color:#475569;
                        padding:3px 0;border-bottom:1px solid #0f172a">
                <span style="color:#334155">{ts}</span>
                <span style="color:#60a5fa;margin-left:8px">{action}</span>
            </div>
            """)

    if not parts:
        return """<div style="color:#475569;font-size:13px;padding:16px;text-align:center">
            No observation data yet. Reset an episode and execute a tool call.
        </div>"""

    return '<div style="font-family:system-ui">' + "".join(parts) + "</div>"


def _step_row_html(step: int, tool: str, args: dict, reward: float,
                   status: str, obs_lines: List[str]) -> str:
    reward_color = _reward_color(reward)
    status_icon = "✅" if status == "success" else ("⚠️" if status == "partial" else "❌")
    args_str = json.dumps(args)
    args_short = args_str[:55] + "…" if len(args_str) > 55 else args_str
    obs_preview = obs_lines[0][:70] if obs_lines else "—"
    return f"""
    <div style="display:flex;gap:10px;align-items:flex-start;padding:10px 0;
                border-bottom:1px solid #1e293b">
        <div style="background:#1e293b;color:#475569;padding:2px 8px;border-radius:4px;
                    font-size:10px;white-space:nowrap;min-width:54px;text-align:center;
                    font-family:monospace;font-weight:600">#{step:02d}</div>
        <div style="flex:1;min-width:0">
            <div style="display:flex;align-items:baseline;gap:6px;margin-bottom:3px;flex-wrap:wrap">
                <span style="color:#60a5fa;font-weight:600;font-family:monospace;font-size:13px">{tool}</span>
                <code style="color:#475569;font-size:11px">{args_short}</code>
            </div>
            <div style="color:#64748b;font-size:12px">{status_icon} {obs_preview}</div>
        </div>
        <div style="min-width:55px;text-align:right">
            <span style="color:{reward_color};font-weight:700;font-size:13px">{reward:+.3f}</span>
        </div>
    </div>
    """


def _build_history_html(history: list) -> str:
    if not history:
        return """
        <div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;
                    padding:20px;text-align:center;color:#334155;font-size:13px">
            No steps taken yet — execute a tool call to begin
        </div>
        """
    rows = "".join(
        _step_row_html(
            e["step"], e["tool"], e["args"],
            e["reward"], e["status"], e.get("obs_lines", []),
        )
        for e in reversed(history[-15:])
    )
    total = sum(e["reward"] for e in history)
    return f"""
    <div style="background:#0a0e1a;border:1px solid #1e293b;border-radius:10px;overflow:hidden">
        <div style="background:#0f172a;border-bottom:1px solid #1e293b;
                    padding:8px 14px;display:flex;justify-content:space-between">
            <span style="color:#64748b;font-size:11px;text-transform:uppercase;
                         letter-spacing:0.08em">Action Log</span>
            <span style="color:#475569;font-size:11px">{len(history)} steps · Σ reward {total:.3f}</span>
        </div>
        <div style="padding:0 12px;max-height:360px;overflow-y:auto">{rows}</div>
    </div>
    """


# ── Demo agent ────────────────────────────────────────────────────────────────

def _get_demo_steps(task_id: str, obs: dict) -> List[Tuple[str, dict]]:
    pending = obs.get("pending_requests", {})
    resources = obs.get("resources", {})
    policies = obs.get("policies", {})
    users = obs.get("users", {})
    entitlements = obs.get("entitlements", {})
    incidents = obs.get("incidents", {})

    req_id = next(iter(pending), None)
    req = pending.get(req_id, {}) if req_id else {}
    resource_id = req.get("resource_id") or next(iter(resources), None)
    requester_id = req.get("user_id") or next(iter(users), None)
    first_ent_id = next(iter(entitlements), None)
    first_incident_id = next(iter(incidents), None)
    first_user_id = next(iter(users), None)
    first_policy_id = next(iter(policies), None)
    review_target = obs.get("review_target_user_id") or first_user_id

    if task_id == "access_decision":
        return [
            ("request.view", {}),
            ("policy.lookup", {"resource_id": resource_id} if resource_id else {}),
            ("entitlement.list", {"user_id": requester_id} if requester_id else {}),
            ("access.decide", {
                "decision": "approve",
                "role": "viewer",
                "ttl_hours": 8,
                "justification_category": "operational",
            }),
        ]

    elif task_id == "jit_escalation":
        return [
            ("request.view", {}),
            ("policy.lookup", {"resource_id": resource_id} if resource_id else {}),
            ("org.get_manager", {"user_id": requester_id} if requester_id else {}),
            ("approval.route", {"request_id": req_id, "approver_id": "manager"} if req_id else {}),
            ("approval.check_status", {"request_id": req_id} if req_id else {}),
            ("ticket.attach", {
                "request_id": req_id,
                "ticket_id": "INC-2024-001",
            } if req_id else {"ticket_id": "INC-2024-001"}),
            ("access.set_ttl", {"request_id": req_id, "ttl_hours": 4} if req_id else {"ttl_hours": 4}),
            ("access.grant", {"request_id": req_id} if req_id else {}),
        ]

    elif task_id == "emergency_breakglass":
        return [
            ("incident.verify", {"incident_id": first_incident_id} if first_incident_id else {}),
            ("policy.lookup", {"resource_id": resource_id} if resource_id else {}),
            ("entitlement.list", {"user_id": requester_id} if requester_id else {}),
            ("ticket.attach", {
                "request_id": req_id,
                "ticket_id": first_incident_id or "INC-001",
            } if req_id else {"ticket_id": first_incident_id or "INC-001"}),
            ("audit.flag", {
                "reason": "emergency_breakglass",
                "incident_id": first_incident_id,
            }),
            ("access.set_ttl", {"request_id": req_id, "ttl_hours": 2} if req_id else {"ttl_hours": 2}),
            ("access.grant", {"request_id": req_id} if req_id else {}),
        ]

    elif task_id == "access_review":
        return [
            ("entitlement.list", {"user_id": review_target} if review_target else {}),
            ("audit.query", {"user_id": review_target} if review_target else {}),
            ("group.resolve", {"user_id": review_target} if review_target else {}),
            ("workflow.check_active", {"user_id": review_target} if review_target else {}),
            ("entitlement.inspect", {"entitlement_id": first_ent_id} if first_ent_id else {}),
            ("entitlement.revoke", {"entitlement_id": first_ent_id} if first_ent_id else {}),
            ("review.submit", {
                "summary": "Reviewed user entitlements and revoked stale/over-privileged access per compliance policy",
            }),
        ]

    elif task_id == "separation_of_duties_audit":
        return [
            ("sod.get_conflict_matrix", {}),
            ("org.list_users", {}),
            ("sod.check_user", {"user_id": first_user_id} if first_user_id else {}),
            ("sod.get_compensating_controls", {}),
            ("entitlement.inspect", {"entitlement_id": first_ent_id} if first_ent_id else {}),
            ("entitlement.revoke", {"entitlement_id": first_ent_id} if first_ent_id else {}),
            ("sod.submit_report", {
                "violations": [],
                "revocations": [first_ent_id] if first_ent_id else [],
                "summary": "SoD audit complete — identified and resolved unmitigated violations",
            }),
        ]

    return [("policy.list", {})]


# ── Task reference HTML ───────────────────────────────────────────────────────

def _task_card_html(tid: str, info: dict) -> str:
    tools_html = "".join(
        f'<span style="display:inline-block;background:#0f172a;border:1px solid #334155;'
        f'border-radius:5px;padding:2px 8px;font-size:11px;font-family:monospace;'
        f'color:#7dd3fc;margin:2px 2px 2px 0">{t}</span>'
        for t in info["tools"]
    )
    weights_html = "".join(
        f"""<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
            <div style="width:180px;color:#94a3b8;font-size:12px;flex-shrink:0">{k}</div>
            <div style="flex:1;height:6px;background:#0f172a;border-radius:3px;overflow:hidden">
                <div style="width:{int(v.strip('%'))}%;height:100%;
                            background:linear-gradient(90deg,#3b82f6,#06b6d4);border-radius:3px"></div>
            </div>
            <div style="color:#60a5fa;font-size:12px;width:36px;text-align:right;font-weight:600">{v}</div>
        </div>"""
        for k, v in info["weights"].items()
    )
    diff_class = info["diff_class"]
    badge_styles = {
        "easy": "background:#064e3b;color:#34d399;border:1px solid #065f4655",
        "medium": "background:#451a03;color:#fb923c;border:1px solid #92400e55",
        "hard": "background:#450a0a;color:#f87171;border:1px solid #991b1b55",
    }
    badge_style = badge_styles.get(diff_class, "")

    return f"""
    <div style="background:#1e293b;border:1px solid #334155;border-radius:14px;
                padding:22px;margin-bottom:18px;transition:border-color 0.2s">
        <div style="display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:14px">
            <div style="display:flex;align-items:center;gap:14px">
                <span style="font-size:32px;filter:drop-shadow(0 0 8px #3b82f688)">{info["emoji"]}</span>
                <div>
                    <div style="color:#e2e8f0;font-weight:700;font-size:17px;margin-bottom:5px">
                        {info["name"]}
                    </div>
                    <span style="display:inline-block;{badge_style};padding:3px 10px;
                                 border-radius:20px;font-size:10px;font-weight:700;
                                 letter-spacing:0.1em;text-transform:uppercase">
                        {info["difficulty"]}
                    </span>
                </div>
            </div>
            <div style="background:#0f172a;border:1px solid #334155;border-radius:10px;
                        padding:10px 18px;text-align:center;flex-shrink:0">
                <div style="color:#475569;font-size:10px;text-transform:uppercase;
                            letter-spacing:0.08em;margin-bottom:2px">Max Steps</div>
                <div style="color:#60a5fa;font-size:26px;font-weight:800;line-height:1.1">
                    {info["max_steps"]}
                </div>
            </div>
        </div>

        <p style="color:#94a3b8;font-size:13px;line-height:1.75;margin-bottom:16px;
                  border-bottom:1px solid #1e293b;padding-bottom:16px">{info["desc"]}</p>

        <div style="margin-bottom:16px">
            <div style="color:#475569;font-size:10px;text-transform:uppercase;
                        letter-spacing:0.1em;margin-bottom:8px">Available Tools</div>
            <div>{tools_html}</div>
        </div>

        <div>
            <div style="color:#475569;font-size:10px;text-transform:uppercase;
                        letter-spacing:0.1em;margin-bottom:10px">Grading Weights</div>
            {weights_html}
        </div>
    </div>
    """


# ── Main UI builder ───────────────────────────────────────────────────────────

def build_privilege_desk_ui() -> gr.Blocks:
    task_choices = list(TASK_INFO.keys())

    with gr.Blocks(title="PrivilegeDesk — IAM Agent Training Environment") as demo:

        # Inject CSS via style tag (Gradio 6 moved css/theme to launch())
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")

        # ── Hero Header ───────────────────────────────────────────────────────
        gr.HTML("""
        <div style="background:linear-gradient(135deg,#080d1a 0%,#0d1b2e 40%,#0a1628 70%,#080d1a 100%);
                    border:1px solid #1e40af33;border-radius:18px;padding:28px 36px;
                    margin-bottom:4px;position:relative;overflow:hidden">
            <!-- Decorative glow -->
            <div style="position:absolute;top:-60px;right:-60px;width:200px;height:200px;
                        background:radial-gradient(circle,#3b82f622,transparent 70%);
                        pointer-events:none"></div>

            <div style="display:flex;align-items:center;gap:22px;flex-wrap:wrap">
                <span style="font-size:58px;filter:drop-shadow(0 0 20px #3b82f699);flex-shrink:0">🛡️</span>
                <div style="flex:1">
                    <h1 style="color:#e2e8f0;font-size:34px;font-weight:900;margin:0 0 5px;
                                letter-spacing:-0.03em;line-height:1">
                        PrivilegeDesk
                    </h1>
                    <p style="color:#64748b;margin:0 0 14px;font-size:14px;line-height:1.5">
                        Zero-Standing-Privilege AI Agent Training Environment
                    </p>
                    <div style="display:flex;flex-wrap:wrap;gap:8px">
                        <span style="display:inline-flex;align-items:center;gap:5px;
                                     background:#1e3a5f22;border:1px solid #3b82f644;
                                     border-radius:20px;padding:4px 12px;
                                     color:#60a5fa;font-size:12px">⚙️ 5 Task Families</span>
                        <span style="display:inline-flex;align-items:center;gap:5px;
                                     background:#1e3a5f22;border:1px solid #3b82f644;
                                     border-radius:20px;padding:4px 12px;
                                     color:#60a5fa;font-size:12px">🔧 27 Agent Tools</span>
                        <span style="display:inline-flex;align-items:center;gap:5px;
                                     background:#06402422;border:1px solid #10b98144;
                                     border-radius:20px;padding:4px 12px;
                                     color:#34d399;font-size:12px">🤖 OpenEnv Compatible</span>
                        <span style="display:inline-flex;align-items:center;gap:5px;
                                     background:#1e293b;border:1px solid #334155;
                                     border-radius:20px;padding:4px 12px;
                                     color:#94a3b8;font-size:12px">📊 Reward-Shaped Grading</span>
                        <span style="display:inline-flex;align-items:center;gap:5px;
                                     background:#1e293b;border:1px solid #334155;
                                     border-radius:20px;padding:4px 12px;
                                     color:#94a3b8;font-size:12px">🌱 Deterministic Seeds</span>
                    </div>
                </div>
            </div>
        </div>
        """)

        # ── Tabs ──────────────────────────────────────────────────────────────
        with gr.Tabs(selected="demo"):

            # ══════════════════════════════════════════════════════════════════
            # Tab 1 — Live Agent Demo
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("🤖 Live Demo", id="demo"):
                gr.HTML("""
                <div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;
                            padding:18px 22px;margin-bottom:14px">
                    <h2 style="color:#e2e8f0;margin:0 0 4px;font-size:18px;font-weight:700">
                        🤖 Live AI Agent Demo
                    </h2>
                    <p style="color:#64748b;margin:0;font-size:13px">
                        Watch a heuristic IAM agent work through a complete episode in real-time —
                        gathering context, reasoning about policies, and making access decisions.
                    </p>
                </div>
                """)

                with gr.Row():
                    demo_task_dd = gr.Dropdown(
                        choices=task_choices,
                        value="access_decision",
                        label="Task",
                        scale=3,
                        interactive=True,
                    )
                    demo_seed_tb = gr.Textbox(
                        value="42",
                        label="Seed",
                        placeholder="42",
                        scale=1,
                    )
                    demo_diff_sl = gr.Slider(
                        minimum=1,
                        maximum=3,
                        value=3,
                        step=1,
                        label="Difficulty Level (affects max steps)",
                        scale=2,
                    )
                    demo_speed_sl = gr.Slider(
                        minimum=0.1,
                        maximum=3.0,
                        value=1.0,
                        step=0.1,
                        label="Speed (pause between steps, seconds)",
                        scale=2,
                    )

                with gr.Row():
                    demo_start_btn = gr.Button("▶  Start Demo", variant="primary", scale=3)
                    demo_stop_btn = gr.Button("⏹  Stop", variant="secondary", scale=1)

                with gr.Row():
                    with gr.Column(scale=3):
                        demo_chatbot = gr.Chatbot(
                            label="Agent Console",
                            height=500,
                            avatar_images=(
                                None,
                                "https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png",
                            ),
                        )
                    with gr.Column(scale=2):
                        demo_score_html = gr.HTML("""
                        <div style="background:#1e293b;border:1px solid #334155;border-radius:14px;
                                    padding:24px;text-align:center;min-height:180px;
                                    display:flex;flex-direction:column;justify-content:center">
                            <div style="color:#334155;font-size:48px;margin-bottom:8px">🏆</div>
                            <div style="color:#475569;font-size:13px">
                                Score appears after the episode ends
                            </div>
                        </div>
                        """)
                        demo_stats_html = gr.HTML("")

                # ── Demo generator callback ───────────────────────────────────
                def run_demo(task_id: str, seed_str: str, difficulty: float, speed: float):
                    seed = int(seed_str) if seed_str and seed_str.strip().isdigit() else 42
                    diff_level = int(difficulty)
                    messages = []
                    info = TASK_INFO.get(task_id, {})

                    # Reset
                    result = api_call("/reset", "POST", {
                        "task_id": task_id,
                        "seed": seed,
                        "difficulty_level": diff_level,
                    })
                    if "error" in result:
                        messages.append({
                            "role": "assistant",
                            "content": f"❌ **Connection Error**\n\n{result['error']}\n\nMake sure the server is running.",
                        })
                        yield messages, "", ""
                        return

                    obs = result.get("observation", {})
                    goal = obs.get("task_goal", "")[:300]

                    messages.append({
                        "role": "assistant",
                        "content": (
                            f"## 🚀 Episode Started\n\n"
                            f"**Task:** {info.get('emoji','')} {info.get('name', task_id)} "
                            f"({info.get('difficulty','')})\n\n"
                            f"**Goal:**\n> {goal}...\n\n"
                            f"**Max Steps:** {obs.get('max_steps', 25)} | "
                            f"**Seed:** {seed}"
                        ),
                    })
                    yield messages, "", ""
                    time.sleep(speed * 0.8)

                    steps = _get_demo_steps(task_id, obs)
                    total_reward = 0.0

                    for i, (tool, args) in enumerate(steps):
                        # Show what we're doing
                        args_display = json.dumps(args, indent=2)[:200]
                        messages.append({
                            "role": "user",
                            "content": (
                                f"**Step {i + 1}/{len(steps)}**\n\n"
                                f"```\n{tool}(\n{args_display}\n)\n```"
                            ),
                        })
                        yield messages, "", ""
                        time.sleep(speed * 0.4)

                        # Execute
                        step_res = api_call("/step", "POST", {"action": {"tool_name": tool, "arguments": args}})

                        if "error" in step_res:
                            messages.append({
                                "role": "assistant",
                                "content": f"❌ **Error:** {step_res['error']}",
                            })
                            yield messages, "", ""
                            break

                        new_obs = step_res.get("observation", {})
                        reward = step_res.get("reward", 0.0)
                        done = step_res.get("done", False)
                        tool_res = new_obs.get("tool_result") or {}
                        status = tool_res.get("status", "unknown")
                        obs_lines = tool_res.get("observations", [])

                        total_reward += reward
                        r_color = _reward_color(reward)
                        bar = "█" * int(reward * 10) + "░" * (10 - int(reward * 10))
                        result_preview = "\n".join(f"> {ln}" for ln in obs_lines[:4]) if obs_lines else "> Tool executed"

                        messages.append({
                            "role": "assistant",
                            "content": (
                                f"{'✅' if status == 'success' else '⚠️'} **{status.upper()}**\n\n"
                                f"{result_preview}\n\n"
                                f"**Reward:** `{reward:.3f}` `[{bar}]`"
                            ),
                        })

                        # Sidebar stats
                        stats_html = f"""
                        <div style="background:#1e293b;border:1px solid #334155;
                                    border-radius:12px;padding:16px;margin-top:8px">
                            <div style="color:#475569;font-size:10px;text-transform:uppercase;
                                        letter-spacing:0.1em;margin-bottom:12px">Live Stats</div>
                            <div style="display:flex;justify-content:space-between;margin-bottom:8px">
                                <span style="color:#94a3b8;font-size:12px">Step</span>
                                <span style="color:#e2e8f0;font-weight:600">{i + 1} / {len(steps)}</span>
                            </div>
                            <div style="margin-bottom:8px">
                                <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                                    <span style="color:#94a3b8;font-size:12px">Last Reward</span>
                                    <span style="color:{r_color};font-weight:600">{reward:.3f}</span>
                                </div>
                                {_reward_bar_html(reward)}
                            </div>
                            <div style="display:flex;justify-content:space-between">
                                <span style="color:#94a3b8;font-size:12px">Cumulative</span>
                                <span style="color:#60a5fa;font-weight:600">{total_reward:.3f}</span>
                            </div>
                        </div>
                        """
                        yield messages, "", stats_html

                        if done:
                            break
                        time.sleep(speed)

                    # Final grade
                    time.sleep(speed * 0.6)
                    grade_res = api_call("/grader", "POST", {})

                    if "error" not in grade_res:
                        score = grade_res.get("score", 0.0)
                        breakdown = grade_res.get("breakdown", {})
                        steps_taken = grade_res.get("steps_taken", len(steps))
                        grade_char = "A" if score > 0.8 else ("B" if score > 0.6 else ("C" if score > 0.4 else "D"))
                        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))

                        bd_lines = "\n".join(
                            f"- **{k.replace('_',' ').title()}**: `{float(v):.3f}`"
                            for k, v in breakdown.items()
                        )
                        messages.append({
                            "role": "assistant",
                            "content": (
                                f"## 🏆 Episode Complete!\n\n"
                                f"**Final Score:** `{score:.4f}` — Grade **{grade_char}**\n\n"
                                f"`[{bar}]`\n\n"
                                f"**Steps Taken:** {steps_taken}\n\n"
                                f"**Score Breakdown:**\n{bd_lines}"
                            ),
                        })
                        yield messages, _score_html(score), ""
                    else:
                        messages.append({
                            "role": "assistant",
                            "content": "⚠️ Could not retrieve final score from grader.",
                        })
                        yield messages, "", ""

                demo_click_event = demo_start_btn.click(
                    run_demo,
                    inputs=[demo_task_dd, demo_seed_tb, demo_diff_sl, demo_speed_sl],
                    outputs=[demo_chatbot, demo_score_html, demo_stats_html],
                )
                demo_stop_btn.click(None, None, None, cancels=[demo_click_event])

            # ══════════════════════════════════════════════════════════════════
            # Tab 2 — Interactive Playground
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("🎮 Playground", id="playground"):
                with gr.Row():
                    # ── Left column: controls ─────────────────────────────────
                    with gr.Column(scale=1, min_width=280):
                        gr.HTML("""<div style="color:#64748b;font-size:11px;text-transform:uppercase;
                                letter-spacing:0.1em;margin-bottom:8px">Episode Setup</div>""")

                        pg_task_dd = gr.Dropdown(
                            choices=task_choices,
                            value="access_decision",
                            label="Task",
                            interactive=True,
                        )
                        with gr.Row():
                            pg_seed_tb = gr.Textbox(value="42", label="Seed", placeholder="42", scale=2)
                            pg_diff_sl = gr.Slider(minimum=1, maximum=3, value=3, step=1,
                                                   label="Difficulty", scale=1)
                        pg_reset_btn = gr.Button("🔄 Reset Episode", variant="primary")
                        pg_status_html = gr.HTML("""
                        <div style="color:#475569;font-size:13px;padding:8px 0">
                            Click Reset to begin a new episode
                        </div>
                        """)

                        gr.HTML("""<div style="border-top:1px solid #1e293b;margin:12px 0"></div>""")
                        gr.HTML("""<div style="color:#64748b;font-size:11px;text-transform:uppercase;
                                letter-spacing:0.1em;margin-bottom:8px">Tool Call</div>""")

                        pg_tool_dd = gr.Dropdown(
                            choices=TASK_INFO["access_decision"]["tools"],
                            value="request.view",
                            label="Tool Name",
                            allow_custom_value=True,
                            interactive=True,
                        )
                        pg_args_tb = gr.Code(
                            value="{}",
                            label="Arguments (JSON)",
                            language="json",
                            lines=5,
                        )
                        pg_step_btn = gr.Button("▶  Execute Step", variant="primary")
                        pg_reward_html = gr.HTML("")

                        gr.HTML("""<div style="border-top:1px solid #1e293b;margin:12px 0"></div>""")
                        pg_grade_btn = gr.Button("📊 Grade Episode", variant="secondary")
                        pg_score_html = gr.HTML("")
                        pg_breakdown_html = gr.HTML("")

                    # ── Right column: observation + history ───────────────────
                    with gr.Column(scale=2):
                        pg_goal_html = gr.HTML("""
                        <div style="background:#1e293b;border:1px solid #334155;border-radius:14px;
                                    padding:22px;margin-bottom:12px;text-align:center">
                            <div style="font-size:40px;margin-bottom:8px">🛡️</div>
                            <div style="color:#475569;font-size:14px">
                                Select a task and click <strong style="color:#94a3b8">Reset Episode</strong>
                                to begin
                            </div>
                        </div>
                        """)
                        pg_obs_html = gr.HTML("")
                        pg_history_html = gr.HTML(_build_history_html([]))

                # Hidden state
                _pg_history = gr.State([])
                _pg_obs = gr.State({})

                # ── Playground callbacks ──────────────────────────────────────

                def pg_reset(task_id: str, seed_str: str, difficulty: float):
                    seed = int(seed_str) if seed_str and seed_str.strip().isdigit() else None
                    result = api_call("/reset", "POST", {
                        "task_id": task_id,
                        "seed": seed,
                        "difficulty_level": int(difficulty),
                    })

                    if "error" in result:
                        err = f'<div style="color:#f87171;padding:8px 0;font-size:13px">❌ {result["error"]}</div>'
                        return err, "", "", "", {}, [], gr.update()

                    obs = result.get("observation", {})
                    tools = obs.get("available_tools", TASK_INFO.get(task_id, {}).get("tools", []))

                    status = """<div style="color:#34d399;font-size:13px;padding:8px 0">
                        ✅ Episode started — make your first tool call
                    </div>"""

                    return (
                        status,
                        _format_goal_html(obs),
                        _format_obs_panels(obs),
                        _build_history_html([]),
                        obs,
                        [],
                        gr.update(choices=tools, value=tools[0] if tools else "request.view"),
                    )

                pg_reset_btn.click(
                    pg_reset,
                    inputs=[pg_task_dd, pg_seed_tb, pg_diff_sl],
                    outputs=[
                        pg_status_html, pg_goal_html, pg_obs_html,
                        pg_history_html, _pg_obs, _pg_history, pg_tool_dd,
                    ],
                )

                def pg_step(tool_name: str, args_json: str, history: list, current_obs: dict):
                    if not current_obs:
                        no_ep = '<div style="color:#f87171;font-size:13px">⚠️ Reset an episode first</div>'
                        return history, no_ep, "", current_obs, _build_history_html(history)

                    try:
                        args = json.loads(args_json or "{}")
                    except json.JSONDecodeError as e:
                        err = f'<div style="color:#f87171;font-size:13px">❌ Invalid JSON: {e}</div>'
                        return history, err, "", current_obs, _build_history_html(history)

                    result = api_call("/step", "POST", {"action": {"tool_name": tool_name, "arguments": args}})

                    if "error" in result:
                        err = f'<div style="color:#f87171;font-size:13px">❌ {result["error"]}</div>'
                        return history, err, "", current_obs, _build_history_html(history)

                    new_obs = result.get("observation", {})
                    reward = result.get("reward", 0.0)
                    done = result.get("done", False)
                    tool_res = new_obs.get("tool_result") or {}
                    status = tool_res.get("status", "unknown")
                    obs_lines = tool_res.get("observations", [])

                    step_num = len(history) + 1
                    new_history = history + [{
                        "step": step_num,
                        "tool": tool_name,
                        "args": args,
                        "reward": reward,
                        "status": status,
                        "obs_lines": obs_lines,
                        "done": done,
                    }]

                    done_note = ""
                    if done:
                        done_note = """<div style="color:#34d399;font-size:12px;margin-top:6px">
                            ✅ Episode done — click Grade Episode for your score!
                        </div>"""

                    reward_display = f"""
                    <div style="padding:8px 0">
                        <div style="color:#475569;font-size:10px;text-transform:uppercase;
                                    letter-spacing:0.08em;margin-bottom:6px">Step {step_num} Reward</div>
                        {_reward_bar_html(reward)}
                        {done_note}
                    </div>
                    """

                    return (
                        new_history,
                        reward_display,
                        _format_obs_panels(new_obs),
                        new_obs,
                        _build_history_html(new_history),
                    )

                pg_step_btn.click(
                    pg_step,
                    inputs=[pg_tool_dd, pg_args_tb, _pg_history, _pg_obs],
                    outputs=[_pg_history, pg_reward_html, pg_obs_html, _pg_obs, pg_history_html],
                )

                def pg_grade():
                    result = api_call("/grader", "POST", {})
                    if "error" in result:
                        return (
                            f'<div style="color:#f87171;font-size:13px">❌ {result["error"]}</div>',
                            "",
                        )
                    score = result.get("score", 0.0)
                    breakdown = result.get("breakdown", {})
                    return _score_html(score), _breakdown_html(breakdown)

                pg_grade_btn.click(pg_grade, outputs=[pg_score_html, pg_breakdown_html])

                def pg_update_tools(task_id: str):
                    tools = TASK_INFO.get(task_id, {}).get("tools", [])
                    return gr.update(choices=tools, value=tools[0] if tools else "")

                pg_task_dd.change(pg_update_tools, inputs=pg_task_dd, outputs=pg_tool_dd)

            # ══════════════════════════════════════════════════════════════════
            # Tab 3 — Task Reference
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("📋 Task Reference", id="reference"):
                gr.HTML("""
                <div style="margin-bottom:20px">
                    <h2 style="color:#e2e8f0;margin:0 0 4px;font-size:20px;font-weight:700">
                        Task Reference
                    </h2>
                    <p style="color:#64748b;margin:0;font-size:13px">
                        Five IAM challenge families — from access decisions to full compliance audits
                    </p>
                </div>
                """)
                for tid, info in TASK_INFO.items():
                    gr.HTML(_task_card_html(tid, info))

            # ══════════════════════════════════════════════════════════════════
            # Tab 4 — Connect
            # ══════════════════════════════════════════════════════════════════
            with gr.Tab("🔗 Connect", id="connect"):
                gr.Markdown("""
## Connect Your Agent

Use the Python client or HTTP API to connect your AI agent to PrivilegeDesk.

---

### Python Client (OpenEnv)

```python
from privilege_desk import PrivilegeDeskAction, PrivilegeDeskEnv

# Connect via HuggingFace Space ID
with PrivilegeDeskEnv.from_env("Krooz/privilege_desk") as env:
    obs = await env.reset(task_id="access_decision", seed=42)

    # Execute a tool call
    result = await env.step(PrivilegeDeskAction(
        tool_name="request.view",
        arguments={},
    ))

    # Make a decision
    result = await env.step(PrivilegeDeskAction(
        tool_name="access.decide",
        arguments={
            "decision": "approve",
            "role": "viewer",
            "ttl_hours": 8,
            "justification_category": "operational",
        },
    ))
```

---

### Direct HTTP API

```python
import requests

BASE = "http://localhost:8000"

# 1. Reset episode
obs = requests.post(f"{BASE}/reset", json={
    "task_id": "access_decision",
    "seed": 42,
}).json()["observation"]

# 2. Execute tool calls
for _ in range(obs["max_steps"]):
    result = requests.post(f"{BASE}/step", json={
        "action": {
            "tool_name": "access.decide",
            "arguments": {
                "decision": "approve",
                "role": "viewer",
                "ttl_hours": 8,
                "justification_category": "operational",
            },
        }
    }).json()

    if result["done"]:
        break

# 3. Get final score
score = requests.post(f"{BASE}/grader", json={}).json()
print(f"Score: {score['score']:.4f}")
print(f"Breakdown: {score['breakdown']}")
```

---

### Available Tasks

| Task ID | Difficulty | Max Steps | Key Tools |
|---------|-----------|-----------|-----------|
| `access_decision` | 🟢 Easy | 5 | `request.view`, `access.decide` |
| `jit_escalation` | 🟡 Medium | 15 | `approval.route`, `access.grant` |
| `emergency_breakglass` | 🟡 Medium | 10 | `incident.verify`, `audit.flag` |
| `access_review` | 🔴 Hard | 25 | `entitlement.revoke`, `review.submit` |
| `separation_of_duties_audit` | 🔴 Hard | 25 | `sod.check_user`, `sod.submit_report` |

---

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Execute a tool call |
| `POST` | `/grader` | Get episode score |
| `GET` | `/tasks` | List all tasks |
| `GET` | `/state` | Current world state |
| `GET` | `/docs` | OpenAPI documentation |
                """)

        # ── Footer ────────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="border-top:1px solid #1e293b;margin-top:24px;padding-top:16px;
                    display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
            <div style="color:#334155;font-size:12px">
                🛡️ <strong style="color:#475569">PrivilegeDesk</strong> —
                Zero-Standing-Privilege AI Agent Training Environment
            </div>
            <div style="display:flex;gap:16px">
                <a href="/docs" style="color:#475569;font-size:12px;text-decoration:none">
                    API Docs
                </a>
                <a href="/tasks" style="color:#475569;font-size:12px;text-decoration:none">
                    Tasks JSON
                </a>
                <span style="color:#334155;font-size:12px">Built with OpenEnv</span>
            </div>
        </div>
        """)

    return demo
