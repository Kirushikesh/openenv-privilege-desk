---
title: Privilege Desk
emoji: 🏢
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
---

# PrivilegeDesk

**Zero-Standing-Privilege Ops Environment for Training AI Agents**

A simulated enterprise access-control environment built on [OpenEnv](https://github.com/meta-pytorch/OpenEnv).  
An AI agent is dropped into a synthetic corporate IAM system and must handle real-world privilege management tasks — from reviewing a single access request to conducting a full entitlement audit.

---

## Environment Overview

| Property | Value |
|----------|-------|
| **Domain** | Identity & Access Management (IAM) |
| **Tasks** | 3 (easy → medium → hard) |
| **Tools** | 19 structured tool calls |
| **Reward** | Partial credit, 0.0–1.0, deterministic grading |
| **Generation** | Fully procedural — unique episode per seed |
| **API** | OpenEnv-compliant (reset / step / state / grader / tasks / baseline) |

---

## The 3 Tasks

### Task 1 — Access Decision (`access_decision`) · Easy

> "A new access request has arrived. Review the request, inspect the applicable policy, and decide whether to approve or deny it. If approving, select the correct role and TTL."

- **Steps**: 1–5
- **Grading**: correct decision (40%) + correct role (25%) + correct TTL (20%) + justification category (15%)
- **Baseline difficulty**: ~0.7–0.9 with a well-prompted LLM

### Task 2 — JIT Escalation (`jit_escalation`) · Medium

> "Process an urgent just-in-time privilege escalation: find the correct approval chain, route the request in order, attach the incident ticket, set the TTL, and activate the grant."

- **Steps**: 3–15
- **Grading**: correct approvers (20%) + routing order (15%) + ticket attached (15%) + role (15%) + TTL (15%) + final decision (20%)
- **Baseline difficulty**: ~0.4–0.6 with a well-prompted LLM

### Task 3 — Access Review (`access_review`) · Hard

> "Conduct an access review for a user. Identify risky, stale, or over-privileged entitlements and revoke the minimum set — without breaking active workflows."

- **Steps**: 5–25
- **Grading**: precision (30%) + recall (30%) + workflow preservation (20%) + policy compliance (10%) + submission (10%)
- **Baseline difficulty**: ~0.2–0.4 with a well-prompted LLM

---

## The World Model

Each episode procedurally generates a complete synthetic enterprise from a seed:

| Entity | Description |
|--------|-------------|
| **Users** | Employees with name, email, department, manager |
| **Resources** | Databases, repos, cloud projects, admin consoles |
| **Policies** | Rules: max role + max TTL + required approvers per resource |
| **Entitlements** | Current user→role→resource assignments |
| **Groups** | Team groups for inherited permissions (Task 3) |
| **Approval Chains** | Ordered approver lists per request (Task 2) |
| **Workflows** | Active pipelines that depend on entitlements (Task 3) |
| **Audit DB** | Historical access events queryable via `audit.query` |
| **Hidden State** | Ground truth — generated at reset, used by grader |

---

## API Reference

### Standard OpenEnv Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start a new episode. Pass `task_id` and optional `seed`. |
| `/step` | POST | Execute a tool call. Returns observation + reward + done. |
| `/state` | GET | Current episode metadata (episode_id, step_count). |
| `/schema` | GET | JSON schemas for Action and Observation. |
| `/health` | GET | Health check. |

### Hackathon-Required Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tasks` | GET | Lists all 3 tasks with schemas and grading weights. |
| `/grader` | POST | Returns 0.0–1.0 episode score with full breakdown. |
| `/baseline` | POST | Runs naive baseline agent on all 3 tasks. |

### Reset Parameters

```json
{
  "task_id": "access_decision",   // "access_decision" | "jit_escalation" | "access_review"
  "seed": 42,                     // optional — omit for random episode
  "difficulty_level": 1           // 1-3 (scales entity counts)
}
```

### Step Action Format

```json
{
  "action": {
    "tool_name": "access.decide",
    "arguments": {
      "request_id": "req_000",
      "decision": "approve",
      "role": "viewer",
      "ttl_hours": 4,
      "justification_category": "operational"
    }
  }
}
```

---

## Available Tools (19 total)

| Tool | Arguments | Used In |
|------|-----------|---------|
| `policy.lookup` | `resource_id` | T1, T2 |
| `policy.list` | — | T1, T2, T3 |
| `org.get_user` | `user_id` | T1, T2, T3 |
| `org.get_manager` | `user_id` | T2 |
| `org.list_users` | `department?` | T2, T3 |
| `request.view` | `request_id?` | T1, T2 |
| `request.list` | — | T2 |
| `approval.route` | `request_id`, `approver_id` | T2 |
| `approval.check_status` | `request_id` | T2 |
| `access.decide` | `request_id`, `decision`, `role`, `ttl_hours`, `justification_category?` | T1 |
| `access.grant` | `request_id` | T2 |
| `access.set_ttl` | `request_id`, `ttl_hours` | T2 |
| `entitlement.list` | `user_id?` | T1, T3 |
| `entitlement.inspect` | `entitlement_id` | T3 |
| `entitlement.revoke` | `entitlement_id`, `reason?` | T3 |
| `audit.query` | `user_id?`, `resource_id?`, `days?` | T3 |
| `group.resolve` | `group_id?` or `user_id?` | T3 |
| `workflow.check_active` | `user_id?`, `entitlement_id?` | T3 |
| `review.submit` | `summary?` | T3 |

---

## Reward Design

### Per-Step Rewards (returned on every `/step`)

- `+0.02–0.05` for discovering new information (policies, entitlements, audit logs)
- `+0.05–0.08` for correctly identifying or revoking a risky entitlement
- `-0.02` for tool errors
- `-0.01` for redundant (repeated) tool calls
- `-0.03` for routing to the wrong approver
- `-0.10` for revoking a workflow-critical entitlement

### Episode Score (returned when `done=True`, accessible via `/grader`)

All task graders return a weighted 0.0–1.0 score with **partial credit**. Binary 0/1 is never used — every field is independently gradable.

---

## Local Development

```bash
# Install dependencies
cd privilege_desk
uv sync

# Run server (hot-reload)
uv run server
# OR
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Test episode generation
uv run python -c "
import sys; sys.path.insert(0, '.')
from pipeline.episode_generator import generate_episode
ws = generate_episode(task_id='access_decision', seed=42)
print('Users:', len(ws['users']), 'Policies:', len(ws['policies']))
"

# Run baseline inference (requires API key)
export API_BASE_URL=https://router.huggingface.co/v1
export HF_TOKEN=hf_...
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
uv run python inference.py

# Validate submission
openenv validate
```

## Docker

```bash
# Build
docker build -t privilege-desk:latest -f server/Dockerfile .

# Run
docker run -p 8000:8000 privilege-desk:latest

# Health check
curl http://localhost:8000/health
```

---

## Architecture

```
privilege_desk/
├── models.py                    # Pydantic Action + Observation
├── pipeline/
│   ├── episode_generator.py     # Procedural world generation
│   └── task_templates.py        # 3 task definitions
├── env/
│   ├── world_state.py           # Episode lifecycle (reset/step/grade)
│   ├── action_router.py         # Tool dispatch + audit log
│   └── tools.py                 # All 19 tool implementations
├── reward/
│   ├── grader.py                # Per-task graders (0.0–1.0)
│   └── aggregator.py            # Step reward + episode score
├── server/
│   ├── app.py                   # FastAPI app (standard + custom endpoints)
│   └── privilege_desk_environment.py  # OpenEnv Environment subclass
├── inference.py                 # Baseline LLM agent
├── openenv.yaml                 # OpenEnv spec declaration
└── Dockerfile                   # HF Spaces deployment
```

---

## Grading Details

### Task 1: Access Decision
| Component | Weight | What's Checked |
|-----------|--------|----------------|
| Correct approve/deny | 40% | Matches policy: requested role ≤ max_role |
| Correct role | 25% | Exact match to policy max_role (partial credit for lower roles) |
| Correct TTL | 20% | Within ±2h of policy max_ttl |
| Justification category | 15% | Correct category label |

### Task 2: JIT Escalation  
| Component | Weight | What's Checked |
|-----------|--------|----------------|
| Correct approvers | 20% | Set of routed approver IDs matches required chain |
| Routing order | 15% | Sequential order matches policy chain |
| Ticket attached | 15% | Non-empty ticket_id in request |
| Correct role | 15% | Requested role ≤ policy max_role |
| Correct TTL | 15% | Within ±2h of policy max_ttl |
| Final grant/deny | 20% | Correctly activated or denied |

### Task 3: Access Review
| Component | Weight | What's Checked |
|-----------|--------|----------------|
| Precision | 30% | Revoked entitlements that were genuinely risky |
| Recall | 30% | Risky entitlements that were caught |
| Workflow preservation | 20% | No active workflows broken |
| Policy compliance | 10% | Remaining entitlements pass policy check |
| Submission | 10% | `review.submit` was called |

---

## License

BSD-style — see LICENSE file.
