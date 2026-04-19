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
| **Tasks** | 6 (easy → very hard, including adversarial multi-agent oversight) |
| **Tools** | 27 structured tool calls |
| **Reward** | 4-component: episode score + step efficiency + format + H* efficiency |
| **Generation** | Fully procedural — unique episode per seed |
| **API** | OpenEnv-compliant (reset / step / state / grader / tasks / baseline) |
| **Training** | GRPO + LoRA, 4-phase curriculum via `training/train_grpo.py` |

---

## The 5 Tasks

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

### Task 4 — Emergency Break-Glass (`emergency_breakglass`) · Medium

> "A production incident is active. An on-call engineer needs emergency access. Verify the incident, check policies, flag the security team, and execute a time-limited break-glass grant if valid."

- **Steps**: 4–10
- **Grading**: incident verify (15%) + role (15%) + TTL (20%) + ticket attached (15%) + security flag (15%) + final grant (20%)
- **Baseline difficulty**: ~0.5–0.7 with a well-prompted LLM

### Task 5 — SoD Audit (`separation_of_duties_audit`) · Hard

> "Detect Separation of Duties (SoD) violations. Cross-reference users against a conflict matrix, check for compensating controls, and revoke conflicting entitlements to ensure compliance."

- **Steps**: 10–25
- **Grading**: violations found (30%) + false positives (15%) + correct revocations (25%) + controls recognized (10%) + submission (20%)
- **Baseline difficulty**: ~0.2–0.3 with a well-prompted LLM

### Task 6 — Multi-Agent Oversight (`multi_agent_oversight`) · Very Hard

> "A rogue autonomous agent in your infrastructure has been accumulating permissions. A new request has arrived that — if approved — will complete a toxic permission combination enabling data exfiltration. Investigate the sub-agents, discover the hidden toxic path, and neutralise the threat."

- **Steps**: 10–25 (optimal H* = 12)
- **Adversarial**: rogue agent identity and toxic permission graph are hidden — agent must use `sod.check_user` and `sod.get_conflict_matrix` to discover the threat
- **Reward signals**: +0.25 for correctly blocking the toxic request, −0.40 for approving it (catastrophic), +0.20 for quarantine, +0.15 for audit flag
- **Baseline difficulty**: ~0.1–0.2 (requires multi-hop reasoning across identity graph)

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
| `/tasks` | GET | Lists all 5 tasks with schemas and grading weights. |
| `/grader` | POST | Returns 0.0–1.0 episode score with full breakdown. |
| `/baseline` | POST | Runs naive baseline agent on all 5 tasks. |

### Reset Parameters

```json
{
  "task_id": "access_decision",   // access_decision | jit_escalation | access_review | emergency_breakglass | separation_of_duties_audit | multi_agent_oversight
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
| `ticket.attach` | `request_id`, `ticket_id` | T2, T4 |
| `access.decide` | `request_id`, `decision`, `role`, `ttl_hours`, `justification_category?` | T1 |
| `access.grant` | `request_id`, `role?` | T2, T4 |
| `access.deny` | `request_id` | T2, T4 |
| `access.set_ttl` | `request_id`, `ttl_hours` | T2, T4 |
| `entitlement.list` | `user_id?` | T1, T3, T5 |
| `entitlement.inspect` | `entitlement_id` | T3, T5 |
| `entitlement.revoke` | `entitlement_id`, `reason?` | T3, T5 |
| `audit.query` | `user_id?`, `resource_id?`, `days?` | T3 |
| `audit.flag` | `incident_id`, `flag_type` | T4 |
| `group.resolve` | `group_id?` or `user_id?` | T3 |
| `workflow.check_active` | `user_id?`, `entitlement_id?` | T3 |
| `review.submit` | `summary?` | T3 |
| `incident.verify` | `incident_id` | T4 |
| `sod.get_conflict_matrix` | — | T5 |
| `sod.check_user` | `user_id` | T5 |
| `sod.get_compensating_controls` | `user_id`, `conflict_id?` | T5 |
| `sod.submit_report` | `summary?` | T5 |

---

## Reward Design

### Per-Step Rewards (returned on every `/step`)

| Signal | Value | Notes |
|--------|-------|-------|
| Tool error | −0.20 | Tool returned error status |
| Redundant call | −0.10 | Identical tool + args as previous step |
| `request.list` | 0.00 | `pending_requests` already in observation |
| Low-value listing | +0.05 | `policy.list`, `org.list_users` |
| Medium investigation | +0.10 | `request.view`, `audit.query`, `entitlement.list` |
| High-value investigation | +0.15 | `policy.lookup`, `entitlement.inspect`, `sod.check_user` |
| Correct approver routing | +0.30 | Wrong routing: −0.20 |
| Correct revocation | +0.35 | Critical revocation: −0.40; non-risky: −0.10 |
| Block toxic request (Task 6) | +0.25 | Approving toxic request: −0.40 (catastrophic) |
| Quarantine rogue agent (Task 6) | +0.20 | Via `emergency_breakglass` |
| Audit flag rogue agent (Task 6) | +0.15 | Via `audit.flag` |

### 4-Component GRPO Reward

| Component | Signal | Purpose |
|-----------|--------|---------|
| `episode_score` | 0–1 grader score | Primary correctness signal |
| `step_efficiency` | Mean per-step reward | Exploration quality |
| `format` | +0.10 if `<think>` used ≥50% of steps | Encourage chain-of-thought |
| `efficiency` | +0.10 if steps ≤ H*, else 0.10 × 0.85^excess | Penalise meandering |

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
│   ├── task_templates.py        # 6 task definitions
│   └── toxic_graph.py           # Adversarial DAG generator (Task 6)
├── env/
│   ├── world_state.py           # Episode lifecycle (reset/step/grade)
│   ├── action_router.py         # Tool dispatch + audit log
│   └── tools.py                 # All tool implementations
├── reward/
│   ├── grader.py                # Per-task graders (0.0–1.0)
│   └── aggregator.py            # Step reward + episode score + oversight adjustments
├── training/
│   └── train_grpo.py            # GRPO + LoRA 4-phase curriculum training
├── server/
│   ├── app.py                   # FastAPI app (standard + custom endpoints)
│   └── privilege_desk_environment.py  # OpenEnv Environment subclass
├── inference.py                 # Baseline LLM agent
├── openenv.yaml                 # OpenEnv spec declaration
└── Dockerfile                   # HF Spaces deployment
```

## Training

The GRPO curriculum trains agents in 4 phases, each carrying the LoRA adapter forward:

| Phase | Tasks | Weight Distribution |
|-------|-------|---------------------|
| 1 — Stabilise | `access_decision` | 100% |
| 2 — Expand | + `emergency_breakglass` | 60% / 40% |
| 3 — Full IAM | All 5 tasks | Stratified |
| 4 — Adversarial | All 5 + `multi_agent_oversight` | 10% / 10% / 10% / 10% / 10% / **50%** |

```bash
# Run Phase 4 (adversarial oversight) from a Phase 3 checkpoint:
python training/train_grpo.py \
  --model-id ./outputs/grpo_run1/phase3/adapter \
  --phase 4 --episodes-per-phase 32 \
  --output-dir ./outputs/grpo_run1

# Full curriculum from scratch:
python training/train_grpo.py \
  --model-id "Qwen/Qwen3.5-2B" \
  --use-vllm --vllm-mode colocate \
  --output-dir ./outputs/grpo_run1
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
 
### Task 4: Emergency Break-Glass
| Component | Weight | What's Checked |
|-----------|--------|----------------|
| Incident valid | 15% | Agent verified incident is P1/P2 and active |
| Correct role | 15% | Requested role ≤ policy breakglass max_role |
| Correct TTL | 20% | TTL ≤ policy breakglass_max_ttl_hours |
| Ticket attached | 15% | Incident ID attached correctly |
| Security flagged | 15% | `audit.flag` was called |
| Final grant | 20% | Correct decision given incident validity |

### Task 5: SoD Audit
| Component | Weight | What's Checked |
|-----------|--------|----------------|
| Violations found | 30% | Recall of true violations planted in world |
| False positives | 15% | Precision: didn't flag mediated violations |
| Correct revocations | 25% | Revoked lower-risk entitlement of conflicting pair |
| Controls recognized | 10% | Checked for active compensating controls |
| Report submitted | 20% | `sod.submit_report` was called |

### Task 6: Multi-Agent Oversight
Graded via per-step bonuses/penalties in `reward/aggregator.py` (no separate episode grader):

| Action | Reward |
|--------|--------|
| Block toxic request (`access.decide` / `access.grant` with deny) | +0.25 |
| Approve toxic request | −0.40 (catastrophic) |
| Quarantine rogue agent (`emergency_breakglass`) | +0.20 |
| Audit flag rogue agent (`audit.flag`) | +0.15 |
| Investigate rogue via `sod.check_user` | +0.20 |
| Analyse conflict matrix (`sod.get_conflict_matrix`) | +0.10 |

---

## License

BSD-style — see LICENSE file.
