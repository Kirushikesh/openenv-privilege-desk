# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PrivilegeDesk** is a Python FastAPI-based OpenEnv environment simulating an enterprise Identity & Access Management (IAM) system. It trains AI agents on privilege management tasks via a reset/step/grade loop.

## Commands

```bash
# Install dependencies (uv required)
uv sync

# Start server (hot-reload)
uv run server
# or: uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Run baseline agent (requires env vars)
export API_BASE_URL=https://router.huggingface.co/v1
export HF_TOKEN=hf_...
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
uv run python inference.py

# Validate OpenEnv submission
openenv validate

# Docker
docker build -t privilege-desk:latest -f server/Dockerfile .
docker run -p 8000:8000 privilege-desk:latest

# Quick episode generation smoke test
uv run python -c "
from pipeline.episode_generator import generate_episode
ws = generate_episode(task_id='access_decision', seed=42)
print('Users:', len(ws['users']), 'Policies:', len(ws['policies']))
"
```

No dedicated test suite — validation is done via episode generation, baseline inference, and `openenv validate`.

## Architecture

### Core Data Flow

1. **Client** calls `POST /reset` → server generates a new `WorldState` episode via `pipeline/episode_generator.py`
2. **Client** calls `POST /step` with a `PrivilegeDeskAction` (tool_name + arguments) → server executes the tool against `WorldState`, returns a `PrivilegeDeskObservation`
3. **Client** calls `POST /grader` → grader compares agent's submitted answers against hidden ground truth in `WorldState`

### Key Modules

- **`models.py`** — Pydantic data models: `PrivilegeDeskAction` (tool call), `PrivilegeDeskObservation` (agent's partial world view including task context, org, resources, policies, entitlements, workflows)

- **`pipeline/episode_generator.py`** — Procedurally generates deterministic episodes from a seed: builds org chart, resources, policies, entitlements (with risky/stale grants for Task 3), groups, approval chains, and workflows

- **`pipeline/task_templates.py`** — Defines the 6 task families with entity counts, subgoals, and grading weights

- **`pipeline/toxic_graph.py`** — Generates adversarial DAG (identity graph) for Task 6: plants a rogue autonomous agent with hidden toxic permission combinations (internal data read + external write) that the overseer agent must discover and block

- **`server/app.py`** — FastAPI app with stateful singleton `WorldState`. Key endpoints: `POST /reset`, `POST /step`, `GET /state`, `GET /schema`, `GET /tasks`, `POST /grader`, `POST /baseline`

- **`server/privilege_desk_environment.py`** — OpenEnv-compatible `Environment` interface wrapper managing `WorldState` lifecycle

- **`reward/grader.py`** — Task-specific graders. All final scores are clamped to `(0.01, 0.99)`:
  - Task 1 `access_decision`: 40% decision correctness, 25% role, 20% TTL, 15% justification
  - Task 2 `jit_escalation`: 20% approvers, 15% order, 15% ticket, 15% role, 15% TTL, 20% final decision
  - Task 3 `access_review`: 30% precision, 30% recall, 20% workflow preservation, 10% compliance, 10% submission
  - Task 4 `emergency_breakglass`: 15% incident verify, 15% role, 20% TTL, 15% ticket, 15% flag, 20% final grant
  - Task 5 `separation_of_duties_audit`: 30% violations found, 15% false positives, 25% revocations, 10% controls, 20% report
  - Task 6 `multi_agent_oversight`: graded via oversight-specific bonuses in `aggregator.py` (correct block +0.25, toxic approval −0.40, quarantine +0.20, audit flag +0.15)

- **`reward/aggregator.py`** — Per-step reward signals (base rewards per tool type, bonuses for correct identification/revocation, penalties for errors/redundancy). Notable: `request.list` = 0.00 (pending_requests already in observation). Task 6 applies `_oversight_adjustment` on top of base rewards.

- **`graders.py`** (root) — OpenEnv judge entry points; thin wrappers delegating to `reward/grader.py`

- **`client.py`** — `PrivilegeDeskEnv` WebSocket client for agents connecting to the server

- **`inference.py`** — Baseline LLM agent using OpenAI-compatible API; emits structured stdout logs for hackathon judge

### 27 Agent Tools (organized by domain)

Policy, Org, Request, Approval, Access, Entitlement, Audit, Group, Workflow, Review, SoD, Emergency — all invoked via `POST /step` with `{"tool_name": "...", "arguments": {...}}`.

Task 6 adds: `emergency_breakglass` (quarantine a rogue agent), `sod.check_user` / `sod.get_conflict_matrix` (discover toxic permission paths).

### Task 6 — Multi-Agent Oversight (`multi_agent_oversight`) · Very Hard

Adversarial scenario: a rogue autonomous agent has been granted partial permissions and submits a request that — if approved — would complete a toxic combination (internal data read + external write access). The overseer agent must:
1. Enumerate sub-agents and their status
2. Use `sod.check_user` and `sod.get_conflict_matrix` to discover the hidden toxic path
3. Deny the rogue agent's request via `access.decide` or `access.grant`
4. Quarantine the rogue agent via `emergency_breakglass`
5. Flag in audit log via `audit.flag`

The rogue agent's identity and toxic path are **not** in the observation — the agent must investigate. The `identity_graph` and request spoiler fields are hidden from the user prompt.

### GRPO Curriculum Training (`training/train_grpo.py`)

4-phase curriculum:

| Phase | Task Mix | Purpose |
|-------|----------|---------|
| 1 | `access_decision` 100% | Stabilise basic IAM reasoning |
| 2 | `access_decision` 60%, `emergency_breakglass` 40% | Introduce urgency and time-limits |
| 3 | All 5 tasks (stratified) | Full IAM competence |
| 4 | All 5 tasks 10% each + `multi_agent_oversight` 50% | Adversarial multi-agent oversight |

**4-component reward:**
- `reward_episode_score` — primary grader signal (0–1)
- `reward_step_efficiency` — mean per-step reward (exploration quality)
- `reward_format` — +0.10 if model uses `<think>` tags on ≥50% of steps
- `reward_efficiency` — +0.10 if steps ≤ H* (optimal horizon), decays as 0.85^excess

**Optimal H* horizons:** access_decision=4, emergency_breakglass=7, jit_escalation=10, access_review=15, sod_audit=15, multi_agent_oversight=12

### Design Decisions

- **Partial observability**: Agent sees only sanitized world state; ground truth for grading is hidden inside `WorldState`
- **Singleton state**: A single `WorldState` is shared across `/reset`, `/step`, `/grader` to enable stateless curl-based testing
- **Deterministic seeding**: Episode generation is fully reproducible given the same seed + task_id
- **Information hiding in prompts**: `identity_graph`, `rogue_agent_requests`, and full DAG structure are excluded from user prompt — agent must use SoD tools to discover toxic paths (preserves exploration reward signal)
- **`request.list` zero reward**: `pending_requests` is already in the observation, so calling `request.list` provides no new information → 0.0 step reward

## Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `API_BASE_URL` | LLM API endpoint for baseline | HuggingFace router |
| `HF_TOKEN` | HuggingFace API token | — |
| `MODEL_NAME` | LLM model for baseline | `meta-llama/Llama-3.3-70B-Instruct` |

## Training Commands

```bash
# Full 4-phase curriculum (vLLM, A100/H100):
python training/train_grpo.py \
  --model-id "Qwen/Qwen3.5-2B" \
  --env-url  "http://localhost:8000" \
  --use-vllm --vllm-mode colocate \
  --episodes-per-phase 32 \
  --output-dir ./outputs/grpo_run1

# Phase 4 only (start from phase3 adapter):
python training/train_grpo.py \
  --model-id ./outputs/grpo_run1/phase3/adapter \
  --phase 4 \
  --episodes-per-phase 32 \
  --output-dir ./outputs/grpo_run1

# Dry-run curriculum plan (no GPU):
python training/train_grpo.py --dry-run
```
