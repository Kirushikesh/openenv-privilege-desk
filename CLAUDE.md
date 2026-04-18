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

- **`pipeline/task_templates.py`** — Defines the 3 task families with entity counts, subgoals, and grading weights

- **`server/app.py`** — FastAPI app with stateful singleton `WorldState`. Key endpoints: `POST /reset`, `POST /step`, `GET /state`, `GET /schema`, `GET /tasks`, `POST /grader`, `POST /baseline`

- **`server/privilege_desk_environment.py`** — OpenEnv-compatible `Environment` interface wrapper managing `WorldState` lifecycle

- **`reward/grader.py`** — Task-specific graders. All final scores are clamped to `(0.01, 0.99)`:
  - Task 1 `access_decision`: 40% decision correctness, 25% role, 20% TTL, 15% justification
  - Task 2 `jit_escalation`: 20% approvers, 15% order, 15% ticket, 15% role, 15% TTL, 20% final decision
  - Task 3 `access_review`: 30% precision, 30% recall, 20% workflow preservation, 10% compliance, 10% submission

- **`reward/aggregator.py`** — Per-step reward signals (base rewards per tool type, bonuses for correct identification/revocation, penalties for errors/redundancy, floor 0.01)

- **`graders.py`** (root) — OpenEnv judge entry points; thin wrappers delegating to `reward/grader.py`

- **`client.py`** — `PrivilegeDeskEnv` WebSocket client for agents connecting to the server

- **`inference.py`** — Baseline LLM agent using OpenAI-compatible API; emits structured stdout logs for hackathon judge

### 19 Agent Tools (organized by domain)

Policy, Org, Request, Approval, Access, Entitlement, Audit, Group, Workflow, Review — all invoked via `POST /step` with `{"tool_name": "...", "arguments": {...}}`.

### Design Decisions

- **Partial observability**: Agent sees only sanitized world state; ground truth for grading is hidden inside `WorldState`
- **Singleton state**: A single `WorldState` is shared across `/reset`, `/step`, `/grader` to enable stateless curl-based testing
- **Deterministic seeding**: Episode generation is fully reproducible given the same seed + task_id

## Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `API_BASE_URL` | LLM API endpoint for baseline | HuggingFace router |
| `HF_TOKEN` | HuggingFace API token | — |
| `MODEL_NAME` | LLM model for baseline | `meta-llama/Llama-3.3-70B-Instruct` |
