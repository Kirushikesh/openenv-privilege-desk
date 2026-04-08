# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""PrivilegeDesk Environment Client — WebSocket-based client for the IAM ops environment."""

from typing import Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import PrivilegeDeskAction, PrivilegeDeskObservation


class PrivilegeDeskEnv(
    EnvClient[PrivilegeDeskAction, PrivilegeDeskObservation, State]
):
    """
    WebSocket client for the PrivilegeDesk environment.

    Maintains a persistent connection to the environment server so the agent
    can take multiple steps with low latency. Each instance has its own
    isolated session on the server.

    Example — Task 1 (Access Decision):
        >>> with PrivilegeDeskEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset(task_id="access_decision", seed=42)
        ...     obs = result.observation
        ...     print(obs.task_goal)
        ...
        ...     action = PrivilegeDeskAction(
        ...         tool_name="access.decide",
        ...         arguments={"request_id": "req_000", "decision": "approve",
        ...                    "role": "viewer", "ttl_hours": 4}
        ...     )
        ...     result = client.step(action)
        ...     print(result.done, result.reward)

    Example — from Docker:
        >>> client = PrivilegeDeskEnv.from_docker_image("privilege-desk:latest")
        >>> try:
        ...     result = client.reset(task_id="access_review")
        ...     result = client.step(PrivilegeDeskAction(tool_name="entitlement.list", arguments={}))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: PrivilegeDeskAction) -> Dict:
        """Convert PrivilegeDeskAction to the JSON payload for the step message."""
        return {
            "tool_name": action.tool_name,
            "arguments": action.arguments,
        }

    def _parse_result(self, payload: Dict) -> StepResult[PrivilegeDeskObservation]:
        """Parse a server step response into a typed StepResult."""
        obs_data = payload.get("observation", {})

        observation = PrivilegeDeskObservation(
            # Task context
            task_id=obs_data.get("task_id", ""),
            task_goal=obs_data.get("task_goal", ""),
            step=obs_data.get("step", 0),
            max_steps=obs_data.get("max_steps", 25),
            current_time=obs_data.get("current_time", ""),
            available_tools=obs_data.get("available_tools", []),
            # Org & resources
            users=obs_data.get("users", {}),
            org_graph=obs_data.get("org_graph", {}),
            resources=obs_data.get("resources", {}),
            policies=obs_data.get("policies", {}),
            groups=obs_data.get("groups", {}),
            # Access state
            entitlements=obs_data.get("entitlements", {}),
            pending_requests=obs_data.get("pending_requests", {}),
            approval_chains=obs_data.get("approval_chains", {}),
            workflows=obs_data.get("workflows", {}),
            # Objectives & logs
            objectives=obs_data.get("objectives", []),
            audit_log=obs_data.get("audit_log", []),
            notifications=obs_data.get("notifications", []),
            review_target_user_id=obs_data.get("review_target_user_id"),
            tool_result=obs_data.get("tool_result"),
            # OpenEnv base fields
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse a server state response into a State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
