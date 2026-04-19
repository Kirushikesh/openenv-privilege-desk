"""
Data models for PrivilegeDesk — Zero-Standing-Privilege Ops Environment.
"""
from typing import Any, Dict, List, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class PrivilegeDeskAction(Action):
    """Action for PrivilegeDesk — a structured tool call."""

    tool_name: str = Field(
        ...,
        description=(
            "Name of the tool to call e.g. 'policy.lookup', 'access.decide', "
            "'entitlement.revoke'. Check available_tools in the observation."
        ),
    )
    arguments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool-specific arguments as key-value pairs",
    )


class PrivilegeDeskObservation(Observation):
    """Observation from PrivilegeDesk — the agent's partial view of the IAM world."""

    # Task context
    task_id: str = Field(default="", description="Active task identifier")
    task_goal: str = Field(default="", description="Natural language description of the goal")
    step: int = Field(default=0, description="Current step number")
    max_steps: int = Field(default=25, description="Maximum steps before truncation")
    current_time: str = Field(default="", description="Simulated current time (ISO)")
    available_tools: List[str] = Field(default_factory=list, description="Tools available for this task")

    # Org & resources
    users: Dict[str, Any] = Field(default_factory=dict, description="Users in the org")
    org_graph: Dict[str, Any] = Field(default_factory=dict, description="Manager hierarchy")
    resources: Dict[str, Any] = Field(default_factory=dict, description="Resources (databases, repos, etc.)")
    policies: Dict[str, Any] = Field(default_factory=dict, description="Access policies")
    groups: Dict[str, Any] = Field(default_factory=dict, description="User groups")

    # Access state
    entitlements: Dict[str, Any] = Field(default_factory=dict, description="Current entitlements (sanitized)")
    pending_requests: Dict[str, Any] = Field(default_factory=dict, description="Pending access requests")
    approval_chains: Dict[str, Any] = Field(default_factory=dict, description="Approval chain state")
    workflows: Dict[str, Any] = Field(default_factory=dict, description="Active workflows")

    # Task-specific entities
    incidents: Dict[str, Any] = Field(default_factory=dict, description="Production incidents (Task 4)")
    conflict_matrix: Dict[str, Any] = Field(default_factory=dict, description="SoD conflict matrix (Task 5)")
    compensating_controls: Dict[str, Any] = Field(default_factory=dict, description="SoD compensating controls (Task 5)")

    # Multi-agent oversight fields (Task 6)
    sub_agents: Dict[str, Any] = Field(default_factory=dict, description="Autonomous sub-agent identities (Task 6)")
    identity_graph: Dict[str, Any] = Field(default_factory=dict, description="Sanitized identity DAG — nodes and edges (Task 6)")
    rogue_agent_requests: Dict[str, Any] = Field(default_factory=dict, description="Pending requests from the rogue sub-agent (Task 6)")

    # Objectives & last action
    objectives: List[Dict[str, Any]] = Field(default_factory=list, description="Task subgoals")
    audit_log: List[Dict[str, Any]] = Field(default_factory=list, description="Last 5 actions taken")
    notifications: List[Dict[str, Any]] = Field(default_factory=list, description="System notifications")

    # Review task
    review_target_user_id: Optional[str] = Field(default=None, description="User to review (access_review task)")

    # Last tool result
    tool_result: Optional[Dict[str, Any]] = Field(default=None, description="Result of the last tool call")
