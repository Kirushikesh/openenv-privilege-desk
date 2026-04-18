"""
Action Router — dispatches tool calls to handlers and manages state updates.
"""
from datetime import datetime
from typing import Any, Dict

from .tools import TOOL_REGISTRY, get_available_tools


class ActionRouter:
    """Routes agent actions to tool handlers."""

    def __init__(self, world_state: Dict[str, Any]):
        self.world_state = world_state
        self.step_count = 0

    def dispatch(self, action_dict: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = action_dict.get("tool_name", "")
        arguments = action_dict.get("arguments", {})
        timestamp = action_dict.get("timestamp", datetime.now().isoformat())

        self.step_count += 1

        if not tool_name:
            return self._make_error("No tool_name specified")

        # Check tool is registered
        if tool_name not in TOOL_REGISTRY:
            avail = self.world_state.get("available_tools", get_available_tools())
            return self._make_error(
                f"Unknown tool '{tool_name}'. Available: {avail}"
            )

        # Check tool is available for this task
        task_tools = self.world_state.get("available_tools", [])
        if task_tools and tool_name not in task_tools:
            return self._make_error(
                f"Tool '{tool_name}' is not available for this task. "
                f"Available: {task_tools}"
            )

        # Execute
        handler = TOOL_REGISTRY[tool_name]
        try:
            result = handler(self.world_state, arguments)
        except Exception as e:
            result = self._make_error(f"Tool execution error: {str(e)}")

        # Apply state delta
        state_delta = result.get("state_delta", {})
        if state_delta:
            self._apply_state_delta(state_delta)

        # Log to audit trail
        self._log_action(tool_name, arguments, result, timestamp)

        return result

    def _apply_state_delta(self, state_delta: Dict[str, Any]):
        """Apply dot-notation state mutations."""
        for key, value in state_delta.items():
            if key.startswith("_"):
                self.world_state[key] = value
                continue

            parts = key.split(".")
            target = self.world_state
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]

            final_key = parts[-1]
            if isinstance(value, dict) and isinstance(target.get(final_key), dict):
                target[final_key].update(value)
            else:
                target[final_key] = value

    def _log_action(self, tool_name, arguments, result, timestamp):
        if "audit_log" not in self.world_state:
            self.world_state["audit_log"] = []
        self.world_state["audit_log"].append({
            "step": self.step_count,
            "timestamp": timestamp,
            "tool_name": tool_name,
            "arguments": arguments,
            "status": result.get("status", "unknown"),
            "observations": result.get("observations", []),
        })

        # Track tool usage stats
        cs = self.world_state.setdefault("completion_state", {})
        stats = cs.setdefault("tools_used", {})
        stats[tool_name] = stats.get(tool_name, 0) + 1

    def _make_error(self, message: str) -> Dict[str, Any]:
        return {
            "status": "error",
            "result": {"error": message},
            "observations": [f"ERROR: {message}"],
            "state_delta": {},
        }
