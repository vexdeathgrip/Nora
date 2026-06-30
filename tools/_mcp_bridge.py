"""MCP tool bridge — state management for the discover_tools interactive flow.

The 3-step discover_tools flow:
  Step 1: list()     → model sees available MCP tools
  Step 2: tool_name=X → model picks a tool, it becomes "pending"
  Step 3: call tool  → tool is registered, pending cleared after 2 uses

This module holds the shared state between the discover_tools handler and
the conversation loop, which both need to know what tool is pending.
"""

import threading

# Thread-safe state
_lock = threading.Lock()
_agent_tools_ref = None
_pending_tool_name = None
_pending_tool_schema = None
_pending_tool_calls = 0  # how many times the model has called this tool
_NUDGE_THRESHOLD = 2     # clear after 2 successful calls OR 2 nudges


def set_agent_tools_ref(tools_list):
    """Store a reference to the agent's tools list so the handler can inject new schemas."""
    global _agent_tools_ref
    with _lock:
        _agent_tools_ref = tools_list


def get_agent_tools_ref():
    """Return the stored agent tools reference."""
    return _agent_tools_ref


def set_pending_tool(name, schema):
    """Mark a tool as pending after the model selects it.

    Args:
        name: Full MCP tool name (e.g. "mcp_universal_list_tools")
        schema: The tool's JSON schema dict
    """
    global _pending_tool_name, _pending_tool_schema, _pending_tool_calls
    with _lock:
        _pending_tool_name = name
        _pending_tool_schema = schema
        _pending_tool_calls = 0


def get_pending_tool():
    """Return (tool_name, schema) if a tool is pending, else (None, None).

    Args:
        name: Full MCP tool name (e.g. "mcp_universal_list_tools")
        schema: The tool's JSON schema dict

    Returns:
        Tuple of (tool_name, schema) or (None, None)
    """
    with _lock:
        return (_pending_tool_name, _pending_tool_schema)


def increment_pending_tool_calls():
    """Increment the call counter for the pending tool.

    Returns:
        True if the counter reached the threshold (should auto-clear),
        False otherwise.
    """
    global _pending_tool_calls
    with _lock:
        _pending_tool_calls += 1
        return _pending_tool_calls >= _NUDGE_THRESHOLD


def clear_pending():
    """Clear all pending tool state."""
    global _pending_tool_name, _pending_tool_schema, _pending_tool_calls
    with _lock:
        _pending_tool_name = None
        _pending_tool_schema = None
        _pending_tool_calls = 0


def get_pending_tool_calls():
    """Return the number of times the model has called the pending tool."""
    with _lock:
        return _pending_tool_calls


def set_pending_tool_calls(count):
    """Set the call counter directly (used by conversation loop after tool execution)."""
    global _pending_tool_calls
    with _lock:
        _pending_tool_calls = count
