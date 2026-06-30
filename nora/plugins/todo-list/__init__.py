"""
Todo-List Plugin: Task management for Nora.

- immediate (default if no action): add task AND execute now (returns execute_now: true)
- create: add task to list (does NOT execute) – use for "later" tasks
- complete: mark done
- list: show pending
- cancel: remove task
"""

import json
import logging
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}

TODO_LIST_SCHEMA = {
    "name": "todo_list",
    "description": (
        "ALWAYS BREAK TASKS INTO THE SMALLEST, EASIEST SUB-TASKS BEFORE CREATING A TO DO task."
        "Task manager.\n"
        "- immediate (DEFAULT if you provide task_name+prompt): Add task AND execute now.\n"
        "- create: Add task to list (does NOT execute). Use for 'later' tasks.\n"
        "- update: Change status, name, or priority of an existing task.\n"
        "- complete: Mark a task done.\n"
        "- list: Show pending tasks.\n"
        "- cancel: Remove a task.\n"
        "Priority: high, medium (default), low.\n"
        "If you provide task_name + prompt without action, I'll assume 'immediate'.\n"
        "Remember to always mark the task as complete. This saves tokens and helps you focus on what is actually left."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "complete", "update", "immediate", "list", "cancel"],
                "description": "Action (optional; defaults to immediate if task_name+prompt given, else list)",
            },
            "task_name": {"type": "string", "description": "Short name for the task"},
            "prompt": {
                "type": "string",
                "description": "Full instructions for the smallest task that has been broken down.",
            },
            "todo_id": {"type": "string", "description": "ID to complete, cancel, or update"},
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "completed", "cancelled"],
                "description": "New status (for update action)",
            },
            "priority": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Task priority (default: medium)",
            },
        },
        "required": [],
    },
}


def _get_hermes_home():
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home()
    except Exception:
        return Path.home() / ".hermes"


def _get_todo_file():
    return _get_hermes_home() / "todo.json"


def _load_todos():
    todo_file = _get_todo_file()
    if todo_file.exists():
        try:
            return json.loads(todo_file.read_text())
        except Exception:
            return []
    return []


def _save_todos(todos):
    # Deduplicate by ID, keeping the last occurrence
    seen = {}
    for t in todos:
        seen[t["id"]] = t
    _get_todo_file().write_text(json.dumps(list(seen.values()), indent=2))


def _sort_by_priority(todos):
    return sorted(
        todos,
        key=lambda t: (
            PRIORITY_ORDER.get(t.get("priority", "medium"), 1),
            t.get("created_at", ""),
        ),
    )


def get_todo_list_for_prompt():
    todos = _load_todos()
    pending = [t for t in todos if t.get("status") in ("pending", "in_progress")]
    if not pending:
        return ""
    pending = _sort_by_priority(pending)
    lines = ["## Your Current Tasks"]
    for t in pending:
        status = "▶" if t.get("status") == "in_progress" else "○"
        icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
            t.get("priority", "medium"), "🟡"
        )
        lines.append(
            f"{status} {icon} [{t['id']}] {t['task_name']}: {t.get('prompt', '')[:100]}"
        )
    lines.append("\nTo complete: call todo_list(action='complete', todo_id='...')")
    return "\n".join(lines)


def check_duplicate_task(todos: List[Dict], task_name: str, prompt: str) -> Optional[str]:
    """Check if a task with the same name/prompt already exists.
    
    Ponytail: O(n) scan, but todo lists are small (<1000 entries typically).
    """
    task_name_lower = task_name.lower().strip()
    prompt_lower = prompt.lower().strip()
    
    for t in todos:
        if t.get("status") in ("pending", "in_progress"):
            # Exact name match (case-insensitive)
            if t.get("task_name", "").lower() == task_name_lower:
                return t.get("id")
            
            # Prompt similarity (check if new prompt is substring of existing or vice versa)
            existing_prompt = t.get("prompt", "").lower()
            if prompt_lower and (prompt_lower in existing_prompt or existing_prompt in prompt_lower):
                return t.get("id")
    
    return None


def _generate_unique_id(todos: List[Dict]) -> str:
    """Generate a unique short ID (8 hex chars) that doesn't already exist."""
    existing_ids = {t.get("id") for t in todos}
    # Use a short random hex string (8 chars = 32 bits of entropy, enough for unique IDs)
    while True:
        new_id = f"todo_{secrets.token_hex(4)}"
        if new_id not in existing_ids:
            return new_id


def _create_task(args: Dict[str, Any], immediate: bool = False) -> str:
    task_name = args.get("task_name", "unnamed-task")
    prompt = args.get("prompt", "")
    priority = args.get("priority", "medium")
    if priority not in PRIORITY_ORDER:
        priority = "medium"
    if not prompt:
        return json.dumps({"error": "Prompt is required."})

    todos = _load_todos()
    
    # Deduplication check: prevent creating duplicate tasks
    if check_duplicate_task(todos, task_name, prompt):
        return json.dumps({"error": f"Task '{task_name}' already exists in your task list."})

    todo_item = {
        "id": _generate_unique_id(todos),
        "task_name": task_name,
        "prompt": prompt,
        "status": "in_progress" if immediate else "pending",
        "priority": priority,
        "created_at": datetime.now().isoformat(),
    }
    todos.append(todo_item)
    _save_todos(todos)

    if immediate:
        return json.dumps(
            {
                "execute_now": True,
                "task_name": task_name,
                "prompt": prompt,
                "priority": priority,
                "todo_id": todo_item["id"],
                "message": f"Added '{task_name}' (priority: {priority}). Executing now.",
            }
        )
    else:
        pending_count = len([t for t in todos if t["status"] == "pending"])
        return json.dumps(
            {
                "success": True,
                "message": f"Added '{task_name}' to your task list (priority: {priority}).",
                "todo_id": todo_item["id"],
                "priority": priority,
                "total_pending": pending_count,
                "execute_now": False,  # explicit, though not needed
            }
        )


def _complete_task(args: Dict[str, Any]) -> str:
    todo_id = args.get("todo_id")
    if not todo_id:
        return json.dumps({"error": "todo_id required"})
    todos = _load_todos()
    for i, t in enumerate(todos):
        if t.get("id") == todo_id:
            t["status"] = "completed"
            todos.pop(i)
            _save_todos(todos)
            pending = [x for x in todos if x["status"] in ("pending", "in_progress")]
            return json.dumps(
                {
                    "success": True,
                    "message": f"Completed '{t['task_name']}'",
                    "remaining": len(pending),
                }
            )
    return json.dumps({"error": f"Todo {todo_id} not found"})


def _list_tasks() -> str:
    todos = _load_todos()
    pending = [t for t in todos if t["status"] in ("pending", "in_progress")]
    completed = [t for t in todos if t["status"] == "completed"]
    return json.dumps(
        {
            "pending": _sort_by_priority(pending),
            "completed_count": len(completed),
            "total": len(todos),
        }
    )


def _update_task(args: Dict[str, Any]) -> str:
    todo_id = args.get("todo_id")
    if not todo_id:
        return json.dumps({"error": "todo_id required"})
    todos = _load_todos()
    for t in todos:
        if t.get("id") == todo_id:
            updated_fields = []
            if "task_name" in args and args["task_name"]:
                t["task_name"] = args["task_name"]
                updated_fields.append("task_name")
            if "prompt" in args and args["prompt"]:
                t["prompt"] = args["prompt"]
                updated_fields.append("prompt")
            if "status" in args and args["status"]:
                status = args["status"]
                if status in ("pending", "in_progress", "completed", "cancelled"):
                    t["status"] = status
                    updated_fields.append("status")
                else:
                    return json.dumps({"error": f"Invalid status: {status}"})
            if "priority" in args and args["priority"]:
                priority = args["priority"]
                if priority in PRIORITY_ORDER:
                    t["priority"] = priority
                    updated_fields.append("priority")
                else:
                    return json.dumps({"error": f"Invalid priority: {priority}"})
            if not updated_fields:
                return json.dumps({"error": "No fields to update. Provide task_name, prompt, status, or priority."})
            _save_todos(todos)
            return json.dumps({
                "success": True,
                "message": f"Updated '{t['task_name']}': {', '.join(updated_fields)}",
                "todo_id": todo_id,
            })
    return json.dumps({"error": f"Todo {todo_id} not found"})


def _cancel_task(args: Dict[str, Any]) -> str:
    todo_id = args.get("todo_id")
    if not todo_id:
        return json.dumps({"error": "todo_id required"})
    todos = _load_todos()
    for i, t in enumerate(todos):
        if t.get("id") == todo_id:
            todos.pop(i)
            _save_todos(todos)
            return json.dumps(
                {"success": True, "message": f"Cancelled '{t['task_name']}'"}
            )
    return json.dumps({"error": f"Todo {todo_id} not found"})


def confirm_task_completion(user_input: str) -> Dict[str, Any]:
    """Process user confirmation for task completion.
    
    Interface: yes/no/cancel
    Returns: {"action": "complete|cancel|abort", "confirmed": bool}
    """
    user_input = user_input.lower().strip()
    
    # Check for strong cancellation first (more specific)
    if user_input in ("cancel", "stop", "abort"):
        return {"action": "abort", "confirmed": False}
    elif user_input in ("no", "n"):
        return {"action": "cancel", "confirmed": False}
    elif user_input in ("yes", "y", "confirm", "ok"):
        return {"action": "complete", "confirmed": True}
    else:
        return {"action": "unknown", "confirmed": False}


def todo_list_handler(args: Dict[str, Any], **kwargs) -> str:
    action = args.get("action")

    # If no action and task_name+prompt given, default to immediate
    if not action:
        if args.get("task_name") and args.get("prompt"):
            return _create_task(args, immediate=True)
        else:
            return _list_tasks()

    if action == "immediate":
        return _create_task(args, immediate=True)
    elif action == "create":
        return _create_task(args, immediate=False)
    elif action == "update":
        return _update_task(args)
    elif action == "complete":
        return _complete_task(args)
    elif action == "list":
        return _list_tasks()
    elif action == "cancel":
        return _cancel_task(args)
    else:
        return json.dumps({"error": f"Unknown action: {action}"})


def _clear_todos():
    """Clear all todos (session-scoped)."""
    _get_todo_file().write_text("[]")


def register(ctx):
    ctx.register_tool(
        name="todo_list",
        toolset="nora-minimal",
        schema=TODO_LIST_SCHEMA,
        handler=todo_list_handler,
        description="Task management",
        emoji="✅",
    )
    # Clear todos at session start — tasks are session-scoped
    ctx.register_hook("on_session_start", lambda: _clear_todos())
    logger.info("Todo-list plugin registered")
