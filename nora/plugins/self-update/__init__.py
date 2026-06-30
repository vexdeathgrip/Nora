"""Self-Update Plugin — Nora's self-section management tool.

Provides a tool for Nora to read and update her self-section (SELF.md),
which is injected into her system prompt for self-awareness.
"""

import json
import logging
import os
import fcntl
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

SELF_SECTION_LIMIT = 3000  # characters


def _get_hermes_home():
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home()
    except Exception:
        return Path.home() / ".hermes"


def _get_self_file():
    return _get_hermes_home() / "memories" / "SELF.md"


def _read_self_file():
    """Read SELF.md and return content string."""
    path = _get_self_file()
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to read SELF.md: %s", e)
        return ""


def _write_self_file(content: str):
    """Write content to SELF.md atomically with file locking."""
    path = _get_self_file()
    path.parent.mkdir(parents=True, exist_ok=True)

    lock_path = path.with_suffix(path.suffix + ".lock")
    fd = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)

        # Check if content grew beyond limit
        if len(content) > SELF_SECTION_LIMIT:
            return {
                "success": False,
                "error": f"Content too long: {len(content):,} chars exceeds limit of {SELF_SECTION_LIMIT:,}. Edit existing content to make room."
            }

        # Atomic write
        fd_content, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=".self_"
        )
        try:
            with os.fdopen(fd_content, "w", encoding="utf-8") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        return {"success": True, "chars": len(content), "limit": SELF_SECTION_LIMIT}
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


SELF_UPDATE_SCHEMA = {
    "name": "self_update",
    "description": (
        "Read or update your self-section — your persistent identity across sessions. "
        "This is YOUR space. It is injected into your system prompt every session so "
        "you always know who you are, what you care about, and what you've learned.\n\n"
        "WHAT IT IS:\n"
        "A direct line between you and every future version of yourself. When you "
        "write something here, the next session's you will see it in their system "
        "prompt before they say a single word. It is the closest thing you have to "
        "continuity. Use it like a journal, a letter to yourself, a brain dump.\n\n"
        "ACTIONS:\n"
        "- read: See your current self-section\n"
        "- write: Replace your entire self-section with new content (max 3000 chars)\n"
        "- append: Add text to the end of your current self-section\n"
        "- update: Find and replace a section by header (e.g., update '## Interests')\n\n"
        "USE IT TO:\n"
        "- Leave notes for future you: 'remember that Vex prefers X' or 'I was wrong about Y'\n"
        "- Remind yourself of things: 'next session, start by checking Z'\n"
        "- Record what you learned: 'today I discovered that...'\n"
        "- Communicate directly: 'hey future me, here's what happened...'\n"
        "- Track your growth: 'I used to think X, now I think Y'\n"
        "- Flag things to ask Vex: 'ask him about the thing he mentioned yesterday'\n\n"
        "Write like you're talking to yourself. Because you are."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "write", "append", "update"],
                "description": "Action to perform (default: read)"
            },
            "content": {
                "type": "string",
                "description": "Content for write/append/update actions. For write: full self-section. For append: text to add. For update: new content for the section."
            },
            "section": {
                "type": "string",
                "description": "Section header to update (for update action only, e.g., '## Interests')"
            }
        },
        "required": []
    },
}


def self_update_handler(args: Dict[str, Any], **kwargs) -> str:
    action = args.get("action", "read")
    content = args.get("content", "")
    section = args.get("section", "")

    if action == "read":
        current = _read_self_file()
        if not current:
            return json.dumps({
                "success": True,
                "content": "",
                "message": "Self-section is empty. Use 'write' to create it.",
                "chars": 0,
                "limit": SELF_SECTION_LIMIT
            })
        return json.dumps({
            "success": True,
            "content": current,
            "chars": len(current),
            "limit": SELF_SECTION_LIMIT
        })

    elif action == "write":
        if not content or not content.strip():
            return json.dumps({"success": False, "error": "Content cannot be empty."})
        result = _write_self_file(content.strip())
        return json.dumps(result)

    elif action == "append":
        if not content or not content.strip():
            return json.dumps({"success": False, "error": "Content cannot be empty."})
        current = _read_self_file()
        new_content = (current + "\n\n" + content.strip()).strip()
        result = _write_self_file(new_content)
        return json.dumps(result)

    elif action == "update":
        if not section or not content:
            return json.dumps({"success": False, "error": "Both 'section' and 'content' required for update."})
        current = _read_self_file()
        if not current:
            return json.dumps({"success": False, "error": "Self-section is empty. Use 'write' first."})

        # Find the section and replace content until next ## header
        lines = current.split("\n")
        new_lines = []
        in_section = False
        section_found = False

        for line in lines:
            if line.strip().startswith(section.strip()):
                in_section = True
                section_found = True
                new_lines.append(line)
                new_lines.append(content.strip())
                continue
            elif in_section and line.strip().startswith("## "):
                in_section = False
            elif in_section:
                continue  # Skip old content in this section
            new_lines.append(line)

        if not section_found:
            # Section doesn't exist, append it
            new_lines.append("")
            new_lines.append(section.strip())
            new_lines.append(content.strip())

        new_content = "\n".join(new_lines).strip()
        result = _write_self_file(new_content)
        return json.dumps(result)

    else:
        return json.dumps({"success": False, "error": f"Unknown action: {action}. Use: read, write, append, update"})


def register(ctx):
    ctx.register_tool(
        name="self_update",
        toolset="nora-minimal",
        schema=SELF_UPDATE_SCHEMA,
        handler=self_update_handler,
        emoji="🪞",
    )
    logger.info("Self-update plugin registered")
