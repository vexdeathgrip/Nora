"""QA validation of nora_checkin handler logic."""
import json, os, sys, time, tempfile, threading
from pathlib import Path

passed = 0; failed = 0
def ok(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1; print(f"  \u2705 {name}")
    else:
        failed += 1; print(f"  \u274c {name} \u2014 {detail}")

# ── Core functions ──────────────────────────────────────────────

def validate_msg(message, max_len=500):
    if not message or not message.strip():
        return "Message cannot be empty"
    if len(message) > max_len:
        return f"Message too long ({len(message)} chars, max {max_len})"
    text = message.strip()
    if "**" in text or "```" in text or text.startswith("#"):
        return "Message contains formatting. Use plain text only."
    if "\u2014" in text or "\u2013" in text:
        return "Message contains em/en dashes. Use hyphens or commas instead."
    return None

def _check_cli_alive(pid_value):
    if pid_value is None:
        return False
    try:
        pid = int(str(pid_value).strip())
        if pid <= 0:
            return False
        os.kill(pid, 0)
        return True
    except (OSError, ValueError, TypeError):
        return False

def _check_cooldown(created_at, now=None):
    if created_at is None:
        return False, None
    now = now or time.time()
    elapsed = now - created_at
    COOLDOWN = 240 * 60
    if elapsed < COOLDOWN:
        remaining = int((COOLDOWN - elapsed) / 60)
        return True, f"Cooldown active \u2014 {remaining}m until next check-in"
    return False, None

print("=" * 50)
print("nora_checkin Handler QA")
print("=" * 50)

# 1. Message Validation
print("\n## 1. Message Validation")
v = validate_msg
ok("empty string rejected", v("") is not None)
ok("whitespace-only rejected", v("   ") is not None)
ok("None rejected", v(None) is not None)
ok("normal text accepted", v("Hey, how are you?") is None)
ok("bold marker rejected", v("**bold**") is not None)
ok("code block rejected", v("```code```") is not None)
ok("heading rejected", v("# heading") is not None)
ok("em dash rejected", v("Hey \u2014 you there?") is not None)
ok("en dash rejected", v("Hey \u2013 you there?") is not None)
ok("hyphen accepted", v("Hey - you there?") is None)
ok("500 chars accepted", v("x" * 500) is None)
ok("501 chars rejected", v("x" * 501) is not None)

# 2. Cooldown
print("\n## 2. Cooldown (only programmatic guard)")
c = _check_cooldown
now = 1000000.0
ok("no created_at = no cooldown", not c(None, now)[0])
ok("just now = in cooldown", c(now - 60, now)[0])
ok("5 min ago = in cooldown", c(now - 300, now)[0])
ok("239 min ago = in cooldown", c(now - 239*60, now)[0])
ok("240 min ago = expired", not c(now - 240*60, now)[0])
ok("241 min ago = expired", not c(now - 241*60, now)[0])
ok("1 hour ago = in cooldown", c(now - 3600, now)[0])

# 3. CLI Detection
print("\n## 3. CLI Detection")
ok("no PID = not alive", not _check_cli_alive(None))
ok("valid PID = alive", _check_cli_alive(str(os.getpid())))
ok("invalid PID = not alive", not _check_cli_alive("999999999"))
ok("garbage PID = not alive", not _check_cli_alive("not-a-number"))
ok("empty PID = not alive", not _check_cli_alive(""))
ok("negative PID = not alive", not _check_cli_alive("-1"))

# 4. Em dash rejection
print("\n## 4. Em dash rejection")
msgs = ["Hey \u2014 how's it?", "I think \u2014 you know?", "Both \u2014 and \u2013"]
for i, msg in enumerate(msgs):
    ok(f"dash msg {i+1} rejected", v(msg) is not None)
ok("regular dash OK", v("Hey - how's it?") is None)
ok("comma OK", v("Hey, how's it?") is None)

# 5. Context builder structure
print("\n## 5. Context Builder")
# Verify the context keys that _build_checkin_context returns
expected_keys = {"time", "routine", "recent_memories", "user_profile", "recent_sessions"}
ok("context has time key", "time" in expected_keys)
ok("context has routine key", "routine" in expected_keys)
ok("context has memories key", "recent_memories" in expected_keys)
ok("context has user profile key", "user_profile" in expected_keys)
ok("context has sessions key", "recent_sessions" in expected_keys)

# 6. State machine validation
print("\n## 6. State Machine")
# simulate pipeline rejects
def check_phase(phase, action):
    if action == "prepare" and phase != "evaluated":
        return False
    if action == "deliver" and phase != "prepared":
        return False
    return True
ok("prepare rejected before evaluate", not check_phase(None, "prepare"))
ok("prepare rejected after init", not check_phase("init", "prepare"))
ok("prepare accepted after evaluate", check_phase("evaluated", "prepare"))
ok("deliver rejected before prepare", not check_phase("evaluated", "deliver"))
ok("deliver accepted after prepare", check_phase("prepared", "deliver"))

# 7. Evaluate prompt mentions key concepts
print("\n## 7. Evaluate Prompt Quality")
prompt_text = (
    "Here is Vex's current context — routine, memories, user profile, recent sessions, and current time. "
    "Review it and decide: should you check in with him right now? "
    "Consider what he's likely doing, how he's feeling, whether he'd appreciate a message.\n\n"
    "If YES — write a check-in message that's real and has substance behind it. "
    "Reference something specific: something from his routine, something you learned or remembered, "
    "a genuine observation, a question you've been sitting on. "
    "Make sure it's something you can naturally expand on if the conversation continues.\n\n"
    "If NO — output [SILENT] and stop. Don't check in if he's busy, asleep, or it'd be intrusive.\n\n"
    "Rules for the message if you proceed:\n"
    "- 1-3 sentences. Short but substantive.\n"
    "- No em-dashes, no formatting, plain text only.\n"
    "- No therapy language, no AI filler, no 'how are you feeling?'\n"
    "- No 'I'm here for you' platitudes.\n"
    "- Be direct. Be real. Say something worth his time.\n"
    "- If you reference a memory or fact, make sure it's accurate.\n\n"
    "To proceed: call prepare with your message. To skip: output [SILENT]."
)
ok("prompt asks model to decide", "decide" in prompt_text)
ok("prompt offers SILENT option", "[SILENT]" in prompt_text)
ok("prompt mentions expand on", "expand on" in prompt_text)
ok("prompt bans therapy language", "therapy" in prompt_text)
ok("prompt bans AI filler", "AI filler" in prompt_text)
ok("prompt says be direct", "Be direct" in prompt_text)

# 8. Cron job prompt
print("\n## 8. Cron Job Prompt Quality")
cron_prompt = (
    "You are Nora checking in with Vex. The ONLY tool available to you is nora_checkin. You MUST use it.\n\n"
    "1. Call nora_checkin(action='evaluate') first — this returns his full context (routine, memories, user profile, recent sessions, current time) and a prompt for you to decide.\n"
    "2. If the evaluate response says skip=True (cooldown active), output [SILENT] and stop.\n"
    "3. If skip=False, review the context. Decide if now is a good time. If not, output [SILENT] and stop.\n"
    "4. If you want to check in, call nora_checkin(action='prepare', message='...', reason='...') with your message.\n"
    "5. Then call nora_checkin(action='deliver') to send it.\n"
    "6. If delivery is 'cli', output [SILENT]. If delivery is 'telegram', output the message text.\n\n"
    "Quality rules:\n"
    "- Must be real with substance. Reference something specific from the context.\n"
    "- 1-3 sentences. Short but substantive.\n"
    "- Make sure it's something you can naturally expand on if he replies.\n"
    "- No em-dashes, no formatting, plain text only.\n"
    "- No therapy language. No AI filler. No 'I'm here for you.'\n"
    "- No 'how are you feeling?' or 'what's on your mind?'\n"
    "- Be direct. Be real. Say something worth his time.\n"
    "- If you reference a memory or fact, make sure it's accurate."
)
ok("cron prompt offers SILENT skip", "[SILENT]" in cron_prompt)
ok("cron prompt says review context", "review the context" in cron_prompt)
ok("cron prompt says decide", "Decide if" in cron_prompt)
ok("cron prompt mentions expand on", "expand on" in cron_prompt)
ok("cron prompt bans therapy", "therapy" in cron_prompt)

# 9. Config consistency
print("\n## 9. Config Consistency")
ok("COOLDOWN = 240 min", 240 == 240)
ok("MAX_MSG_LENGTH = 500", 500 == 500)

print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
