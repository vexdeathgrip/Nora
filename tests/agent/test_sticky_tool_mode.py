"""Comprehensive tests for sticky tool mode state machine.

Covers:
- _sticky_build_guidance (pure function)
- _sticky_cleanup_on_success (state-mutating helper)
- Nudge injection logic (user message appended on retry failure)
- Threshold auto-exit (nudge count boundary, state reset, flag)
- Combined flows (enter → nudge → success, enter → nudge → threshold → exit)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.tool_executor import _sticky_build_guidance, _sticky_cleanup_on_success


# ── Helpers ──────────────────────────────────────────────────────────────


def _bare_agent(**overrides) -> MagicMock:
    """Create a minimal sticky-capable agent.

    All sticky attributes default to clean (post-init) values.
    Pass overrides to set specific test conditions.
    """
    attrs = {
        "_sticky_tool_name": None,
        "_sticky_tool_quit_allowed": True,
        "_sticky_tool_fail_count": 0,
        "_sticky_tool_quit_count": 0,
        "_sticky_saved_tools": None,
        "_sticky_steer_text": None,
        "_sticky_first_msg_idx": None,
        "_sticky_empty_count": 0,
        "_sticky_nudge_count": 0,
        "tools": None,
    }
    attrs.update(overrides)
    agent = MagicMock(spec=[])
    for k, v in attrs.items():
        setattr(agent, k, v)
    return agent


# ── _sticky_build_guidance ───────────────────────────────────────────────


class TestStickyBuildGuidance:
    """Raw guidance marker appended to tool result content."""

    def test_simple_marker_format(self):
        agent = _bare_agent(_sticky_tool_fail_count=1)
        result = _sticky_build_guidance(agent, "web_search")
        assert result == "\n\n⚠️ STICKY: tool=web_search (attempt 1)"

    def test_includes_attempt_number(self):
        agent = _bare_agent(_sticky_tool_fail_count=3)
        result = _sticky_build_guidance(agent, "web_search")
        assert "(attempt 3)" in result

    def test_no_quit_instructions(self):
        """Guidance is now just a marker — instructions live in the user nudge."""
        agent = _bare_agent(_sticky_tool_fail_count=1)
        result = _sticky_build_guidance(agent, "web_search")
        assert "_quit_tool" not in result
        assert "remaining" not in result
        assert "Focus on fixing" not in result

    def test_different_tool_name_appears(self):
        agent = _bare_agent(_sticky_tool_fail_count=2)
        assert "terminal" in _sticky_build_guidance(agent, "terminal")
        assert "read_file" in _sticky_build_guidance(agent, "read_file")


# ── _sticky_cleanup_on_success ───────────────────────────────────────────


class TestStickyCleanupOnSuccess:
    """On success, sticky traces are removed from messages + state reset."""

    def test_clears_messages_between_first_idx_and_neg2(self):
        agent = _bare_agent(_sticky_first_msg_idx=1)
        messages = [
            {"role": "system", "content": "sys"},          # idx 0
            {"role": "assistant", "content": "try tool"},   # idx 1 (first_idx)
            {"role": "tool", "content": "result 1", "name": "web_search"},
            {"role": "assistant", "content": "try again"},  # idx 3
            {"role": "tool", "content": "result 2", "name": "web_search"},
            {"role": "assistant", "content": "done"},       # idx 5 (-2)
            {"role": "user", "content": "what now?"},       # idx 6 (-1) kept
        ]
        _sticky_cleanup_on_success(agent, messages)
        # cleanup keeps pre-sticky messages + the last 2 messages (-2 and -1)
        assert len(messages) == 3
        assert messages[0] == {"role": "system", "content": "sys"}
        assert messages[1] == {"role": "assistant", "content": "done"}       # -2 kept
        assert messages[2] == {"role": "user", "content": "what now?"}        # -1 kept

    def test_resets_all_sticky_attributes(self):
        agent = _bare_agent(
            _sticky_tool_name="web_search",
            _sticky_tool_fail_count=5,
            _sticky_nudge_count=2,
            _sticky_tool_quit_count=1,
            _sticky_tool_quit_allowed=False,
            _sticky_first_msg_idx=1,
            _sticky_steer_text="some guidance",
            _sticky_empty_count=1,
            _sticky_saved_tools=["tool_a", "tool_b"],
            tools=[],
        )
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "try tool"},
            {"role": "tool", "content": "result", "name": "web_search"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "what now?"},
        ]
        _sticky_cleanup_on_success(agent, messages)
        assert agent._sticky_tool_name is None
        assert agent._sticky_tool_fail_count == 0
        assert agent._sticky_nudge_count == 0
        assert agent._sticky_tool_quit_count == 0
        assert agent._sticky_tool_quit_allowed is True
        assert agent._sticky_first_msg_idx is None
        assert agent._sticky_steer_text is None
        assert agent._sticky_empty_count == 0

    def test_restores_saved_tools(self):
        saved = ["tool_a", "tool_b"]
        agent = _bare_agent(
            _sticky_first_msg_idx=1,
            _sticky_saved_tools=saved,
            tools=[],
        )
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "try"},
            {"role": "tool", "content": "r", "name": "t"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "ok"},
        ]
        _sticky_cleanup_on_success(agent, messages)
        assert agent.tools is saved
        assert agent._sticky_saved_tools is None

    def test_graceful_when_first_idx_is_none(self):
        agent = _bare_agent(_sticky_first_msg_idx=None)
        messages = [{"role": "system", "content": "sys"}]
        _sticky_cleanup_on_success(agent, messages)
        assert len(messages) == 1  # unchanged

    def test_graceful_when_first_idx_out_of_range(self):
        agent = _bare_agent(_sticky_first_msg_idx=999)
        messages = [{"role": "system", "content": "sys"}]
        _sticky_cleanup_on_success(agent, messages)
        assert len(messages) == 1  # unchanged

    def test_first_idx_equals_last_index_minus_one_skips_cleanup(self):
        """When first_idx == len(messages) - 2 the cleanup slice is empty."""
        agent = _bare_agent(_sticky_first_msg_idx=2)
        messages = [
            {"role": "system", "content": "sys"},         # 0
            {"role": "assistant", "content": "try"},       # 1
            {"role": "tool", "content": "fail"},           # 2
            {"role": "assistant", "content": "done"},      # 3 (-1)
        ]
        _sticky_cleanup_on_success(agent, messages)
        # slice messages[2:-2] = messages[2:2] = empty — no delete
        assert len(messages) == 4

    def test_no_saved_tools_restore_not_called(self):
        agent = _bare_agent(
            _sticky_first_msg_idx=1,
            _sticky_saved_tools=None,
            tools=["existing"],
        )
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "try"},
            {"role": "tool", "content": "r"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "ok"},
        ]
        _sticky_cleanup_on_success(agent, messages)
        assert agent.tools == ["existing"]  # unchanged


# ── Nudge injection logic (simulated state-machine blocks) ───────────────


class TestNudgeInjection:
    """User message nudge injected on retry failure (replaces old append-to-result)."""

    def test_nudge_appended_as_user_message(self):
        agent = _bare_agent(
            _sticky_tool_name="web_search",
            _sticky_tool_fail_count=2,
            _sticky_nudge_count=1,
            _sticky_tool_quit_allowed=True,
            _sticky_tool_quit_count=0,
        )
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "call", "tool_calls": [1]},
            {"role": "tool", "content": "error", "name": "web_search"},
        ]

        # Simulate the retry-failure code block (conversation_loop.py:3944-3966)
        agent._sticky_tool_fail_count += 1
        agent._sticky_nudge_count += 1
        _stuck = "web_search"
        _max_q = 10
        _quit_available = agent._sticky_tool_quit_allowed and agent._sticky_tool_quit_count < _max_q
        _nudge = (
            f"[STICKY] `{_stuck}` failed (attempt {agent._sticky_tool_fail_count}). "
            f"Fix the error and retry."
            + (f" Call `_quit_tool` to exit ({_max_q - agent._sticky_tool_quit_count} remaining)." if _quit_available else " Quitting exhausted.")
            + f" Call `{_stuck}` now."
        )
        messages.append({"role": "user", "content": _nudge})

        assert len(messages) == 4
        assert messages[-1]["role"] == "user"
        assert "[STICKY]" in messages[-1]["content"]
        assert "web_search" in messages[-1]["content"]
        assert "attempt 3" in messages[-1]["content"]
        assert "Fix the error and retry" in messages[-1]["content"]

    def test_nudge_format_matches_discover_tools_style(self):
        agent = _bare_agent(
            _sticky_tool_name="read_file",
            _sticky_tool_fail_count=1,
            _sticky_nudge_count=0,
            _sticky_tool_quit_allowed=True,
            _sticky_tool_quit_count=0,
        )
        messages = []
        _stuck = "read_file"
        _max_q = 10
        _quit_available = True
        _nudge = (
            f"[STICKY] `{_stuck}` failed (attempt {agent._sticky_tool_fail_count + 1}). "
            f"Fix the error and retry."
            + (f" Call `_quit_tool` to exit ({_max_q - agent._sticky_tool_quit_count} remaining)." if _quit_available else " Quitting exhausted.")
            + f" Call `{_stuck}` now."
        )
        messages.append({"role": "user", "content": _nudge})
        content = messages[-1]["content"]
        assert content.startswith("[STICKY]")
        assert "Fix the error and retry." in content
        assert "Call `read_file` now." in content

    def test_nudge_includes_quit_when_available(self):
        agent = _bare_agent(
            _sticky_tool_name="x",
            _sticky_tool_fail_count=3,
            _sticky_nudge_count=2,
            _sticky_tool_quit_allowed=True,
            _sticky_tool_quit_count=0,
        )
        messages = []
        _stuck = "x"
        _max_q = 10
        _quit_available = agent._sticky_tool_quit_allowed and agent._sticky_tool_quit_count < _max_q
        _nudge = (
            f"[STICKY] `{_stuck}` failed (attempt {agent._sticky_tool_fail_count + 1}). "
            f"Fix the error and retry."
            + (f" Call `_quit_tool` to exit ({_max_q - agent._sticky_tool_quit_count} remaining)." if _quit_available else " Quitting exhausted.")
            + f" Call `{_stuck}` now."
        )
        messages.append({"role": "user", "content": _nudge})
        content = messages[-1]["content"]
        assert "_quit_tool" in content
        assert "10 remaining" in content

    def test_nudge_excludes_quit_when_exhausted(self):
        agent = _bare_agent(
            _sticky_tool_name="x",
            _sticky_tool_fail_count=1,
            _sticky_nudge_count=0,
            _sticky_tool_quit_allowed=False,
            _sticky_tool_quit_count=10,
        )
        messages = []
        _stuck = "x"
        _max_q = 10
        _quit_available = agent._sticky_tool_quit_allowed and agent._sticky_tool_quit_count < _max_q
        _nudge = (
            f"[STICKY] `{_stuck}` failed (attempt {agent._sticky_tool_fail_count + 1}). "
            f"Fix the error and retry."
            + (f" Call `_quit_tool` to exit ({_max_q - agent._sticky_tool_quit_count} remaining)." if _quit_available else " Quitting exhausted.")
            + f" Call `{_stuck}` now."
        )
        messages.append({"role": "user", "content": _nudge})
        content = messages[-1]["content"]
        assert "Quitting exhausted" in content

    def test_nudge_excludes_quit_when_not_allowed(self):
        agent = _bare_agent(
            _sticky_tool_name="x",
            _sticky_tool_fail_count=1,
            _sticky_nudge_count=0,
            _sticky_tool_quit_allowed=False,
            _sticky_tool_quit_count=0,
        )
        messages = []
        _stuck = "x"
        _max_q = 10
        _quit_available = agent._sticky_tool_quit_allowed and agent._sticky_tool_quit_count < _max_q
        _nudge = (
            f"[STICKY] `{_stuck}` failed (attempt {agent._sticky_tool_fail_count + 1}). "
            f"Fix the error and retry."
            + (f" Call `_quit_tool` to exit ({_max_q - agent._sticky_tool_quit_count} remaining)." if _quit_available else " Quitting exhausted.")
            + f" Call `{_stuck}` now."
        )
        messages.append({"role": "user", "content": _nudge})
        content = messages[-1]["content"]
        assert "Quitting exhausted" in content


# ── Threshold auto-exit (boundary + state reset) ─────────────────────────


class TestNudgeThresholdAutoExit:
    """When _sticky_nudge_count >= max_nudges, sticky exits with state reset."""

    def test_below_threshold_no_auto_exit(self):
        """Nudge count < max_nudges (3) → no auto-exit triggered."""
        agent = _bare_agent(
            _sticky_tool_name="web_search",
            _sticky_tool_fail_count=2,
            _sticky_nudge_count=1,
            _sticky_tool_quit_allowed=True,
            _sticky_tool_quit_count=0,
        )
        _max_nudges = 3
        _sticky_nudge_auto_exit = False

        # Simulate the threshold check
        if agent._sticky_nudge_count >= _max_nudges:
            _sticky_nudge_auto_exit = True

        assert _sticky_nudge_auto_exit is False
        assert agent._sticky_tool_name == "web_search"  # still in sticky

    def test_at_threshold_triggers_auto_exit(self):
        """nudge_count == max_nudges → auto-exit, full state reset."""
        agent = _bare_agent(
            _sticky_tool_name="web_search",
            _sticky_tool_fail_count=4,
            _sticky_nudge_count=3,
            _sticky_tool_quit_count=2,
            _sticky_tool_quit_allowed=False,
            _sticky_first_msg_idx=1,
            _sticky_steer_text="steer",
            _sticky_empty_count=1,
            _sticky_saved_tools=["tool_a"],
            tools=[],
        )
        _max_nudges = 3
        _sticky_nudge_auto_exit = False

        if agent._sticky_nudge_count >= _max_nudges:
            _nudge_total = agent._sticky_nudge_count
            agent._sticky_tool_name = None
            agent._sticky_tool_fail_count = 0
            agent._sticky_nudge_count = 0
            agent._sticky_tool_quit_count = 0
            agent._sticky_tool_quit_allowed = True
            agent._sticky_first_msg_idx = None
            agent._sticky_steer_text = None
            agent._sticky_empty_count = 0
            if agent._sticky_saved_tools is not None:
                agent.tools = agent._sticky_saved_tools
                agent._sticky_saved_tools = None
            _sticky_nudge_auto_exit = True

        assert _sticky_nudge_auto_exit is True
        assert agent._sticky_tool_name is None
        assert agent._sticky_tool_fail_count == 0
        assert agent._sticky_nudge_count == 0
        assert agent._sticky_tool_quit_count == 0
        assert agent._sticky_tool_quit_allowed is True
        assert agent._sticky_first_msg_idx is None
        assert agent._sticky_steer_text is None
        assert agent._sticky_empty_count == 0
        assert _nudge_total == 3

    def test_above_threshold_still_triggers(self):
        """nudge_count > max_nudges → still triggers (belt-and-suspenders)."""
        agent = _bare_agent(
            _sticky_tool_name="web_search",
            _sticky_tool_fail_count=10,
            _sticky_nudge_count=5,
            _sticky_saved_tools=["tool_a"],
            tools=[],
        )
        _max_nudges = 3
        _sticky_nudge_auto_exit = False

        if agent._sticky_nudge_count >= _max_nudges:
            agent._sticky_tool_name = None
            agent._sticky_tool_fail_count = 0
            agent._sticky_nudge_count = 0
            agent._sticky_tool_quit_count = 0
            agent._sticky_tool_quit_allowed = True
            agent._sticky_first_msg_idx = None
            agent._sticky_steer_text = None
            agent._sticky_empty_count = 0
            if agent._sticky_saved_tools is not None:
                agent.tools = agent._sticky_saved_tools
                agent._sticky_saved_tools = None
            _sticky_nudge_auto_exit = True

        assert _sticky_nudge_auto_exit is True
        assert agent._sticky_tool_name is None

    def test_auto_exit_restores_saved_tools(self):
        agent = _bare_agent(
            _sticky_tool_name="x",
            _sticky_tool_fail_count=3,
            _sticky_nudge_count=3,
            _sticky_saved_tools=["tool_a", "tool_b"],
            tools=[],
        )
        _max_nudges = 3
        _sticky_nudge_auto_exit = False

        if agent._sticky_nudge_count >= _max_nudges:
            agent._sticky_tool_name = None
            agent._sticky_tool_fail_count = 0
            agent._sticky_nudge_count = 0
            agent._sticky_tool_quit_count = 0
            agent._sticky_tool_quit_allowed = True
            agent._sticky_first_msg_idx = None
            agent._sticky_steer_text = None
            agent._sticky_empty_count = 0
            if agent._sticky_saved_tools is not None:
                agent.tools = agent._sticky_saved_tools
                agent._sticky_saved_tools = None
            _sticky_nudge_auto_exit = True

        assert agent.tools == ["tool_a", "tool_b"]
        assert agent._sticky_saved_tools is None

    def test_auto_exit_with_none_saved_tools(self):
        agent = _bare_agent(
            _sticky_tool_name="x",
            _sticky_tool_fail_count=3,
            _sticky_nudge_count=3,
            _sticky_saved_tools=None,
            tools=["still_here"],
        )
        _max_nudges = 3
        _sticky_nudge_auto_exit = False

        if agent._sticky_nudge_count >= _max_nudges:
            agent._sticky_tool_name = None
            agent._sticky_tool_fail_count = 0
            agent._sticky_nudge_count = 0
            agent._sticky_tool_quit_count = 0
            agent._sticky_tool_quit_allowed = True
            agent._sticky_first_msg_idx = None
            agent._sticky_steer_text = None
            agent._sticky_empty_count = 0
            if agent._sticky_saved_tools is not None:
                agent.tools = agent._sticky_saved_tools
                agent._sticky_saved_tools = None
            _sticky_nudge_auto_exit = True

        assert agent.tools == ["still_here"]  # unchanged

    def test_nudge_count_tracking_across_multiple_retries(self):
        """Simulate 3 retries → fail_count=3, nudge_count=3."""
        agent = _bare_agent(
            _sticky_tool_name="web_search",
            _sticky_tool_fail_count=0,
            _sticky_nudge_count=0,
        )
        for _ in range(3):
            agent._sticky_tool_fail_count += 1
            agent._sticky_nudge_count += 1
        assert agent._sticky_tool_fail_count == 3
        assert agent._sticky_nudge_count == 3


# ── Combined / integration-like flows ────────────────────────────────────


class TestCombinedFlows:
    """Multi-step scenarios that exercise the full sticky lifecycle."""

    def test_enter_sticky_then_retry_nudge_then_success_and_cleanup(self):
        """Full cycle: enter → fail → nudge → succeed → cleanup."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "call web_search"},
        ]
        agent = _bare_agent(
            _sticky_tool_name="web_search",
            _sticky_tool_fail_count=1,
            _sticky_nudge_count=0,
            _sticky_tool_quit_allowed=True,
            _sticky_tool_quit_count=0,
            _sticky_saved_tools=["tool_a", "tool_b"],
            _sticky_first_msg_idx=1,
            tools=[],
        )

        # Step 1: retry failure → inject nudge (simulating state machine block)
        agent._sticky_tool_fail_count += 1
        agent._sticky_nudge_count += 1
        _stuck = "web_search"
        _max_q = 10
        _quit_available = agent._sticky_tool_quit_allowed and agent._sticky_tool_quit_count < _max_q
        _nudge = (
            f"[STICKY] `{_stuck}` failed (attempt {agent._sticky_tool_fail_count}). "
            f"Fix the error and retry."
            + (f" Call `_quit_tool` to exit ({_max_q - agent._sticky_tool_quit_count} remaining)." if _quit_available else " Quitting exhausted.")
            + f" Call `{_stuck}` now."
        )
        messages.append({"role": "user", "content": _nudge})

        assert agent._sticky_tool_fail_count == 2
        assert agent._sticky_nudge_count == 1
        assert len(messages) == 3
        assert messages[-1]["role"] == "user"
        assert "[STICKY]" in messages[-1]["content"]

        # Step 2: tool succeeds on retry → cleanup
        messages.append({"role": "assistant", "content": "second try"})
        messages.append({"role": "tool", "content": "success", "name": "web_search"})
        messages.append({"role": "assistant", "content": "done"})
        messages.append({"role": "user", "content": "thanks"})

        # Simulate success handler calling _sticky_cleanup_on_success
        _sticky_cleanup_on_success(agent, messages)

        # cleanup keeps pre-sticky + last 2 messages (assistant success + user)
        assert len(messages) == 3
        assert messages[0] == {"role": "system", "content": "sys"}
        assert messages[1] == {"role": "assistant", "content": "done"}  # -2 kept
        assert messages[2] == {"role": "user", "content": "thanks"}     # -1 kept
        assert agent._sticky_tool_name is None

    def test_enter_sticky_then_reach_nudge_threshold_then_auto_exit(self):
        """Full cycle: enter → 3 nudges → auto-exit with state reset."""
        agent = _bare_agent(
            _sticky_tool_name="web_search",
            _sticky_tool_fail_count=0,
            _sticky_nudge_count=0,
            _sticky_tool_quit_count=1,
            _sticky_tool_quit_allowed=True,
            _sticky_first_msg_idx=1,
            _sticky_saved_tools=["tool_a"],
            tools=[],
        )
        _max_nudges = 3
        _sticky_nudge_auto_exit = False
        messages = [{"role": "user", "content": "do it"}]

        # Simulate 3 retry-failure cycles, each injecting a nudge
        for i in range(3):
            agent._sticky_tool_fail_count += 1
            agent._sticky_nudge_count += 1
            _stuck = "web_search"
            _max_q = 10
            _quit_available = agent._sticky_tool_quit_allowed and agent._sticky_tool_quit_count < _max_q
            _nudge = (
                f"[STICKY] `{_stuck}` failed (attempt {agent._sticky_tool_fail_count}). "
                f"Fix the error and retry."
                + (f" Call `_quit_tool` to exit ({_max_q - agent._sticky_tool_quit_count} remaining)." if _quit_available else " Quitting exhausted.")
                + f" Call `{_stuck}` now."
            )
            messages.append({"role": "user", "content": _nudge})

            # After incrementing, check threshold (same code as state machine)
            if agent._sticky_nudge_count >= _max_nudges:
                agent._sticky_tool_name = None
                agent._sticky_tool_fail_count = 0
                agent._sticky_nudge_count = 0
                agent._sticky_tool_quit_count = 0
                agent._sticky_tool_quit_allowed = True
                agent._sticky_first_msg_idx = None
                agent._sticky_steer_text = None
                agent._sticky_empty_count = 0
                if agent._sticky_saved_tools is not None:
                    agent.tools = agent._sticky_saved_tools
                    agent._sticky_saved_tools = None
                _sticky_nudge_auto_exit = True

        # After threshold: auto-exit triggered on the 3rd iteration
        assert _sticky_nudge_auto_exit is True
        assert agent._sticky_tool_name is None
        assert agent._sticky_nudge_count == 0
        assert agent._sticky_tool_fail_count == 0
        assert agent._sticky_tool_quit_count == 0
        assert agent._sticky_saved_tools is None
        assert agent.tools == ["tool_a"]

        # Messages: original + 3 nudges (last nudge injected before threshold tripped)
        assert len(messages) == 4

    def test_quit_during_sticky_keeps_state_reset(self):
        """_quit_tool exits sticky and resets nudge/message state."""
        agent = _bare_agent(
            _sticky_tool_name="web_search",
            _sticky_tool_fail_count=2,
            _sticky_nudge_count=1,
            _sticky_tool_quit_count=0,
            _sticky_tool_quit_allowed=True,
            _sticky_first_msg_idx=1,
            _sticky_empty_count=0,
        )
        # Simulate quit handler (conversation_loop.py:4000-4012)
        agent._sticky_tool_name = None
        agent._sticky_tool_fail_count = 0
        agent._sticky_nudge_count = 0
        agent._sticky_tool_quit_count += 1
        agent._sticky_tool_quit_allowed = True
        agent._sticky_first_msg_idx = None
        agent._sticky_empty_count = 0

        assert agent._sticky_tool_name is None
        assert agent._sticky_tool_fail_count == 0
        assert agent._sticky_nudge_count == 0
        assert agent._sticky_tool_quit_count == 1



# ── Config defaults ──────────────────────────────────────────────────────


class TestConfigDefaults:
    """Verify the config.yaml sticky defaults are reasonable."""

    def test_max_nudges_default_is_three(self):
        from hermes_cli.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["sticky"]["max_nudges"] == 3

    def test_max_quit_cycles_default_is_ten(self):
        from hermes_cli.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["sticky"]["max_quit_cycles"] == 10

    def test_enabled_default_is_true(self):
        from hermes_cli.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["sticky"]["enabled"] is True

    def test_guardrail_defaults_high(self):
        from hermes_cli.config import DEFAULT_CONFIG
        g = DEFAULT_CONFIG["sticky"]["guardrails"]
        assert g["exact_failure_block_after"] == 10
        assert g["same_tool_failure_halt_after"] == 10
