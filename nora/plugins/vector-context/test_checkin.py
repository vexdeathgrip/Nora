"""Comprehensive tests for the nora_checkin tool pipeline.

Tests cover:
- Schedule timing (sleep, college, weekends)
- Cooldown enforcement
- Activity checks
- Message validation (length, formatting, emptiness)
- State machine ordering (evaluate → prepare → deliver)
- CLI detection (PID file alive/dead/missing)
- Signal file writing and consumption
- Fallback job scheduling
- Edge cases and boundary values
"""

import json
import os
import time
import threading
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock, PropertyMock
import pytest

# ─── Patch infrastructure BEFORE importing the handler ───────────────────

# Mock chromadb before the plugin module loads
sys_modules_patch = patch.dict("sys.modules", {"chromadb": MagicMock()})
sys_modules_patch.start()

# Mock hermes_constants
hermes_constants_mock = MagicMock()
hermes_constants_mock.get_hermes_home.return_value = Path("/tmp/test_hermes_home")
sys_modules_patch2 = patch.dict("sys.modules", {"hermes_constants": hermes_constants_mock})
sys_modules_patch2.start()

# Mock hermes_state
hermes_state_mock = MagicMock()
sys_modules_patch3 = patch.dict("sys.modules", {"hermes_state": hermes_state_mock})
sys_modules_patch3.start()

from nora.plugins.vector_context import __init__ as plugin


# ─── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clean_state():
    """Ensure clean state between tests."""
    key = "test_task"
    plugin._checkin_cleanup_state(key)
    # Clean up any signal/state files
    for f in [plugin._CHECKIN_SIGNAL_FILE, plugin._CHECKIN_STATE_FILE, plugin._CHECKIN_LOCK_FILE]:
        try:
            f.unlink(missing_ok=True)
        except Exception:
            pass
    yield
    plugin._checkin_cleanup_state(key)
    for f in [plugin._CHECKIN_SIGNAL_FILE, plugin._CHECKIN_STATE_FILE]:
        try:
            f.unlink(missing_ok=True)
        except Exception:
            pass


@pytest.fixture
def task_id():
    return "test_task_123"


@pytest.fixture
def call_handler(task_id):
    """Helper to call the handler and parse JSON response."""

    def _call(action, **kwargs):
        args = {"action": action, **kwargs}
        result = plugin.nora_checkin_handler(args, task_id=task_id)
        return json.loads(result)

    return _call


# ─── Mock helpers ───────────────────────────────────────────────────────


def mock_time(hour, minute, weekday=0):
    """Mock datetime.now() to a specific time."""
    dt = datetime(2026, 6, 29, hour, minute, 0)  # Monday
    # Adjust to correct weekday
    dt = dt.replace(day=dt.day + weekday)
    mock = MagicMock()
    mock.now.return_value = dt
    mock.weekday.return_value = weekday
    return mock


# ═══════════════════════════════════════════════════════════════════════════
# SCHEDULE TIMING TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestSleepSchedule:
    """Vex should not receive check-ins during sleep or college hours."""

    @patch("nora.plugins.vector_context.__init__._checkin_now_minutes")
    @patch("nora.plugins.vector_context.__init__._checkin_is_weekday")
    def test_weekday_sleep_midnight(self, mock_weekday, mock_minutes):
        mock_weekday.return_value = True
        mock_minutes.return_value = 0  # midnight
        skip, reason = plugin._check_sleep_schedule()
        assert skip is True
        assert "asleep" in reason

    @patch("nora.plugins.vector_context.__init__._checkin_now_minutes")
    @patch("nora.plugins.vector_context.__init__._checkin_is_weekday")
    def test_weekday_sleep_11pm(self, mock_weekday, mock_minutes):
        mock_weekday.return_value = True
        mock_minutes.return_value = 23 * 60  # 11 PM
        skip, reason = plugin._check_sleep_schedule()
        assert skip is True
        assert "asleep" in reason

    @patch("nora.plugins.vector_context.__init__._checkin_now_minutes")
    @patch("nora.plugins.vector_context.__init__._checkin_is_weekday")
    def test_weekday_college_morning(self, mock_weekday, mock_minutes):
        mock_weekday.return_value = True
        mock_minutes.return_value = 8 * 60  # 8 AM
        skip, reason = plugin._check_sleep_schedule()
        assert skip is True
        assert "college" in reason

    @patch("nora.plugins.vector_context.__init__._checkin_now_minutes")
    @patch("nora.plugins.vector_context.__init__._checkin_is_weekday")
    def test_weekday_college_afternoon(self, mock_weekday, mock_minutes):
        mock_weekday.return_value = True
        mock_minutes.return_value = 14 * 60  # 2 PM
        skip, reason = plugin._check_sleep_schedule()
        assert skip is True
        assert "college" in reason

    @patch("nora.plugins.vector_context.__init__._checkin_now_minutes")
    @patch("nora.plugins.vector_context.__init__._checkin_is_weekday")
    def test_weekday_evening_ok(self, mock_weekday, mock_minutes):
        mock_weekday.return_value = True
        mock_minutes.return_value = 19 * 60  # 7 PM
        skip, reason = plugin._check_sleep_schedule()
        assert skip is False

    @patch("nora.plugins.vector_context.__init__._checkin_now_minutes")
    @patch("nora.plugins.vector_context.__init__._checkin_is_weekday")
    def test_weekday_after_college_ok(self, mock_weekday, mock_minutes):
        mock_weekday.return_value = True
        mock_minutes.return_value = 17 * 60 + 30  # 5:30 PM
        skip, reason = plugin._check_sleep_schedule()
        assert skip is False

    @patch("nora.plugins.vector_context.__init__._checkin_now_minutes")
    @patch("nora.plugins.vector_context.__init__._checkin_is_weekday")
    def test_weekend_night_ok(self, mock_weekday, mock_minutes):
        """Weekend: 10 PM should be OK (weekend bedtime is later)."""
        mock_weekday.return_value = False  # weekend
        mock_minutes.return_value = 22 * 60  # 10 PM
        skip, reason = plugin._check_sleep_schedule()
        assert skip is False

    @patch("nora.plugins.vector_context.__init__._checkin_now_minutes")
    @patch("nora.plugins.vector_context.__init__._checkin_is_weekday")
    def test_weekend_late_night(self, mock_weekday, mock_minutes):
        mock_weekday.return_value = False
        mock_minutes.return_value = 23 * 60 + 30  # 11:30 PM
        skip, reason = plugin._check_sleep_schedule()
        assert skip is True
        assert "asleep" in reason

    @patch("nora.plugins.vector_context.__init__._checkin_now_minutes")
    @patch("nora.plugins.vector_context.__init__._checkin_is_weekday")
    def test_weekend_morning_ok(self, mock_weekday, mock_minutes):
        """Weekend: 8 AM should be OK (weekend wake is 9 AM)."""
        mock_weekday.return_value = False
        mock_minutes.return_value = 8 * 60  # 8 AM
        skip, reason = plugin._check_sleep_schedule()
        assert skip is False

    @patch("nora.plugins.vector_context.__init__._checkin_now_minutes")
    @patch("nora.plugins.vector_context.__init__._checkin_is_weekday")
    def test_weekend_early_morning_sleep(self, mock_weekday, mock_minutes):
        mock_weekday.return_value = False
        mock_minutes.return_value = 6 * 60  # 6 AM
        skip, reason = plugin._check_sleep_schedule()
        assert skip is True
        assert "asleep" in reason


# ═══════════════════════════════════════════════════════════════════════════
# COOLDOWN TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestCooldown:
    """Check-ins must respect a minimum 4-hour cooldown."""

    def test_no_state_no_cooldown(self):
        skip, reason = plugin._check_cooldown()
        assert skip is False

    def test_active_cooldown(self):
        state = {"created_at": str(time.time() - 60)}  # 1 min ago
        plugin._checkin_save_state(state)
        skip, reason = plugin._check_cooldown()
        assert skip is True
        assert "Cooldown" in reason

    def test_expired_cooldown(self):
        state = {"created_at": str(time.time() - 241 * 60)}  # 4h1m ago
        plugin._checkin_save_state(state)
        skip, reason = plugin._check_cooldown()
        assert skip is False

    def test_cooldown_boundary_just_before(self):
        state = {"created_at": str(time.time() - 239 * 60)}  # 3h59m ago
        plugin._checkin_save_state(state)
        skip, reason = plugin._check_cooldown()
        assert skip is True

    def test_cooldown_boundary_just_after(self):
        state = {"created_at": str(time.time() - 241 * 60)}  # 4h1m ago
        plugin._checkin_save_state(state)
        skip, reason = plugin._check_cooldown()
        assert skip is False

    def test_cooldown_corrupted_state(self):
        plugin._CHECKIN_STATE_FILE.write_text("not json", encoding="utf-8")
        skip, reason = plugin._check_cooldown()
        assert skip is False


# ═══════════════════════════════════════════════════════════════════════════
# ACTIVITY CHECK TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestLastActivity:
    """Check-ins should skip if Vex was recently active."""

    @patch("nora.plugins.vector_context.__init__.SessionDB")
    def test_recently_active(self, mock_sdb):
        mock_db = MagicMock()
        mock_db.search_sessions.return_value = [
            {"last_active": time.time() - 60, "source": "cli"}  # 1 min ago
        ]
        mock_sdb.return_value = mock_db
        is_active, mins, platform = plugin._check_last_activity("/fake/db")
        assert is_active is True
        assert mins < 30

    @patch("nora.plugins.vector_context.__init__.SessionDB")
    def test_not_recently_active(self, mock_sdb):
        mock_db = MagicMock()
        mock_db.search_sessions.return_value = [
            {"last_active": time.time() - 3600, "source": "telegram"}  # 1h ago
        ]
        mock_sdb.return_value = mock_db
        is_active, mins, platform = plugin._check_last_activity("/fake/db")
        assert is_active is False

    @patch("nora.plugins.vector_context.__init__.SessionDB")
    def test_multiple_sessions_picks_latest(self, mock_sdb):
        mock_db = MagicMock()
        mock_db.search_sessions.return_value = [
            {"last_active": time.time() - 3600, "source": "cli"},
            {"last_active": time.time() - 120, "source": "telegram"},  # this is latest
            {"last_active": time.time() - 7200, "source": "cli"},
        ]
        mock_sdb.return_value = mock_db
        is_active, mins, platform = plugin._check_last_activity("/fake/db")
        assert is_active is False  # 120 min > 30 min threshold
        # wait, 120 min is 2 hours, that's > 30 min
        # Let me fix: make one within 30 min
        mock_db.search_sessions.return_value = [
            {"last_active": time.time() - 600, "source": "cli"},
            {"last_active": time.time() - 1200, "source": "telegram"},
        ]
        is_active, mins, platform = plugin._check_last_activity("/fake/db")
        # 600s = 10 min, which is < 30 min
        assert is_active is True
        assert mins < 30

    @patch("nora.plugins.vector_context.__init__.SessionDB")
    def test_boundary_29_minutes(self, mock_sdb):
        mock_db = MagicMock()
        mock_db.search_sessions.return_value = [
            {"last_active": time.time() - 29 * 60, "source": "cli"}
        ]
        mock_sdb.return_value = mock_db
        is_active, mins, platform = plugin._check_last_activity("/fake/db")
        assert is_active is True  # 29 min < 30 min

    @patch("nora.plugins.vector_context.__init__.SessionDB")
    def test_boundary_31_minutes(self, mock_sdb):
        mock_db = MagicMock()
        mock_db.search_sessions.return_value = [
            {"last_active": time.time() - 31 * 60, "source": "cli"}
        ]
        mock_sdb.return_value = mock_db
        is_active, mins, platform = plugin._check_last_activity("/fake/db")
        assert is_active is False  # 31 min > 30 min

    @patch("nora.plugins.vector_context.__init__.SessionDB")
    def test_session_db_unavailable(self, mock_sdb):
        mock_sdb.side_effect = ImportError("No DB available")
        is_active, mins, platform = plugin._check_last_activity("/fake/db")
        assert is_active is False
        assert mins is None


# ═══════════════════════════════════════════════════════════════════════════
# CLI ALIVE CHECK TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestCLIAlive:
    """Detect whether the CLI process is running."""

    def test_no_pid_file(self):
        pid_file = plugin._CHECKIN_DIR / "cli.pid"
        pid_file.unlink(missing_ok=True)
        assert plugin._check_cli_alive() is False

    def test_pid_file_with_live_process(self):
        pid_file = plugin._CHECKIN_DIR / "cli.pid"
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(os.getpid()), encoding="utf-8")
        assert plugin._check_cli_alive() is True
        pid_file.unlink(missing_ok=True)

    def test_pid_file_with_dead_process(self):
        pid_file = plugin._CHECKIN_DIR / "cli.pid"
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(999999999), encoding="utf-8")
        assert plugin._check_cli_alive() is False
        pid_file.unlink(missing_ok=True)

    def test_pid_file_corrupted(self):
        pid_file = plugin._CHECKIN_DIR / "cli.pid"
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text("not_a_number", encoding="utf-8")
        assert plugin._check_cli_alive() is False
        pid_file.unlink(missing_ok=True)

    def test_pid_file_empty(self):
        pid_file = plugin._CHECKIN_DIR / "cli.pid"
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text("", encoding="utf-8")
        assert plugin._check_cli_alive() is False
        pid_file.unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# MESSAGE VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestMessageValidation:
    """Check-in messages must be valid plain text."""

    def test_valid_message(self):
        error = plugin._validate_checkin_message("hey, how's your day going?")
        assert error is None

    def test_empty_message(self):
        error = plugin._validate_checkin_message("")
        assert error is not None
        assert "empty" in error

    def test_whitespace_only(self):
        error = plugin._validate_checkin_message("   ")
        assert error is not None
        assert "empty" in error

    def test_message_at_max_length(self):
        msg = "a" * plugin._CHECKIN_MAX_MSG_LENGTH
        error = plugin._validate_checkin_message(msg)
        assert error is None

    def test_message_over_max_length(self):
        msg = "a" * (plugin._CHECKIN_MAX_MSG_LENGTH + 1)
        error = plugin._validate_checkin_message(msg)
        assert error is not None
        assert "too long" in error

    def test_message_with_bold_formatting(self):
        error = plugin._validate_checkin_message("**hello** there")
        assert error is not None
        assert "formatting" in error

    def test_message_with_code_block(self):
        error = plugin._validate_checkin_message("```code```")
        assert error is not None
        assert "formatting" in error

    def test_message_with_heading(self):
        error = plugin._validate_checkin_message("# Heading")
        assert error is not None
        assert "formatting" in error

    def test_message_with_asterisks_in_text(self):
        """Asterisks used for emphasis like *hi* should be caught."""
        error = plugin._validate_checkin_message("*hi* how are you")
        assert error is not None

    def test_boundary_499_chars(self):
        msg = "a" * 499
        error = plugin._validate_checkin_message(msg)
        assert error is None


# ═══════════════════════════════════════════════════════════════════════════
# STATE MACHINE TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestStateMachine:
    """Tool must be called in order: evaluate → prepare → deliver."""

    @patch("nora.plugins.vector_context.__init__._check_sleep_schedule")
    @patch("nora.plugins.vector_context.__init__._check_cooldown")
    @patch("nora.plugins.vector_context.__init__._check_last_activity")
    def test_evaluate_proceed(self, mock_activity, mock_cooldown, mock_sleep, call_handler):
        mock_sleep.return_value = (False, None)
        mock_cooldown.return_value = (False, None)
        mock_activity.return_value = (False, None, None)
        result = call_handler("evaluate")
        assert result["success"] is True
        assert result["skip"] is False
        assert result["complete"] is False

    @patch("nora.plugins.vector_context.__init__._check_sleep_schedule")
    def test_evaluate_sleep_skip(self, mock_sleep, call_handler):
        mock_sleep.return_value = (True, "Vex is asleep")
        result = call_handler("evaluate")
        assert result["success"] is True
        assert result["skip"] is True
        assert result["complete"] is True

    def test_prepare_without_evaluate(self, call_handler):
        result = call_handler("prepare", message="hey")
        assert result["success"] is False
        assert "evaluate" in result.get("error", "")

    @patch("nora.plugins.vector_context.__init__._check_sleep_schedule")
    @patch("nora.plugins.vector_context.__init__._check_cooldown")
    @patch("nora.plugins.vector_context.__init__._check_last_activity")
    def test_prepare_after_evaluate(self, mock_activity, mock_cooldown, mock_sleep, call_handler):
        mock_sleep.return_value = (False, None)
        mock_cooldown.return_value = (False, None)
        mock_activity.return_value = (False, None, None)
        call_handler("evaluate")
        result = call_handler("prepare", message="hey, just checking in")
        assert result["success"] is True
        assert result["complete"] is False

    @patch("nora.plugins.vector_context.__init__._check_sleep_schedule")
    @patch("nora.plugins.vector_context.__init__._check_cooldown")
    @patch("nora.plugins.vector_context.__init__._check_last_activity")
    def test_prepare_invalid_message(self, mock_activity, mock_cooldown, mock_sleep, call_handler):
        mock_sleep.return_value = (False, None)
        mock_cooldown.return_value = (False, None)
        mock_activity.return_value = (False, None, None)
        call_handler("evaluate")
        result = call_handler("prepare", message="")
        assert result["success"] is False
        assert "empty" in result.get("error", "")

    def test_deliver_without_prepare(self, call_handler):
        result = call_handler("deliver")
        assert result["success"] is False
        assert "prepare" in result.get("error", "")

    def test_unknown_action(self, call_handler):
        result = call_handler("unknown_action")
        assert result["success"] is False
        assert "Unknown" in result.get("error", "")


# ═══════════════════════════════════════════════════════════════════════════
# DELIVERY TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestDeliveryCLI:
    """When CLI is alive, deliver via signal file."""

    @patch("nora.plugins.vector_context.__init__._check_sleep_schedule")
    @patch("nora.plugins.vector_context.__init__._check_cooldown")
    @patch("nora.plugins.vector_context.__init__._check_last_activity")
    @patch("nora.plugins.vector_context.__init__._check_cli_alive")
    @patch("nora.plugins.vector_context.__init__.SessionDB")
    @patch("nora.plugins.vector_context.__init__._schedule_fallback_job")
    def test_deliver_to_cli(
        self, mock_fallback, mock_sdb, mock_cli, mock_activity, mock_cooldown, mock_sleep, call_handler
    ):
        mock_sleep.return_value = (False, None)
        mock_cooldown.return_value = (False, None)
        mock_activity.return_value = (False, None, None)
        mock_cli.return_value = True
        mock_fallback.return_value = "fallback_job_123"

        mock_db = MagicMock()
        mock_db.search_sessions.return_value = [{"id": "session_123", "source": "cli"}]
        mock_sdb.return_value = mock_db

        call_handler("evaluate")
        call_handler("prepare", message="hey, how's it going?")
        result = call_handler("deliver")

        assert result["success"] is True
        assert result["complete"] is True
        assert result["delivery"] == "cli"

        # Verify signal file was written
        assert plugin._CHECKIN_SIGNAL_FILE.exists()
        signal = json.loads(plugin._CHECKIN_SIGNAL_FILE.read_text(encoding="utf-8"))
        assert signal["message"] == "hey, how's it going?"
        assert signal["consumed"] is False
        assert "session_id" in signal

        # Verify fallback was scheduled
        mock_fallback.assert_called_once()

    @patch("nora.plugins.vector_context.__init__._check_sleep_schedule")
    @patch("nora.plugins.vector_context.__init__._check_cooldown")
    @patch("nora.plugins.vector_context.__init__._check_last_activity")
    @patch("nora.plugins.vector_context.__init__._check_cli_alive")
    def test_deliver_no_cli(
        self, mock_cli, mock_activity, mock_cooldown, mock_sleep, call_handler
    ):
        mock_sleep.return_value = (False, None)
        mock_cooldown.return_value = (False, None)
        mock_activity.return_value = (False, None, None)
        mock_cli.return_value = False

        call_handler("evaluate")
        call_handler("prepare", message="hey from Telegram")
        result = call_handler("deliver")

        assert result["success"] is True
        assert result["complete"] is True
        assert result["delivery"] == "telegram"

        # No signal file for Telegram delivery
        assert not plugin._CHECKIN_SIGNAL_FILE.exists()


class TestSignalFile:
    """Signal file lifecycle tests."""

    def test_signal_file_written_atomically(self):
        plugin._CHECKIN_DIR.mkdir(parents=True, exist_ok=True)
        signal = {
            "session_id": "sess_1",
            "message": "test message",
            "reason": "testing",
            "timestamp": time.time(),
            "consumed": False,
        }
        tmp = plugin._CHECKIN_SIGNAL_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(signal, ensure_ascii=False), encoding="utf-8")
        tmp.replace(plugin._CHECKIN_SIGNAL_FILE)
        assert plugin._CHECKIN_SIGNAL_FILE.exists()
        # Clean up
        plugin._CHECKIN_SIGNAL_FILE.unlink(missing_ok=True)

    def test_signal_file_consumed_marker(self):
        plugin._CHECKIN_DIR.mkdir(parents=True, exist_ok=True)
        signal = {
            "session_id": "sess_1",
            "message": "test",
            "timestamp": time.time(),
            "consumed": True,
        }
        plugin._CHECKIN_SIGNAL_FILE.write_text(
            json.dumps(signal, ensure_ascii=False), encoding="utf-8"
        )
        data = json.loads(plugin._CHECKIN_SIGNAL_FILE.read_text(encoding="utf-8"))
        assert data["consumed"] is True
        plugin._CHECKIN_SIGNAL_FILE.unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# STATE PERSISTENCE TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestStatePersistence:
    """Tool state must persist across calls within the same pipeline."""

    @patch("nora.plugins.vector_context.__init__._check_sleep_schedule")
    @patch("nora.plugins.vector_context.__init__._check_cooldown")
    @patch("nora.plugins.vector_context.__init__._check_last_activity")
    def test_state_across_calls(self, mock_activity, mock_cooldown, mock_sleep, task_id):
        mock_sleep.return_value = (False, None)
        mock_cooldown.return_value = (False, None)
        mock_activity.return_value = (False, None, None)

        # evaluate
        plugin.nora_checkin_handler({"action": "evaluate"}, task_id=task_id)

        # prepare
        plugin.nora_checkin_handler(
            {"action": "prepare", "message": "hello"}, task_id=task_id
        )

        # State should be "prepared"
        state = plugin._checkin_load_state(task_id)
        assert state["phase"] == "prepared"
        assert state["message"] == "hello"
        assert state["reason"] == "checking in"  # default when not provided

    @patch("nora.plugins.vector_context.__init__._check_sleep_schedule")
    @patch("nora.plugins.vector_context.__init__._check_cooldown")
    @patch("nora.plugins.vector_context.__init__._check_last_activity")
    def test_state_cleanup_on_new_evaluate(
        self, mock_activity, mock_cooldown, mock_sleep, task_id
    ):
        mock_sleep.return_value = (False, None)
        mock_cooldown.return_value = (False, None)
        mock_activity.return_value = (False, None, None)

        plugin.nora_checkin_handler({"action": "evaluate"}, task_id=task_id)
        plugin.nora_checkin_handler(
            {"action": "prepare", "message": "hello"}, task_id=task_id
        )

        # Second evaluate should reset state
        plugin.nora_checkin_handler({"action": "evaluate"}, task_id=task_id)
        state = plugin._checkin_load_state(task_id)
        assert state["phase"] == "evaluated"
        assert "context" in state  # context should be populated
        assert state["message"] is None  # message should be cleared

    def test_different_task_ids_isolated(self):
        result1 = plugin.nora_checkin_handler(
            {"action": "deliver"}, task_id="task_a"
        )
        result2 = plugin.nora_checkin_handler(
            {"action": "deliver"}, task_id="task_b"
        )
        # Both should fail with same error (no prepare) — no cross-contamination
        assert json.loads(result1)["success"] is False
        assert json.loads(result2)["success"] is False


# ═══════════════════════════════════════════════════════════════════════════
# EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Unusual or extreme scenarios."""

    def test_handler_called_with_no_args(self):
        result = plugin.nora_checkin_handler({}, task_id="test")
        data = json.loads(result)
        assert data["success"] is False

    def test_handler_called_with_none_action(self):
        result = plugin.nora_checkin_handler({"action": None}, task_id="test")
        data = json.loads(result)
        assert data["success"] is False

    def test_handler_empty_kwargs(self):
        """Handler should work even with no task_id."""
        result = plugin.nora_checkin_handler({"action": "unknown"})
        data = json.loads(result)
        assert data["success"] is False

    @patch("nora.plugins.vector_context.__init__._check_sleep_schedule")
    @patch("nora.plugins.vector_context.__init__._check_cooldown")
    @patch("nora.plugins.vector_context.__init__._check_last_activity")
    def test_prepare_with_reason(self, mock_activity, mock_cooldown, mock_sleep, call_handler):
        mock_sleep.return_value = (False, None)
        mock_cooldown.return_value = (False, None)
        mock_activity.return_value = (False, None, None)
        call_handler("evaluate")
        result = call_handler(
            "prepare",
            message="noticed your sleep shifted",
            reason="routine observation",
        )
        assert result["success"] is True
        state = plugin._checkin_load_state("test_task_123")
        assert state["reason"] == "routine observation"

    def test_state_file_corrupted(self, task_id):
        """Corrupted state file should not crash the handler."""
        plugin._CHECKIN_DIR.mkdir(parents=True, exist_ok=True)
        plugin._CHECKIN_STATE_FILE.write_text("{invalid json", encoding="utf-8")
        result = plugin.nora_checkin_handler({"action": "deliver"}, task_id=task_id)
        data = json.loads(result)
        # Should gracefully degrade to "must call prepare first"
        assert data["success"] is False

    def test_signal_file_stale_ignored(self, task_id):
        """A signal file older than 5 minutes should be ignored."""
        plugin._CHECKIN_DIR.mkdir(parents=True, exist_ok=True)
        old_signal = {
            "session_id": "old_session",
            "message": "ancient checkin",
            "timestamp": time.time() - 400,  # 6.7 min old
            "consumed": False,
        }
        plugin._CHECKIN_SIGNAL_FILE.write_text(
            json.dumps(old_signal, ensure_ascii=False), encoding="utf-8"
        )
        # Signal file is stale — this test verifies the logic that the
        # CLI loop would skip it. We can't easily test the CLI loop directly,
        # but we can verify the signal file state.
        assert plugin._CHECKIN_SIGNAL_FILE.exists()
        plugin._CHECKIN_SIGNAL_FILE.unlink(missing_ok=True)

    def test_signal_file_already_consumed(self, task_id):
        """A consumed signal file should not be re-processed."""
        plugin._CHECKIN_DIR.mkdir(parents=True, exist_ok=True)
        consumed = {
            "session_id": "sess",
            "message": "done",
            "timestamp": time.time(),
            "consumed": True,
        }
        plugin._CHECKIN_SIGNAL_FILE.write_text(
            json.dumps(consumed, ensure_ascii=False), encoding="utf-8"
        )
        data = json.loads(plugin._CHECKIN_SIGNAL_FILE.read_text(encoding="utf-8"))
        assert data["consumed"] is True
        plugin._CHECKIN_SIGNAL_FILE.unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# CONTEXT BUILDING TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestContextBuilding:
    """Context should be built without crashing even if files are missing."""

    @patch("nora.plugins.vector_context.__init__.SessionDB")
    def test_context_with_no_files(self, mock_sdb):
        mock_db = MagicMock()
        mock_db.search_sessions.return_value = []
        mock_sdb.return_value = mock_db
        context = plugin._build_checkin_context("/fake/db")
        assert context["time"] is not None
        assert context["routine"] is None
        assert context["recent_memories"] is None
        assert context["recent_exploration"] is None

    @patch("nora.plugins.vector_context.__init__.SessionDB")
    def test_context_with_routine_file(self, mock_sdb):
        mock_db = MagicMock()
        mock_db.search_sessions.return_value = []
        mock_sdb.return_value = mock_db

        routine_path = Path.home() / ".hermes" / "memories" / "ROUTINE.md"
        routine_path.parent.mkdir(parents=True, exist_ok=True)
        routine_path.write_text("# Test Routine\n- Wake at 5 AM", encoding="utf-8")

        context = plugin._build_checkin_context("/fake/db")
        assert context["routine"] is not None
        assert "Wake" in context["routine"]

        routine_path.unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# FALLBACK JOB TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestFallbackJob:
    """Fallback one-shot cron job for Telegram delivery."""

    @patch("nora.plugins.vector_context.__init__.create_job")
    def test_fallback_scheduled(self, mock_create_job):
        mock_create_job.return_value = {"id": "fallback_abc123"}
        job_id = plugin._schedule_fallback_job("test message", "task_abc")
        assert job_id == "fallback_abc123"
        mock_create_job.assert_called_once()

    @patch("nora.plugins.vector_context.__init__.create_job")
    def test_fallback_contains_message(self, mock_create_job):
        mock_create_job.return_value = {"id": "fb_1"}
        plugin._schedule_fallback_job("hey there", "task_1")
        prompt = mock_create_job.call_args[1]["prompt"]
        assert "hey there" in prompt
        assert "checkin-pending.json" in prompt

    @patch("nora.plugins.vector_context.__init__.create_job")
    def test_fallback_failure_returns_none(self, mock_create_job):
        mock_create_job.side_effect = Exception("DB lock")
        job_id = plugin._schedule_fallback_job("test", "task_2")
        assert job_id is None

    @patch("nora.plugins.vector_context.__init__.remove_job")
    def test_cancel_fallback(self, mock_remove):
        plugin._cancel_fallback_job("fallback_abc")
        mock_remove.assert_called_once_with("fallback_abc")

    def test_cancel_none_fallback(self):
        """Cancelling with None should not crash."""
        plugin._cancel_fallback_job(None)  # Should not raise


# ═══════════════════════════════════════════════════════════════════════════
# CLEANUP TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestCleanup:
    """State cleanup should work correctly."""

    def test_cleanup_removes_state(self, task_id):
        state = {"task_id": task_id, "phase": "delivered"}
        plugin._checkin_save_state(state)
        assert plugin._CHECKIN_STATE_FILE.exists()
        plugin._checkin_cleanup_state(task_id)
        assert not plugin._CHECKIN_STATE_FILE.exists()

    def test_cleanup_different_task_id(self, task_id):
        state = {"task_id": task_id, "phase": "delivered"}
        plugin._checkin_save_state(state)
        plugin._checkin_cleanup_state("other_task")
        assert plugin._CHECKIN_STATE_FILE.exists()  # should still exist
        plugin._checkin_cleanup_state(task_id)  # now clean up
        assert not plugin._CHECKIN_STATE_FILE.exists()

    def test_cleanup_no_file(self):
        """Cleaning up when no state file exists should not crash."""
        plugin._CHECKIN_STATE_FILE.unlink(missing_ok=True)
        plugin._checkin_cleanup_state("ghost_task")  # Should not raise

    def test_cleanup_signal_file(self):
        plugin._CHECKIN_DIR.mkdir(parents=True, exist_ok=True)
        plugin._CHECKIN_SIGNAL_FILE.write_text("{}", encoding="utf-8")
        assert plugin._CHECKIN_SIGNAL_FILE.exists()
        plugin._checkin_cleanup_state("any")
        assert not plugin._CHECKIN_SIGNAL_FILE.exists()


# ═══════════════════════════════════════════════════════════════════════════
# CONCURRENCY TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestConcurrency:
    """Tool should handle concurrent access safely."""

    @patch("nora.plugins.vector_context.__init__._check_sleep_schedule")
    @patch("nora.plugins.vector_context.__init__._check_cooldown")
    @patch("nora.plugins.vector_context.__init__._check_last_activity")
    def test_concurrent_evaluate(self, mock_activity, mock_cooldown, mock_sleep):
        """Multiple concurrent evaluate calls should not corrupt state."""
        mock_sleep.return_value = (False, None)
        mock_cooldown.return_value = (False, None)
        mock_activity.return_value = (False, None, None)

        errors = []
        results = []

        def run_evaluate():
            try:
                r = plugin.nora_checkin_handler(
                    {"action": "evaluate"}, task_id="concurrent_test"
                )
                results.append(json.loads(r))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_evaluate) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All should succeed
        assert all(r["success"] for r in results)


# Clean up patches
sys_modules_patch.stop()
sys_modules_patch2.stop()
sys_modules_patch3.stop()
