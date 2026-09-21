from __future__ import annotations

import sys
import time

import pytest

from sih26158.process_control import (
    ManagedProcessExecutor,
    ProcessCancelledError,
    ProcessTimeoutError,
)


def test_managed_process_emits_heartbeats_and_captures_output(tmp_path) -> None:
    heartbeats: list[float] = []
    process_state = tmp_path / ".active_process.json"
    executor = ManagedProcessExecutor(
        timeout_s=2,
        cancel_requested=lambda: False,
        heartbeat=lambda: heartbeats.append(time.monotonic()),
        heartbeat_interval_s=0.01,
        poll_interval_s=0.01,
        process_state_path=process_state,
    )

    result = executor.run(
        [sys.executable, "-c", "import time; time.sleep(0.05); print('ready')"],
        capture_output=True,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "ready"
    assert len(heartbeats) >= 2
    assert not process_state.exists()


def test_managed_process_honours_cancellation_and_terminates_quickly(tmp_path) -> None:
    started = time.monotonic()
    process_state = tmp_path / ".active_process.json"
    executor = ManagedProcessExecutor(
        timeout_s=5,
        cancel_requested=lambda: time.monotonic() - started > 0.05,
        heartbeat=lambda: None,
        poll_interval_s=0.01,
        terminate_grace_s=0.05,
        process_state_path=process_state,
    )

    with pytest.raises(ProcessCancelledError, match="terminated external command"):
        executor.run([sys.executable, "-c", "import time; time.sleep(5)"])

    assert time.monotonic() - started < 1
    assert not process_state.exists()


def test_managed_process_enforces_shared_stage_deadline() -> None:
    executor = ManagedProcessExecutor(
        timeout_s=0.05,
        cancel_requested=lambda: False,
        heartbeat=lambda: None,
        poll_interval_s=0.01,
        terminate_grace_s=0.05,
    )

    with pytest.raises(ProcessTimeoutError, match="timeout"):
        executor.run([sys.executable, "-c", "import time; time.sleep(5)"])
