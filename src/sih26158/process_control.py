from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import IO


class ProcessCancelledError(RuntimeError):
    """Raised after a cancellation request terminates an external process tree."""


class ProcessTimeoutError(RuntimeError):
    """Raised after a stage deadline terminates an external process tree."""


class ManagedProcessExecutor:
    """Run external commands with cancellation, a shared deadline and heartbeats."""

    def __init__(
        self,
        *,
        timeout_s: float,
        cancel_requested: Callable[[], bool],
        heartbeat: Callable[[], None],
        heartbeat_interval_s: float = 10.0,
        poll_interval_s: float = 0.25,
        terminate_grace_s: float = 5.0,
    ) -> None:
        self.deadline = time.monotonic() + timeout_s
        self.timeout_s = timeout_s
        self.cancel_requested = cancel_requested
        self.heartbeat = heartbeat
        self.heartbeat_interval_s = heartbeat_interval_s
        self.poll_interval_s = poll_interval_s
        self.terminate_grace_s = terminate_grace_s

    def _terminate_tree(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "nt":  # pragma: no cover - exercised on Windows hosts
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(process.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            process.terminate()
        try:
            process.wait(timeout=self.terminate_grace_s)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            if os.name == "nt":  # pragma: no cover - exercised on Windows hosts
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            process.kill()
        process.wait()

    def run(
        self,
        command: Sequence[str],
        *,
        stdout: int | IO[str] | None = None,
        stderr: int | IO[str] | None = None,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        if self.cancel_requested():
            raise ProcessCancelledError("Run cancellation was requested before command start")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise ProcessTimeoutError(
                f"Stage exceeded its {self.timeout_s:.0f}-second execution timeout"
            )
        if capture_output:
            stdout = subprocess.PIPE
            stderr = subprocess.PIPE
        creationflags = 0
        start_new_session = os.name != "nt"
        if os.name == "nt":  # pragma: no cover - exercised on Windows hosts
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        process = subprocess.Popen(
            list(command),
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=start_new_session,
            creationflags=creationflags,
        )
        last_heartbeat = 0.0
        captured_stdout: str | None = None
        captured_stderr: str | None = None
        while True:
            now = time.monotonic()
            if self.cancel_requested():
                self._terminate_tree(process)
                raise ProcessCancelledError(
                    f"Run cancellation terminated external command: {Path(command[0]).name}"
                )
            if now >= self.deadline:
                self._terminate_tree(process)
                raise ProcessTimeoutError(
                    f"Stage exceeded its {self.timeout_s:.0f}-second timeout while running "
                    f"{Path(command[0]).name}"
                )
            if now - last_heartbeat >= self.heartbeat_interval_s:
                try:
                    self.heartbeat()
                except Exception:
                    self._terminate_tree(process)
                    raise
                last_heartbeat = now
            wait_s = min(self.poll_interval_s, max(self.deadline - now, 0.01))
            try:
                captured_stdout, captured_stderr = process.communicate(timeout=wait_s)
                break
            except subprocess.TimeoutExpired:
                continue
        return subprocess.CompletedProcess(
            list(command), process.returncode, captured_stdout, captured_stderr
        )
