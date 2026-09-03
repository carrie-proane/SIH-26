"""Optional AI surface-completion integration with explicit uncertainty provenance.

The repository deliberately does not bundle an untrained network or silently download
weights. A completion runtime is integrated through a small command-line protocol so a
trained model can be added without coupling it to the FastAPI process. Every output is
visual-only and remains ineligible for metric measurement.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..infrastructure.storage import atomic_json, runtime_environment

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class SurfaceCompletionError(RuntimeError):
    """A configured completion provider failed its explicit runtime contract."""


@dataclass(frozen=True)
class SurfaceCompletionContext:
    run_dir: Path
    source_geometry: Path
    source_geometry_kind: str
    camera_poses: Path
    selected_frames: Path
    model_path: Path | None
    sample_count: int
    coordinate_frame: str = "LOCAL_ENU_METRES"


@dataclass
class SurfaceCompletionResult:
    status: str
    provider: str
    artifacts: list[Path] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)
    reason: str | None = None


class SurfaceCompletionProvider(ABC):
    """Inference boundary for a model that predicts unobserved geometry."""

    name = "abstract"

    @abstractmethod
    def availability(self, context: SurfaceCompletionContext) -> tuple[bool, str]:
        raise NotImplementedError

    @abstractmethod
    def run(self, context: SurfaceCompletionContext, request_path: Path) -> list[Path]:
        raise NotImplementedError


class UnavailableSurfaceCompletionProvider(SurfaceCompletionProvider):
    name = "unavailable"

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def availability(self, context: SurfaceCompletionContext) -> tuple[bool, str]:
        del context
        return False, self.reason

    def run(self, context: SurfaceCompletionContext, request_path: Path) -> list[Path]:
        del context, request_path
        raise SurfaceCompletionError(self.reason)


class ExternalSurfaceCompletionProvider(SurfaceCompletionProvider):
    """Run an isolated model executable using a versioned JSON request.

    The executable receives ``--request`` and ``--output-dir``. It must produce
    ``completed_mesh.ply`` and ``uncertainty.json`` in the output directory.
    """

    name = "external"

    def __init__(
        self,
        binary: str | None = None,
        *,
        runner: CommandRunner = subprocess.run,
        timeout_s: int = 3600,
    ) -> None:
        self.binary = binary or os.getenv("SIH_SURFACE_COMPLETION_BIN", "")
        self.runner = runner
        self.timeout_s = timeout_s

    def availability(self, context: SurfaceCompletionContext) -> tuple[bool, str]:
        if not self.binary:
            return False, "SIH_SURFACE_COMPLETION_BIN is not configured."
        resolved = shutil.which(self.binary)
        if resolved is None and not Path(self.binary).is_file():
            return False, f"Surface-completion executable was not found: {self.binary}"
        if context.model_path is None:
            return False, "No local surface-completion model weights were configured."
        if not context.model_path.is_file():
            return False, f"Surface-completion weights were not found: {context.model_path}"
        return True, "External completion runtime and local weights are available."

    @staticmethod
    def _validate_ply(path: Path) -> None:
        if not path.is_file() or path.stat().st_size == 0:
            raise SurfaceCompletionError("The completion runtime did not produce completed_mesh.ply.")
        with path.open("rb") as stream:
            if stream.read(4) != b"ply\n":
                raise SurfaceCompletionError("completed_mesh.ply does not have a valid PLY header.")

    @staticmethod
    def _validate_uncertainty(path: Path) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SurfaceCompletionError("uncertainty.json is missing or invalid JSON.") from exc
        required = {"schema_version", "semantics", "summary"}
        if not isinstance(payload, dict) or not required.issubset(payload):
            raise SurfaceCompletionError(
                "uncertainty.json must declare schema_version, semantics, and summary."
            )
        if payload.get("semantics") != "HIGHER_IS_MORE_UNCERTAIN":
            raise SurfaceCompletionError(
                "uncertainty.json must use HIGHER_IS_MORE_UNCERTAIN semantics."
            )

    def run(self, context: SurfaceCompletionContext, request_path: Path) -> list[Path]:
        output_dir = context.run_dir / "completion"
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = context.run_dir / "logs" / "surface_completion.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.binary,
            "--request",
            str(request_path),
            "--output-dir",
            str(output_dir),
        ]
        result = self.runner(
            command,
            cwd=context.run_dir,
            capture_output=True,
            text=True,
            check=False,
            timeout=self.timeout_s,
        )
        log_path.write_text(
            f"command: {' '.join(command)}\nexit_code: {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}\n",
            encoding="utf-8",
        )
        if result.returncode:
            raise SurfaceCompletionError(
                f"Surface-completion runtime failed with exit code {result.returncode}; "
                "inspect logs/surface_completion.log."
            )
        mesh_path = output_dir / "completed_mesh.ply"
        uncertainty_path = output_dir / "uncertainty.json"
        self._validate_ply(mesh_path)
        self._validate_uncertainty(uncertainty_path)
        return [mesh_path, uncertainty_path, log_path]


def _request_payload(context: SurfaceCompletionContext, provider: str) -> dict[str, object]:
    def relative(path: Path) -> str:
        return path.resolve().relative_to(context.run_dir.resolve()).as_posix()

    return {
        "schema_version": "1.0",
        "task": "GENERATIVE_SURFACE_COMPLETION",
        "provider": provider,
        "inputs": {
            "source_geometry": relative(context.source_geometry),
            "source_geometry_kind": context.source_geometry_kind,
            "camera_poses": relative(context.camera_poses),
            "selected_frames": relative(context.selected_frames),
            "coordinate_frame": context.coordinate_frame,
        },
        "model": {
            "weights_path": str(context.model_path) if context.model_path else None,
            "sample_count": context.sample_count,
        },
        "required_outputs": {
            "mesh": "completion/completed_mesh.ply",
            "uncertainty": "completion/uncertainty.json",
            "uncertainty_semantics": "HIGHER_IS_MORE_UNCERTAIN",
        },
        "scientific_contract": {
            "observed_geometry_must_be_preserved": True,
            "generated_geometry_label": "AI_ASSISTED_NOT_MEASURABLE",
            "measurement_eligible": False,
            "multiple_hypotheses_recommended": context.sample_count > 1,
        },
    }


def run_surface_completion_stage(
    context: SurfaceCompletionContext,
    provider: SurfaceCompletionProvider,
) -> SurfaceCompletionResult:
    """Run an optional provider without invalidating the observed reconstruction."""

    request_path = context.run_dir / "completion_request.json"
    report_path = context.run_dir / "completion_report.json"
    atomic_json(request_path, _request_payload(context, provider.name))
    available, reason = provider.availability(context)
    if not available:
        warning = {"code": "SURFACE_COMPLETION_UNAVAILABLE", "message": reason}
        atomic_json(
            report_path,
            {
                "schema_version": "1.0",
                "status": "BLOCKED",
                "provider": provider.name,
                "reason": reason,
                "measurement_eligible": False,
                "generated_geometry_label": "AI_ASSISTED_NOT_MEASURABLE",
                "runtime_environment": runtime_environment(),
            },
        )
        return SurfaceCompletionResult(
            status="BLOCKED",
            provider=provider.name,
            artifacts=[request_path, report_path],
            warnings=[warning],
            reason=reason,
        )
    try:
        produced = provider.run(context, request_path)
    except (OSError, subprocess.SubprocessError, SurfaceCompletionError) as exc:
        reason = str(exc)
        warning = {"code": "SURFACE_COMPLETION_FAILED", "message": reason}
        atomic_json(
            report_path,
            {
                "schema_version": "1.0",
                "status": "FAILED",
                "provider": provider.name,
                "reason": reason,
                "measurement_eligible": False,
                "generated_geometry_label": "AI_ASSISTED_NOT_MEASURABLE",
                "runtime_environment": runtime_environment(),
            },
        )
        return SurfaceCompletionResult(
            status="FAILED",
            provider=provider.name,
            artifacts=[request_path, report_path],
            warnings=[warning],
            reason=reason,
        )
    atomic_json(
        report_path,
        {
            "schema_version": "1.0",
            "status": "COMPLETED",
            "provider": provider.name,
            "source_geometry": context.source_geometry_kind,
            "sample_count": context.sample_count,
            "measurement_eligible": False,
            "generated_geometry_label": "AI_ASSISTED_NOT_MEASURABLE",
            "artifacts": [path.relative_to(context.run_dir).as_posix() for path in produced],
            "runtime_environment": runtime_environment(),
        },
    )
    return SurfaceCompletionResult(
        status="COMPLETED",
        provider=provider.name,
        artifacts=[request_path, report_path, *produced],
    )
