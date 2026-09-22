from __future__ import annotations

import hashlib
import json
import math
import mimetypes
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import uuid
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import BinaryIO, Self

from .models import (
    ArtifactEntry,
    InputAsset,
    MeasurementCreate,
    MeasurementRecord,
    ProjectManifest,
    ProvenanceOrigin,
    RunConfig,
    RunRecord,
    utc_now,
)

ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,80}$")


class RunExecutionLock:
    """Cross-process advisory lock whose OS handle is released after a crash."""

    def __init__(self, path: Path, *, blocking: bool = False) -> None:
        self.path = path
        self.blocking = blocking
        self._stream: BinaryIO | None = None
        self.acquired = False

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a+b")
        self._stream.seek(0, os.SEEK_END)
        if self._stream.tell() == 0:
            self._stream.write(b"0")
            self._stream.flush()
        try:
            if os.name == "nt":  # pragma: no cover - exercised on Windows CI/hosts
                import msvcrt

                self._stream.seek(0)
                mode = msvcrt.LK_LOCK if self.blocking else msvcrt.LK_NBLCK
                msvcrt.locking(self._stream.fileno(), mode, 1)
            else:
                import fcntl

                operation = fcntl.LOCK_EX
                if not self.blocking:
                    operation |= fcntl.LOCK_NB
                fcntl.flock(self._stream.fileno(), operation)
        except (BlockingIOError, OSError):
            self._stream.close()
            self._stream = None
            return self
        self.acquired = True
        return self

    def __exit__(self, *_: object) -> None:
        if self._stream is None:
            return
        try:
            if self.acquired:
                if os.name == "nt":  # pragma: no cover - exercised on Windows CI/hosts
                    import msvcrt

                    self._stream.seek(0)
                    msvcrt.locking(self._stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        finally:
            self._stream.close()
            self._stream = None
            self.acquired = False


def combine_provenance(
    video_origin: ProvenanceOrigin, telemetry_origin: ProvenanceOrigin
) -> ProvenanceOrigin:
    """Return the most conservative immutable evidence classification."""
    origins = {video_origin, telemetry_origin}
    if ProvenanceOrigin.SYNTHETIC in origins:
        return ProvenanceOrigin.SYNTHETIC
    if ProvenanceOrigin.UNKNOWN in origins:
        return ProvenanceOrigin.UNKNOWN
    if ProvenanceOrigin.DERIVED in origins:
        return ProvenanceOrigin.DERIVED
    return ProvenanceOrigin.REAL


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def runtime_environment() -> dict[str, str | None]:
    environment: dict[str, str | None] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for package in (
        "fastapi",
        "numpy",
        "opencv-python-headless",
        "opencv-python",
        "Pillow",
        "pydantic",
        "trimesh",
        "ultralytics",
    ):
        try:
            environment[package] = version(package)
        except PackageNotFoundError:
            environment[package] = None
    for tool, flag in (("ffmpeg", "-version"), ("ffprobe", "-version"), ("colmap", "-h")):
        executable = shutil.which(tool)
        if executable is None:
            environment[tool] = None
            continue
        try:
            completed = subprocess.run(
                [executable, flag],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            output = (completed.stdout or completed.stderr).splitlines()
            environment[tool] = output[0][:240] if output else "installed"
        except (OSError, subprocess.TimeoutExpired):
            environment[tool] = "installed; version check failed"
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout.strip()
        )
        environment["git_revision"] = revision
        environment["git_dirty"] = str(dirty).lower()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        environment["git_revision"] = None
        environment["git_dirty"] = None
    return environment


class ProjectStore:
    def __init__(self, root: str | Path = "data/projects") -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _validate_id(self, value: str) -> str:
        if not ID_PATTERN.fullmatch(value):
            raise ValueError("Invalid identifier")
        return value

    def project_dir(self, project_id: str) -> Path:
        return self.root / self._validate_id(project_id)

    def run_dir(self, project_id: str, run_id: str) -> Path:
        return self.project_dir(project_id) / "runs" / self._validate_id(run_id)

    @staticmethod
    def new_id(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    def _copy_asset(
        self,
        input_dir: Path,
        role: str,
        filename: str,
        source: Path | BinaryIO,
        media_type: str | None = None,
        origin: ProvenanceOrigin = ProvenanceOrigin.UNKNOWN,
    ) -> InputAsset:
        safe_name = Path(filename).name
        if not safe_name or safe_name in {".", ".."}:
            raise ValueError("Invalid asset filename")
        target = input_dir / f"{role}_{safe_name}"
        if isinstance(source, Path):
            shutil.copyfile(source, target)
        else:
            with target.open("wb") as output:
                shutil.copyfileobj(source, output)
        return InputAsset(
            role=role,  # type: ignore[arg-type]
            original_name=safe_name,
            relative_path=str(target.relative_to(input_dir.parent)),
            size_bytes=target.stat().st_size,
            sha256=sha256_file(target),
            media_type=media_type or mimetypes.guess_type(safe_name)[0] or "application/octet-stream",
            origin=origin,
        )

    def create_project(
        self,
        *,
        name: str,
        description: str,
        video_name: str,
        video: Path | BinaryIO,
        telemetry_name: str,
        telemetry: Path | BinaryIO,
        data_classification: str = "PUBLIC_DEMO",
        video_origin: ProvenanceOrigin | str = ProvenanceOrigin.UNKNOWN,
        telemetry_origin: ProvenanceOrigin | str = ProvenanceOrigin.UNKNOWN,
    ) -> ProjectManifest:
        video_origin = ProvenanceOrigin(video_origin)
        telemetry_origin = ProvenanceOrigin(telemetry_origin)
        source_provenance = combine_provenance(video_origin, telemetry_origin)
        project_id = self.new_id("prj")
        project_dir = self.project_dir(project_id)
        input_dir = project_dir / "input"
        with self._lock:
            input_dir.mkdir(parents=True, exist_ok=False)
            try:
                assets = [
                    self._copy_asset(
                        input_dir, "video", video_name, video, origin=video_origin
                    ),
                    self._copy_asset(
                        input_dir,
                        "telemetry",
                        telemetry_name,
                        telemetry,
                        origin=telemetry_origin,
                    ),
                ]
                manifest = ProjectManifest(
                    project_id=project_id,
                    name=name,
                    description=description,
                    data_classification=data_classification,  # type: ignore[arg-type]
                    assets=assets,
                    source_provenance=source_provenance,
                    video_origin=video_origin,
                    telemetry_origin=telemetry_origin,
                )
                atomic_json(project_dir / "manifest.json", manifest.model_dump(mode="json"))
                return manifest
            except Exception:
                shutil.rmtree(project_dir, ignore_errors=True)
                raise

    def get_project(self, project_id: str) -> ProjectManifest:
        path = self.project_dir(project_id) / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(project_id)
        return ProjectManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def list_projects(self) -> list[ProjectManifest]:
        projects: list[ProjectManifest] = []
        for path in sorted(self.root.glob("*/manifest.json")):
            try:
                projects.append(ProjectManifest.model_validate_json(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return sorted(projects, key=lambda item: item.created_at, reverse=True)

    def create_run(self, project_id: str, config: RunConfig) -> RunRecord:
        if config.matcher != "SIFT":
            raise ValueError(
                "SUPERPOINT_LIGHTGLUE is not an executable reconstruction provider; use SIFT. "
                "The learned matcher remains an offline comparison experiment."
            )
        project = self.get_project(project_id)
        run_id = self.new_id("run")
        directory = self.run_dir(project_id, run_id)
        directory.mkdir(parents=True, exist_ok=False)
        for child in ("frames", "masks", "sparse", "dense", "logs"):
            (directory / child).mkdir()
        record = RunRecord(
            project_id=project_id,
            run_id=run_id,
            config_version=config.config_version,
            config=config,
            environment=runtime_environment(),
            synthetic_fixture=project.source_provenance == ProvenanceOrigin.SYNTHETIC,
            source_provenance=project.source_provenance,
            video_origin=project.video_origin,
            telemetry_origin=project.telemetry_origin,
            requested_matcher=config.matcher,
        )
        self.save_run(record)
        return record

    def create_linked_run(self, source_run_id: str, config: RunConfig) -> RunRecord:
        source = self.get_run(source_run_id)
        if source.status.value != "COMPLETED":
            raise ValueError("Only completed runs can be used as immutable rerun evidence")
        record = self.create_run(source.project_id, config)
        record.derived_from_run_id = source.run_id
        self.save_run(record)
        return record

    def get_run(self, run_id: str) -> RunRecord:
        self._validate_id(run_id)
        matches = list(self.root.glob(f"*/runs/{run_id}/run_manifest.json"))
        if len(matches) != 1:
            raise FileNotFoundError(run_id)
        return RunRecord.model_validate_json(matches[0].read_text(encoding="utf-8"))

    def list_runs(self) -> list[RunRecord]:
        records: list[RunRecord] = []
        for path in sorted(self.root.glob("*/runs/*/run_manifest.json")):
            try:
                records.append(RunRecord.model_validate_json(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return records

    def execution_lock(self, run_id: str, *, blocking: bool = False) -> RunExecutionLock:
        record = self.get_run(run_id)
        return RunExecutionLock(
            self.run_dir(record.project_id, record.run_id) / ".execution.lock",
            blocking=blocking,
        )

    def heavy_job_locks(self, limit: int) -> list[RunExecutionLock]:
        """Return process-safe resource slots shared by every run in this data root."""

        if limit < 1:
            raise ValueError("heavy job limit must be at least one")
        directory = self.root / ".resource_locks"
        return [RunExecutionLock(directory / f"heavy-{index}.lock") for index in range(limit)]

    def save_run(self, record: RunRecord) -> None:
        record.updated_at = utc_now()
        with self._lock:
            atomic_json(
                self.run_dir(record.project_id, record.run_id) / "run_manifest.json",
                record.model_dump(mode="json"),
            )

    def register_artifacts(self, record: RunRecord, paths: Iterable[Path]) -> None:
        directory = self.run_dir(record.project_id, record.run_id)
        existing = {item.relative_path: item for item in record.artifacts}
        for path in paths:
            resolved = path.resolve()
            if not resolved.is_relative_to(directory) or not resolved.is_file():
                raise ValueError(f"Artifact is outside the run folder: {path}")
            relative = str(resolved.relative_to(directory))
            existing[relative] = ArtifactEntry(
                name=resolved.name,
                relative_path=relative,
                media_type=mimetypes.guess_type(resolved.name)[0] or "application/octet-stream",
                size_bytes=resolved.stat().st_size,
                sha256=sha256_file(resolved),
                url=f"/api/runs/{record.run_id}/artifacts/{relative}",
            )
        record.artifacts = sorted(existing.values(), key=lambda item: item.relative_path)
        self.save_run(record)

    def resolve_declared_artifact(self, run_id: str, artifact_path: str) -> Path:
        record = self.get_run(run_id)
        clean = Path(artifact_path)
        if clean.is_absolute() or ".." in clean.parts:
            raise FileNotFoundError(artifact_path)
        declared = {item.relative_path for item in record.artifacts}
        if artifact_path not in declared:
            raise FileNotFoundError(artifact_path)
        path = (self.run_dir(record.project_id, run_id) / clean).resolve()
        if not path.is_file() or not path.is_relative_to(self.run_dir(record.project_id, run_id)):
            raise FileNotFoundError(artifact_path)
        return path

    def _measurements_path(self, record: RunRecord) -> Path:
        return self.run_dir(record.project_id, record.run_id) / "measurements.json"

    @staticmethod
    def _ply_counts(path: Path) -> tuple[int | None, int | None]:
        vertex_count: int | None = None
        face_count: int | None = None
        with path.open("rb") as stream:
            for raw in stream:
                try:
                    line = raw.decode("ascii").strip()
                except UnicodeDecodeError as exc:
                    raise ValueError("PLY header is not ASCII") from exc
                if line.startswith("element vertex "):
                    vertex_count = int(line.split()[-1])
                elif line.startswith("element face "):
                    face_count = int(line.split()[-1])
                elif line == "end_header":
                    return vertex_count, face_count
        raise ValueError("PLY is missing end_header")

    def list_measurements(self, run_id: str) -> list[MeasurementRecord]:
        record = self.get_run(run_id)
        path = self._measurements_path(record)
        if not path.is_file():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("measurements", []) if isinstance(payload, dict) else []
        return [MeasurementRecord.model_validate(item) for item in rows]

    def add_measurement(self, run_id: str, value: MeasurementCreate) -> MeasurementRecord:
        """Persist a backend-computed measurement bound to one declared geometry hash."""

        record = self.get_run(run_id)
        declared = {item.relative_path: item for item in record.artifacts}
        artifact = declared.get(value.geometry_artifact_path)
        if artifact is None:
            raise ValueError("geometry artifact is not declared by this run")
        if artifact.sha256 != value.geometry_artifact_sha256:
            raise ValueError("geometry artifact hash does not match the declared run artifact")
        geometry = self.resolve_declared_artifact(run_id, value.geometry_artifact_path)
        if sha256_file(geometry) != value.geometry_artifact_sha256:
            raise ValueError("geometry artifact no longer matches its declared checksum")
        if geometry.suffix.lower() != ".ply":
            raise ValueError("measurement references currently require a validated PLY artifact")

        vertex_count, face_count = self._ply_counts(geometry)
        for endpoint in (value.start, value.end):
            if endpoint.point_id is not None and (
                vertex_count is None or endpoint.point_id >= vertex_count
            ):
                raise ValueError("measurement point reference is outside the geometry artifact")
            if endpoint.face_id is not None and (
                face_count is None or endpoint.face_id >= face_count
            ):
                raise ValueError("measurement face reference is outside the geometry artifact")

        delta = [
            end - start
            for start, end in zip(value.start.coordinates, value.end.coordinates, strict=True)
        ]
        if value.measurement_kind == "HORIZONTAL":
            distance = math.hypot(delta[0], delta[1])
        elif value.measurement_kind == "VERTICAL":
            distance = abs(delta[2])
        else:
            distance = math.sqrt(sum(component * component for component in delta))

        observed_paths = {"sparse/sparse_local.ply", "sparse/sparse.ply"}
        inferred = any(
            token in value.geometry_artifact_path.lower()
            for token in ("inferred", "completion", "generated")
        )
        provenance = "INFERRED" if inferred else (
            "OBSERVED" if value.geometry_artifact_path in observed_paths else "UNKNOWN"
        )
        eligible = (
            provenance == "OBSERVED"
            and value.geometry_artifact_path == "sparse/sparse_local.ply"
            and value.coordinate_frame == "LOCAL_ENU_METRES"
            and record.source_provenance != ProvenanceOrigin.SYNTHETIC
            and record.config.execution_mode != "SYNTHETIC_DEMO"
        )
        if provenance != "OBSERVED":
            reason = "Only original observed sparse geometry is measurement eligible."
        elif value.geometry_artifact_path != "sparse/sparse_local.ply":
            reason = "Measurement requires the aligned local-metric sparse artifact."
        elif value.coordinate_frame != "LOCAL_ENU_METRES":
            reason = "Coordinate frame is not the run's local metric frame."
        elif record.source_provenance == ProvenanceOrigin.SYNTHETIC:
            reason = "Synthetic fixtures are not measurement evidence."
        elif record.config.execution_mode == "SYNTHETIC_DEMO":
            reason = "Synthetic execution mode is not measurement evidence."
        else:
            reason = "Observed local-metric geometry with a validated artifact hash."
        measurement = MeasurementRecord(
            **value.model_dump(mode="python"),
            measurement_id=self.new_id("msr"),
            run_id=run_id,
            backend_distance_m=distance,
            geometry_provenance=provenance,
            measurement_eligible=eligible,
            eligibility_reason=reason,
        )
        measurements = self.list_measurements(run_id)
        measurements.append(measurement)
        atomic_json(
            self._measurements_path(record),
            {
                "schema_version": "1.0",
                "run_id": run_id,
                "measurements": [item.model_dump(mode="json") for item in measurements],
            },
        )
        return measurement
