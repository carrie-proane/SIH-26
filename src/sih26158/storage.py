from __future__ import annotations

import hashlib
import json
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
from typing import BinaryIO

from .models import (
    ArtifactEntry,
    InputAsset,
    ProjectManifest,
    ProvenanceOrigin,
    RunConfig,
    RunRecord,
    utc_now,
)

ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,80}$")


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
    for package in ("fastapi", "numpy", "pydantic"):
        try:
            environment[package] = version(package)
        except PackageNotFoundError:
            environment[package] = None
    for tool, flag in (("ffprobe", "-version"), ("colmap", "-h")):
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

    def create_run(self, project_id: str, config: RunConfig) -> RunRecord:
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
        )
        self.save_run(record)
        return record

    def get_run(self, run_id: str) -> RunRecord:
        self._validate_id(run_id)
        matches = list(self.root.glob(f"*/runs/{run_id}/run_manifest.json"))
        if len(matches) != 1:
            raise FileNotFoundError(run_id)
        return RunRecord.model_validate_json(matches[0].read_text(encoding="utf-8"))

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
