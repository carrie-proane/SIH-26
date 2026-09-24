from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from frames.contact_sheet import create_contact_sheet
from frames.extractor import extract_frames
from frames.selector import (
    FRAME_SCORE_COLUMNS,
    FrameQualityThresholds,
    SelectionWeights,
    select_keyframes,
)
from telemetry.csv_parser import parse_csv
from telemetry.models import sha256_file as telemetry_sha256_file
from telemetry.models import write_csv as write_telemetry_csv
from telemetry.srt_parser import parse_srt

from .ai_integration import (
    CompletionRequest,
    resolve_segmentation_device,
    run_completion,
    segmentation_fingerprint,
    segmentation_settings,
    write_completion_not_run,
    write_coverage_report,
    write_segmentation_status,
)
from .benchmark import build_benchmark_report
from .capabilities import collect_server_capabilities, synthetic_capability_profile
from .colmap import ColmapRunner, ExternalToolError, ReconstructionResult, write_matcher_benchmark
from .confidence import validate_point_confidence_for_ply
from .dense import (
    DenseContext,
    UnavailableProvider,
    run_dense_stage,
    select_dense_provider,
)
from .exports import build_export_readiness, export_artifact_paths, write_export_readiness
from .geo import SimilarityTransform, geodetic_to_enu, transform_ply
from .models import (
    MatcherMetrics,
    OffsetSource,
    ProvenanceOrigin,
    RunCheckpoint,
    RunRecord,
    RunStatus,
    StageCheckpoint,
    StageEvent,
    utc_now,
)
from .process_control import (
    ManagedProcessExecutor,
    ProcessCancelledError,
    ProcessTimeoutError,
)
from .report import build_quality_report, write_quality_report
from .scene_policy import analyze_scene
from .segmentation import run_managed_segmentation
from .storage import ProjectStore, atomic_json, sha256_file
from .sync import calibrate_telemetry_offset


class PipelineError(RuntimeError):
    pass


def _is_synthetic_demo(record: RunRecord) -> bool:
    return record.config.execution_mode == "SYNTHETIC_DEMO"


def _effective_worker_threads(requested: int) -> int:
    """Clamp explicit threads to scheduler/affinity limits without guessing shared capacity."""

    host_count = os.cpu_count() or 1
    try:
        affinity_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity_count = host_count
    try:
        scheduler_count = int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or 0)
    except ValueError:
        scheduler_count = 0
    limits = [value for value in (host_count, affinity_count, scheduler_count) if value > 0]
    available = min(limits) if limits else 1
    if requested > 0:
        return min(requested, available)
    # On an allocated/cgroup-constrained server, make the limit explicit. On an
    # unconstrained workstation, zero retains the installed tool's default.
    if scheduler_count > 0 or affinity_count < host_count:
        return available
    return 0


def _asset_path(store: ProjectStore, record: RunRecord, role: str) -> Path:
    project = store.get_project(record.project_id)
    matches = [asset for asset in project.assets if asset.role == role]
    if len(matches) != 1:
        raise PipelineError(f"Project must declare exactly one {role} asset")
    return store.project_dir(record.project_id) / matches[0].relative_path


def _synthetic_ply(path: Path) -> None:
    points = [
        (-2, -1, 0, 32, 191, 107),
        (-1, -1, 0.1, 32, 191, 107),
        (0, -1, 0.2, 245, 158, 11),
        (1, -1, 0.1, 245, 158, 11),
        (2, -1, 0, 239, 68, 68),
        (-2, 1, 0, 32, 191, 107),
        (-1, 1, 0.2, 32, 191, 107),
        (0, 1, 0.4, 245, 158, 11),
        (1, 1, 0.2, 239, 68, 68),
        (2, 1, 0, 239, 68, 68),
    ]
    header = (
        "ply\nformat ascii 1.0\ncomment SYNTHETIC_DEMO - not reconstruction evidence\n"
        f"element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    path.write_text(
        header + "\n".join(" ".join(map(str, row)) for row in points) + "\n", encoding="utf-8"
    )


def _temporal_coverage(
    frames: list[dict[str, object]],
    duration_s: float,
    interval_s: float,
    registered_names: set[str] | None = None,
) -> dict[str, object]:
    """Report time-binned candidate/selection/registration coverage without inventing area coverage."""

    duration_s = max(
        duration_s, max((float(row.get("timestamp_s", 0)) for row in frames), default=0)
    )
    interval_count = max(1, int(np.ceil(max(duration_s, 0.001) / interval_s)))
    intervals: list[dict[str, object]] = []
    for index in range(interval_count):
        start = index * interval_s
        end = min(duration_s, (index + 1) * interval_s)
        contained = [
            row
            for row in frames
            if start <= float(row.get("timestamp_s", 0))
            and (index == interval_count - 1 or float(row.get("timestamp_s", 0)) < end)
        ]
        selected = [row for row in contained if bool(row.get("selected", False))]
        item: dict[str, object] = {
            "start_s": round(start, 6),
            "end_s": round(end, 6),
            "candidate_count": len(contained),
            "selected_count": len(selected),
        }
        if registered_names is not None:
            item["registered_count"] = sum(
                Path(str(row.get("image_name", ""))).name in registered_names for row in selected
            )
        intervals.append(item)

    def zero_ranges(field: str) -> list[dict[str, float]]:
        ranges: list[dict[str, float]] = []
        range_start: float | None = None
        for item in intervals:
            if int(item.get(field, 0)) == 0 and range_start is None:
                range_start = float(item["start_s"])
            if int(item.get(field, 0)) > 0 and range_start is not None:
                ranges.append({"start_s": range_start, "end_s": float(item["start_s"])})
                range_start = None
        if range_start is not None:
            ranges.append({"start_s": range_start, "end_s": duration_s})
        return ranges

    return {
        "status": "DIAGNOSTIC_ONLY",
        "interval_s": interval_s,
        "video_duration_s": duration_s,
        "intervals": intervals,
        "long_unrepresented_selected_intervals": zero_ranges("selected_count"),
        "long_unrepresented_registered_intervals": (
            zero_ranges("registered_count") if registered_names is not None else None
        ),
        "interpretation": (
            "Temporal registration coverage is a capture diagnostic, not a denominator-based "
            "visible-surface completeness measurement."
        ),
    }


class PipelineRunner:
    def __init__(self, store: ProjectStore, max_workers: int = 2) -> None:
        self.store = store
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sih-run")
        self._submitted: dict[str, Future[RunRecord]] = {}
        self._submitted_lock = threading.Lock()
        self._stage_started_monotonic: dict[tuple[str, str], float] = {}
        try:
            configured_heavy_limit = int(os.getenv("SIH_HEAVY_JOB_LIMIT", "1"))
        except ValueError:
            configured_heavy_limit = 1
        self.heavy_job_limit = max(1, configured_heavy_limit)
        self.heavy_job_wait_timeout_s = max(
            1.0, float(os.getenv("SIH_HEAVY_JOB_WAIT_TIMEOUT_S", "86400"))
        )

    def submit(self, run_id: str) -> bool:
        """Submit once per process; the filesystem lock covers other processes."""

        with self._submitted_lock:
            existing = self._submitted.get(run_id)
            if existing is not None and not existing.done():
                return False
            future = self.executor.submit(self.run, run_id)
            self._submitted[run_id] = future

        def discard(completed: Future[RunRecord]) -> None:
            with self._submitted_lock:
                if self._submitted.get(run_id) is completed:
                    self._submitted.pop(run_id, None)

        future.add_done_callback(discard)
        return True

    def shutdown(self, *, wait: bool = True) -> None:
        self.executor.shutdown(wait=wait, cancel_futures=False)

    def _cancel_path(self, record: RunRecord) -> Path:
        return self.store.run_dir(record.project_id, record.run_id) / ".cancel_requested"

    def _cancel_requested(self, record: RunRecord) -> bool:
        return self._cancel_path(record).is_file()

    def _raise_if_cancelled(self, record: RunRecord) -> None:
        if self._cancel_requested(record):
            raise ProcessCancelledError("Run cancellation was requested")

    def _heartbeat(self, run_id: str) -> None:
        record = self.store.get_run(run_id)
        if record.status not in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
            record.last_heartbeat_at = utc_now()
            self.store.save_run(record)

    def _managed_executor(self, record: RunRecord, timeout_s: float) -> ManagedProcessExecutor:
        return ManagedProcessExecutor(
            timeout_s=timeout_s,
            cancel_requested=lambda: self._cancel_requested(record),
            heartbeat=lambda: self._heartbeat(record.run_id),
            heartbeat_interval_s=record.config.command_heartbeat_s,
            process_state_path=(
                self.store.run_dir(record.project_id, record.run_id) / ".active_process.json"
            ),
        )

    def _surviving_process_state(self, record: RunRecord) -> dict[str, object] | None:
        path = self.store.run_dir(record.project_id, record.run_id) / ".active_process.json"
        if not path.is_file():
            return None
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            pid = int(state["pid"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise PipelineError(
                "An unreadable active-process marker requires operator review before recovery."
            ) from exc
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            path.unlink(missing_ok=True)
            return None
        except PermissionError:
            return state
        return state

    def request_cancel(self, run_id: str) -> tuple[RunRecord, bool]:
        record = self.store.get_run(run_id)
        if record.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
            return record, False
        if self._cancel_path(record).is_file():
            return record, True
        self._cancel_path(record).write_text(utc_now() + "\n", encoding="utf-8")
        record.cancel_requested_at = utc_now()
        record.last_heartbeat_at = utc_now()
        record.events.append(
            StageEvent(
                stage=record.stage,
                status="STARTED",
                progress=record.progress,
                message="Cancellation requested; active external processes will stop safely.",
            )
        )
        self.store.save_run(record)
        return record, True

    def _checkpoint_path(self, record: RunRecord) -> Path:
        return self.store.run_dir(record.project_id, record.run_id) / ".pipeline_checkpoint.json"

    @staticmethod
    def _fingerprint(value: object) -> str:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _tool_signature(names: tuple[str, ...]) -> dict[str, object]:
        signatures: dict[str, object] = {}
        for name in names:
            path = shutil.which(name)
            if path is None:
                signatures[name] = None
                continue
            try:
                stat = Path(path).stat()
                signatures[name] = {
                    "path": str(Path(path).resolve()),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            except OSError:
                signatures[name] = {"path": path, "stat": "unavailable"}
        return signatures

    def _stage_fingerprints(
        self, record: RunRecord, checkpoint: RunCheckpoint, stage: str
    ) -> tuple[str, str, str]:
        project = self.store.get_project(record.project_id)
        project_inputs = {
            asset.role: {"sha256": asset.sha256, "size_bytes": asset.size_bytes}
            for asset in project.assets
        }
        ancestor_names = {
            "INGEST": (),
            "PREPROCESS": ("INGEST",),
            "SEGMENTATION": ("PREPROCESS",),
            "SPARSE": ("PREPROCESS", "SEGMENTATION"),
            # Dense consumes selected frames/masks as well as the sparse model.
            "DENSE": ("PREPROCESS", "SEGMENTATION", "SPARSE"),
            "COVERAGE": ("PREPROCESS", "SPARSE", "DENSE"),
            "COMPLETION": ("COVERAGE", "DENSE", "SPARSE"),
            # Reporting consumes frame/coverage diagnostics even when dense is disabled.
            "REPORT": (
                "PREPROCESS",
                "SEGMENTATION",
                "SPARSE",
                "DENSE",
                "COVERAGE",
                "COMPLETION",
            ),
        }[stage]
        dependencies = {
            name: checkpoint.completed[name].artifacts
            for name in ancestor_names
            if name in checkpoint.completed
        }
        config = record.config.model_dump(mode="json")
        fields = {
            "INGEST": (),
            "PREPROCESS": (
                "preprocessing_run",
                "force_include_frame_indices",
                "force_exclude_frame_indices",
                "frame_min_laplacian_variance",
                "frame_min_exposure_score",
                "frame_relative_sharpness_floor",
                "frame_min_feature_count",
                "frame_min_feature_grid_coverage",
                "frame_max_parallax_fraction",
                "reconstruction_target",
                "masking_mode",
                "max_candidate_frames",
                "max_selected_frames",
                "processing_max_image_dimension",
            ),
            "SEGMENTATION": (
                "enable_segmentation",
                "segmentation_model_path",
                "segmentation_model_name",
                "segmentation_model_version",
                "segmentation_device",
                "segmentation_allow_cpu_fallback",
                "segmentation_confidence",
                "segmentation_iou_threshold",
                "segmentation_image_size",
                "segmentation_mask_dilation_px",
                "segmentation_mask_erosion_px",
                "segmentation_excluded_classes",
                "segmentation_timeout_s",
                "reconstruction_target",
                "masking_mode",
            ),
            "SPARSE": (
                "profile",
                "matcher",
                "camera_model",
                "camera_model_policy",
                "camera_params",
                "camera_params_reference",
                "refine_intrinsics",
                "max_reconstruction_retries",
                "sequential_overlap",
                "use_gpu",
                "telemetry_offset_s",
                "telemetry_offset_source",
                "local_origin",
                "matching_strategy",
                "vocab_tree_path",
                "sparse_timeout_s",
                "command_heartbeat_s",
                "worker_threads",
            ),
            "DENSE": (
                "profile",
                "enable_dense_reconstruction",
                "dense_provider",
                "dense_timeout_s",
                "command_heartbeat_s",
                "worker_threads",
                "reconstruction_target",
                "masking_mode",
            ),
            "COVERAGE": ("enable_coverage_analysis",),
            "COMPLETION": ("enable_completion", "completion_timeout_s"),
            "REPORT": ("known_distance_m", "measured_distance_m", "coverage_interval_s"),
        }[stage]
        tool_names = {
            "INGEST": ("ffprobe",),
            "PREPROCESS": ("ffmpeg", "ffprobe"),
            "SEGMENTATION": (),
            "SPARSE": ("colmap",),
            "DENSE": ("colmap", "InterfaceCOLMAP", "DensifyPointCloud", "TextureMesh"),
            "COVERAGE": (),
            "COMPLETION": (),
            "REPORT": (),
        }[stage]
        input_material: dict[str, object] = {"project_inputs": project_inputs}
        if stage != "INGEST":
            input_material["dependency_artifacts"] = dependencies
        if stage == "PREPROCESS" and record.config.preprocessing_run:
            handoff = Path(record.config.preprocessing_run)
            input_material["preprocessing_handoff"] = {
                name: sha256_file(handoff / name) if (handoff / name).is_file() else None
                for name in (
                    "keyframes.json",
                    "frame_scores.csv",
                    "normalized_telemetry.csv",
                    "normalized_telemetry.meta.json",
                )
            }
        if stage == "SEGMENTATION":
            input_material["segmentation"] = segmentation_fingerprint(
                record, self.store.run_dir(record.project_id, record.run_id)
            )
        resource_signature = (
            {
                "effective_worker_threads": _effective_worker_threads(record.config.worker_threads),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            }
            if stage in {"SEGMENTATION", "SPARSE", "DENSE", "COMPLETION"}
            else {}
        )
        stage_contract = (
            "3.0" if stage in {"SEGMENTATION", "COVERAGE", "COMPLETION", "REPORT"} else "2.0"
        )
        return (
            self._fingerprint(input_material),
            self._fingerprint(
                {name: config[name] for name in fields} | {"contract": stage_contract}
            ),
            self._fingerprint(
                self._tool_signature(tool_names)
                | {"contract": stage_contract, "resources": resource_signature}
            ),
        )

    def _load_checkpoint(self, record: RunRecord) -> RunCheckpoint:
        path = self._checkpoint_path(record)
        if not path.is_file():
            return RunCheckpoint(run_id=record.run_id)
        try:
            checkpoint = RunCheckpoint.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return RunCheckpoint(run_id=record.run_id)
        return (
            checkpoint
            if checkpoint.run_id == record.run_id
            else RunCheckpoint(run_id=record.run_id)
        )

    def _save_checkpoint(self, record: RunRecord, checkpoint: RunCheckpoint) -> None:
        checkpoint.updated_at = utc_now()
        atomic_json(self._checkpoint_path(record), checkpoint.model_dump(mode="json"))

    def _begin_checkpoint_stage(
        self,
        record: RunRecord,
        checkpoint: RunCheckpoint,
        stage: str,
    ) -> None:
        checkpoint.active_stage = stage
        checkpoint.active_stage_started_at = utc_now()
        self._stage_started_monotonic[(record.run_id, stage)] = time.monotonic()
        if stage == "PREPROCESS" and record.processing_started_at is None:
            record.processing_started_at = checkpoint.active_stage_started_at
        record.checkpoint_stage = stage
        record.last_heartbeat_at = utc_now()
        self._save_checkpoint(record, checkpoint)
        self.store.save_run(record)

    def _complete_checkpoint_stage(
        self,
        record: RunRecord,
        checkpoint: RunCheckpoint,
        stage: str,
        paths: list[Path],
        warnings: list[dict[str, str]],
    ) -> None:
        run_dir = self.store.run_dir(record.project_id, record.run_id)
        declared = {item.relative_path: item for item in record.artifacts}
        artifacts: dict[str, str] = {}
        for path in paths:
            resolved = path.resolve()
            if resolved.is_file() and resolved.is_relative_to(run_dir):
                relative = str(resolved.relative_to(run_dir))
                artifact = declared.get(relative)
                if artifact is not None:
                    artifacts[relative] = artifact.sha256
        # Some downstream stages deliberately enrich an upstream JSON artifact
        # (telemetry metadata in ingest; registered-frame status in keyframes).
        # Refresh only hashes that were explicitly re-registered, at the stage
        # completion boundary, before fingerprinting the dependent stage.
        for prior in checkpoint.completed.values():
            for relative_path in list(prior.artifacts):
                artifact = declared.get(relative_path)
                if artifact is not None:
                    prior.artifacts[relative_path] = artifact.sha256
        input_fingerprint, config_fingerprint, environment_fingerprint = self._stage_fingerprints(
            record, checkpoint, stage
        )
        started = self._stage_started_monotonic.pop((record.run_id, stage), None)
        duration_s = max(0.0, time.monotonic() - started) if started is not None else None
        checkpoint.completed[stage] = StageCheckpoint(
            stage=stage,  # type: ignore[arg-type]
            artifacts=artifacts,
            warnings=list(warnings),
            input_fingerprint=input_fingerprint,
            configuration_fingerprint=config_fingerprint,
            environment_fingerprint=environment_fingerprint,
            started_at=checkpoint.active_stage_started_at,
            duration_s=duration_s,
        )
        checkpoint.active_stage = None
        checkpoint.active_stage_started_at = None
        checkpoint.warnings = list(warnings)
        record.checkpoint_stage = stage
        if duration_s is not None:
            record.stage_timings_s[stage] = duration_s
        record.last_heartbeat_at = utc_now()
        self._save_checkpoint(record, checkpoint)
        self.store.save_run(record)

    def _checkpoint_stage_valid(
        self,
        record: RunRecord,
        checkpoint: RunCheckpoint,
        stage: str,
    ) -> bool:
        completed = checkpoint.completed.get(stage)
        if completed is None or not completed.artifacts:
            return False
        expected = self._stage_fingerprints(record, checkpoint, stage)
        if (
            completed.input_fingerprint,
            completed.configuration_fingerprint,
            completed.environment_fingerprint,
        ) != expected:
            return False
        run_dir = self.store.run_dir(record.project_id, record.run_id)
        declared = {item.relative_path: item.sha256 for item in record.artifacts}
        for relative_path, expected_sha256 in completed.artifacts.items():
            path = run_dir / relative_path
            if (
                not path.is_file()
                or declared.get(relative_path) != expected_sha256
                or sha256_file(path) != expected_sha256
            ):
                return False
        return True

    @contextmanager
    def _heavy_job_slot(self, record: RunRecord):
        """Limit expensive work across runs and processes; OS locks recover after crashes."""

        started = time.monotonic()
        while True:
            self._raise_if_cancelled(record)
            for candidate in self.store.heavy_job_locks(self.heavy_job_limit):
                slot = candidate.__enter__()
                if slot.acquired:
                    try:
                        yield slot.path.name
                    finally:
                        slot.__exit__(None, None, None)
                    return
                slot.__exit__(None, None, None)
            if time.monotonic() - started >= self.heavy_job_wait_timeout_s:
                raise PipelineError(
                    "Timed out waiting for a global heavy-job slot; another run may still be active."
                )
            self._heartbeat(record.run_id)
            time.sleep(0.1)

    @contextmanager
    def _accelerator_slot(self, record: RunRecord, device: str):
        """Serialize use of one scheduler-visible accelerator allocation across runs."""

        if device == "cpu":
            yield None
            return
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "scheduler-default")
        identity = hashlib.sha256(f"{device}:{visible}".encode()).hexdigest()[:16]
        started = time.monotonic()
        while True:
            self._raise_if_cancelled(record)
            with self.store.resource_lock(f"accelerator-{identity}") as slot:
                if slot.acquired:
                    yield f"{device}:{visible}"
                    return
            if time.monotonic() - started >= self.heavy_job_wait_timeout_s:
                raise PipelineError(
                    f"Timed out waiting for the allocated {device} device; another run is active."
                )
            self._heartbeat(record.run_id)
            time.sleep(0.1)

    def recover_interrupted_runs(self) -> list[str]:
        recovered: list[str] = []
        active = {
            RunStatus.QUEUED,
            RunStatus.INGESTING,
            RunStatus.PREPROCESSING,
            RunStatus.RECONSTRUCTING,
            RunStatus.REPORTING,
        }
        for record in self.store.list_runs():
            if record.status not in active:
                continue
            with self.store.execution_lock(record.run_id) as execution_lock:
                if not execution_lock.acquired:
                    continue
                if self._surviving_process_state(record) is not None:
                    continue
                record = self.store.get_run(record.run_id)
                record.stage = RunStatus.QUEUED
                record.status = RunStatus.QUEUED
                record.failure_reason = None
                record.recovery_count += 1
                record.last_heartbeat_at = utc_now()
                record.events.append(
                    StageEvent(
                        stage=RunStatus.QUEUED,
                        status="STARTED",
                        progress=record.progress,
                        message="Recovered interrupted run from its last valid checkpoint.",
                    )
                )
                self.store.save_run(record)
            if self.submit(record.run_id):
                recovered.append(record.run_id)
        return recovered

    def resume(self, run_id: str) -> tuple[RunRecord, bool]:
        record = self.store.get_run(run_id)
        if record.status == RunStatus.COMPLETED:
            return record, False
        with self._submitted_lock:
            submitted = self._submitted.get(run_id)
            if submitted is not None and not submitted.done():
                return record, False
        with self.store.execution_lock(run_id) as execution_lock:
            if not execution_lock.acquired:
                return self.store.get_run(run_id), False
            record = self.store.get_run(run_id)
            record.stage = RunStatus.QUEUED
            record.status = RunStatus.QUEUED
            record.failure_reason = None
            record.cancel_requested_at = None
            record.cancelled_at = None
            record.recovery_count += 1
            record.last_heartbeat_at = utc_now()
            record.events.append(
                StageEvent(
                    stage=RunStatus.QUEUED,
                    status="STARTED",
                    progress=record.progress,
                    message="Run queued for checkpoint-safe resume.",
                )
            )
            self.store.save_run(record)
            self._cancel_path(record).unlink(missing_ok=True)
        return self.store.get_run(run_id), self.submit(run_id)

    def _transition(
        self,
        record: RunRecord,
        stage: RunStatus,
        progress: int,
        message: str,
        event_status: str = "STARTED",
    ) -> None:
        record.stage = stage
        record.status = stage
        record.progress = progress
        record.last_heartbeat_at = utc_now()
        record.events.append(
            StageEvent(stage=stage, status=event_status, progress=progress, message=message)  # type: ignore[arg-type]
        )
        self.store.save_run(record)

    def _capture_capabilities(self, record: RunRecord) -> Path:
        run_dir = self.store.run_dir(record.project_id, record.run_id)
        profile = (
            synthetic_capability_profile(record.config)
            if _is_synthetic_demo(record)
            else collect_server_capabilities(data_root=self.store.root, config=record.config)
        )
        path = run_dir / "server_capabilities.json"
        profile["resource_policy"] = {
            "heavy_job_limit": self.heavy_job_limit,
            "heavy_job_limit_source": "SIH_HEAVY_JOB_LIMIT",
            "worker_threads_requested": record.config.worker_threads,
            "worker_threads_effective": _effective_worker_threads(record.config.worker_threads),
            "scheduler_environment": {
                name: os.environ.get(name)
                for name in (
                    "CUDA_VISIBLE_DEVICES",
                    "SLURM_JOB_ID",
                    "SLURM_CPUS_PER_TASK",
                    "SLURM_GPUS",
                )
                if os.environ.get(name) is not None
            },
        }
        atomic_json(path, profile)
        record.capability_profile_path = "server_capabilities.json"
        record.effective_sparse_gpu = bool(profile["sparse"]["effective_gpu"])
        record.selected_dense_provider = profile["dense"]["selected_provider"]
        self.store.save_run(record)
        return path

    def _probe(self, record: RunRecord) -> tuple[Path, list[dict[str, str]]]:
        video = _asset_path(self.store, record, "video")
        project = self.store.get_project(record.project_id)
        warnings: list[dict[str, str]] = []
        ffprobe = shutil.which("ffprobe")
        if _is_synthetic_demo(record):
            probe = {
                "format": {"filename": video.name, "format_name": "synthetic_fixture"},
                "streams": [],
            }
            warnings.append(
                {
                    "code": "FFPROBE_SKIPPED",
                    "message": "Synthetic demo did not inspect a real codec.",
                }
            )
        elif ffprobe:
            result = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_format",
                    "-show_streams",
                    "-of",
                    "json",
                    str(video),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode:
                raise PipelineError(
                    "ffprobe could not read the video. Verify that the file is a supported, non-corrupt "
                    "MP4/MOV and inspect logs before retrying."
                )
            probe = json.loads(result.stdout)
        else:
            raise PipelineError(
                "ffprobe is not installed or not on PATH. Install FFmpeg and verify `ffprobe -version`; "
                "ingest will not continue with unknown frame timing."
            )
        payload = {
            "project_id": record.project_id,
            "run_id": record.run_id,
            "stage": "INGESTING",
            "status": "COMPLETED",
            "created_at": record.created_at,
            "config_version": record.config_version,
            "video_probe": probe,
            "input_assets": [asset.model_dump(mode="json") for asset in project.assets],
            "source_provenance": record.source_provenance,
            "video_origin": record.video_origin,
            "telemetry_origin": record.telemetry_origin,
            "genuine_real_evidence": record.source_provenance == ProvenanceOrigin.REAL,
            "warnings": warnings,
        }
        path = self.store.run_dir(record.project_id, record.run_id) / "ingest_report.json"
        atomic_json(path, payload)
        return path, warnings

    def _preprocess_contract(self, record: RunRecord) -> list[Path]:
        run_dir = self.store.run_dir(record.project_id, record.run_id)
        if _is_synthetic_demo(record):
            keyframes = [
                {
                    "frame_index": i,
                    "timestamp_s": i * 0.5,
                    "selected": True,
                    "source": "SYNTHETIC_DEMO",
                }
                for i in range(10)
            ]
            keyframes_path = run_dir / "keyframes.json"
            atomic_json(keyframes_path, {"synthetic_fixture": True, "frames": keyframes})
            scores_path = run_dir / "frame_scores.csv"
            with scores_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    ["frame_index", "timestamp_s", "blur_score", "exposure_score", "selected"]
                )
                for item in keyframes:
                    writer.writerow([item["frame_index"], item["timestamp_s"], 0.8, 0.9, True])
            telemetry_path = run_dir / "normalized_telemetry.csv"
            with telemetry_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    [
                        "timestamp_s",
                        "lat",
                        "lon",
                        "alt_m",
                        "alt_source",
                        "fix_quality",
                        "source_row",
                    ]
                )
                for i in range(10):
                    writer.writerow(
                        [
                            i * 0.5,
                            28.6139,
                            77.209 + i * 0.000005,
                            42,
                            "synthetic",
                            "ok",
                            i,
                        ]
                    )
            telemetry_meta_path = run_dir / "normalized_telemetry.meta.json"
            atomic_json(
                telemetry_meta_path,
                {
                    "schema_version": "1.0",
                    "source_file": "synthetic_telemetry.csv",
                    "source_format": "synthetic",
                    "source_dialect": "deterministic_smoke_fixture",
                    "parser_version": "0.1.0",
                    "row_count": 10,
                    "duration_s": 4.5,
                    "sample_rate_hz_estimated": 2.0,
                    "time_origin": "synthetic_video_start",
                    "coordinate_frame": "WGS84",
                    "altitude_reference": "relative_to_launch",
                    "warnings": [
                        {
                            "code": "SYNTHETIC_TELEMETRY",
                            "count": 10,
                            "detail": "Generated data; never present as a real flight.",
                        }
                    ],
                    "field_coverage": {"lat": 1.0, "lon": 1.0, "alt_m": 1.0},
                },
            )
            return [keyframes_path, scores_path, telemetry_path, telemetry_meta_path]
        handoff_problem: str | None = None
        if record.config.preprocessing_run:
            source = Path(record.config.preprocessing_run).resolve()
            required = [
                source / "keyframes.json",
                source / "frame_scores.csv",
                source / "normalized_telemetry.csv",
                source / "normalized_telemetry.meta.json",
                source / "frames",
            ]
            missing = [path.name for path in required if not path.exists()]
            if not missing:
                try:
                    payload = json.loads(required[0].read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    missing.append("valid keyframes.json")
                    payload = {}
                frame_rows = payload.get("frames", []) if isinstance(payload, dict) else []
                if not isinstance(frame_rows, list):
                    missing.append("keyframes.json frames list")
                    frame_rows = []
                selected_frame_rows = [
                    item
                    for item in frame_rows
                    if isinstance(item, dict) and item.get("selected", True)
                ]
                if len(selected_frame_rows) < 3:
                    missing.append("at least three selected keyframes")
                copied_frames: list[Path] = []
                for item in frame_rows:
                    if not isinstance(item, dict) or not item.get("selected", True):
                        continue
                    raw_name = item.get("image_name") or item.get("filename")
                    safe_name = Path(str(raw_name or "")).name
                    source_image = required[4] / safe_name
                    if not safe_name or not source_image.is_file():
                        missing.append(f"frames/{safe_name or '<missing image_name>'}")
                        continue
                    destination = run_dir / "frames" / safe_name
                    shutil.copyfile(source_image, destination)
                    copied_frames.append(destination)
                    item["image_name"] = safe_name
                    item["image_url"] = f"/api/runs/{record.run_id}/artifacts/frames/{safe_name}"
                    item.pop("path", None)
                    item.pop("frame_path", None)
                if missing:
                    handoff_problem = (
                        f"Configured handoff {source} was incomplete; missing: "
                        f"{', '.join(sorted(set(missing)))}. Generated a dataset-specific handoff "
                        "from this run's immutable inputs instead."
                    )
                    for copied in copied_frames:
                        copied.unlink(missing_ok=True)
                    return self._preprocess_uploaded_inputs(record, handoff_problem)
                atomic_json(run_dir / "keyframes.json", payload)
                shutil.copyfile(required[1], run_dir / "frame_scores.csv")
                shutil.copyfile(required[2], run_dir / "normalized_telemetry.csv")
                shutil.copyfile(required[3], run_dir / "normalized_telemetry.meta.json")
                artifacts = [
                    run_dir / "keyframes.json",
                    run_dir / "frame_scores.csv",
                    run_dir / "normalized_telemetry.csv",
                    run_dir / "normalized_telemetry.meta.json",
                    *copied_frames,
                ]
                for optional_name in ("frame_index.csv", "contact_sheet.png"):
                    optional = source / optional_name
                    if optional.is_file():
                        shutil.copyfile(optional, run_dir / optional_name)
                        artifacts.append(run_dir / optional_name)
                return artifacts
            handoff_problem = (
                f"Configured handoff {source} was incomplete; missing: {', '.join(missing)}. "
                "Generated a dataset-specific handoff from this run's immutable inputs instead."
            )
        return self._preprocess_uploaded_inputs(record, handoff_problem)

    def _preprocess_uploaded_inputs(
        self, record: RunRecord, handoff_problem: str | None = None
    ) -> list[Path]:
        """Create a scored, dataset-specific handoff from immutable uploaded inputs."""
        run_dir = self.store.run_dir(record.project_id, record.run_id)
        video = _asset_path(self.store, record, "video")
        telemetry = _asset_path(self.store, record, "telemetry")
        preprocessing_dir = run_dir / "preprocessing"
        preprocessing_dir.mkdir(parents=True, exist_ok=True)
        try:
            extraction = extract_frames(
                video,
                run_dir,
                frames_subdir="preprocessing/candidates",
                max_candidates=record.config.max_candidate_frames,
                max_image_dimension=record.config.processing_max_image_dimension,
            )
        except (ValueError, OSError) as exc:
            raise PipelineError(f"Automatic frame extraction failed: {exc}") from exc
        candidates = extraction.frames
        if len(candidates) < 3:
            raise PipelineError(
                "Automatic preprocessing extracted fewer than three frames; provide a longer "
                "video or a complete dataset-specific preprocessing handoff."
            )
        self._transition(
            record,
            RunStatus.PREPROCESSING,
            36,
            f"Decoded {len(candidates)} timestamped candidate frames.",
        )
        target_frames = min(record.config.max_selected_frames, len(candidates))
        try:
            rows = select_keyframes(
                candidates,
                run_dir,
                target_frames=target_frames,
                weights=SelectionWeights(),
                force_include=set(record.config.force_include_frame_indices),
                force_exclude=set(record.config.force_exclude_frame_indices),
                quality_thresholds=FrameQualityThresholds(
                    min_laplacian_variance=record.config.frame_min_laplacian_variance,
                    min_exposure_score=record.config.frame_min_exposure_score,
                    relative_sharpness_floor=record.config.frame_relative_sharpness_floor,
                    min_feature_count=record.config.frame_min_feature_count,
                    min_feature_grid_coverage=(record.config.frame_min_feature_grid_coverage),
                    max_parallax_fraction=record.config.frame_max_parallax_fraction,
                ),
            )
        except ValueError as exc:
            raise PipelineError(f"Automatic frame selection failed: {exc}") from exc
        contact_sheet_path = create_contact_sheet(rows, run_dir / "contact_sheet.png")
        self._transition(
            record,
            RunStatus.PREPROCESSING,
            44,
            f"Scored candidates and selected {sum(bool(row['selected']) for row in rows)} frames.",
        )

        frames_dir = run_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        selected_paths: list[Path] = []
        keyframes: list[dict[str, object]] = []
        extracted_by_index = {item.frame_index: item for item in candidates}
        for row in rows:
            candidate = Path(str(row["frame_path"]))
            selected = bool(row["selected"])
            if selected:
                destination = frames_dir / candidate.name
                shutil.copyfile(candidate, destination)
                selected_paths.append(destination)
            safe_row = dict(row)
            safe_row["source_video"] = video.name
            safe_row["frame_path"] = f"preprocessing/candidates/{candidate.name}"
            safe_row["image_name"] = candidate.name
            safe_row["path"] = safe_row["frame_path"]
            safe_row["source"] = "AUTO_SCORED_SELECTION_FROM_UPLOADED_VIDEO"
            extracted = extracted_by_index[int(row["frame_index"])]
            safe_row["image_transform"] = {
                "source_width": extracted.source_width,
                "source_height": extracted.source_height,
                "output_width": extracted.output_width,
                "output_height": extracted.output_height,
                "rotation_degrees_clockwise": extraction.rotation_degrees,
                "uniform_resize_scale": extracted.resize_scale,
                "cropped": False,
            }
            if selected:
                safe_row["image_url"] = (
                    f"/api/runs/{record.run_id}/artifacts/frames/{candidate.name}"
                )
            keyframes.append(safe_row)
        keyframes_path = run_dir / "keyframes.json"
        atomic_json(
            keyframes_path,
            {
                "schema_version": "1.0",
                "selection_method": "QUALITY_GATED_BLUR_EXPOSURE_REDUNDANCY",
                "frame_quality_gate": {
                    "minimum_laplacian_variance": (record.config.frame_min_laplacian_variance),
                    "minimum_exposure_score": record.config.frame_min_exposure_score,
                    "relative_sharpness_floor": (record.config.frame_relative_sharpness_floor),
                    "minimum_feature_count": record.config.frame_min_feature_count,
                    "minimum_feature_grid_coverage": (
                        record.config.frame_min_feature_grid_coverage
                    ),
                    "maximum_parallax_fraction": record.config.frame_max_parallax_fraction,
                    "candidate_count": len(keyframes),
                    "eligible_count": sum(bool(item["quality_eligible"]) for item in keyframes),
                    "rejected_count": sum(not bool(item["quality_eligible"]) for item in keyframes),
                    "selected_count": sum(bool(item["selected"]) for item in keyframes),
                },
                "override_actions": {
                    "force_include": record.config.force_include_frame_indices,
                    "force_exclude": record.config.force_exclude_frame_indices,
                },
                "resource_limits": {
                    "max_candidate_frames": record.config.max_candidate_frames,
                    "max_selected_frames": record.config.max_selected_frames,
                    "processing_max_image_dimension": record.config.processing_max_image_dimension,
                    "streaming_decode": True,
                    "full_resolution_frames_retained_in_memory": False,
                },
                "temporal_coverage": _temporal_coverage(
                    keyframes,
                    extraction.source_frame_count / extraction.fps,
                    record.config.coverage_interval_s,
                ),
                "frames": keyframes,
            },
        )
        scores_path = run_dir / "frame_scores.csv"
        with scores_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FRAME_SCORE_COLUMNS)
            writer.writeheader()
            writer.writerows(
                {column: item.get(column, "") for column in FRAME_SCORE_COLUMNS}
                for item in keyframes
            )

        suffix = telemetry.suffix.lower()
        parsed = parse_srt(telemetry) if suffix == ".srt" else parse_csv(telemetry)
        usable = [
            record for record in parsed.records if record.has_fix and record.alt_m is not None
        ]
        if len(usable) < 3:
            raise PipelineError(
                "Uploaded telemetry could not produce at least three usable latitude, longitude, "
                f"and altitude samples ({parsed.warnings.summary()})."
            )
        if handoff_problem:
            parsed.warnings.add("HANDOFF_FALLBACK", handoff_problem)
        duration_s = (
            extraction.source_frame_count / extraction.fps
            if extraction.source_frame_count > 0
            else max((frame.timestamp_s for frame in candidates), default=0.0)
        )
        selected_rows = [row for row in rows if row["selected"]]
        quality_eligible_rows = [row for row in rows if row["quality_eligible"]]
        if len(candidates) < 60:
            parsed.warnings.add(
                "TOO_FEW_FRAME_CANDIDATES",
                f"Only {len(candidates)} candidate frames were decoded; 60 or more are preferred.",
            )
        if len(selected_rows) < 60:
            parsed.warnings.add(
                "TOO_FEW_SELECTED_FRAMES",
                f"Only {len(selected_rows)} frames were selected; the target range is 60-120.",
            )
        rejected_count = len(rows) - len(quality_eligible_rows)
        if rejected_count:
            parsed.warnings.add(
                "FRAME_QUALITY_REJECTIONS",
                f"{rejected_count} candidate frame(s) were excluded by absolute/adaptive "
                "sharpness or exposure gates.",
            )
        if duration_s < 20:
            parsed.warnings.add(
                "VERY_SHORT_VIDEO",
                f"Video duration is approximately {duration_s:.2f}s; controlled passes should be 30-60s.",
            )
        averages = {
            name: float(np.mean([float(row[name]) for row in rows]))
            for name in (
                "blur_score",
                "exposure_score",
                "redundancy_score",
                "feature_count",
                "feature_grid_coverage",
                "parallax_fraction",
            )
        }
        if averages["blur_score"] < 0.3:
            parsed.warnings.add("EXCESSIVE_BLUR", "Mean normalized sharpness is below 0.30.")
        if averages["exposure_score"] < 0.4:
            parsed.warnings.add("POOR_EXPOSURE", "Mean exposure score is below 0.40.")
        if averages["redundancy_score"] < 0.25:
            parsed.warnings.add("HIGHLY_REDUNDANT_FOOTAGE", "Mean uniqueness score is below 0.25.")
        if averages["feature_grid_coverage"] < 0.15:
            parsed.warnings.add(
                "POOR_FEATURE_DISTRIBUTION",
                "Detected visual features occupy less than 15% of the diagnostic image grid.",
            )
        if averages["parallax_fraction"] < 0.002:
            parsed.warnings.add(
                "INSUFFICIENT_PARALLAX",
                "Median adjacent visual displacement is very low; footage may be mostly stationary or rotational.",
            )
        telemetry_duration = parsed.duration_s
        if telemetry_duration and abs(telemetry_duration - duration_s) > max(2.0, duration_s * 0.1):
            parsed.warnings.add(
                "TELEMETRY_DURATION_MISMATCH",
                f"Video is ~{duration_s:.2f}s but telemetry is ~{telemetry_duration:.2f}s.",
            )
        for warning in extraction.warnings:
            parsed.warnings.add(warning, "Decoded timestamps were unavailable for some frames.")
        telemetry_path = run_dir / "normalized_telemetry.csv"
        write_telemetry_csv(parsed.records, telemetry_path)
        telemetry_meta_path = run_dir / "normalized_telemetry.meta.json"
        metadata = parsed.meta(telemetry_sha256_file(telemetry))
        metadata["preprocessing_source"] = "AUTO_FROM_IMMUTABLE_RUN_INPUTS"
        metadata["frame_selection_method"] = "QUALITY_GATED_BLUR_EXPOSURE_REDUNDANCY"
        metadata["candidate_frame_count"] = len(candidates)
        metadata["selected_frame_count"] = len(selected_rows)
        metadata["video_duration_s"] = duration_s
        metadata["quality_score_means"] = averages
        metadata["frame_quality_gate"] = {
            "candidate_count": len(rows),
            "eligible_count": len(quality_eligible_rows),
            "rejected_count": rejected_count,
            "selected_count": len(selected_rows),
            "minimum_laplacian_variance": record.config.frame_min_laplacian_variance,
            "minimum_exposure_score": record.config.frame_min_exposure_score,
            "relative_sharpness_floor": record.config.frame_relative_sharpness_floor,
            "minimum_feature_count": record.config.frame_min_feature_count,
            "minimum_feature_grid_coverage": record.config.frame_min_feature_grid_coverage,
            "maximum_parallax_fraction": record.config.frame_max_parallax_fraction,
            "operator_override_count": sum(str(row["override"]) != "NONE" for row in rows),
        }
        atomic_json(telemetry_meta_path, metadata)
        self._transition(
            record,
            RunStatus.PREPROCESSING,
            50,
            f"Normalized {len(parsed.records)} telemetry samples and finalized preprocessing.",
        )
        return [
            run_dir / "frame_index.csv",
            scores_path,
            keyframes_path,
            contact_sheet_path,
            telemetry_path,
            telemetry_meta_path,
            *selected_paths,
        ]

    def _merge_telemetry_metadata(self, record: RunRecord, warnings: list[dict[str, str]]) -> Path:
        run_dir = self.store.run_dir(record.project_id, record.run_id)
        meta_path = run_dir / "normalized_telemetry.meta.json"
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PipelineError(
                "normalized_telemetry.meta.json is missing or invalid; rerun the telemetry parser "
                "and preserve its schema/version/warnings sidecar."
            ) from exc
        if metadata.get("schema_version") != "1.0":
            raise PipelineError("Only normalized telemetry schema_version 1.0 is supported.")
        for warning in metadata.get("warnings", []):
            if not isinstance(warning, dict):
                continue
            warnings.append(
                {
                    "code": str(warning.get("code", "TELEMETRY_WARNING")),
                    "message": str(warning.get("detail", warning.get("code", "Telemetry warning"))),
                }
            )
            if str(warning.get("code")) == "SYNTHETIC_TELEMETRY":
                record.telemetry_origin = ProvenanceOrigin.SYNTHETIC
                record.source_provenance = ProvenanceOrigin.SYNTHETIC
                record.synthetic_fixture = True
                self.store.save_run(record)
        ingest_path = run_dir / "ingest_report.json"
        ingest = json.loads(ingest_path.read_text(encoding="utf-8"))
        ingest["telemetry_normalization"] = metadata
        ingest["warnings"] = warnings
        ingest["source_provenance"] = record.source_provenance
        ingest["video_origin"] = record.video_origin
        ingest["telemetry_origin"] = record.telemetry_origin
        ingest["genuine_real_evidence"] = record.source_provenance == ProvenanceOrigin.REAL
        atomic_json(ingest_path, ingest)
        return ingest_path

    def _align_to_local_metric(
        self, record: RunRecord, warnings: list[dict[str, str]]
    ) -> list[Path]:
        run_dir = self.store.run_dir(record.project_id, record.run_id)
        if _is_synthetic_demo(record):
            transform_path = run_dir / "local_transform.json"
            sync_path = run_dir / "sync_report.json"
            sync_report = {
                "schema_version": "1.0",
                "telemetry_offset_s": 0.0,
                "offset_source": OffsetSource.NOT_APPLICABLE,
                "rmse_before_m": 0.0,
                "rmse_after_m": 0.0,
                "matched_camera_count": 10,
                "inlier_count": 10,
                "synthetic_fixture": True,
            }
            atomic_json(sync_path, sync_report)
            atomic_json(
                transform_path,
                {
                    "synthetic_fixture": True,
                    "coordinate_frame": "LOCAL_ENU_METRES",
                    "scale": 1.0,
                    "rotation": np.eye(3).tolist(),
                    "translation_m": [0.0, 0.0, 0.0],
                    "inlier_count": 10,
                    "residuals_m": [0.0] * 10,
                    "telemetry_offset_s": 0.0,
                    "offset_source": OffsetSource.NOT_APPLICABLE,
                    "rmse_before_m": 0.0,
                    "rmse_after_m": 0.0,
                },
            )
            record.telemetry_offset_s = 0.0
            record.offset_source = OffsetSource.NOT_APPLICABLE
            record.rmse_before_m = 0.0
            record.rmse_after_m = 0.0
            record.matched_camera_count = 10
            record.inlier_count = 10
            self.store.save_run(record)
            local_ply = run_dir / "sparse" / "sparse_local.ply"
            shutil.copyfile(run_dir / "sparse" / "sparse.ply", local_ply)
            return [transform_path, sync_path, local_ply]

        keyframe_payload = json.loads((run_dir / "keyframes.json").read_text(encoding="utf-8"))
        keyframes = (
            keyframe_payload.get("frames", [])
            if isinstance(keyframe_payload, dict)
            else keyframe_payload
        )
        if not isinstance(keyframes, list):
            raise PipelineError(
                "keyframes.json must be a list or an object containing a frames list."
            )
        timestamps: dict[str, float] = {}
        for item in keyframes:
            if not isinstance(item, dict) or not item.get("selected", True):
                continue
            name = item.get("image_name") or item.get("filename")
            if name is not None and item.get("timestamp_s") is not None:
                timestamps[Path(str(name)).name] = float(item["timestamp_s"])
        if len(timestamps) < 3:
            raise PipelineError(
                "At least three selected keyframes must declare image_name/filename and timestamp_s "
                "for metric alignment."
            )

        telemetry: list[dict[str, str]] = []
        with (run_dir / "normalized_telemetry.csv").open(newline="", encoding="utf-8") as stream:
            telemetry = list(csv.DictReader(stream))
        required = {
            "timestamp_s",
            "lat",
            "lon",
            "alt_m",
            "alt_source",
            "fix_quality",
            "source_row",
        }
        if not telemetry or not required.issubset(telemetry[0]):
            raise PipelineError(
                "normalized_telemetry.csv does not match Yosha's schema v1 column contract."
            )
        telemetry = [
            row
            for row in telemetry
            if row["lat"].strip()
            and row["lon"].strip()
            and row["alt_m"].strip()
            and row["fix_quality"] != "missing"
        ]
        if len(telemetry) < 3:
            raise PipelineError(
                "Fewer than three telemetry rows contain a usable coordinate fix; metric alignment "
                "cannot be solved."
            )
        telemetry.sort(key=lambda row: float(row["timestamp_s"]))
        times = np.array([float(row["timestamp_s"]) for row in telemetry])
        if np.any(np.diff(times) <= 0):
            raise PipelineError("Normalized telemetry timestamps must be strictly increasing.")
        origin = record.config.local_origin or (
            float(telemetry[0]["lat"]),
            float(telemetry[0]["lon"]),
            float(telemetry[0]["alt_m"]),
        )
        enu = np.array(
            [
                geodetic_to_enu(float(row["lat"]), float(row["lon"]), float(row["alt_m"]), origin)
                for row in telemetry
            ]
        )

        pose_rows: list[dict[str, str]]
        pose_path = run_dir / "camera_poses.csv"
        with pose_path.open(newline="", encoding="utf-8") as stream:
            pose_rows = list(csv.DictReader(stream))
        sfm_points: list[list[float]] = []
        frame_times: list[float] = []
        eligible_rows: list[dict[str, str]] = []
        for row in pose_rows:
            timestamp = timestamps.get(Path(row["image_name"]).name)
            if timestamp is None:
                continue
            sfm_points.append([float(row["sfm_x"]), float(row["sfm_y"]), float(row["sfm_z"])])
            frame_times.append(timestamp)
            eligible_rows.append(row | {"timestamp_s": str(timestamp)})
        if len(sfm_points) < 3:
            raise PipelineError(
                "Fewer than three registered cameras could be joined to telemetry timestamps; "
                "verify keyframe filenames and synchronization."
            )
        calibration = calibrate_telemetry_offset(
            np.array(sfm_points),
            np.array(frame_times),
            times,
            enu,
            manual_offset_s=record.config.telemetry_offset_s,
            manual_source=record.config.telemetry_offset_source,
        )
        selected = calibration.selected
        transform = selected.transform
        selected_sfm = np.array(sfm_points)[selected.matched_indices]
        aligned = transform.apply(selected_sfm)
        matched_rows = [eligible_rows[index] for index in selected.matched_indices]
        with pose_path.open("w", newline="", encoding="utf-8") as stream:
            fields = list(matched_rows[0]) + [
                "telemetry_timestamp_s",
                "x_m",
                "y_m",
                "z_m",
                "alignment_residual_m",
                "alignment_inlier",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index, row in enumerate(matched_rows):
                row.update(
                    {
                        "telemetry_timestamp_s": (float(row["timestamp_s"]) + selected.offset_s),
                        "x_m": aligned[index, 0],
                        "y_m": aligned[index, 1],
                        "z_m": aligned[index, 2],
                        "alignment_residual_m": transform.residuals_m[index],
                        "alignment_inlier": bool(transform.inliers[index]),
                    }
                )
                writer.writerow(row)
        transform_path = run_dir / "local_transform.json"
        sync_path = run_dir / "sync_report.json"
        sync_report = calibration.as_report()
        if calibration.ambiguity_status == "AMBIGUOUS":
            warnings.append(
                {
                    "code": "TELEMETRY_OFFSET_AMBIGUOUS",
                    "message": (
                        "Multiple bounded time offsets produced nearly equivalent alignment fits; "
                        "the selected minimum is not evidence of precise synchronization."
                    ),
                }
            )
        selected_keyframes = [
            item for item in keyframes if isinstance(item, dict) and item.get("selected", True)
        ]
        registered_names = {
            Path(row["image_name"]).name for row in pose_rows if row.get("image_name")
        }
        for item in keyframes:
            if not isinstance(item, dict):
                continue
            if not item.get("selected", True):
                item["reconstruction_status"] = "NOT_SELECTED"
                item["reconstruction_exclusion_reason"] = (
                    item.get("quality_rejection_reasons") or "FRAME_SELECTION_POLICY"
                )
                continue
            image_name = Path(str(item.get("image_name") or item.get("filename") or "")).name
            if image_name in registered_names:
                item["reconstruction_status"] = "REGISTERED_SELECTED_COMPONENT"
                item["reconstruction_exclusion_reason"] = None
            else:
                item["reconstruction_status"] = "NOT_REGISTERED_IN_SELECTED_COMPONENT"
                item["reconstruction_exclusion_reason"] = (
                    "COLMAP did not register this selected frame in the delivered component."
                )
        try:
            preprocessing_meta = json.loads(
                (run_dir / "normalized_telemetry.meta.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            preprocessing_meta = {}
        altitude_reference = preprocessing_meta.get("altitude_reference", "unknown")
        sync_report["altitude_reference"] = altitude_reference
        sync_report["altitude_reference_status"] = (
            "DECLARED_NON_SURVEY_REFERENCE"
            if str(altitude_reference).lower() in {"relative", "relative_to_launch", "unknown"}
            else "DECLARED"
        )
        if sync_report["altitude_reference_status"] == "DECLARED_NON_SURVEY_REFERENCE":
            warnings.append(
                {
                    "code": "ALTITUDE_REFERENCE_UNCERTAIN",
                    "message": (
                        f"Telemetry altitude reference is {altitude_reference!r}; vertical alignment "
                        "residuals are not independent vertical-accuracy evidence."
                    ),
                }
            )
        video_duration_s = float(
            preprocessing_meta.get("video_duration_s")
            or max((float(item.get("timestamp_s", 0)) for item in keyframes), default=0)
        )
        keyframe_payload["temporal_coverage"] = _temporal_coverage(
            keyframes,
            video_duration_s,
            record.config.coverage_interval_s,
            registered_names,
        )
        out_of_range: list[int] = []
        latitudes = np.array([float(row["lat"]) for row in telemetry])
        longitudes = np.array([float(row["lon"]) for row in telemetry])
        altitudes = np.array([float(row["alt_m"]) for row in telemetry])
        for item in selected_keyframes:
            shifted_timestamp = float(item["timestamp_s"]) + selected.offset_s
            item["telemetry_timestamp_s"] = shifted_timestamp
            if shifted_timestamp < times[0] or shifted_timestamp > times[-1]:
                item["telemetry_status"] = "OUT_OF_RANGE"
                out_of_range.append(int(item["frame_index"]))
                continue
            item["telemetry_status"] = "INTERPOLATED"
            item["telemetry"] = {
                "lat": float(np.interp(shifted_timestamp, times, latitudes)),
                "lon": float(np.interp(shifted_timestamp, times, longitudes)),
                "alt_m": float(np.interp(shifted_timestamp, times, altitudes)),
            }
        sync_report["out_of_range_frame_indices"] = out_of_range
        sync_report["selected_keyframe_count"] = len(selected_keyframes)
        sync_report["telemetry_covered_keyframe_count"] = len(selected_keyframes) - len(
            out_of_range
        )
        sync_report["telemetry_coverage_fraction"] = (
            (len(selected_keyframes) - len(out_of_range)) / len(selected_keyframes)
            if selected_keyframes
            else 0.0
        )
        if out_of_range:
            warnings.append(
                {
                    "code": "TELEMETRY_EXTRAPOLATION_REJECTED",
                    "message": (
                        f"{len(out_of_range)} selected frame(s) fell outside telemetry coverage "
                        "after applying the run-specific offset; no extrapolated position was used."
                    ),
                }
            )
        atomic_json(run_dir / "keyframes.json", keyframe_payload)
        atomic_json(sync_path, sync_report)
        atomic_json(
            transform_path,
            transform.as_dict()
            | {
                "coordinate_frame": "LOCAL_ENU_METRES",
                "origin_wgs84": {"lat": origin[0], "lon": origin[1], "alt_m": origin[2]},
                "outlier_image_names": [
                    matched_rows[index]["image_name"]
                    for index, keep in enumerate(transform.inliers)
                    if not keep
                ],
                "telemetry_offset_s": selected.offset_s,
                "offset_source": calibration.source,
                "rmse_before_m": calibration.before.rmse_m,
                "rmse_after_m": selected.rmse_m,
                "offset_ambiguity_status": calibration.ambiguity_status,
                "altitude_reference": altitude_reference,
            },
        )
        record.telemetry_offset_s = selected.offset_s
        record.offset_source = calibration.source
        record.rmse_before_m = calibration.before.rmse_m
        record.rmse_after_m = selected.rmse_m
        record.matched_camera_count = len(selected.matched_indices)
        record.inlier_count = selected.inlier_count
        self.store.save_run(record)
        local_ply = run_dir / "sparse" / "sparse_local.ply"
        transform_ply(run_dir / "sparse" / "sparse.ply", local_ply, transform)
        return [pose_path, transform_path, sync_path, local_ply, run_dir / "keyframes.json"]

    def _reconstruct(self, record: RunRecord) -> ReconstructionResult:
        run_dir = self.store.run_dir(record.project_id, record.run_id)
        if _is_synthetic_demo(record):
            ply = run_dir / "sparse" / "sparse.ply"
            _synthetic_ply(ply)
            poses = run_dir / "camera_poses.csv"
            with poses.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["frame_index", "timestamp_s", "x_m", "y_m", "z_m", "source"])
                for i in range(10):
                    writer.writerow([i, i * 0.5, i * 0.5, 0, 2, "SYNTHETIC_DEMO"])
            metrics = MatcherMetrics(
                matcher="SIFT_SYNTHETIC_PLACEHOLDER",
                eligible_frames=10,
                registered_frames=9,
                median_reprojection_error_px=0.9,
                p95_reprojection_error_px=1.4,
                runtime_s=0.01,
            )
            return ReconstructionResult(metrics, [ply, poses], [])
        if record.config.matcher != "SIFT":
            raise PipelineError(
                "Requested matcher SUPERPOINT_LIGHTGLUE is not implemented. No fallback was "
                "executed; create a linked rerun configured with SIFT."
            )
        record.requested_matcher = record.config.matcher
        record.executed_matcher = "SIFT"
        self.store.save_run(record)
        if record.config.camera_params and record.config.camera_params_reference == "SOURCE_VIDEO":
            payload = json.loads((run_dir / "keyframes.json").read_text(encoding="utf-8"))
            selected = [
                item
                for item in payload.get("frames", [])
                if isinstance(item, dict) and item.get("selected", True)
            ]
            transformed = [
                item.get("image_transform", {})
                for item in selected
                if item.get("image_transform", {}).get("rotation_degrees_clockwise", 0) != 0
                or abs(
                    float(item.get("image_transform", {}).get("uniform_resize_scale", 1.0)) - 1.0
                )
                > 1e-9
                or item.get("image_transform", {}).get("cropped", False)
            ]
            if transformed:
                raise PipelineError(
                    "Source-video camera parameters cannot be applied after frame rotation, resize, "
                    "or crop without an explicit intrinsic transform. Supply calibration for "
                    "PROCESSED_FRAMES or disable the image transform."
                )
        effective_config = record.config.model_copy(
            update={
                "use_gpu": record.effective_sparse_gpu,
                "worker_threads": _effective_worker_threads(record.config.worker_threads),
            }
        )
        return ColmapRunner(
            executor=self._managed_executor(record, record.config.sparse_timeout_s)
        ).run(run_dir / "frames", run_dir, effective_config)

    def _write_sparse_mask_usage(self, record: RunRecord, result: ReconstructionResult) -> Path:
        run_dir = self.store.run_dir(record.project_id, record.run_id)
        segmentation_path = run_dir / "segmentation_status.json"
        segmentation = (
            json.loads(segmentation_path.read_text(encoding="utf-8"))
            if segmentation_path.is_file()
            else {}
        )
        applied = any("--ImageReader.mask_path" in command for command in result.commands)
        expected = segmentation.get("status") == "completed"
        if expected and not applied and not record.synthetic_fixture:
            raise PipelineError(
                "Segmentation completed but sparse reconstruction did not receive the mask path"
            )
        payload = {
            "schema_version": "1.0",
            "segmentation_status_sha256": (
                sha256_file(segmentation_path) if segmentation_path.is_file() else None
            ),
            "masking_expected": expected,
            "masks_passed_to_sparse_backend": applied,
            "backend": "SYNTHETIC_DEMO" if record.synthetic_fixture else "COLMAP",
            "mask_flag": "--ImageReader.mask_path" if applied else None,
            "status": (
                "NOT_APPLICABLE_SYNTHETIC"
                if record.synthetic_fixture
                else "APPLIED"
                if applied
                else "EXPLICIT_UNMASKED"
            ),
            "warning": (
                "Synthetic execution does not prove real backend mask consumption."
                if record.synthetic_fixture
                else None
            ),
        }
        path = run_dir / "sparse" / "mask_usage.json"
        atomic_json(path, payload)
        return path

    def _run_optional_dense(
        self,
        record: RunRecord,
        result: ReconstructionResult,
        alignment_report: dict[str, object],
        warnings: list[dict[str, str]],
    ) -> list[Path]:
        if not record.config.enable_dense_reconstruction:
            return []
        run_dir = self.store.run_dir(record.project_id, record.run_id)
        residuals = np.asarray(alignment_report.get("residuals_m", []), dtype=float)
        inlier_count = int(alignment_report.get("inlier_count", 0))
        transform = SimilarityTransform(
            scale=float(alignment_report.get("scale", 1.0)),
            rotation=np.asarray(alignment_report.get("rotation", np.eye(3)), dtype=float),
            translation=np.asarray(
                alignment_report.get("translation_m", [0.0, 0.0, 0.0]), dtype=float
            ),
            inliers=np.ones(len(residuals), dtype=bool),
            residuals_m=residuals,
        )
        selection_path = run_dir / "sparse" / "model_selection.json"
        selected_model = ""
        if selection_path.is_file():
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            selected_model = str(selection.get("selected_model", ""))
        sparse_model_dir = run_dir / "sparse" / "filtered_model"
        if not sparse_model_dir.is_dir() or not any(sparse_model_dir.iterdir()):
            sparse_model_dir = run_dir / "sparse" / "model" / selected_model
        scene_analysis_path = run_dir / "scene_analysis.json"
        scene_analysis = (
            json.loads(scene_analysis_path.read_text(encoding="utf-8"))
            if scene_analysis_path.is_file()
            else {}
        )
        mask_dir = run_dir / "masks" / "reconstruction"
        if scene_analysis.get("masking_decision") != "APPLIED":
            mask_dir = None
        context = DenseContext(
            run_dir=run_dir,
            frames_dir=run_dir / "frames",
            sparse_model_dir=sparse_model_dir,
            registered_images=result.metrics.registered_frames,
            transform=transform,
            mask_dir=mask_dir,
            reconstruction_target=record.config.reconstruction_target,
            scene_analysis=scene_analysis,
            profile=record.config.profile,
            worker_threads=_effective_worker_threads(record.config.worker_threads),
        )
        gate_reasons: list[str] = []
        if record.synthetic_fixture:
            gate_reasons.append("synthetic fixtures cannot produce real dense evidence")
        if result.metrics.registration_rate < 0.8:
            gate_reasons.append("sparse registration is below the 80% gate")
        geometry_path = run_dir / "sparse" / "geometry_diagnostics.json"
        if geometry_path.is_file():
            geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
            if not geometry.get("measurement_geometry_gate_passed", False):
                gate_reasons.append(
                    "sparse geometric diagnostics found weak tracks, triangulation, or camera baseline"
                )
        if inlier_count < 3 or transform.scale <= 0:
            gate_reasons.append("local metric alignment gate did not produce three valid inliers")
        if not sparse_model_dir.is_dir() and not record.synthetic_fixture:
            gate_reasons.append("selected sparse COLMAP model directory is unavailable")
        if scene_analysis.get("dense_suitability") == "BLOCKED_WITHOUT_MASK" and mask_dir is None:
            gate_reasons.append(
                "scene diagnostics found extreme unstable/featureless content and no mask was applied"
            )
        if record.config.reconstruction_target == "PRIMARY_SUBJECT" and mask_dir is None:
            gate_reasons.append(
                "PRIMARY_SUBJECT reconstruction requires a complete applied mask set"
            )
        selected_preference = record.config.dense_provider
        if selected_preference == "auto":
            selected_preference = (
                "auto" if mask_dir is not None else record.selected_dense_provider or "unavailable"
            )
        if selected_preference == "unavailable":
            gate_reasons.append("server capability snapshot found no executable dense provider")
        provider = (
            UnavailableProvider("; ".join(gate_reasons))
            if gate_reasons
            else select_dense_provider(
                selected_preference,
                require_masks=mask_dir is not None,
                executor=self._managed_executor(record, record.config.dense_timeout_s),
            )
        )
        dense_result = run_dense_stage(context, provider)
        warnings.extend(dense_result.warnings)
        return dense_result.artifacts

    def run(self, run_id: str) -> RunRecord:
        with self.store.execution_lock(run_id) as execution_lock:
            if not execution_lock.acquired:
                return self.store.get_run(run_id)
            record = self.store.get_run(run_id)
            surviving = self._surviving_process_state(record)
            if surviving is not None:
                record.failure_reason = (
                    "Recovery deferred: a previously managed child process is still alive "
                    f"(pid {surviving.get('pid')}). No duplicate reconstruction was launched."
                )
                self.store.save_run(record)
                return record
            return self._run_locked(run_id)

    def _run_locked(self, run_id: str) -> RunRecord:
        record = self.store.get_run(run_id)
        if record.status == RunStatus.COMPLETED:
            return record
        checkpoint = self._load_checkpoint(record)
        warnings: list[dict[str, str]] = []
        reused_stages: list[str] = []
        resume_chain = True
        run_dir = self.store.run_dir(record.project_id, record.run_id)
        try:
            self._raise_if_cancelled(record)
            if resume_chain and self._checkpoint_stage_valid(record, checkpoint, "INGEST"):
                reused_stages.append("INGEST")
                warnings = list(checkpoint.completed["INGEST"].warnings)
            else:
                resume_chain = False
                self._begin_checkpoint_stage(record, checkpoint, "INGEST")
                self._transition(
                    record, RunStatus.INGESTING, 10, "Inspecting immutable input assets."
                )
                ingest, warnings = self._probe(record)
                capabilities = self._capture_capabilities(record)
                capability_payload = json.loads(capabilities.read_text(encoding="utf-8"))
                warnings.extend(capability_payload.get("warnings", []))
                self.store.register_artifacts(record, [ingest, capabilities])
                self._complete_checkpoint_stage(
                    record, checkpoint, "INGEST", [ingest, capabilities], warnings
                )

            self._raise_if_cancelled(record)
            scene_path = run_dir / "scene_analysis.json"
            if resume_chain and self._checkpoint_stage_valid(record, checkpoint, "PREPROCESS"):
                reused_stages.append("PREPROCESS")
                warnings = list(checkpoint.completed["PREPROCESS"].warnings)
            else:
                resume_chain = False
                self._begin_checkpoint_stage(record, checkpoint, "PREPROCESS")
                self._transition(
                    record,
                    RunStatus.PREPROCESSING,
                    30,
                    "Preparing scored frame selection and normalized telemetry.",
                )
                handoff = self._preprocess_contract(record)
                self.store.register_artifacts(record, handoff)
                keyframes_path = run_dir / "keyframes.json"
                keyframe_payload = json.loads(keyframes_path.read_text(encoding="utf-8"))
                frame_rows = keyframe_payload.get("frames", [])
                scene_path, _, scene_warnings = analyze_scene(
                    run_dir,
                    frame_rows,
                    reconstruction_target=record.config.reconstruction_target,
                    masking_mode=record.config.masking_mode,
                )
                warnings.extend(scene_warnings)
                self.store.register_artifacts(record, [scene_path])
                updated_ingest = self._merge_telemetry_metadata(record, warnings)
                self.store.register_artifacts(record, [updated_ingest])
                self._complete_checkpoint_stage(
                    record,
                    checkpoint,
                    "PREPROCESS",
                    [*handoff, scene_path, updated_ingest],
                    warnings,
                )

            self._raise_if_cancelled(record)
            if resume_chain and self._checkpoint_stage_valid(record, checkpoint, "SEGMENTATION"):
                reused_stages.append("SEGMENTATION")
                warnings = list(checkpoint.completed["SEGMENTATION"].warnings)
            else:
                resume_chain = False
                self._begin_checkpoint_stage(record, checkpoint, "SEGMENTATION")
                keyframes_path = run_dir / "keyframes.json"
                keyframe_payload = json.loads(keyframes_path.read_text(encoding="utf-8"))
                frame_rows = keyframe_payload.get("frames", [])
                segmentation_artifacts: list[Path] = []
                segmentation_warnings: list[dict[str, str]] = []
                requested_device = record.config.segmentation_device
                executed_device: str | None = None
                comparison: dict[str, object] | None = None
                if record.config.enable_segmentation:
                    executed_device, device_warnings = resolve_segmentation_device(record.config)
                    segmentation_warnings.extend(device_warnings)
                    if executed_device is not None:
                        settings = segmentation_settings(record.config, device=executed_device)
                        with self._heavy_job_slot(record), self._accelerator_slot(
                            record, executed_device
                        ):
                            segmentation_artifacts, provider_warnings = run_managed_segmentation(
                                run_dir,
                                record.run_id,
                                frame_rows,
                                record.config.segmentation_model_path,
                                reconstruction_target=record.config.reconstruction_target,
                                masking_mode=record.config.masking_mode,
                                settings=settings,
                                executor=self._managed_executor(
                                    record, record.config.segmentation_timeout_s
                                ),
                                accelerator_authorized=executed_device != "cpu",
                            )
                        segmentation_warnings.extend(provider_warnings)
                        comparison_path = run_dir / "segmentation_comparison.json"
                        if comparison_path.is_file():
                            comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
                    else:
                        comparison = {
                            "status": "BLOCKED"
                            if record.config.masking_mode == "REQUIRED"
                            else "UNAVAILABLE_FALLBACK",
                            "selected_frame_count": sum(
                                bool(item.get("selected", True))
                                for item in frame_rows
                                if isinstance(item, dict)
                            ),
                            "comparison": device_warnings[0]["message"],
                            "execution": {},
                        }
                else:
                    comparison = {
                        "status": "DISABLED",
                        "selected_frame_count": sum(
                            bool(item.get("selected", True))
                            for item in frame_rows
                            if isinstance(item, dict)
                        ),
                        "comparison": "Segmentation was not enabled for this run.",
                        "execution": {},
                    }
                    if any(item.get("code", "").startswith("SCENE_") for item in warnings):
                        segmentation_warnings.append(
                            {
                                "code": "SCENE_MASK_RECOMMENDATION_NOT_APPLIED",
                                "message": (
                                    "Scene analysis recommended masking, but segmentation was not "
                                    "enabled; reconstruction remains explicitly unmasked."
                                ),
                            }
                        )
                status_path = write_segmentation_status(
                    record,
                    run_dir,
                    comparison,
                    segmentation_warnings,
                    requested_device=requested_device,
                    executed_device=executed_device,
                )
                warnings.extend(segmentation_warnings)
                atomic_json(keyframes_path, keyframe_payload)
                self.store.register_artifacts(
                    record, [keyframes_path, status_path, *segmentation_artifacts]
                )
                blocking = [
                    warning["message"]
                    for warning in segmentation_warnings
                    if warning["code"].startswith("REQUIRED_SEGMENTATION")
                    or (
                        warning["code"] == "SEGMENTATION_DEVICE_UNAVAILABLE"
                        and record.config.masking_mode == "REQUIRED"
                    )
                ]
                if blocking:
                    raise PipelineError(
                        "Required reconstruction masking is unavailable: " + blocking[0]
                    )
                self._complete_checkpoint_stage(
                    record,
                    checkpoint,
                    "SEGMENTATION",
                    [keyframes_path, status_path, *segmentation_artifacts],
                    warnings,
                )

            self._raise_if_cancelled(record)
            metrics_path = run_dir / "sparse_metrics.json"
            alignment_path = run_dir / "local_transform.json"
            if resume_chain and self._checkpoint_stage_valid(record, checkpoint, "SPARSE"):
                reused_stages.append("SPARSE")
                warnings = list(checkpoint.completed["SPARSE"].warnings)
                metrics = MatcherMetrics.model_validate_json(
                    metrics_path.read_text(encoding="utf-8")
                )
                result = ReconstructionResult(metrics, [], [])
                alignment_report = json.loads(alignment_path.read_text(encoding="utf-8"))
            else:
                resume_chain = False
                self._begin_checkpoint_stage(record, checkpoint, "SPARSE")
                self._transition(
                    record, RunStatus.RECONSTRUCTING, 55, "Running sparse reconstruction."
                )
                with self._heavy_job_slot(record):
                    result = self._reconstruct(record)
                mask_usage_path = self._write_sparse_mask_usage(record, result)
                self.store.register_artifacts(record, [*result.artifacts, mask_usage_path])
                alignment = self._align_to_local_metric(record, warnings)
                self.store.register_artifacts(record, alignment)
                atomic_json(
                    metrics_path,
                    result.metrics.model_dump(mode="json")
                    | {
                        "registration_rate": result.metrics.registration_rate,
                        "synthetic_fixture": record.synthetic_fixture,
                    },
                )
                benchmark_path = run_dir / "matcher_benchmark.json"
                write_matcher_benchmark(benchmark_path, result.metrics, None)
                self.store.register_artifacts(record, [metrics_path, benchmark_path])
                alignment_report = json.loads(alignment_path.read_text(encoding="utf-8"))
                self._complete_checkpoint_stage(
                    record,
                    checkpoint,
                    "SPARSE",
                    [
                        *result.artifacts,
                        mask_usage_path,
                        *alignment,
                        metrics_path,
                        benchmark_path,
                    ],
                    warnings,
                )

            self._raise_if_cancelled(record)
            if record.config.enable_dense_reconstruction:
                if resume_chain and self._checkpoint_stage_valid(record, checkpoint, "DENSE"):
                    reused_stages.append("DENSE")
                    warnings = list(checkpoint.completed["DENSE"].warnings)
                else:
                    resume_chain = False
                    self._begin_checkpoint_stage(record, checkpoint, "DENSE")
                    self._transition(
                        record,
                        RunStatus.RECONSTRUCTING,
                        75,
                        "Running optional dense visual reconstruction after sparse metric gates.",
                    )
                    with self._heavy_job_slot(record):
                        dense_artifacts = self._run_optional_dense(
                            record, result, alignment_report, warnings
                        )
                    self.store.register_artifacts(record, dense_artifacts)
                    self._complete_checkpoint_stage(
                        record, checkpoint, "DENSE", dense_artifacts, warnings
                    )

            self._raise_if_cancelled(record)
            if resume_chain and self._checkpoint_stage_valid(record, checkpoint, "COVERAGE"):
                reused_stages.append("COVERAGE")
                warnings = list(checkpoint.completed["COVERAGE"].warnings)
            else:
                resume_chain = False
                self._begin_checkpoint_stage(record, checkpoint, "COVERAGE")
                project = self.store.get_project(record.project_id)
                raw_video = next(asset for asset in project.assets if asset.role == "video")
                coverage_path = write_coverage_report(
                    record, run_dir, video_sha256=raw_video.sha256
                )
                self.store.register_artifacts(record, [coverage_path])
                coverage_payload = json.loads(coverage_path.read_text(encoding="utf-8"))
                if coverage_payload.get("status") != "COMPLETED":
                    warnings.append(
                        {
                            "code": "COVERAGE_UNAVAILABLE",
                            "message": str(
                                coverage_payload.get("failure_reason")
                                or "Coverage analysis was unavailable"
                            ),
                        }
                    )
                self._complete_checkpoint_stage(
                    record, checkpoint, "COVERAGE", [coverage_path], warnings
                )

            self._raise_if_cancelled(record)
            if resume_chain and self._checkpoint_stage_valid(record, checkpoint, "COMPLETION"):
                reused_stages.append("COMPLETION")
                warnings = list(checkpoint.completed["COMPLETION"].warnings)
            else:
                resume_chain = False
                self._begin_checkpoint_stage(record, checkpoint, "COMPLETION")
                if record.config.enable_completion:
                    with self._heavy_job_slot(record):
                        completion_payload, completion_paths, _ = run_completion(
                            record,
                            run_dir,
                            CompletionRequest(),
                            cancel_requested=lambda: self._cancel_path(record).exists(),
                        )
                else:
                    completion_path = write_completion_not_run(
                        record, run_dir, "Completion was disabled by configuration."
                    )
                    completion_payload = json.loads(completion_path.read_text(encoding="utf-8"))
                    completion_paths = [completion_path]
                self.store.register_artifacts(record, completion_paths)
                if completion_payload.get("status") not in {"completed", "not_run"}:
                    warnings.append(
                        {
                            "code": "COMPLETION_NOT_PRODUCED",
                            "message": str(
                                completion_payload.get("failure_reason")
                                or "No inferred completion geometry was produced"
                            ),
                        }
                    )
                self._complete_checkpoint_stage(
                    record, checkpoint, "COMPLETION", completion_paths, warnings
                )

            self._raise_if_cancelled(record)
            quality_path = run_dir / "quality_report.json"
            if not (resume_chain and self._checkpoint_stage_valid(record, checkpoint, "REPORT")):
                self._begin_checkpoint_stage(record, checkpoint, "REPORT")
                self._transition(
                    record, RunStatus.REPORTING, 85, "Generating trust and known-distance report."
                )
                confidence_available = False
                if any(
                    artifact.relative_path == "point_confidence.json"
                    for artifact in record.artifacts
                ):
                    try:
                        validate_point_confidence_for_ply(
                            run_dir / "point_confidence.json",
                            run_dir / "sparse" / "sparse_local.ply",
                        )
                        confidence_available = True
                    except ValueError as exc:
                        warnings.append(
                            {"code": "INVALID_CONFIDENCE_ARTIFACT", "message": str(exc)}
                        )
                geometry_path = run_dir / "sparse" / "geometry_diagnostics.json"
                camera_selection_path = run_dir / "sparse" / "camera_model_selection.json"
                keyframe_report = json.loads(
                    (run_dir / "keyframes.json").read_text(encoding="utf-8")
                )
                dense_report_path = run_dir / "dense_report.json"
                report = build_quality_report(
                    record,
                    result.metrics,
                    warnings,
                    alignment=alignment_report,
                    confidence_available=confidence_available,
                    scene_analysis=(
                        json.loads(scene_path.read_text(encoding="utf-8"))
                        if scene_path.is_file()
                        else None
                    ),
                    frame_quality=keyframe_report.get("frame_quality_gate", {}),
                    geometry_diagnostics=(
                        json.loads(geometry_path.read_text(encoding="utf-8"))
                        if geometry_path.is_file()
                        else None
                    ),
                    camera_model_selection=(
                        json.loads(camera_selection_path.read_text(encoding="utf-8"))
                        if camera_selection_path.is_file()
                        else None
                    ),
                    temporal_coverage=keyframe_report.get("temporal_coverage", {}),
                    dense_report=(
                        json.loads(dense_report_path.read_text(encoding="utf-8"))
                        if dense_report_path.is_file()
                        else None
                    ),
                )
                write_quality_report(quality_path, report)
                export_path = run_dir / "export_readiness.json"
                export_started = time.monotonic()
                export_report = build_export_readiness(record, run_dir)
                write_export_readiness(export_path, export_report)
                generated_exports = export_artifact_paths(export_report, run_dir)
                export_duration_s = max(0.0, time.monotonic() - export_started)
                record.processing_completed_at = utc_now()
                self.store.save_run(record)
                report["execution"]["processing_completed_at"] = record.processing_completed_at
                write_quality_report(quality_path, report)
                self.store.register_artifacts(
                    record, [quality_path, export_path, *generated_exports]
                )
                capabilities_path = run_dir / "server_capabilities.json"
                capabilities_payload = (
                    json.loads(capabilities_path.read_text(encoding="utf-8"))
                    if capabilities_path.is_file()
                    else {}
                )
                benchmark_path = run_dir / "benchmark_report.json"
                report_started = self._stage_started_monotonic.get((record.run_id, "REPORT"))
                active_report_duration_s = (
                    max(0.0, time.monotonic() - report_started)
                    if report_started is not None
                    else None
                )
                atomic_json(
                    benchmark_path,
                    build_benchmark_report(
                        record,
                        checkpoint,
                        json.loads((run_dir / "ingest_report.json").read_text(encoding="utf-8")),
                        keyframe_report,
                        capabilities_payload,
                        reused_stages=reused_stages,
                        active_report_duration_s=active_report_duration_s,
                        export_duration_s=export_duration_s,
                        export_report=export_report,
                        dense_report=(
                            json.loads(dense_report_path.read_text(encoding="utf-8"))
                            if dense_report_path.is_file()
                            else None
                        ),
                    ),
                )
                self.store.register_artifacts(record, [benchmark_path])
                self._complete_checkpoint_stage(
                    record,
                    checkpoint,
                    "REPORT",
                    [quality_path, export_path, benchmark_path, *generated_exports],
                    warnings,
                )
            if record.processing_completed_at is None:
                record.processing_completed_at = utc_now()
            self._transition(record, RunStatus.COMPLETED, 100, "Run completed.", "COMPLETED")
            return record
        except ProcessCancelledError as exc:
            record = self.store.get_run(run_id)
            record.failure_reason = None
            record.stage = RunStatus.CANCELLED
            record.status = RunStatus.CANCELLED
            record.cancelled_at = utc_now()
            record.last_heartbeat_at = utc_now()
            record.events.append(
                StageEvent(
                    stage=RunStatus.CANCELLED,
                    status="CANCELLED",
                    progress=record.progress,
                    message=str(exc),
                )
            )
            self.store.save_run(record)
            return record
        except (
            PipelineError,
            ExternalToolError,
            ProcessTimeoutError,
            ValueError,
            OSError,
            json.JSONDecodeError,
        ) as exc:
            record.failure_reason = str(exc)
            record.stage = RunStatus.FAILED
            record.status = RunStatus.FAILED
            record.events.append(
                StageEvent(
                    stage=RunStatus.FAILED,
                    status="FAILED",
                    progress=record.progress,
                    message=str(exc),
                )
            )
            self.store.save_run(record)
            return record
