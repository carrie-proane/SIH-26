from __future__ import annotations

import csv
import json
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .confidence import classify_observed_point
from .models import MatcherMetrics, RunConfig
from .process_control import ManagedProcessExecutor


class ExternalToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReconstructionResult:
    metrics: MatcherMetrics
    artifacts: list[Path]
    commands: list[list[str]]


@dataclass(frozen=True)
class SparseModelCandidate:
    path: Path
    registered_images: int
    median_reprojection_error_px: float | None
    p95_reprojection_error_px: float | None

    def sort_key(self) -> tuple[int, float, str]:
        error = (
            self.median_reprojection_error_px
            if self.median_reprojection_error_px is not None
            else float("inf")
        )
        return (-self.registered_images, error, str(self.path))


@dataclass
class CameraModelAttempt:
    attempt_id: str
    camera_model: str
    workspace: Path
    recovery: bool
    status: str
    commands: list[list[str]]
    sparse_model: SparseModelCandidate | None = None
    intrinsics: dict[str, object] | None = None
    failure_reason: str | None = None

    def valid(self) -> bool:
        return (
            self.status == "COMPLETED"
            and self.sparse_model is not None
            and bool((self.intrinsics or {}).get("plausible", False))
        )

    def sort_key(self) -> tuple[int, float, str]:
        if self.sparse_model is None:
            return (0, float("inf"), self.attempt_id)
        return self.sparse_model.sort_key()


def choose_matcher(sift: MatcherMetrics, learned: MatcherMetrics | None) -> tuple[str, str]:
    if learned is None:
        return (
            "SIFT",
            "Learned matcher was unavailable; the CPU-safe SIFT baseline remains selected.",
        )
    registration_gain = learned.registration_rate - sift.registration_rate
    reprojection_gain = sift.median_reprojection_error_px - learned.median_reprojection_error_px
    if registration_gain > 0.005 or (registration_gain >= 0 and reprojection_gain > 0.05):
        return learned.matcher, (
            "Learned matcher selected because it improved registered-frame rate or median "
            "reprojection error without reducing registration."
        )
    return "SIFT", (
        "SIFT retained because the learned matcher did not improve registered-frame rate or "
        "reprojection error."
    )


def write_matcher_benchmark(
    output: Path, sift: MatcherMetrics, learned: MatcherMetrics | None
) -> dict[str, object]:
    selected, reason = choose_matcher(sift, learned)
    report: dict[str, object] = {
        "baseline": sift.model_dump(mode="json") | {"registration_rate": sift.registration_rate},
        "learned": None
        if learned is None
        else learned.model_dump(mode="json") | {"registration_rate": learned.registration_rate},
        "selected_matcher": selected,
        "decision": reason,
        "policy": "Keep SuperPoint+LightGlue only when registration or reprojection improves.",
        "learned_status": "BLOCKED_UNAVAILABLE" if learned is None else "EXECUTED",
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


class ColmapRunner:
    def __init__(
        self,
        binary: str = "colmap",
        executor: ManagedProcessExecutor | None = None,
    ) -> None:
        self.binary = binary
        self.executor = executor

    def doctor(self) -> str:
        location = shutil.which(self.binary)
        if location is None:
            raise ExternalToolError(
                "COLMAP is not installed or not on PATH. Install COLMAP, verify `colmap -h`, "
                "then rerun. The pipeline will not substitute synthetic geometry for a real run."
            )
        return location

    def build_commands(
        self,
        frames: Path,
        run_dir: Path,
        config: RunConfig,
        *,
        workspace: Path | None = None,
        recovery: bool = False,
    ) -> list[list[str]]:
        workspace = workspace or run_dir
        database = workspace / "sparse" / "database.db"
        model = workspace / "sparse" / "model"
        model.mkdir(parents=True, exist_ok=True)
        gpu = "1" if config.use_gpu else "0"
        feature_command = [
            self.binary,
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(frames),
            "--ImageReader.camera_model",
            config.camera_model,
            "--ImageReader.single_camera",
            "1",
            "--FeatureExtraction.use_gpu",
            gpu,
            "--SiftExtraction.max_num_features",
            (
                "32768"
                if recovery
                else "16384"
                if config.profile in {"balanced", "accurate", "diagnostic"}
                else "8192"
            ),
        ]
        if config.worker_threads:
            feature_command.extend(["--FeatureExtraction.num_threads", str(config.worker_threads)])
        if recovery:
            feature_command.extend(["--SiftExtraction.peak_threshold", "0.003"])
        if config.profile == "accurate":
            feature_command.extend(
                [
                    "--SiftExtraction.estimate_affine_shape",
                    "1",
                    "--SiftExtraction.domain_size_pooling",
                    "1",
                ]
            )
        if config.camera_params:
            feature_command.extend(["--ImageReader.camera_params", config.camera_params])
        mask_dir = run_dir / "masks" / "reconstruction"
        image_names = (
            [
                path.name
                for path in frames.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
            if frames.is_dir()
            else []
        )
        segmentation_status_path = run_dir / "segmentation_status.json"
        segmentation_status: dict[str, object] = {}
        if segmentation_status_path.is_file():
            try:
                candidate = json.loads(segmentation_status_path.read_text(encoding="utf-8"))
                if isinstance(candidate, dict):
                    segmentation_status = candidate
            except (OSError, json.JSONDecodeError):
                segmentation_status = {}
        masks_required = segmentation_status.get("status") == "completed"
        if masks_required:
            if not image_names:
                raise ExternalToolError("Applied segmentation has no reconstruction frames")
            for name in image_names:
                image = cv2.imread(str(frames / name), cv2.IMREAD_GRAYSCALE)
                mask_path = mask_dir / f"{name}.png"
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if image is None or mask is None or image.shape != mask.shape:
                    raise ExternalToolError(
                        f"Applied segmentation mask is missing or malformed for {name}"
                    )
                if not set(np.unique(mask).tolist()).issubset({0, 255}):
                    raise ExternalToolError(f"Applied segmentation mask is not binary for {name}")
            feature_command.extend(["--ImageReader.mask_path", str(mask_dir)])
        exhaustive = (
            recovery
            or config.matching_strategy == "EXHAUSTIVE"
            or (config.matching_strategy == "AUTO" and config.profile == "accurate")
        )
        matching_command = [
            self.binary,
            "exhaustive_matcher" if exhaustive else "sequential_matcher",
            "--database_path",
            str(database),
            "--FeatureMatching.use_gpu",
            gpu,
            "--FeatureMatching.guided_matching",
            "1",
            "--TwoViewGeometry.max_error",
            "3",
        ]
        if config.worker_threads:
            matching_command.extend(["--FeatureMatching.num_threads", str(config.worker_threads)])
        if not exhaustive:
            matching_command.extend(
                [
                    "--SequentialMatching.overlap",
                    str(config.sequential_overlap),
                    "--SequentialMatching.quadratic_overlap",
                    "1",
                ]
            )
            if config.vocab_tree_path and Path(config.vocab_tree_path).is_file():
                matching_command.extend(
                    [
                        "--SequentialMatching.loop_detection",
                        "1",
                        "--SequentialMatching.vocab_tree_path",
                        config.vocab_tree_path,
                    ]
                )
        mapper_command = [
            self.binary,
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(frames),
            "--output_path",
            str(model),
            "--Mapper.filter_max_reproj_error",
            "3",
            "--Mapper.filter_min_tri_angle",
            "2",
            "--Mapper.ba_refine_focal_length",
            "1" if config.refine_intrinsics else "0",
            "--Mapper.ba_refine_principal_point",
            "0",
            "--Mapper.ba_refine_extra_params",
            "1" if config.refine_intrinsics else "0",
            "--Mapper.ba_global_max_num_iterations",
            "100" if config.profile == "accurate" else "75",
        ]
        if config.worker_threads:
            mapper_command.extend(["--Mapper.num_threads", str(config.worker_threads)])
        if recovery:
            mapper_command.extend(
                [
                    "--Mapper.init_min_num_inliers",
                    "60",
                    "--Mapper.max_reg_trials",
                    "5",
                ]
            )
        return [feature_command, matching_command, mapper_command]

    def _refine_model(
        self, model_dir: Path, run_dir: Path, config: RunConfig, log: Path
    ) -> tuple[Path, list[list[str]]]:
        """Run a final global adjustment and conservative geometric point filter."""

        refined = run_dir / "sparse" / "refined_model"
        filtered = run_dir / "sparse" / "filtered_model"
        refined.mkdir(parents=True, exist_ok=True)
        filtered.mkdir(parents=True, exist_ok=True)
        commands = [
            [
                self.binary,
                "bundle_adjuster",
                "--input_path",
                str(model_dir),
                "--output_path",
                str(refined),
                "--BundleAdjustment.refine_focal_length",
                "1" if config.refine_intrinsics else "0",
                "--BundleAdjustment.refine_principal_point",
                "0",
                "--BundleAdjustment.refine_extra_params",
                "1" if config.refine_intrinsics else "0",
                "--BundleAdjustment.refine_points3D",
                "1",
                "--BundleAdjustmentCeres.max_num_iterations",
                "150" if config.profile == "accurate" else "100",
            ],
            [
                self.binary,
                "point_filtering",
                "--input_path",
                str(refined),
                "--output_path",
                str(filtered),
                "--min_track_len",
                "2",
                "--max_reproj_error",
                "3",
                "--min_tri_angle",
                "2",
            ],
        ]
        for command in commands:
            self._execute(command, log)
        return filtered, commands

    def _execute(self, command: list[str], log: Path) -> None:
        with log.open("a", encoding="utf-8") as stream:
            stream.write("$ " + " ".join(command) + "\n")
            if self.executor is None:
                result = subprocess.run(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            else:
                result = self.executor.run(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
        if result.returncode:
            raise ExternalToolError(
                f"COLMAP command failed with exit code {result.returncode}. Inspect {log.name}; "
                "verify overlap, intrinsics, blur, and selected-frame spacing."
            )

    @staticmethod
    def _read_registered_image_count(images_bin: Path) -> int:
        try:
            with images_bin.open("rb") as stream:
                raw = stream.read(8)
        except OSError as exc:
            raise ExternalToolError(f"Unable to inspect COLMAP model: {images_bin}") from exc
        if len(raw) != 8:
            raise ExternalToolError(f"Malformed COLMAP images.bin: {images_bin}")
        return int(struct.unpack("<Q", raw)[0])

    @staticmethod
    def _read_point_errors(points_bin: Path) -> list[float]:
        if not points_bin.is_file():
            return []
        errors: list[float] = []
        try:
            with points_bin.open("rb") as stream:
                count_raw = stream.read(8)
                if len(count_raw) != 8:
                    raise ExternalToolError(f"Malformed COLMAP points3D.bin: {points_bin}")
                point_count = struct.unpack("<Q", count_raw)[0]
                record_size = struct.calcsize("<QdddBBBd")
                for _ in range(point_count):
                    record = stream.read(record_size)
                    if len(record) != record_size:
                        raise ExternalToolError(f"Malformed COLMAP points3D.bin: {points_bin}")
                    error = float(struct.unpack("<QdddBBBd", record)[-1])
                    errors.append(error)
                    track_length_raw = stream.read(8)
                    if len(track_length_raw) != 8:
                        raise ExternalToolError(f"Malformed COLMAP points3D.bin: {points_bin}")
                    track_length = struct.unpack("<Q", track_length_raw)[0]
                    stream.seek(int(track_length) * 8, 1)
        except OSError as exc:
            raise ExternalToolError(f"Unable to inspect COLMAP model: {points_bin}") from exc
        return errors

    @classmethod
    def inspect_sparse_models(cls, model_root: Path) -> list[SparseModelCandidate]:
        candidates: list[SparseModelCandidate] = []
        for images_bin in sorted(model_root.glob("*/images.bin")):
            model_dir = images_bin.parent
            errors = cls._read_point_errors(model_dir / "points3D.bin")
            candidates.append(
                SparseModelCandidate(
                    path=model_dir,
                    registered_images=cls._read_registered_image_count(images_bin),
                    median_reprojection_error_px=(float(np.median(errors)) if errors else None),
                    p95_reprojection_error_px=(
                        float(np.percentile(errors, 95)) if errors else None
                    ),
                )
            )
        return candidates

    @classmethod
    def select_best_model(
        cls, model_root: Path, report_path: Path | None = None
    ) -> SparseModelCandidate:
        candidates = cls.inspect_sparse_models(model_root)
        if not candidates:
            raise ExternalToolError(
                "COLMAP completed without a sparse model. Check colmap.log for verified-match and "
                "camera-model diagnostics."
            )
        selected = min(candidates, key=SparseModelCandidate.sort_key)
        if report_path is not None:
            report = {
                "schema_version": "1.0",
                "selection_order": [
                    "highest registered_images",
                    "lowest median_reprojection_error_px",
                    "lexical path",
                ],
                "candidates": [
                    {
                        "path": str(candidate.path.relative_to(model_root)),
                        "registered_images": candidate.registered_images,
                        "median_reprojection_error_px": candidate.median_reprojection_error_px,
                        "p95_reprojection_error_px": candidate.p95_reprojection_error_px,
                    }
                    for candidate in sorted(candidates, key=lambda item: str(item.path))
                ],
                "selected_model": str(selected.path.relative_to(model_root)),
                "rationale": (
                    f"Selected {selected.path.relative_to(model_root)} with "
                    f"{selected.registered_images} registered images. Candidates are ranked by "
                    "registered-image count descending, median reprojection error ascending, then "
                    "lexical path."
                ),
            }
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return selected

    @staticmethod
    def _camera_model_candidates(config: RunConfig) -> list[str]:
        if config.camera_model_policy == "FIXED":
            return [config.camera_model]
        candidates = [config.camera_model, "SIMPLE_RADIAL", "RADIAL", "OPENCV"]
        return list(dict.fromkeys(candidates))

    @staticmethod
    def _inspect_intrinsics(cameras_txt: Path) -> dict[str, object]:
        cameras: list[dict[str, object]] = []
        reasons: list[str] = []
        for raw in cameras_txt.read_text(encoding="utf-8").splitlines():
            if not raw or raw.startswith("#"):
                continue
            values = raw.split()
            if len(values) < 8:
                reasons.append("MALFORMED_CAMERA_RECORD")
                continue
            model = values[1]
            width, height = int(values[2]), int(values[3])
            params = [float(value) for value in values[4:]]
            if model in {"SIMPLE_RADIAL", "RADIAL"} and len(params) >= 4:
                focal_x = focal_y = params[0]
                principal_x, principal_y = params[1:3]
                radial = params[3:]
                tangential: list[float] = []
            elif model == "OPENCV" and len(params) >= 8:
                focal_x, focal_y, principal_x, principal_y = params[:4]
                radial = params[4:6]
                tangential = params[6:8]
            else:
                reasons.append(f"UNSUPPORTED_OR_MALFORMED_MODEL:{model}")
                continue
            scale = float(max(width, height, 1))
            focal_ratio = min(focal_x, focal_y) / scale
            focal_aspect_ratio = max(focal_x, focal_y) / max(min(focal_x, focal_y), 1e-12)
            principal_offset = float(
                np.hypot(principal_x - width / 2, principal_y - height / 2) / scale
            )
            max_radial = max((abs(value) for value in radial), default=0.0)
            max_tangential = max((abs(value) for value in tangential), default=0.0)
            if not 0.2 <= focal_ratio <= 5.0:
                reasons.append(f"IMPLAUSIBLE_FOCAL_RATIO:{focal_ratio:.6f}")
            if focal_aspect_ratio > 2.0:
                reasons.append(f"IMPLAUSIBLE_FOCAL_ASPECT:{focal_aspect_ratio:.6f}")
            if principal_offset > 0.25:
                reasons.append(f"IMPLAUSIBLE_PRINCIPAL_POINT:{principal_offset:.6f}")
            if max_radial > 2.0:
                reasons.append(f"UNSTABLE_RADIAL_DISTORTION:{max_radial:.6f}")
            if max_tangential > 0.5:
                reasons.append(f"UNSTABLE_TANGENTIAL_DISTORTION:{max_tangential:.6f}")
            cameras.append(
                {
                    "camera_id": int(values[0]),
                    "model": model,
                    "width": width,
                    "height": height,
                    "focal_ratio": focal_ratio,
                    "focal_aspect_ratio": focal_aspect_ratio,
                    "principal_offset_fraction": principal_offset,
                    "max_abs_radial_distortion": max_radial,
                    "max_abs_tangential_distortion": max_tangential,
                }
            )
        if not cameras:
            reasons.append("NO_VALID_CAMERA_RECORDS")
        return {
            "plausible": bool(cameras) and not reasons,
            "camera_count": len(cameras),
            "cameras": cameras,
            "rejection_reasons": sorted(set(reasons)),
            "bounds": {
                "focal_ratio": [0.2, 5.0],
                "max_focal_aspect_ratio": 2.0,
                "max_principal_offset_fraction": 0.25,
                "max_abs_radial_distortion": 2.0,
                "max_abs_tangential_distortion": 0.5,
            },
        }

    def _run_camera_attempt(
        self,
        frames: Path,
        run_dir: Path,
        config: RunConfig,
        *,
        attempt_id: str,
        camera_model: str,
        recovery: bool,
        log: Path,
    ) -> CameraModelAttempt:
        workspace = run_dir / "sparse" / "attempts" / attempt_id
        # Attempt workspaces are private scratch state, never declared evidence.
        # COLMAP's feature extractor appends to an existing database, so reusing
        # a workspace left by a killed process can mix two executions or fail on
        # duplicate images. Restart the bounded attempt from a clean boundary;
        # completed, declared sparse exports elsewhere in the run remain intact.
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        attempt_config = config.model_copy(update={"camera_model": camera_model})
        commands = self.build_commands(
            frames, run_dir, attempt_config, workspace=workspace, recovery=recovery
        )
        attempt = CameraModelAttempt(
            attempt_id=attempt_id,
            camera_model=camera_model,
            workspace=workspace,
            recovery=recovery,
            status="STARTED",
            commands=commands,
        )
        try:
            with log.open("a", encoding="utf-8") as stream:
                stream.write(
                    f"\n# camera-model attempt {attempt_id}: {camera_model}; recovery={recovery}\n"
                )
            for command in commands:
                self._execute(command, log)
            selection_path = workspace / "sparse" / "model_selection.json"
            attempt.sparse_model = self.select_best_model(
                workspace / "sparse" / "model", selection_path
            )
            text_model = workspace / "sparse" / "text_model"
            text_model.mkdir(exist_ok=True)
            converter = [
                self.binary,
                "model_converter",
                "--input_path",
                str(attempt.sparse_model.path),
                "--output_path",
                str(text_model),
                "--output_type",
                "TXT",
            ]
            attempt.commands.append(converter)
            self._execute(converter, log)
            attempt.intrinsics = self._inspect_intrinsics(text_model / "cameras.txt")
            attempt.status = "COMPLETED"
        except (ExternalToolError, OSError, ValueError) as exc:
            attempt.status = "FAILED"
            attempt.failure_reason = str(exc)
        return attempt

    @staticmethod
    def _attempt_payload(attempt: CameraModelAttempt, root: Path) -> dict[str, object]:
        sparse = attempt.sparse_model
        component_report = attempt.workspace / "sparse" / "model_selection.json"
        try:
            component_candidates = json.loads(component_report.read_text(encoding="utf-8")).get(
                "candidates", []
            )
        except (OSError, json.JSONDecodeError):
            component_candidates = []
        return {
            "attempt_id": attempt.attempt_id,
            "camera_model": attempt.camera_model,
            "recovery": attempt.recovery,
            "status": attempt.status,
            "workspace": str(attempt.workspace.relative_to(root)),
            "registered_images": sparse.registered_images if sparse else 0,
            "median_reprojection_error_px": (
                sparse.median_reprojection_error_px if sparse else None
            ),
            "p95_reprojection_error_px": sparse.p95_reprojection_error_px if sparse else None,
            "selected_sparse_model": (
                str(sparse.path.relative_to(root)) if sparse is not None else None
            ),
            "intrinsics": attempt.intrinsics,
            "failure_reason": attempt.failure_reason,
            "commands": attempt.commands,
            "disconnected_components": component_candidates,
            "component_count": len(component_candidates),
        }

    @staticmethod
    def _retry_required(
        attempt: CameraModelAttempt, eligible_images: int
    ) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        sparse = attempt.sparse_model
        if sparse is None:
            return True, ["NO_SPARSE_MODEL"]
        registration_rate = sparse.registered_images / eligible_images if eligible_images else 0.0
        if registration_rate < 0.8:
            reasons.append("REGISTRATION_BELOW_80_PERCENT")
        if sparse.median_reprojection_error_px is None or sparse.median_reprojection_error_px > 1.5:
            reasons.append("MEDIAN_REPROJECTION_ABOVE_1_5_PX")
        if sparse.p95_reprojection_error_px is None or sparse.p95_reprojection_error_px > 4.0:
            reasons.append("P95_REPROJECTION_ABOVE_4_PX")
        return bool(reasons), reasons

    def run(self, frames: Path, run_dir: Path, config: RunConfig) -> ReconstructionResult:
        self.doctor()
        images = [p for p in frames.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if len(images) < 3:
            raise ExternalToolError(
                "Fewer than three selected images were provided. Supply Yosha's retained frames "
                "and keyframes.json before reconstruction."
            )
        log = run_dir / "logs" / "colmap.log"
        started = time.monotonic()
        attempts: list[CameraModelAttempt] = []
        for index, camera_model in enumerate(self._camera_model_candidates(config)):
            safe_model = re.sub(r"[^a-z0-9]+", "_", camera_model.lower()).strip("_")
            attempts.append(
                self._run_camera_attempt(
                    frames,
                    run_dir,
                    config,
                    attempt_id=f"{index:02d}_{safe_model}",
                    camera_model=camera_model,
                    recovery=False,
                    log=log,
                )
            )
        valid_attempts = [attempt for attempt in attempts if attempt.valid()]
        initial_selected = (
            min(valid_attempts, key=CameraModelAttempt.sort_key) if valid_attempts else None
        )
        retry_reasons: list[str] = []
        if initial_selected is not None:
            retry_required, retry_reasons = self._retry_required(initial_selected, len(images))
            if retry_required and config.max_reconstruction_retries:
                retry_model = re.sub(
                    r"[^a-z0-9]+", "_", initial_selected.camera_model.lower()
                ).strip("_")
                attempts.append(
                    self._run_camera_attempt(
                        frames,
                        run_dir,
                        config,
                        attempt_id=f"retry_01_{retry_model}",
                        camera_model=initial_selected.camera_model,
                        recovery=True,
                        log=log,
                    )
                )
        valid_attempts = [attempt for attempt in attempts if attempt.valid()]
        selected_attempt = (
            min(valid_attempts, key=CameraModelAttempt.sort_key) if valid_attempts else None
        )
        camera_selection_path = run_dir / "sparse" / "camera_model_selection.json"
        camera_report = {
            "schema_version": "1.0",
            "policy": config.camera_model_policy,
            "candidate_models": self._camera_model_candidates(config),
            "maximum_recovery_retries": config.max_reconstruction_retries,
            "retry_triggered": len(attempts) > len(self._camera_model_candidates(config)),
            "retry_trigger_reasons": retry_reasons,
            "selection_order": [
                "reject malformed or physically implausible intrinsics",
                "highest registered_images",
                "lowest median_reprojection_error_px",
                "lexical attempt path",
            ],
            "attempts": [self._attempt_payload(attempt, run_dir) for attempt in attempts],
            "selected_attempt": selected_attempt.attempt_id if selected_attempt else None,
            "selected_camera_model": selected_attempt.camera_model if selected_attempt else None,
            "rationale": (
                "Selected the valid attempt using registered-image count descending, median "
                "reprojection error ascending, and lexical attempt path. One recovery attempt "
                "is permitted only when the initial winner misses registration or reprojection gates."
            ),
        }
        camera_selection_path.write_text(
            json.dumps(camera_report, indent=2) + "\n", encoding="utf-8"
        )
        if selected_attempt is None or selected_attempt.sparse_model is None:
            raise ExternalToolError(
                "Every bounded camera-model attempt failed or produced implausible intrinsics. "
                "Inspect sparse/camera_model_selection.json and logs/colmap.log; no model was "
                "silently accepted."
            )
        selection_path = run_dir / "sparse" / "model_selection.json"
        selection_report = {
            "schema_version": "1.0",
            "selection_order": [
                "valid camera intrinsics",
                "highest registered_images",
                "lowest median_reprojection_error_px",
                "lexical path",
            ],
            "candidates": [
                {
                    "path": (
                        str(attempt.sparse_model.path.relative_to(run_dir / "sparse"))
                        if attempt.sparse_model
                        else None
                    ),
                    "camera_model": attempt.camera_model,
                    "registered_images": (
                        attempt.sparse_model.registered_images if attempt.sparse_model else 0
                    ),
                    "median_reprojection_error_px": (
                        attempt.sparse_model.median_reprojection_error_px
                        if attempt.sparse_model
                        else None
                    ),
                    "p95_reprojection_error_px": (
                        attempt.sparse_model.p95_reprojection_error_px
                        if attempt.sparse_model
                        else None
                    ),
                    "valid_intrinsics": bool((attempt.intrinsics or {}).get("plausible", False)),
                    "status": attempt.status,
                }
                for attempt in attempts
            ],
            "selected_model": str(
                selected_attempt.sparse_model.path.relative_to(run_dir / "sparse")
            ),
            "selected_camera_model": selected_attempt.camera_model,
            "rationale": camera_report["rationale"],
            "excluded_components": [
                {
                    "attempt_id": attempt.attempt_id,
                    "camera_model": attempt.camera_model,
                    "path": str(
                        (
                            attempt.workspace / "sparse" / "model" / str(component.get("path"))
                        ).relative_to(run_dir / "sparse")
                    ),
                    "registered_images": component.get("registered_images"),
                    "reason": (
                        "DISCONNECTED_COMPONENT_NOT_SELECTED; no supported alignment was available, "
                        "so it was not merged into the delivered model."
                    ),
                }
                for attempt in attempts
                for component in self._attempt_payload(attempt, run_dir).get(
                    "disconnected_components", []
                )
                if not (
                    attempt is selected_attempt
                    and str(component.get("path")) == selected_attempt.sparse_model.path.name
                )
            ],
            "component_policy": (
                "Every disconnected mapper component is reported. Only the deterministic winner is "
                "delivered; components are never merged without a supported alignment."
            ),
        }
        selection_path.write_text(json.dumps(selection_report, indent=2) + "\n", encoding="utf-8")
        commands = [command for attempt in attempts for command in attempt.commands]
        model_dir, refinement_commands = self._refine_model(
            selected_attempt.sparse_model.path, run_dir, config, log
        )
        commands.extend(refinement_commands)
        text_model = run_dir / "sparse" / "text_model"
        text_model.mkdir(exist_ok=True)
        final_converter = [
            self.binary,
            "model_converter",
            "--input_path",
            str(model_dir),
            "--output_path",
            str(text_model),
            "--output_type",
            "TXT",
        ]
        commands.append(final_converter)
        self._execute(final_converter, log)
        ply = run_dir / "sparse" / "sparse.ply"
        confidence_path = run_dir / "point_confidence.json"
        self._export_points_and_confidence(
            text_model / "points3D.txt",
            text_model / "images.txt",
            ply,
            confidence_path,
        )
        poses = run_dir / "camera_poses.csv"
        self._export_camera_poses(text_model / "images.txt", poses)
        geometry_report = run_dir / "sparse" / "geometry_diagnostics.json"
        self._write_geometry_diagnostics(
            text_model / "points3D.txt",
            text_model / "images.txt",
            geometry_report,
            eligible_images=len(images),
        )
        selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
        selection_payload["refined_export_model"] = str(model_dir.relative_to(run_dir / "sparse"))
        selection_payload["refinement"] = (
            "Global bundle adjustment followed by track/reprojection/triangulation filtering"
        )
        selection_path.write_text(json.dumps(selection_payload, indent=2) + "\n", encoding="utf-8")
        analyzer_command = [self.binary, "model_analyzer", "--path", str(model_dir)]
        commands.append(analyzer_command)
        analysis = (
            subprocess.run(
                analyzer_command,
                capture_output=True,
                text=True,
                check=False,
            )
            if self.executor is None
            else self.executor.run(analyzer_command, capture_output=True)
        )
        (run_dir / "sparse" / "model_analysis.txt").write_text(
            analysis.stdout + analysis.stderr, encoding="utf-8"
        )
        commands_path = run_dir / "sparse" / "sparse_commands.json"
        commands_path.write_text(
            json.dumps({"schema_version": "1.0", "commands": commands}, indent=2) + "\n",
            encoding="utf-8",
        )
        values = {
            k.lower().replace(" ", "_"): v
            for k, v in re.findall(r"^([^:]+):\s*(.+)$", analysis.stdout, re.MULTILINE)
        }
        registered = self._read_registered_image_count(model_dir / "images.bin")
        analyzer_error = float(str(values.get("mean_reprojection_error", "0")).split()[0])
        refined_errors = self._read_point_errors(model_dir / "points3D.bin")
        median_error = float(np.median(refined_errors)) if refined_errors else None
        p95_error = float(np.percentile(refined_errors, 95)) if refined_errors else None
        metrics = MatcherMetrics(
            matcher="SIFT",
            eligible_frames=len(images),
            registered_frames=registered,
            median_reprojection_error_px=max(
                0.0, analyzer_error if median_error is None else median_error
            ),
            p95_reprojection_error_px=max(0.0, analyzer_error if p95_error is None else p95_error),
            runtime_s=time.monotonic() - started,
        )
        return ReconstructionResult(
            metrics,
            [
                ply,
                poses,
                log,
                run_dir / "sparse" / "model_analysis.txt",
                selection_path,
                camera_selection_path,
                confidence_path,
                geometry_report,
                commands_path,
            ],
            commands,
        )

    @classmethod
    def _write_geometry_diagnostics(
        cls,
        points_txt: Path,
        images_txt: Path,
        output: Path,
        *,
        eligible_images: int,
    ) -> dict[str, object]:
        centres = cls._camera_centres(images_txt)
        track_lengths: list[int] = []
        errors: list[float] = []
        angles: list[float] = []
        positions: list[np.ndarray] = []
        for raw in points_txt.read_text(encoding="utf-8").splitlines():
            if not raw or raw.startswith("#"):
                continue
            values = raw.split()
            if len(values) < 12:
                continue
            position = np.asarray([float(value) for value in values[1:4]], dtype=float)
            image_ids = [int(values[index]) for index in range(8, len(values), 2)]
            positions.append(position)
            track_lengths.append(len(image_ids))
            errors.append(float(values[7]))
            angles.append(cls._triangulation_angle(position, image_ids, centres))
        camera_values = np.asarray(list(centres.values()), dtype=float)
        point_values = np.asarray(positions, dtype=float)
        camera_span = (
            float(np.linalg.norm(np.ptp(camera_values, axis=0))) if len(camera_values) else 0.0
        )
        scene_span = (
            float(np.linalg.norm(np.ptp(point_values, axis=0))) if len(point_values) else 0.0
        )
        baseline_ratio = camera_span / scene_span if scene_span > 0 else 0.0
        if len(camera_values) >= 3:
            singular = np.linalg.svd(
                camera_values - np.mean(camera_values, axis=0), compute_uv=False
            )
            camera_rank_ratio = float(singular[1] / singular[0]) if singular[0] > 0 else 0.0
        else:
            camera_rank_ratio = 0.0
        registered = len(centres)
        registration_rate = registered / eligible_images if eligible_images else 0.0
        median_track = float(np.median(track_lengths)) if track_lengths else 0.0
        median_angle = float(np.median(angles)) if angles else 0.0
        median_error = float(np.median(errors)) if errors else float("inf")
        p95_error = float(np.percentile(errors, 95)) if errors else float("inf")
        warnings: list[dict[str, str]] = []
        if registration_rate < 0.8:
            warnings.append(
                {
                    "code": "LOW_REGISTRATION",
                    "message": "Fewer than 80% of selected frames registered.",
                }
            )
        if median_angle < 2:
            warnings.append(
                {
                    "code": "LOW_TRIANGULATION_ANGLE",
                    "message": "Median point triangulation angle is below 2 degrees; depth is weakly constrained.",
                }
            )
        if median_track < 3:
            warnings.append(
                {
                    "code": "SHORT_FEATURE_TRACKS",
                    "message": "Median reconstructed point track has fewer than three supporting views.",
                }
            )
        if baseline_ratio < 0.01:
            warnings.append(
                {
                    "code": "MOSTLY_ROTATIONAL_CAPTURE",
                    "message": "Camera-centre span is very small relative to the reconstructed scene.",
                }
            )
        if camera_rank_ratio < 0.02:
            warnings.append(
                {
                    "code": "DEGENERATE_CAMERA_PATH",
                    "message": "Registered camera centres have very limited two-dimensional spread.",
                }
            )
        gates = {
            "registration_80_percent": registration_rate >= 0.8,
            "median_reprojection_1_5_px": median_error <= 1.5,
            "p95_reprojection_4_px": p95_error <= 4,
            "median_track_length_3": median_track >= 3,
            "median_triangulation_angle_2_deg": median_angle >= 2,
            "camera_baseline_ratio_0_01": baseline_ratio >= 0.01,
        }
        payload: dict[str, object] = {
            "schema_version": "1.0",
            "status": "PASS" if all(gates.values()) else "LIMITED",
            "measurement_geometry_gate_passed": all(gates.values()),
            "eligible_images": eligible_images,
            "registered_images": registered,
            "registration_rate": registration_rate,
            "point_count": len(positions),
            "median_track_length": median_track,
            "median_triangulation_angle_deg": median_angle,
            "median_reprojection_error_px": median_error if errors else None,
            "p95_reprojection_error_px": p95_error if errors else None,
            "camera_span_scene_ratio": baseline_ratio,
            "camera_secondary_spread_ratio": camera_rank_ratio,
            "gates": gates,
            "warnings": warnings,
            "interpretation": "These diagnostics constrain geometric reliability; they do not independently verify metric scale.",
        }
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload

    @staticmethod
    def _quaternion_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
        norm = np.linalg.norm([qw, qx, qy, qz])
        if norm == 0:
            raise ExternalToolError("COLMAP returned a zero-length camera quaternion.")
        qw, qx, qy, qz = np.array([qw, qx, qy, qz]) / norm
        return np.array(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ]
        )

    @classmethod
    def _export_camera_poses(cls, images_txt: Path, output: Path) -> None:
        if not images_txt.is_file():
            raise ExternalToolError("COLMAP text export did not produce images.txt.")
        rows: list[list[object]] = []
        image_line = True
        for raw in images_txt.read_text(encoding="utf-8").splitlines():
            if not raw or raw.startswith("#"):
                continue
            if image_line:
                values = raw.split()
                if len(values) < 10:
                    raise ExternalToolError("Malformed COLMAP images.txt camera record.")
                qw, qx, qy, qz = map(float, values[1:5])
                translation = np.array(list(map(float, values[5:8])))
                rotation = cls._quaternion_rotation(qw, qx, qy, qz)
                center = -(rotation.T @ translation)
                rows.append([values[0], values[9], *center.tolist(), qw, qx, qy, qz])
            image_line = not image_line
        if not rows:
            raise ExternalToolError("COLMAP sparse model contains no registered camera poses.")
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["image_id", "image_name", "sfm_x", "sfm_y", "sfm_z", "qw", "qx", "qy", "qz"]
            )
            writer.writerows(rows)

    @classmethod
    def _camera_centres(cls, images_txt: Path) -> dict[int, np.ndarray]:
        centres: dict[int, np.ndarray] = {}
        image_line = True
        for raw in images_txt.read_text(encoding="utf-8").splitlines():
            if not raw or raw.startswith("#"):
                continue
            if image_line:
                values = raw.split()
                if len(values) < 10:
                    raise ExternalToolError("Malformed COLMAP images.txt camera record.")
                qw, qx, qy, qz = map(float, values[1:5])
                translation = np.array(list(map(float, values[5:8])))
                rotation = cls._quaternion_rotation(qw, qx, qy, qz)
                centres[int(values[0])] = -(rotation.T @ translation)
            image_line = not image_line
        return centres

    @staticmethod
    def _triangulation_angle(
        point: np.ndarray, image_ids: list[int], camera_centres: dict[int, np.ndarray]
    ) -> float:
        rays = []
        for image_id in sorted(set(image_ids)):
            centre = camera_centres.get(image_id)
            if centre is None:
                continue
            ray = centre - point
            norm = float(np.linalg.norm(ray))
            if norm > 0:
                rays.append(ray / norm)
        maximum = 0.0
        for first_index, first in enumerate(rays):
            for second in rays[first_index + 1 :]:
                cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
                maximum = max(maximum, float(np.degrees(np.arccos(cosine))))
        return maximum

    @classmethod
    def _export_points_and_confidence(
        cls, points_txt: Path, images_txt: Path, ply_path: Path, confidence_path: Path
    ) -> None:
        """Export PLY and confidence together in ascending COLMAP point-ID order."""
        if not points_txt.is_file() or not images_txt.is_file():
            raise ExternalToolError("COLMAP text export lacks points3D.txt or images.txt.")
        camera_centres = cls._camera_centres(images_txt)
        points: list[tuple[int, np.ndarray, tuple[int, int, int], float, list[int]]] = []
        for raw in points_txt.read_text(encoding="utf-8").splitlines():
            if not raw or raw.startswith("#"):
                continue
            values = raw.split()
            if len(values) < 8 or (len(values) - 8) % 2:
                raise ExternalToolError("Malformed COLMAP points3D.txt point record.")
            source_id = int(values[0])
            position = np.array(list(map(float, values[1:4])))
            color = tuple(map(int, values[4:7]))
            error = float(values[7])
            image_ids = [int(values[index]) for index in range(8, len(values), 2)]
            points.append((source_id, position, color, error, image_ids))
        if not points:
            raise ExternalToolError("COLMAP sparse model contains no triangulated points.")
        points.sort(key=lambda item: item[0])
        ply_lines = [
            "ply",
            "format ascii 1.0",
            "comment vertex order is ascending COLMAP POINT3D_ID",
            f"element vertex {len(points)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]
        confidence_records: list[dict[str, object]] = []
        for vertex_id, (_, position, color, error, image_ids) in enumerate(points):
            angle = cls._triangulation_angle(position, image_ids, camera_centres)
            track_length = len(image_ids)
            ply_lines.append(
                " ".join(
                    [*(f"{value:.12g}" for value in position), *(str(value) for value in color)]
                )
            )
            confidence_records.append(
                {
                    "point_id": vertex_id,
                    "supporting_views": len(set(image_ids)),
                    "track_length": track_length,
                    "reprojection_error": error,
                    "triangulation_angle": angle,
                    "confidence_class": classify_observed_point(track_length, error, angle).value,
                }
            )
        ply_path.write_text("\n".join(ply_lines) + "\n", encoding="utf-8")
        confidence_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "point_order": "PLY_VERTEX_ORDER",
                    "points": confidence_records,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
