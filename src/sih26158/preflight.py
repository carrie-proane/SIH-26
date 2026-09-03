from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from telemetry.csv_parser import parse_csv
from telemetry.srt_parser import parse_srt

Runner = Callable[..., subprocess.CompletedProcess[str]]
Which = Callable[[str], str | None]

OPENMVS_TOOLS = (
    "InterfaceCOLMAP",
    "DensifyPointCloud",
    "ReconstructMesh",
    "RefineMesh",
    "TextureMesh",
)


def _run(
    runner: Runner,
    command: list[str],
    *,
    timeout_s: float = 10,
) -> tuple[int | None, str]:
    try:
        completed = runner(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)
    return completed.returncode, ((completed.stdout or "") + (completed.stderr or "")).strip()


def _check(
    check_id: str,
    status: str,
    summary: str,
    **details: Any,
) -> dict[str, Any]:
    return {"id": check_id, "status": status, "summary": summary, "details": details}


def _first_line(output: str) -> str | None:
    return next((line.strip() for line in output.splitlines() if line.strip()), None)


def _memory_bytes() -> int | None:
    try:
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    return pages * page_size


def _probe_tool(
    name: str,
    version_args: list[str],
    *,
    runner: Runner,
    which: Which,
    required: bool = True,
) -> tuple[dict[str, Any], str | None, str]:
    path = which(name)
    missing_status = "BLOCKED" if required else "WARN"
    if path is None:
        return (
            _check(f"tool.{name}", missing_status, f"{name} executable is unavailable"),
            None,
            "",
        )
    returncode, output = _run(runner, [path, *version_args])
    if returncode not in {0, None}:
        return (
            _check(
                f"tool.{name}",
                missing_status,
                f"{name} capability probe failed",
                path=path,
                exit_code=returncode,
            ),
            path,
            output,
        )
    if returncode is None:
        return (
            _check(
                f"tool.{name}",
                missing_status,
                f"{name} capability probe could not execute",
                path=path,
                error=output,
            ),
            path,
            output,
        )
    return (
        _check(
            f"tool.{name}",
            "PASS",
            f"{name} executable is available",
            path=path,
            version=_first_line(output),
        ),
        path,
        output,
    )


def _colmap_sparse_capabilities(
    path: str | None,
    *,
    runner: Runner,
) -> tuple[dict[str, Any], dict[str, str]]:
    required = {
        "feature_extractor": (
            "--database_path",
            "--image_path",
            "--FeatureExtraction.use_gpu",
        ),
        "exhaustive_matcher": ("--database_path", "--FeatureMatching.use_gpu"),
        "sequential_matcher": ("--database_path", "--FeatureMatching.use_gpu"),
        "mapper": ("--database_path", "--image_path", "--output_path"),
        "bundle_adjuster": ("--input_path", "--output_path"),
        "model_analyzer": ("--path",),
        "model_converter": ("--input_path", "--output_path", "--output_type"),
    }
    help_by_command: dict[str, str] = {}
    if path is None:
        return _check(
            "colmap.sparse_capabilities",
            "BLOCKED",
            "Sparse COLMAP commands cannot be inspected without COLMAP",
        ), help_by_command

    missing: dict[str, list[str]] = {}
    failed: list[str] = []
    for command, flags in required.items():
        returncode, output = _run(runner, [path, command, "-h"])
        help_by_command[command] = output
        if returncode != 0:
            failed.append(command)
            continue
        absent = [flag for flag in flags if flag not in output]
        if absent:
            missing[command] = absent
    if failed or missing:
        return _check(
            "colmap.sparse_capabilities",
            "BLOCKED",
            "Installed COLMAP does not satisfy the sparse command contract",
            failed_commands=failed,
            missing_flags=missing,
        ), help_by_command
    return _check(
        "colmap.sparse_capabilities",
        "PASS",
        "Installed COLMAP exposes the current sparse command contract",
        inspected_commands=list(required),
    ), help_by_command


def _colmap_dense_capability(
    path: str | None,
    colmap_root_help: str,
    *,
    runner: Runner,
) -> dict[str, Any]:
    if path is None:
        return _check(
            "dense.colmap_cuda", "WARN", "CUDA COLMAP dense reconstruction is unavailable"
        )
    if "without CUDA" in colmap_root_help:
        return _check(
            "dense.colmap_cuda",
            "WARN",
            "Installed COLMAP explicitly reports a non-CUDA build",
        )
    required = {
        "image_undistorter": ("--image_path", "--input_path", "--output_path"),
        "patch_match_stereo": ("--workspace_path", "--PatchMatchStereo.gpu_index"),
        "stereo_fusion": ("--workspace_path", "--output_path"),
        "poisson_mesher": ("--input_path", "--output_path"),
    }
    missing: dict[str, list[str]] = {}
    failed: list[str] = []
    for command, flags in required.items():
        returncode, output = _run(runner, [path, command, "-h"])
        if returncode != 0:
            failed.append(command)
            continue
        absent = [flag for flag in flags if flag not in output]
        if absent:
            missing[command] = absent
    if failed or missing:
        return _check(
            "dense.colmap_cuda",
            "WARN",
            "Installed COLMAP cannot satisfy the CUDA dense command contract",
            failed_commands=failed,
            missing_flags=missing,
        )
    return _check(
        "dense.colmap_cuda",
        "PASS",
        "Installed COLMAP exposes CUDA dense commands",
        inspected_commands=list(required),
    )


def _openmvs_capability(*, which: Which) -> dict[str, Any]:
    paths = {tool: which(tool) for tool in OPENMVS_TOOLS}
    missing = [tool for tool, path in paths.items() if path is None]
    if missing:
        return _check(
            "dense.openmvs",
            "WARN",
            "OpenMVS external fallback is incomplete",
            missing_tools=missing,
            paths=paths,
        )
    return _check(
        "dense.openmvs",
        "PASS",
        "OpenMVS external command suite is available",
        paths=paths,
    )


def _nvidia_cuda(*, runner: Runner, which: Which, required: bool) -> dict[str, Any]:
    path = which("nvidia-smi")
    status = "BLOCKED" if required else "WARN"
    if path is None:
        return _check(
            "gpu.nvidia_cuda",
            status,
            "No NVIDIA CUDA device probe is available",
            note="Apple Metal and Intel integrated GPUs are not CUDA devices.",
        )
    returncode, output = _run(
        runner,
        [
            path,
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
    )
    if returncode != 0 or not output:
        return _check(
            "gpu.nvidia_cuda",
            status,
            "NVIDIA CUDA device probe failed",
            path=path,
            exit_code=returncode,
            output=output[:500],
        )
    return _check(
        "gpu.nvidia_cuda",
        "PASS",
        "NVIDIA CUDA device is available",
        devices=output.splitlines(),
    )


def _resource_checks(
    data_root: Path,
    *,
    minimum_free_disk_gb: float,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    cpu_count = os.cpu_count() or 0
    checks.append(
        _check(
            "resource.cpu",
            "PASS" if cpu_count >= 2 else "WARN",
            f"{cpu_count} logical CPU(s) detected",
            logical_cpu_count=cpu_count,
        )
    )
    memory_bytes = _memory_bytes()
    memory_gb = memory_bytes / 1_000_000_000 if memory_bytes is not None else None
    checks.append(
        _check(
            "resource.memory",
            "WARN" if memory_gb is None or memory_gb < 8 else "PASS",
            "Physical memory could not be measured"
            if memory_gb is None
            else f"{memory_gb:.1f} GB physical memory detected",
            physical_memory_bytes=memory_bytes,
        )
    )
    try:
        data_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".preflight-", dir=data_root):
            pass
    except OSError as exc:
        checks.append(
            _check(
                "resource.data_root_writable",
                "BLOCKED",
                "Data root is not writable",
                path=str(data_root.resolve()),
                error=str(exc),
            )
        )
        return checks
    checks.append(
        _check(
            "resource.data_root_writable",
            "PASS",
            "Data root is writable",
            path=str(data_root.resolve()),
        )
    )
    usage = shutil.disk_usage(data_root)
    free_gb = usage.free / 1_000_000_000
    checks.append(
        _check(
            "resource.free_disk",
            "PASS" if free_gb >= minimum_free_disk_gb else "BLOCKED",
            f"{free_gb:.1f} GB free disk space available",
            free_bytes=usage.free,
            total_bytes=usage.total,
            minimum_free_bytes=int(minimum_free_disk_gb * 1_000_000_000),
        )
    )
    return checks


def _video_check(
    video: Path | None,
    ffprobe_path: str | None,
    *,
    runner: Runner,
) -> tuple[dict[str, Any], float | None]:
    if video is None:
        return _check("input.video", "SKIPPED", "No video was supplied for input preflight"), None
    if not video.is_file():
        return _check(
            "input.video", "BLOCKED", "Video file does not exist", path=str(video)
        ), None
    if ffprobe_path is None:
        return _check(
            "input.video", "BLOCKED", "Video cannot be inspected without ffprobe", path=str(video)
        ), None
    returncode, output = _run(
        runner,
        [
            ffprobe_path,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-print_format",
            "json",
            str(video),
        ],
        timeout_s=30,
    )
    if returncode != 0:
        return _check(
            "input.video",
            "BLOCKED",
            "ffprobe could not decode the video container",
            path=str(video),
            exit_code=returncode,
            output=output[:1000],
        ), None
    try:
        payload = json.loads(output)
        stream = next(item for item in payload.get("streams", []) if item.get("codec_type") == "video")
        duration = float(payload.get("format", {}).get("duration") or stream.get("duration"))
    except (json.JSONDecodeError, StopIteration, TypeError, ValueError):
        return _check(
            "input.video",
            "BLOCKED",
            "No valid video stream and duration were reported",
            path=str(video),
        ), None
    details = {
        "path": str(video.resolve()),
        "size_bytes": video.stat().st_size,
        "codec": stream.get("codec_name"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "pixel_format": stream.get("pix_fmt"),
        "average_frame_rate": stream.get("avg_frame_rate"),
        "duration_s": duration,
        "rotation": stream.get("tags", {}).get("rotate"),
    }
    if duration <= 0 or not stream.get("width") or not stream.get("height"):
        return _check(
            "input.video", "BLOCKED", "Video stream metadata is incomplete", **details
        ), duration
    return _check("input.video", "PASS", "Video stream is readable", **details), duration


def _telemetry_check(
    telemetry: Path | None,
) -> tuple[dict[str, Any], float | None]:
    if telemetry is None:
        return _check(
            "input.telemetry", "SKIPPED", "No telemetry was supplied for input preflight"
        ), None
    if not telemetry.is_file():
        return _check(
            "input.telemetry",
            "BLOCKED",
            "Telemetry file does not exist",
            path=str(telemetry),
        ), None
    try:
        result = parse_srt(telemetry) if telemetry.suffix.lower() == ".srt" else parse_csv(telemetry)
    except OSError as exc:
        return _check(
            "input.telemetry",
            "BLOCKED",
            "Telemetry file could not be read",
            path=str(telemetry),
            error=str(exc),
        ), None
    coverage = result.field_coverage()
    details = {
        "path": str(telemetry.resolve()),
        "source_format": result.source_format,
        "source_dialect": result.source_dialect,
        "row_count": len(result.records),
        "duration_s": result.duration_s,
        "sample_rate_hz_estimated": result.estimated_rate_hz,
        "field_coverage": coverage,
        "warnings": result.warnings.as_list(),
    }
    if not result.records or coverage["lat"] == 0 or coverage["lon"] == 0:
        return _check(
            "input.telemetry",
            "BLOCKED",
            "Telemetry has no usable synchronized position records",
            **details,
        ), result.duration_s
    status = "WARN" if result.warnings else "PASS"
    return _check(
        "input.telemetry",
        status,
        "Telemetry parsed with declared limitations" if result.warnings else "Telemetry is readable",
        **details,
    ), result.duration_s


def collect_preflight(
    *,
    data_root: str | Path = "data/projects",
    video: str | Path | None = None,
    telemetry: str | Path | None = None,
    dense_provider: str = "auto",
    require_dense: bool = False,
    require_gpu: bool = False,
    minimum_free_disk_gb: float = 10.0,
    runner: Runner = subprocess.run,
    which: Which = shutil.which,
) -> dict[str, Any]:
    """Inspect this exact host and optional inputs without running reconstruction."""

    checks: list[dict[str, Any]] = []
    ffmpeg_check, _, _ = _probe_tool(
        "ffmpeg", ["-version"], runner=runner, which=which
    )
    ffprobe_check, ffprobe_path, _ = _probe_tool(
        "ffprobe", ["-version"], runner=runner, which=which
    )
    colmap_check, colmap_path, colmap_help = _probe_tool(
        "colmap", ["-h"], runner=runner, which=which
    )
    checks.extend((ffmpeg_check, ffprobe_check, colmap_check))
    sparse_check, _ = _colmap_sparse_capabilities(colmap_path, runner=runner)
    checks.append(sparse_check)
    checks.extend(
        _resource_checks(Path(data_root), minimum_free_disk_gb=minimum_free_disk_gb)
    )

    cuda_check = _nvidia_cuda(runner=runner, which=which, required=require_gpu)
    colmap_dense = _colmap_dense_capability(colmap_path, colmap_help, runner=runner)
    openmvs_dense = _openmvs_capability(which=which)
    checks.extend((cuda_check, colmap_dense, openmvs_dense))

    provider_available = {
        "colmap": colmap_dense["status"] == "PASS" and cuda_check["status"] == "PASS",
        "openmvs": openmvs_dense["status"] == "PASS",
    }
    dense_available = (
        any(provider_available.values())
        if dense_provider == "auto"
        else provider_available.get(dense_provider, False)
    )
    checks.append(
        _check(
            "dense.requested_provider",
            "PASS" if dense_available else ("BLOCKED" if require_dense else "WARN"),
            "Requested dense execution path is available"
            if dense_available
            else "Requested dense execution path is unavailable; sparse execution remains possible",
            requested=dense_provider,
            providers=provider_available,
            required=require_dense,
        )
    )

    video_check, video_duration = _video_check(
        Path(video) if video is not None else None,
        ffprobe_path,
        runner=runner,
    )
    telemetry_check, telemetry_duration = _telemetry_check(
        Path(telemetry) if telemetry is not None else None
    )
    checks.extend((video_check, telemetry_check))
    if video_duration is not None and telemetry_duration is not None:
        delta = abs(video_duration - telemetry_duration)
        tolerance = max(2.0, video_duration * 0.2)
        checks.append(
            _check(
                "input.duration_coverage",
                "PASS" if delta <= tolerance else "WARN",
                "Video and telemetry durations overlap plausibly"
                if delta <= tolerance
                else "Video and telemetry durations differ materially; synchronization requires review",
                video_duration_s=video_duration,
                telemetry_duration_s=telemetry_duration,
                absolute_difference_s=delta,
                tolerance_s=tolerance,
            )
        )
    else:
        checks.append(
            _check(
                "input.duration_coverage",
                "SKIPPED",
                "Video/telemetry duration comparison requires both valid inputs",
            )
        )

    blockers = [item["id"] for item in checks if item["status"] == "BLOCKED"]
    warnings = [item["id"] for item in checks if item["status"] == "WARN"]
    sparse_blockers = [
        item["id"]
        for item in checks
        if item["status"] == "BLOCKED" and not item["id"].startswith("dense.")
    ]
    ready_for_sparse = not sparse_blockers
    status = "BLOCKED" if blockers else ("READY_WITH_WARNINGS" if warnings else "READY")
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": status,
        "ready_for_sparse": ready_for_sparse,
        "ready_for_dense": ready_for_sparse and dense_available,
        "requested": {
            "data_root": str(Path(data_root).resolve()),
            "video": str(Path(video).resolve()) if video is not None else None,
            "telemetry": str(Path(telemetry).resolve()) if telemetry is not None else None,
            "dense_provider": dense_provider,
            "require_dense": require_dense,
            "require_gpu": require_gpu,
            "minimum_free_disk_gb": minimum_free_disk_gb,
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "blockers": blockers,
        "warnings": warnings,
        "checks": checks,
    }
