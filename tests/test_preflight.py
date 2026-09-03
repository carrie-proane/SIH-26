from __future__ import annotations

import json
import subprocess
from pathlib import Path

from sih26158.capabilities import build_capability_profile
from sih26158.models import RunConfig
from sih26158.preflight import OPENMVS_TOOLS, collect_preflight

COLMAP_FLAGS = (
    "--database_path --image_path --output_path --input_path --path --output_type "
    "--FeatureExtraction.use_gpu --FeatureMatching.use_gpu --workspace_path "
    "--PatchMatchStereo.gpu_index"
)


def _runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
    executable = Path(command[0]).name
    if executable == "ffprobe" and "-show_streams" in command:
        output = json.dumps(
            {
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "h264",
                        "width": 1920,
                        "height": 1080,
                        "pix_fmt": "yuv420p",
                        "avg_frame_rate": "30/1",
                    }
                ],
                "format": {"duration": "20.0"},
            }
        )
    elif executable in {"ffmpeg", "ffprobe"}:
        output = f"{executable} version test"
    elif executable == "colmap" and command[1:] == ["-h"]:
        output = "COLMAP test build with CUDA"
    elif executable == "colmap":
        output = COLMAP_FLAGS
    elif executable == "nvidia-smi":
        output = "Test GPU, 555.1, 24576, 8.9"
    else:
        output = "tool help"
    return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")


def _all_tools(name: str) -> str:
    return f"/fake/{name}"


def _base_tools_only(name: str) -> str | None:
    if name in {"ffmpeg", "ffprobe", "colmap"}:
        return f"/fake/{name}"
    return None


def test_preflight_reports_sparse_dense_resources_and_inputs(tmp_path: Path) -> None:
    video = tmp_path / "capture.mp4"
    video.write_bytes(b"fixture-video")
    telemetry = tmp_path / "capture.csv"
    telemetry.write_text(
        "timestamp_s,lat,lon,alt_m\n"
        "0,28.0,77.0,10\n"
        "10,28.0001,77.0001,11\n"
        "20,28.0002,77.0002,12\n",
        encoding="utf-8",
    )

    report = collect_preflight(
        data_root=tmp_path / "projects",
        video=video,
        telemetry=telemetry,
        require_dense=True,
        require_gpu=True,
        minimum_free_disk_gb=0,
        runner=_runner,
        which=_all_tools,
    )

    checks = {item["id"]: item for item in report["checks"]}
    assert report["status"] in {"READY", "READY_WITH_WARNINGS"}
    assert report["ready_for_sparse"] is True
    assert report["ready_for_dense"] is True
    assert checks["colmap.sparse_capabilities"]["status"] == "PASS"
    assert checks["gpu.nvidia_cuda"]["status"] == "PASS"
    assert checks["dense.openmvs"]["status"] == "PASS"
    assert checks["input.video"]["details"]["width"] == 1920
    assert checks["input.telemetry"]["details"]["row_count"] == 3
    assert checks["input.duration_coverage"]["status"] == "PASS"
    json.dumps(report)


def test_preflight_blocks_when_required_sparse_tools_are_missing(tmp_path: Path) -> None:
    report = collect_preflight(
        data_root=tmp_path / "projects",
        minimum_free_disk_gb=0,
        runner=_runner,
        which=lambda _: None,
    )

    assert report["status"] == "BLOCKED"
    assert report["ready_for_sparse"] is False
    assert "tool.ffmpeg" in report["blockers"]
    assert "tool.ffprobe" in report["blockers"]
    assert "tool.colmap" in report["blockers"]
    assert "colmap.sparse_capabilities" in report["blockers"]


def test_optional_dense_unavailable_preserves_sparse_readiness(tmp_path: Path) -> None:
    report = collect_preflight(
        data_root=tmp_path / "projects",
        dense_provider="auto",
        minimum_free_disk_gb=0,
        runner=lambda command, **kwargs: (
            subprocess.CompletedProcess(command, 0, stdout="COLMAP test build without CUDA", stderr="")
            if Path(command[0]).name == "colmap" and command[1:] == ["-h"]
            else _runner(command, **kwargs)
        ),
        which=_base_tools_only,
    )

    assert report["status"] == "READY_WITH_WARNINGS"
    assert report["ready_for_sparse"] is True
    assert report["ready_for_dense"] is False
    assert "dense.requested_provider" in report["warnings"]


def test_required_dense_unavailable_is_a_blocker(tmp_path: Path) -> None:
    report = collect_preflight(
        data_root=tmp_path / "projects",
        require_dense=True,
        minimum_free_disk_gb=0,
        runner=lambda command, **kwargs: (
            subprocess.CompletedProcess(command, 0, stdout="COLMAP test build without CUDA", stderr="")
            if Path(command[0]).name == "colmap" and command[1:] == ["-h"]
            else _runner(command, **kwargs)
        ),
        which=_base_tools_only,
    )

    assert report["status"] == "BLOCKED"
    assert report["ready_for_sparse"] is True
    assert report["ready_for_dense"] is False
    assert "dense.requested_provider" in report["blockers"]


def test_openmvs_contract_lists_every_external_tool() -> None:
    assert set(OPENMVS_TOOLS) == {
        "InterfaceCOLMAP",
        "DensifyPointCloud",
        "ReconstructMesh",
        "RefineMesh",
        "TextureMesh",
    }


def test_capability_profile_falls_back_to_cpu_sparse_and_openmvs_dense() -> None:
    report = {
        "environment": {"machine": "test"},
        "checks": [
            {"id": "colmap.sparse_capabilities", "status": "PASS"},
            {"id": "gpu.nvidia_cuda", "status": "WARN"},
            {"id": "dense.colmap_cuda", "status": "PASS"},
            {"id": "dense.openmvs", "status": "PASS"},
        ],
    }

    profile = build_capability_profile(
        report,
        RunConfig(use_gpu=True, enable_dense_reconstruction=True, dense_provider="auto"),
    )

    assert profile["sparse"]["selection"] == "COLMAP_CPU"
    assert profile["sparse"]["effective_gpu"] is False
    assert profile["dense"]["selected_provider"] == "openmvs"
    assert profile["dense"]["providers"] == {"colmap": False, "openmvs": True}
    assert profile["warnings"][0]["code"] == "SPARSE_GPU_UNAVAILABLE_CPU_SELECTED"


def test_capability_profile_selects_cuda_colmap_when_verified() -> None:
    report = {
        "environment": {},
        "checks": [
            {"id": "colmap.sparse_capabilities", "status": "PASS"},
            {"id": "gpu.nvidia_cuda", "status": "PASS"},
            {"id": "dense.colmap_cuda", "status": "PASS"},
            {"id": "dense.openmvs", "status": "PASS"},
        ],
    }

    profile = build_capability_profile(
        report,
        RunConfig(use_gpu=True, enable_dense_reconstruction=True, dense_provider="auto"),
    )

    assert profile["sparse"]["selection"] == "COLMAP_GPU"
    assert profile["dense"]["selected_provider"] == "colmap"
