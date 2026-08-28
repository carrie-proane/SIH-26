from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from sih26158.dense import (
    ColmapDenseProvider,
    DenseContext,
    DenseProviderError,
    DenseReconstructionProvider,
    UnavailableProvider,
    run_dense_stage,
    select_dense_provider,
)
from sih26158.geo import SimilarityTransform

PLY = """ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face 1
property list uchar int vertex_indices
end_header
0 0 0 255 0 0
1 0 0 0 255 0
0 1 0 0 0 255
3 0 1 2
"""


def _context(tmp_path: Path) -> DenseContext:
    run_dir = tmp_path / "run"
    frames = run_dir / "frames"
    sparse = run_dir / "sparse" / "0"
    for path in (frames, sparse, run_dir / "logs"):
        path.mkdir(parents=True, exist_ok=True)
    return DenseContext(
        run_dir=run_dir,
        frames_dir=frames,
        sparse_model_dir=sparse,
        registered_images=91,
        transform=SimilarityTransform(
            scale=2.0,
            rotation=np.eye(3),
            translation=np.array([1.0, 2.0, 3.0]),
            inliers=np.ones(3, dtype=bool),
            residuals_m=np.zeros(3),
        ),
    )


def _help(command: str) -> str:
    return {
        "image_undistorter": "--image_path --input_path --output_path --output_type",
        "patch_match_stereo": "--workspace_path --workspace_format --PatchMatchStereo.gpu_index",
        "stereo_fusion": "--workspace_path --workspace_format --input_type --output_path",
        "poisson_mesher": "--input_path --output_path",
        "mesh_simplifier": (
            "--input_path --output_path --MeshSimplification.target_face_ratio"
        ),
        "mesh_texturer": "--workspace_path --input_path --output_path --output_type",
    }[command]


def _successful_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
    if command == ["colmap", "-h"]:
        return subprocess.CompletedProcess(command, 0, "COLMAP 4.1.1 with CUDA", "")
    if len(command) == 3 and command[-1] == "-h":
        return subprocess.CompletedProcess(command, 0, _help(command[1]), "")
    output = (
        Path(command[command.index("--output_path") + 1])
        if "--output_path" in command
        else None
    )
    if command[1] == "image_undistorter":
        assert output is not None
        output.mkdir(parents=True, exist_ok=True)
    elif command[1] in {"stereo_fusion", "poisson_mesher", "mesh_simplifier"}:
        assert output is not None
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(PLY, encoding="ascii")
    elif command[1] == "mesh_texturer":
        assert output is not None
        output.mkdir(parents=True, exist_ok=True)
        (output / "textured.ply").write_text(PLY, encoding="ascii")
        (output / "atlas.png").write_bytes(b"mock-png")
    return subprocess.CompletedProcess(command, 0, "", "")


def test_dense_colmap_success_and_metric_transform(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.setattr("sih26158.dense.shutil.which", lambda _: "/usr/bin/colmap")
    context = _context(tmp_path)
    result = run_dense_stage(context, ColmapDenseProvider(runner=_successful_runner))

    assert result.status == "COMPLETED"
    assert result.dense_point_count == 3
    assert result.vertex_count == 3
    assert result.face_count == 1
    assert result.metric_alignment_preserved is True
    assert result.measurement_eligible is False
    assert (context.run_dir / "dense/fused.ply").is_file()
    assert (context.run_dir / "dense/meshed.ply").is_file()
    assert (context.run_dir / "dense/meshed-poisson-simplified.ply").is_file()
    assert (context.run_dir / "dense/texture/textured.ply").is_file()
    assert (context.run_dir / "dense/texture/atlas.png").is_file()

    fused = (context.run_dir / "dense/fused.ply").read_text(encoding="ascii")
    body = fused.split("end_header\n", 1)[1].splitlines()
    assert body[0].split()[:3] == ["1", "2", "3"]
    assert body[1].split()[:3] == ["3", "2", "3"]

    report = json.loads((context.run_dir / "dense/dense_report.json").read_text())
    assert report["registered_inputs"] == 91
    assert report["available"] is True
    assert report["failure_reason"] is None
    assert report["artifacts"]["fused_ply"] == "dense/fused.ply"
    assert report["artifacts"]["mesh"] == "dense/meshed.ply"
    assert report["artifacts"]["texture_dir"] == "dense/texture/"
    assert report["metric_alignment_preserved"] is True
    assert report["measurement_eligible"] is False
    commands = json.loads((context.run_dir / "dense/dense_commands.json").read_text())
    assert [item[1] for item in commands["commands"]] == [
        "image_undistorter",
        "patch_match_stereo",
        "stereo_fusion",
        "poisson_mesher",
        "mesh_simplifier",
        "mesh_texturer",
    ]


def test_dense_unavailable_records_blocker(tmp_path: Path) -> None:
    context = _context(tmp_path)
    result = run_dense_stage(context, UnavailableProvider("CUDA dense unavailable"))

    assert result.status == "UNAVAILABLE"
    report = json.loads((context.run_dir / "dense/dense_report.json").read_text())
    assert report["status"] == "UNAVAILABLE"
    assert report["warnings"][0]["code"] == "DENSE_PROVIDER_UNAVAILABLE"
    assert report["available"] is False
    assert report["failure_reason"] == "CUDA dense unavailable"
    assert report["artifacts"] == {
        "fused_ply": None,
        "mesh": None,
        "texture_dir": None,
    }
    assert not (context.run_dir / "dense/fused.ply").exists()


class _FailingProvider(DenseReconstructionProvider):
    name = "MOCK_DENSE"

    def availability(self) -> tuple[bool, str]:
        return True, "mocked"

    def run(self, context: DenseContext):  # type: ignore[no-untyped-def]
        raise DenseProviderError("mock PatchMatch failure")


def test_dense_failure_preserves_sparse_success(tmp_path: Path) -> None:
    context = _context(tmp_path)
    sparse_evidence = context.run_dir / "sparse/sparse_local.ply"
    sparse_evidence.write_text(PLY, encoding="ascii")

    result = run_dense_stage(context, _FailingProvider())

    assert result.status == "FAILED_VISUAL_ONLY"
    assert sparse_evidence.read_text(encoding="ascii") == PLY
    report = json.loads((context.run_dir / "dense/dense_report.json").read_text())
    assert report["sparse_evidence_preserved"] is True
    assert report["measurement_eligible"] is False
    assert "mock PatchMatch failure" in report["warnings"][0]["message"]
    assert report["available"] is False
    assert report["failure_reason"] == "mock PatchMatch failure"
    assert set(report) >= {
        "provider",
        "available",
        "runtime_s",
        "registered_inputs",
        "failure_reason",
        "warnings",
        "artifacts",
    }


def test_non_cuda_colmap_is_honestly_unavailable(
    monkeypatch: object,
) -> None:
    monkeypatch.setattr("sih26158.dense.shutil.which", lambda _: "/usr/bin/colmap")

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, "COLMAP 4.1.1 without CUDA", "")

    available, reason = ColmapDenseProvider(runner=runner).availability()
    assert available is False
    assert "CUDA runtime is unavailable" in reason


def test_provider_selection_prefers_colmap_then_openmvs(monkeypatch: object) -> None:
    monkeypatch.setattr(
        "sih26158.dense.ColmapDenseProvider.availability", lambda _: (True, "CUDA")
    )
    monkeypatch.setattr(
        "sih26158.dense.OpenMVSProvider.availability", lambda _: (True, "OpenMVS")
    )
    assert isinstance(select_dense_provider(), ColmapDenseProvider)

    monkeypatch.setattr(
        "sih26158.dense.ColmapDenseProvider.availability", lambda _: (False, "no CUDA")
    )
    from sih26158.dense import OpenMVSProvider

    assert isinstance(select_dense_provider(), OpenMVSProvider)


def test_openmvs_detection_requires_named_executable(monkeypatch: object) -> None:
    from sih26158.dense import OpenMVSProvider

    observed: list[str] = []

    def which(name: str) -> str | None:
        observed.append(name)
        return None

    monkeypatch.setattr("sih26158.dense.shutil.which", which)
    available, reason = OpenMVSProvider().availability()

    assert available is False
    assert observed == ["OpenMVS"]
    assert "OpenMVS executable" in reason


def test_mesh_failure_is_partial_dense_success(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.setattr("sih26158.dense.shutil.which", lambda _: "/usr/bin/colmap")

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[1] == "poisson_mesher" and command[-1] != "-h":
            return subprocess.CompletedProcess(command, 1, "", "mesh failure")
        return _successful_runner(command, **kwargs)

    context = _context(tmp_path)
    result = run_dense_stage(context, ColmapDenseProvider(runner=runner))

    assert result.status == "PARTIAL_SUCCESS"
    assert result.available is True
    assert (context.run_dir / "dense/fused.ply").is_file()
    assert not (context.run_dir / "dense/meshed.ply").exists()
    assert any(warning["code"] == "DENSE_MESH_FAILED" for warning in result.warnings)
