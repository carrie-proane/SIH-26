from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from sih26158.dense import (
    ColmapDenseProvider,
    DenseContext,
    DenseProviderError,
    DenseReconstructionProvider,
    OpenMVSProvider,
    UnavailableProvider,
    assess_texture_atlas,
    dense_profile_settings,
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

TEXTURED_PLY = PLY.replace(
    "format ascii 1.0\n",
    "format ascii 1.0\ncomment TextureFile scene_mesh_texture0.png\n",
)


def _write_atlas(path: Path, rgb: tuple[int, int, int], size: int = 32) -> None:
    image = np.full((size, size, 3), rgb[::-1], dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


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
        "mesh_simplifier": ("--input_path --output_path --MeshSimplification.target_face_ratio"),
        "mesh_texturer": "--workspace_path --input_path --output_path --output_type",
    }[command]


def _successful_runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
    if command == ["colmap", "-h"]:
        return subprocess.CompletedProcess(command, 0, "COLMAP 4.1.1 with CUDA", "")
    if len(command) == 3 and command[-1] == "-h":
        return subprocess.CompletedProcess(command, 0, _help(command[1]), "")
    output = (
        Path(command[command.index("--output_path") + 1]) if "--output_path" in command else None
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


def test_dense_colmap_success_and_metric_transform(tmp_path: Path, monkeypatch: object) -> None:
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
    assert (context.run_dir / "dense/meshed-poisson.ply").is_file()
    assert (context.run_dir / "dense/meshed-poisson-simplified.ply").is_file()
    assert (context.run_dir / "dense/textured/textured.ply").is_file()
    assert (context.run_dir / "dense/textured/atlas.png").is_file()

    fused = (context.run_dir / "dense/fused.ply").read_text(encoding="ascii")
    body = fused.split("end_header\n", 1)[1].splitlines()
    assert body[0].split()[:3] == ["1", "2", "3"]
    assert body[1].split()[:3] == ["3", "2", "3"]

    report = json.loads((context.run_dir / "dense_report.json").read_text())
    assert report["input_registered_images"] == 91
    assert report["metric_alignment_preserved"] is True
    assert report["measurement_eligible"] is False
    commands = json.loads((context.run_dir / "dense_commands.json").read_text())
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
    report = json.loads((context.run_dir / "dense_report.json").read_text())
    assert report["status"] == "UNAVAILABLE"
    assert report["warnings"][0]["code"] == "DENSE_PROVIDER_UNAVAILABLE"
    assert not (context.run_dir / "dense/fused.ply").exists()


def test_openmvs_uses_openmvs_bin_environment_directory(tmp_path: Path, monkeypatch: object) -> None:
    bin_dir = tmp_path / "openmvs-bin"
    bin_dir.mkdir()
    for tool in OpenMVSProvider.tools:
        executable = bin_dir / tool
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        executable.chmod(0o755)
    monkeypatch.setenv("OPENMVS_BIN", str(bin_dir))
    calls: list[list[str]] = []

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "--resolution-level", "")

    provider = OpenMVSProvider(runner=runner)

    available, reason = provider.availability()

    assert available is True
    assert f"OPENMVS_BIN={bin_dir}" in reason
    provider._help("TextureMesh")
    assert calls == [[str(bin_dir / "TextureMesh"), "-h"]]


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
    report = json.loads((context.run_dir / "dense_report.json").read_text())
    assert report["sparse_evidence_preserved"] is True
    assert report["measurement_eligible"] is False
    assert "mock PatchMatch failure" in report["warnings"][0]["message"]


def test_openmvs_existing_outputs_export_to_viewer_contract(tmp_path: Path) -> None:
    context = _context(tmp_path)
    raw_dir = context.run_dir / "dense/openmvs_sfm"
    raw_dir.mkdir(parents=True)
    dense_with_views = """ply
format ascii 1.0
element vertex 2
property float x
property float y
property float z
property list uchar uint view_indices
end_header
0 0 0 2 4 9
1 0 0 1 7
"""
    (raw_dir / "scene_dense.ply").write_text(dense_with_views, encoding="ascii")
    (raw_dir / "scene_mesh.ply").write_text(PLY, encoding="ascii")
    (raw_dir / "scene_mesh_refine.ply").write_text(PLY, encoding="ascii")
    (raw_dir / "scene_mesh_texture.ply").write_text(TEXTURED_PLY, encoding="ascii")
    _write_atlas(raw_dir / "scene_mesh_texture0.png", (90, 120, 150))

    artifacts, points, vertices, faces = OpenMVSProvider.export_existing_outputs(context, raw_dir)

    relative = {str(path.relative_to(context.run_dir)) for path in artifacts}
    assert {
        "dense/fused.ply",
        "dense/meshed-openmvs.ply",
        "dense/meshed-openmvs-refined.ply",
        "dense/textured/model.ply",
        "dense/textured/scene_mesh_texture0.png",
    } <= relative
    assert (points, vertices, faces) == (2, 3, 1)
    fused_body = (
        (context.run_dir / "dense/fused.ply")
        .read_text(encoding="ascii")
        .split("end_header\n", 1)[1]
        .splitlines()
    )
    assert fused_body[0] == "1 2 3 2 4 9"
    assert fused_body[1] == "3 2 3 1 7"


def test_texture_atlas_validation_rejects_primary_colour_clipping(tmp_path: Path) -> None:
    mesh = tmp_path / "scene_mesh_texture.ply"
    mesh.write_text(TEXTURED_PLY, encoding="ascii")
    _write_atlas(tmp_path / "scene_mesh_texture0.png", (255, 0, 0))

    assessment = assess_texture_atlas(mesh)

    assert assessment.accepted is False
    assert assessment.status == "REJECTED"
    assert assessment.primary_clip_fraction == 1.0
    assert "Primary-colour clipping" in assessment.reasons[0]


def test_texture_atlas_validation_measures_openmvs_empty_coverage(tmp_path: Path) -> None:
    mesh = tmp_path / "scene_mesh_texture.ply"
    mesh.write_text(TEXTURED_PLY, encoding="ascii")
    image = np.full((20, 20, 3), (150, 120, 90), dtype=np.uint8)
    image[:5, :] = (39, 127, 255)  # OpenCV BGR for OpenMVS RGB(255, 127, 39).
    assert cv2.imwrite(str(tmp_path / "scene_mesh_texture0.png"), image)

    assessment = assess_texture_atlas(mesh)

    assert assessment.accepted is True
    assert assessment.coverage == 0.75
    assert assessment.empty_fraction == 0.25


def test_non_cuda_colmap_is_honestly_unavailable(
    monkeypatch: object,
) -> None:
    monkeypatch.setattr("sih26158.dense.shutil.which", lambda _: "/usr/bin/colmap")

    def runner(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, "COLMAP 4.1.1 without CUDA", "")

    available, reason = ColmapDenseProvider(runner=runner).availability()
    assert available is False
    assert "non-CUDA" in reason


def test_dense_profiles_make_accurate_full_resolution_and_support_aware() -> None:
    preview = dense_profile_settings("preview")
    accurate = dense_profile_settings("accurate")

    assert preview.resolution_level == 2
    assert accurate.resolution_level == 0
    assert accurate.number_views_fuse >= 3
    assert accurate.filter_point_cloud == 1
    assert accurate.max_texture_size == 8192


def test_openmvs_retries_rejected_atlas_with_conservative_settings(
    tmp_path: Path, monkeypatch: object
) -> None:
    context = _context(tmp_path)
    provider = OpenMVSProvider()
    monkeypatch.setattr("sih26158.dense.shutil.which", lambda _: "/usr/bin/tool")
    monkeypatch.setattr(
        provider,
        "_help",
        lambda *_: (
            "--global-seam-leveling --local-seam-leveling --sharpness-weight "
            "--ignore-mask-label --close-holes"
        ),
    )
    monkeypatch.setattr(provider, "_require_flags", lambda *_: None)
    texture_attempts = 0

    def execute(command: list[str], _: Path) -> None:
        nonlocal texture_attempts
        if command[0] == "DensifyPointCloud":
            Path(command[command.index("-o") + 1]).with_suffix(".ply").write_text(
                PLY, encoding="ascii"
            )
        elif command[0] in {"ReconstructMesh", "RefineMesh"}:
            Path(command[command.index("-o") + 1]).write_text(PLY, encoding="ascii")
        elif command[0] == "TextureMesh":
            texture_attempts += 1
            mesh = Path(command[command.index("-o") + 1])
            mesh.write_text(TEXTURED_PLY, encoding="ascii")
            atlas = mesh.parent / "scene_mesh_texture0.png"
            _write_atlas(atlas, (255, 0, 0) if texture_attempts == 1 else (90, 120, 150))

    monkeypatch.setattr(provider, "_execute", execute)

    result = provider.run(context)

    texture_commands = [command for command in result.commands if command[0] == "TextureMesh"]
    assert len(texture_commands) == 2
    assert "--local-seam-leveling" in texture_commands[1]
    assert texture_commands[1][texture_commands[1].index("--local-seam-leveling") + 1] == "0"
    assert "--global-seam-leveling" not in texture_commands[1]
    assert result.status == "COMPLETED"
    assert result.texture_validation["accepted"] is True
    assert result.texture_validation["atlas_nonempty_fraction"] == 1.0
    assert result.texture_coverage is None
    assert (context.run_dir / "dense/textured/model.ply").is_file()
    assert any(
        warning["code"] == "TEXTURE_ATLAS_CONSERVATIVE_RETRY_ACCEPTED"
        for warning in result.warnings
    )
    assert (
        context.run_dir / "dense/openmvs_sfm/texture_rejected/initial/scene_mesh_texture0.png"
    ).is_file()


def test_openmvs_rejects_bad_retry_but_preserves_dense_model(
    tmp_path: Path, monkeypatch: object
) -> None:
    context = _context(tmp_path)
    provider = OpenMVSProvider()
    monkeypatch.setattr("sih26158.dense.shutil.which", lambda _: "/usr/bin/tool")
    monkeypatch.setattr(
        provider,
        "_help",
        lambda *_: (
            "--global-seam-leveling --local-seam-leveling --sharpness-weight "
            "--ignore-mask-label --close-holes"
        ),
    )
    monkeypatch.setattr(provider, "_require_flags", lambda *_: None)

    def execute(command: list[str], _: Path) -> None:
        if command[0] == "DensifyPointCloud":
            Path(command[command.index("-o") + 1]).with_suffix(".ply").write_text(
                PLY, encoding="ascii"
            )
        elif command[0] in {"ReconstructMesh", "RefineMesh"}:
            Path(command[command.index("-o") + 1]).write_text(PLY, encoding="ascii")
        elif command[0] == "TextureMesh":
            mesh = Path(command[command.index("-o") + 1])
            mesh.write_text(TEXTURED_PLY, encoding="ascii")
            _write_atlas(mesh.parent / "scene_mesh_texture0.png", (0, 255, 0))

    monkeypatch.setattr(provider, "_execute", execute)

    result = provider.run(context)

    assert result.status == "COMPLETED"
    assert result.texture_validation["accepted"] is False
    assert (context.run_dir / "dense/fused.ply").is_file()
    assert (context.run_dir / "dense/meshed-openmvs-refined.ply").is_file()
    assert not (context.run_dir / "dense/textured/model.ply").exists()
    assert any(warning["code"] == "TEXTURE_ATLAS_REJECTED_FINAL" for warning in result.warnings)


def test_openmvs_applies_masks_and_adaptive_primary_subject_filtering(
    tmp_path: Path, monkeypatch: object
) -> None:
    context = _context(tmp_path)
    mask_dir = context.run_dir / "masks" / "reconstruction"
    mask_dir.mkdir(parents=True)
    context = replace(
        context,
        mask_dir=mask_dir,
        reconstruction_target="PRIMARY_SUBJECT",
        scene_analysis={"masking_decision": "APPLIED"},
        profile="accurate",
    )
    provider = OpenMVSProvider()
    monkeypatch.setattr("sih26158.dense.shutil.which", lambda _: "/usr/bin/tool")
    monkeypatch.setattr(provider, "_require_flags", lambda *_: None)

    def execute(command: list[str], _: Path) -> None:
        output_flag = "-o" if "-o" in command else "--output_path"
        if output_flag not in command:
            return
        output = Path(command[command.index(output_flag) + 1])
        if command[0] in {"DensifyPointCloud", "ReconstructMesh", "RefineMesh", "TextureMesh"}:
            output.parent.mkdir(parents=True, exist_ok=True)
            if command[0] == "DensifyPointCloud":
                output.with_suffix(".ply").write_text(PLY, encoding="ascii")
            else:
                output.write_text(PLY, encoding="ascii")

    monkeypatch.setattr(provider, "_execute", execute)
    result = provider.run(context)
    densify = next(command for command in result.commands if command[0] == "DensifyPointCloud")
    reconstruct = next(command for command in result.commands if command[0] == "ReconstructMesh")
    texture = next(command for command in result.commands if command[0] == "TextureMesh")

    assert ["-m", str(mask_dir), "--ignore-mask-label", "0"] == densify[-4:]
    assert densify[densify.index("--resolution-level") + 1] == "0"
    assert densify[densify.index("--number-views-fuse") + 1] == "3"
    assert densify[densify.index("--filter-point-cloud") + 1] == "1"
    assert reconstruct[reconstruct.index("--remove-spurious") + 1] == "50"
    assert reconstruct[reconstruct.index("--close-holes") + 1] == "0"
    assert texture[texture.index("--ignore-mask-label") + 1] == "0"
    assert result.mesh_filtering["adaptive"] is True
    assert result.quality_profile["untextured_face_policy"] == ("DECLARED_EMPTY_COLOR_FILTER")


def test_auto_provider_uses_openmvs_when_colmap_cannot_honor_masks(
    monkeypatch: object,
) -> None:
    monkeypatch.setattr(ColmapDenseProvider, "availability", lambda _: (True, "CUDA"))
    monkeypatch.setattr(ColmapDenseProvider, "_help", lambda *_: "--workspace_path")
    monkeypatch.setattr(OpenMVSProvider, "availability", lambda _: (True, "OpenMVS"))

    provider = select_dense_provider("auto", require_masks=True)

    assert isinstance(provider, OpenMVSProvider)
