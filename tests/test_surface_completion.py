import json
from pathlib import Path

from sih26158.reconstruction.surface_completion import (
    SurfaceCompletionContext,
    SurfaceCompletionProvider,
    UnavailableSurfaceCompletionProvider,
    run_surface_completion_stage,
)


def _context(tmp_path: Path) -> SurfaceCompletionContext:
    geometry = tmp_path / "sparse" / "sparse_local.ply"
    geometry.parent.mkdir()
    geometry.write_text("ply\nformat ascii 1.0\nend_header\n", encoding="utf-8")
    poses = tmp_path / "camera_poses.csv"
    poses.write_text("image_name,x_m,y_m,z_m\nframe.jpg,0,0,0\n", encoding="utf-8")
    keyframes = tmp_path / "keyframes.json"
    keyframes.write_text('{"frames": []}\n', encoding="utf-8")
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"fixture")
    return SurfaceCompletionContext(
        run_dir=tmp_path,
        source_geometry=geometry,
        source_geometry_kind="SPARSE_OBSERVED_SFM",
        camera_poses=poses,
        selected_frames=keyframes,
        model_path=weights,
        sample_count=3,
    )


def test_unavailable_completion_is_reported_without_fake_geometry(tmp_path: Path):
    result = run_surface_completion_stage(
        _context(tmp_path),
        UnavailableSurfaceCompletionProvider("model runtime unavailable"),
    )

    assert result.status == "BLOCKED"
    assert not (tmp_path / "completion" / "completed_mesh.ply").exists()
    report = json.loads((tmp_path / "completion_report.json").read_text(encoding="utf-8"))
    assert report["measurement_eligible"] is False
    assert report["generated_geometry_label"] == "AI_ASSISTED_NOT_MEASURABLE"


class _FixtureProvider(SurfaceCompletionProvider):
    name = "fixture"

    def availability(self, context: SurfaceCompletionContext) -> tuple[bool, str]:
        return context.model_path is not None, "fixture available"

    def run(self, context: SurfaceCompletionContext, request_path: Path) -> list[Path]:
        assert request_path.is_file()
        output = context.run_dir / "completion"
        output.mkdir()
        mesh = output / "completed_mesh.ply"
        mesh.write_text("ply\nformat ascii 1.0\nend_header\n", encoding="utf-8")
        uncertainty = output / "uncertainty.json"
        uncertainty.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "semantics": "HIGHER_IS_MORE_UNCERTAIN",
                    "summary": {"mean": 0.7},
                }
            ),
            encoding="utf-8",
        )
        return [mesh, uncertainty]


def test_completion_contract_publishes_visual_only_artifacts(tmp_path: Path):
    result = run_surface_completion_stage(_context(tmp_path), _FixtureProvider())

    assert result.status == "COMPLETED"
    report = json.loads((tmp_path / "completion_report.json").read_text(encoding="utf-8"))
    request = json.loads((tmp_path / "completion_request.json").read_text(encoding="utf-8"))
    assert report["measurement_eligible"] is False
    assert request["inputs"]["source_geometry"] == "sparse/sparse_local.ply"
    assert request["scientific_contract"]["multiple_hypotheses_recommended"] is True
