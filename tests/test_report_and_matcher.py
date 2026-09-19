from sih26158.colmap import choose_matcher
from sih26158.models import MatcherMetrics
from sih26158.report import known_distance_metrics


def metric(name: str, registered: int, reprojection: float) -> MatcherMetrics:
    return MatcherMetrics(
        matcher=name,
        eligible_frames=100,
        registered_frames=registered,
        median_reprojection_error_px=reprojection,
        p95_reprojection_error_px=reprojection * 2,
        runtime_s=10,
    )


def test_known_distance_gate_is_visible() -> None:
    passed = known_distance_metrics(10, 10.9)
    failed = known_distance_metrics(10, 11.1)
    assert passed["passes_10_percent_gate"] is True
    assert failed["passes_10_percent_gate"] is False


def test_learned_matcher_must_improve_evidence() -> None:
    assert choose_matcher(metric("SIFT", 80, 1.2), metric("SUPERPOINT_LIGHTGLUE", 84, 1.3))[0] == "SUPERPOINT_LIGHTGLUE"
    assert choose_matcher(metric("SIFT", 80, 1.2), metric("SUPERPOINT_LIGHTGLUE", 79, 0.8))[0] == "SIFT"
    assert choose_matcher(metric("SIFT", 80, 1.2), metric("SUPERPOINT_LIGHTGLUE", 80, 1.0))[0] == "SUPERPOINT_LIGHTGLUE"



def test_report_uses_execution_matcher_not_requested_matcher() -> None:
    from sih26158.models import RunConfig, RunRecord
    from sih26158.report import build_quality_report

    record = RunRecord(project_id="p", run_id="r", config=RunConfig(matcher="SUPERPOINT_LIGHTGLUE"))
    metrics = metric("SIFT", 90, 1.0)
    metrics.matcher_actually_used = "sift"
    report = build_quality_report(record, metrics, [])
    assert report["matcher_actually_used"] == "sift"
    metrics.matcher_actually_used = None
    assert build_quality_report(record, metrics, [])["matcher_actually_used"] is None


def test_missing_reference_is_not_validated_and_failed_gate_takes_precedence() -> None:
    from sih26158.models import RunConfig, RunRecord
    from sih26158.report import build_quality_report

    record = RunRecord(project_id="p", run_id="r", config=RunConfig(), source_provenance="REAL")
    alignment = {"scale": 1.0, "inlier_count": 10}
    report = build_quality_report(record, metric("SIFT", 90, 1), [], alignment=alignment)
    assert report["evidence_verdict"] == "NOT_VALIDATED"
    report = build_quality_report(record, metric("SIFT", 40, 3), [], alignment=alignment)
    assert report["evidence_verdict"] == "FAILED"
    assert report["status"] == "COMPLETED"


def test_degenerate_alignment_cannot_pass_even_with_low_residual() -> None:
    from sih26158.models import RunConfig, RunRecord
    from sih26158.report import build_quality_report
    record = RunRecord(project_id="p", run_id="r", source_provenance="REAL",
                       config=RunConfig(known_distance_m=10, measured_distance_m=10))
    alignment = {"scale": 1, "inlier_count": 10, "residuals_m": [0] * 10,
                 "vertical_alignment_verdict": "PASSED", "alignment_identifiability": "degenerate"}
    report = build_quality_report(record, metric("SIFT", 100, 0.1), [], alignment=alignment)
    assert report["evidence_verdict"] == "NOT_VALIDATED"
    assert report["alignment_identifiability"] == "degenerate"
    assert report["metrics"]["metric_alignment"]["residual_label"] == "camera-to-telemetry consistency metric"


def test_known_distance_uses_this_runs_declared_vertices_and_records_provenance(tmp_path) -> None:
    from sih26158.models import ArtifactEntry, RunConfig, RunRecord
    from sih26158.report import build_quality_report
    from sih26158.storage import sha256_file
    cloud = tmp_path / "sparse" / "sparse_local.ply"
    cloud.parent.mkdir()
    cloud.write_text("ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n0 3 4\n")
    config = RunConfig(known_distance_m=5, measured_distance_m=5,
        known_distance_reference_source="tape_measurement",
        known_distance_endpoint_a={"point_id": 0, "description": "bottom step nosing"},
        known_distance_endpoint_b={"point_id": 1, "description": "top step nosing"})
    record = RunRecord(project_id="p", run_id="r", config=config, source_provenance="REAL",
        artifacts=[ArtifactEntry(name="cloud", relative_path="sparse/sparse_local.ply",
            media_type="application/octet-stream", size_bytes=cloud.stat().st_size, sha256=sha256_file(cloud))])
    alignment = {"scale": 1, "inlier_count": 4, "vertical_alignment_verdict": "PASSED",
                 "alignment_identifiability": "well_conditioned"}
    report = build_quality_report(record, metric("SIFT", 100, 0.5), [], alignment=alignment, run_dir=tmp_path)
    check = report["metrics"]["known_distance"]
    assert report["evidence_verdict"] == "PASSED"
    assert check["reconstructed_distance_m"] == 5
    assert check["reference_source"] == "tape_measurement"
    assert check["endpoint_a"]["point_id"] == 0
    assert check["endpoint_b"]["description"] == "top step nosing"
    assert check["computed_at"] and check["cloud_sha256"] and check["run_id"] == "r"
    record.config.known_distance_m = 10
    record.config.measured_distance_m = 10  # Fabricated agreement in config cannot pass.
    report = build_quality_report(record, metric("SIFT", 100, 0.5), [], alignment=alignment, run_dir=tmp_path)
    assert report["metrics"]["known_distance"]["reconstructed_distance_m"] == 5
    assert report["evidence_verdict"] == "FAILED"
    record.artifacts = []
    report = build_quality_report(record, metric("SIFT", 100, 0.5), [], alignment=alignment, run_dir=tmp_path)
    assert report["metrics"]["known_distance"]["reconstructed_distance_m"] is None
    assert report["metrics"]["known_distance"]["note"]
    assert report["evidence_verdict"] == "NOT_VALIDATED"


def test_known_distance_script_preserves_circularity_guard() -> None:
    import subprocess
    import sys
    from pathlib import Path
    result = subprocess.run([sys.executable, str(Path(__file__).parents[1] / "scripts/known_distance_check.py"),
        "--ground-truth", "unused.json", "--scale-id", "M1", "--validate-id", "M1",
        "--scale-points", "0,0,0", "1,0,0", "--validate-points", "0,0,0", "1,0,0"],
        capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert "scale and validation must be different" in result.stdout
