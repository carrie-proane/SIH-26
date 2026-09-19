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
