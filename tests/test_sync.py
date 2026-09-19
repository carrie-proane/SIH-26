import numpy as np

from sih26158.sync import calibrate_telemetry_offset


def _trajectory(times: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [np.sin(0.7 * times), np.cos(0.3 * times), 0.05 * times**2]
    )


def test_bounded_automatic_offset_search_finds_minimum_robust_residual() -> None:
    telemetry_times = np.arange(0.0, 10.01, 0.05)
    telemetry_points = _trajectory(telemetry_times)
    frame_times = np.arange(0.5, 9.01, 0.5)
    sfm_points = _trajectory(frame_times + 0.30)

    result = calibrate_telemetry_offset(
        sfm_points,
        frame_times,
        telemetry_times,
        telemetry_points,
        search_min_s=-0.5,
        search_max_s=0.5,
        search_step_s=0.05,
    )

    assert result.source == "automatic"
    assert result.selected.offset_s == 0.30
    assert result.selected.rmse_m < result.before.rmse_m


def test_offset_without_zero_overlap_can_be_selected_manually_or_automatically() -> None:
    frame_times = np.arange(0, 1.01, 0.1)
    telemetry_times = frame_times + 4
    points = _trajectory(frame_times)
    for options in ({"manual_offset_s": 4}, {"search_min_s": 3.5, "search_max_s": 4.5}):
        result = calibrate_telemetry_offset(points, frame_times, telemetry_times, points, **options)
        assert result.selected.offset_s == 4
        assert result.selected.rmse_m < 1e-10
        assert result.before is None
        assert result.as_report()["rmse_before_m"] is None


def test_zero_offset_baseline_remains_available() -> None:
    times = np.arange(0, 2.01, 0.1)
    result = calibrate_telemetry_offset(_trajectory(times), times, times, _trajectory(times), manual_offset_s=0)
    assert result.before is not None
    assert result.before.rmse_m == result.selected.rmse_m
    assert result.as_report()["rmse_before_m"] < 1e-10


def test_identifiability_flags_line_but_accepts_planar_orbit() -> None:
    times = np.linspace(0, 6, 30)
    line = np.column_stack([times, 0.0001 * np.sin(times), np.zeros(len(times))])
    orbit = np.column_stack([np.sin(times), np.cos(times), np.zeros(len(times))])
    for points, expected in [(line, "degenerate"), (orbit, "well_conditioned")]:
        result = calibrate_telemetry_offset(points, times, times, points, manual_offset_s=0)
        assert result.selected.rmse_m < 1e-10
        assert result.as_report()["alignment_identifiability"] == expected
        assert result.selected.transform.scale > 0
