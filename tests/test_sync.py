import numpy as np

from sih26158.reconstruction.sync import calibrate_telemetry_offset


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
