import numpy as np

from sih26158.geo import (
    SimilarityTransform,
    estimate_similarity,
    geodetic_to_enu,
    robust_similarity,
    transform_ply,
)


def test_enu_origin_is_zero() -> None:
    origin = (28.6139, 77.2090, 42.0)
    assert np.linalg.norm(geodetic_to_enu(*origin, origin)) < 1e-8


def test_similarity_recovers_known_transform() -> None:
    source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    target = 2.5 * (source @ rotation.T) + np.array([10, -4, 3])
    fit = estimate_similarity(source, target)
    assert np.allclose(fit.apply(source), target, atol=1e-9)
    assert abs(fit.scale - 2.5) < 1e-9


def test_robust_similarity_flags_gps_outlier() -> None:
    rng = np.random.default_rng(7)
    source = rng.normal(size=(12, 3))
    target = 1.8 * source + np.array([4, 5, 6])
    target[-1] += 100
    fit = robust_similarity(source, target)
    assert fit.inliers.sum() == 11
    assert not fit.inliers[-1]
    assert np.median(fit.residuals_m[:-1]) < 1e-8


def test_ascii_ply_is_transformed_and_other_properties_survive(tmp_path) -> None:
    source = tmp_path / "source.ply"
    output = tmp_path / "output.ply"
    source.write_text(
        "ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\n"
        "property float z\nproperty uchar red\nend_header\n0 0 0 7\n1 0 0 8\n",
        encoding="ascii",
    )
    transform = SimilarityTransform(
        scale=2,
        rotation=np.eye(3),
        translation=np.array([1, 2, 3]),
        inliers=np.ones(2, dtype=bool),
        residuals_m=np.zeros(2),
    )
    transform_ply(source, output, transform)
    body = output.read_text(encoding="ascii").split("end_header\n", 1)[1].splitlines()
    assert body == ["1 2 3 7", "3 2 3 8"]
