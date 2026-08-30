import struct

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


def test_binary_ply_with_openmvs_vertex_lists_is_transformed(tmp_path) -> None:
    """OpenMVS adds per-vertex view lists after the normal and colour fields."""
    source = tmp_path / "openmvs-dense.ply"
    output = tmp_path / "aligned.ply"
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "element vertex 2\n"
        "property float32 x\n"
        "property float32 y\n"
        "property float32 z\n"
        "property uint8 red\n"
        "property list uint8 uint32 view_indices\n"
        "property list uint8 float32 view_weights\n"
        "element face 1\n"
        "property list uint8 uint32 vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    vertices = b"".join(
        [
            struct.pack("<fffB", 0.0, 0.0, 0.0, 12)
            + struct.pack("<BII", 2, 4, 9)
            + struct.pack("<Bff", 2, 0.25, 0.75),
            struct.pack("<fffB", 1.0, 0.0, 0.0, 34)
            + struct.pack("<BI", 1, 7)
            + struct.pack("<Bf", 1, 1.0),
        ]
    )
    face = struct.pack("<BII", 2, 0, 1)
    source.write_bytes(header + vertices + face)
    transform = SimilarityTransform(
        scale=2,
        rotation=np.eye(3),
        translation=np.array([1, 2, 3]),
        inliers=np.ones(2, dtype=bool),
        residuals_m=np.zeros(2),
    )

    transform_ply(source, output, transform)

    raw = output.read_bytes()
    body = raw.split(b"end_header\n", 1)[1]
    assert struct.unpack_from("<fff", body, 0) == (1.0, 2.0, 3.0)
    second_vertex = struct.calcsize("<fffB") + struct.calcsize("<BII") + struct.calcsize("<Bff")
    assert struct.unpack_from("<fff", body, second_vertex) == (3.0, 2.0, 3.0)
    assert body[second_vertex + struct.calcsize("<fff")] == 34
    assert body.endswith(face)
