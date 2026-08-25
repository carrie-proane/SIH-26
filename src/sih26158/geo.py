from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)


def geodetic_to_ecef(lat_deg: float, lon_deg: float, altitude_m: float) -> NDArray[np.float64]:
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    n = WGS84_A / np.sqrt(1 - WGS84_E2 * sin_lat * sin_lat)
    return np.array(
        [
            (n + altitude_m) * cos_lat * np.cos(lon),
            (n + altitude_m) * cos_lat * np.sin(lon),
            (n * (1 - WGS84_E2) + altitude_m) * sin_lat,
        ],
        dtype=float,
    )


def geodetic_to_enu(
    lat_deg: float,
    lon_deg: float,
    altitude_m: float,
    origin: tuple[float, float, float],
) -> NDArray[np.float64]:
    lat0, lon0, alt0 = origin
    delta = geodetic_to_ecef(lat_deg, lon_deg, altitude_m) - geodetic_to_ecef(lat0, lon0, alt0)
    lat = np.radians(lat0)
    lon = np.radians(lon0)
    rotation = np.array(
        [
            [-np.sin(lon), np.cos(lon), 0],
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        ]
    )
    return rotation @ delta


@dataclass(frozen=True)
class SimilarityTransform:
    scale: float
    rotation: NDArray[np.float64]
    translation: NDArray[np.float64]
    inliers: NDArray[np.bool_]
    residuals_m: NDArray[np.float64]

    def apply(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.scale * (points @ self.rotation.T) + self.translation

    def as_dict(self) -> dict[str, object]:
        return {
            "scale": self.scale,
            "rotation": self.rotation.tolist(),
            "translation_m": self.translation.tolist(),
            "inlier_count": int(self.inliers.sum()),
            "residuals_m": self.residuals_m.tolist(),
        }


def estimate_similarity(source: NDArray[np.float64], target: NDArray[np.float64]) -> SimilarityTransform:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both have shape (N, 3)")
    if len(source) < 3:
        raise ValueError("At least three point pairs are required")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    variance = np.mean(np.sum(source_centered**2, axis=1))
    if variance <= np.finfo(float).eps:
        raise ValueError("Source points are degenerate")
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    predicted = scale * (source @ rotation.T) + translation
    residuals = np.linalg.norm(predicted - target, axis=1)
    return SimilarityTransform(scale, rotation, translation, np.ones(len(source), dtype=bool), residuals)


def robust_similarity(
    source: NDArray[np.float64],
    target: NDArray[np.float64],
    *,
    max_iterations: int = 8,
    sigma: float = 3.0,
) -> SimilarityTransform:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    inliers = np.ones(len(source), dtype=bool)
    for _ in range(max_iterations):
        if inliers.sum() < 3:
            raise ValueError("Fewer than three alignment inliers remain")
        fit = estimate_similarity(source[inliers], target[inliers])
        all_residuals = np.linalg.norm(fit.apply(source) - target, axis=1)
        median = float(np.median(all_residuals[inliers]))
        mad = float(np.median(np.abs(all_residuals[inliers] - median)))
        threshold = max(0.25, median + sigma * 1.4826 * mad)
        updated = all_residuals <= threshold
        if np.array_equal(updated, inliers):
            return SimilarityTransform(fit.scale, fit.rotation, fit.translation, inliers, all_residuals)
        inliers = updated
    fit = estimate_similarity(source[inliers], target[inliers])
    residuals = np.linalg.norm(fit.apply(source) - target, axis=1)
    return SimilarityTransform(fit.scale, fit.rotation, fit.translation, inliers, residuals)


def transform_ply(source: Path, output: Path, transform: SimilarityTransform) -> None:
    """Transform vertex XYZ in an ASCII or binary-little-endian PLY and preserve other data."""
    raw = source.read_bytes()
    marker = b"end_header\n"
    end = raw.find(marker)
    if end < 0:
        marker = b"end_header\r\n"
        end = raw.find(marker)
    if end < 0:
        raise ValueError("PLY is missing end_header")
    body_start = end + len(marker)
    header = raw[:body_start]
    lines = header.decode("ascii").splitlines()
    file_format = next((line.split()[1] for line in lines if line.startswith("format ")), None)
    vertex_count = 0
    vertex_properties: list[tuple[str, str]] = []
    inside_vertices = False
    for line in lines:
        fields = line.split()
        if fields[:2] == ["element", "vertex"]:
            vertex_count = int(fields[2])
            inside_vertices = True
        elif fields and fields[0] == "element" and fields[1] != "vertex":
            inside_vertices = False
        elif inside_vertices and fields[:1] == ["property"]:
            if len(fields) != 3:
                raise ValueError("List properties are not supported inside PLY vertices")
            vertex_properties.append((fields[1], fields[2]))
    names = [name for _, name in vertex_properties]
    if vertex_count <= 0 or not {"x", "y", "z"}.issubset(names):
        raise ValueError("PLY must contain vertex x, y, and z properties")
    xyz_indices = [names.index(axis) for axis in ("x", "y", "z")]

    if file_format == "ascii":
        text_lines = raw[body_start:].decode("ascii").splitlines()
        if len(text_lines) < vertex_count:
            raise ValueError("PLY contains fewer vertices than declared")
        vertices = [line.split() for line in text_lines[:vertex_count]]
        points = np.array(
            [[float(values[index]) for index in xyz_indices] for values in vertices], dtype=float
        )
        aligned = transform.apply(points)
        for row, values in enumerate(vertices):
            for column, value in zip(xyz_indices, aligned[row], strict=True):
                values[column] = f"{value:.9g}"
        output.write_bytes(
            header
            + ("\n".join(" ".join(values) for values in vertices + [line.split() for line in text_lines[vertex_count:]]) + "\n").encode(
                "ascii"
            )
        )
        return

    if file_format != "binary_little_endian":
        raise ValueError(f"Unsupported PLY format: {file_format}")
    formats = {
        "char": "b",
        "int8": "b",
        "uchar": "B",
        "uint8": "B",
        "short": "h",
        "int16": "h",
        "ushort": "H",
        "uint16": "H",
        "int": "i",
        "int32": "i",
        "uint": "I",
        "uint32": "I",
        "float": "f",
        "float32": "f",
        "double": "d",
        "float64": "d",
    }
    try:
        vertex_struct = struct.Struct("<" + "".join(formats[kind] for kind, _ in vertex_properties))
    except KeyError as exc:
        raise ValueError(f"Unsupported PLY vertex type: {exc.args[0]}") from exc
    vertex_bytes = vertex_count * vertex_struct.size
    if len(raw) < body_start + vertex_bytes:
        raise ValueError("PLY binary vertex data is truncated")
    records = [
        list(vertex_struct.unpack_from(raw, body_start + index * vertex_struct.size))
        for index in range(vertex_count)
    ]
    points = np.array([[record[index] for index in xyz_indices] for record in records], dtype=float)
    aligned = transform.apply(points)
    encoded = bytearray()
    for row, record in enumerate(records):
        for column, value in zip(xyz_indices, aligned[row], strict=True):
            record[column] = float(value)
        encoded.extend(vertex_struct.pack(*record))
    output.write_bytes(header + encoded + raw[body_start + vertex_bytes :])
