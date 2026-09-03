"""Compute normalized sharpness, exposure, and adjacent-frame uniqueness scores."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np


def normalize_scores(values: Sequence[float], constant_value: float = 1.0) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if np.isclose(low, high):
        return [constant_value] * len(values)
    return [(value - low) / (high - low) for value in values]


def laplacian_variances(images: Sequence[np.ndarray]) -> list[float]:
    """Return absolute Laplacian variance for reproducible sharpness gates."""

    return [
        float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        for image in images
    ]


def blur_scores(images: Sequence[np.ndarray]) -> list[float]:
    """Return min-max normalized Laplacian variance (higher means sharper)."""

    return normalize_scores(laplacian_variances(images))


def exposure_score(image: np.ndarray) -> float:
    """Score tonal centering/spread while penalizing clipped highlights and shadows."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    histogram /= max(1.0, float(histogram.sum()))
    levels = np.arange(256, dtype=np.float64)
    mean = float(np.dot(histogram, levels))
    variance = float(np.dot(histogram, (levels - mean) ** 2))
    mean_score = max(0.0, 1.0 - abs(mean - 127.5) / 127.5)
    spread_score = min(1.0, variance**0.5 / 64.0)
    shadow_fraction = float(histogram[:5].sum())
    highlight_fraction = float(histogram[251:].sum())
    clipping_penalty = min(
        1.0, max(0.0, shadow_fraction - 0.05) * 4 + max(0.0, highlight_fraction - 0.05) * 4
    )
    return float(
        np.clip((0.65 * mean_score + 0.35 * spread_score) * (1.0 - clipping_penalty), 0.0, 1.0)
    )


def exposure_scores(images: Sequence[np.ndarray]) -> list[float]:
    return [exposure_score(image) for image in images]


def _ssim(first: np.ndarray, second: np.ndarray) -> float:
    """Compute mean structural similarity using the standard local-window formula."""

    if first.shape != second.shape:
        second = cv2.resize(second, (first.shape[1], first.shape[0]), interpolation=cv2.INTER_AREA)
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY).astype(np.float64)
    second_gray = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_first = cv2.GaussianBlur(first_gray, (11, 11), 1.5)
    mu_second = cv2.GaussianBlur(second_gray, (11, 11), 1.5)
    sigma_first = cv2.GaussianBlur(first_gray * first_gray, (11, 11), 1.5) - mu_first**2
    sigma_second = cv2.GaussianBlur(second_gray * second_gray, (11, 11), 1.5) - mu_second**2
    covariance = cv2.GaussianBlur(first_gray * second_gray, (11, 11), 1.5) - mu_first * mu_second
    numerator = (2 * mu_first * mu_second + c1) * (2 * covariance + c2)
    denominator = (mu_first**2 + mu_second**2 + c1) * (sigma_first + sigma_second + c2)
    return float(np.clip(np.mean(numerator / denominator), -1.0, 1.0))


def redundancy_scores(images: Sequence[np.ndarray]) -> list[float]:
    """Return adjacent-frame SSIM dissimilarity (higher means more unique)."""

    if not images:
        return []
    if len(images) == 1:
        return [1.0]
    adjacent_similarity = [_ssim(first, second) for first, second in pairwise(images)]
    raw: list[float] = []
    for index in range(len(images)):
        neighbours = []
        if index:
            neighbours.append(adjacent_similarity[index - 1])
        if index < len(adjacent_similarity):
            neighbours.append(adjacent_similarity[index])
        raw.append(1.0 - sum(neighbours) / len(neighbours))
    return normalize_scores(raw, constant_value=0.0)


def load_images(paths: Sequence[str | Path]) -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not read frame: {path}")
        images.append(image)
    return images
