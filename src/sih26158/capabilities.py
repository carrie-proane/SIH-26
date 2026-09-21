from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .models import RunConfig, utc_now
from .preflight import collect_preflight

Runner = Callable[..., subprocess.CompletedProcess[str]]
Which = Callable[[str], str | None]


def _status(report: dict[str, Any], check_id: str) -> str:
    for check in report.get("checks", []):
        if check.get("id") == check_id:
            return str(check.get("status", "BLOCKED"))
    return "BLOCKED"


def build_capability_profile(
    preflight: dict[str, Any], config: RunConfig
) -> dict[str, Any]:
    sparse_available = _status(preflight, "colmap.sparse_capabilities") == "PASS"
    cuda_device_available = _status(preflight, "gpu.nvidia_cuda") == "PASS"
    colmap_cuda_available = _status(preflight, "colmap.cuda_build") == "PASS"
    cuda_available = cuda_device_available and colmap_cuda_available
    colmap_dense = (
        _status(preflight, "dense.colmap_cuda") == "PASS" and cuda_available
    )
    openmvs_dense = _status(preflight, "dense.openmvs") == "PASS"
    providers = {"colmap": colmap_dense, "openmvs": openmvs_dense}
    if config.dense_provider == "auto":
        selected_dense = "colmap" if colmap_dense else "openmvs" if openmvs_dense else None
    else:
        selected_dense = (
            config.dense_provider if providers.get(config.dense_provider, False) else None
        )
    effective_sparse_gpu = bool(config.use_gpu and cuda_available)
    warnings: list[dict[str, str]] = []
    if config.use_gpu and not effective_sparse_gpu:
        warnings.append(
            {
                "code": "SPARSE_GPU_UNAVAILABLE_CPU_SELECTED",
                "message": (
                    "GPU sparse extraction was requested, but CUDA was not detected; "
                    "COLMAP sparse execution will use its CPU path."
                ),
            }
        )
    if config.enable_dense_reconstruction and selected_dense is None:
        warnings.append(
            {
                "code": "DENSE_PROVIDER_UNAVAILABLE",
                "message": (
                    "No requested dense provider is executable on this host; the sparse run "
                    "remains eligible to complete."
                ),
            }
        )
    return {
        "schema_version": "1.0",
        "captured_at": utc_now(),
        "environment": preflight.get("environment", {}),
        "sparse": {
            "colmap_available": sparse_available,
            "gpu_requested": config.use_gpu,
            "cuda_available": cuda_available,
            "cuda_device_available": cuda_device_available,
            "colmap_cuda_build": colmap_cuda_available,
            "effective_gpu": effective_sparse_gpu,
            "selection": "COLMAP_GPU" if effective_sparse_gpu else "COLMAP_CPU",
        },
        "dense": {
            "requested": config.dense_provider,
            "enabled": config.enable_dense_reconstruction,
            "providers": providers,
            "selected_provider": selected_dense,
        },
        "warnings": warnings,
        "preflight": preflight,
    }


def collect_server_capabilities(
    *,
    data_root: str | Path,
    config: RunConfig,
    runner: Runner = subprocess.run,
    which: Which | None = None,
) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "data_root": data_root,
        "dense_provider": config.dense_provider,
        "require_dense": False,
        "require_gpu": False,
        "minimum_free_disk_gb": 0,
        "runner": runner,
    }
    if which is not None:
        arguments["which"] = which
    return build_capability_profile(collect_preflight(**arguments), config)


def synthetic_capability_profile(config: RunConfig) -> dict[str, Any]:
    """Declare that the orchestration fixture did not exercise host reconstruction tools."""

    return {
        "schema_version": "1.0",
        "captured_at": utc_now(),
        "environment": {},
        "sparse": {
            "colmap_available": None,
            "gpu_requested": config.use_gpu,
            "cuda_available": None,
            "cuda_device_available": None,
            "colmap_cuda_build": None,
            "effective_gpu": False,
            "selection": "NOT_APPLICABLE_SYNTHETIC_FIXTURE",
        },
        "dense": {
            "requested": config.dense_provider,
            "enabled": config.enable_dense_reconstruction,
            "providers": {"colmap": None, "openmvs": None},
            "selected_provider": None,
        },
        "warnings": [
            {
                "code": "CAPABILITY_EXECUTION_NOT_APPLICABLE_SYNTHETIC",
                "message": (
                    "The synthetic orchestration fixture does not execute or validate COLMAP, "
                    "CUDA, or OpenMVS capabilities."
                ),
            }
        ],
        "preflight": None,
    }
