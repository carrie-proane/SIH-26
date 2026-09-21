from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .exports import build_export_readiness, export_artifact_paths, write_export_readiness
from .models import (
    MeasurementCreate,
    MeasurementRecord,
    ProjectManifest,
    ProvenanceOrigin,
    RunConfig,
    RunRecord,
    RunStatus,
)
from .pipeline import PipelineRunner
from .report import measurement_evaluation
from .storage import ProjectStore
from .viewer_manifest import ViewerManifestUnavailable, build_viewer_manifest


def create_app(data_root: str | Path | None = None) -> FastAPI:
    store = ProjectStore(data_root or os.getenv("SIH_DATA_ROOT", "data/projects"))
    runner = PipelineRunner(store)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.recovered_run_ids = runner.recover_interrupted_runs()
        yield
        runner.shutdown(wait=False)

    app = FastAPI(title="SIH26158 Reconstruction API", version="0.1.0", lifespan=lifespan)
    app.state.store = store
    app.state.runner = runner
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "sih26158-api"}

    @app.post("/api/projects", response_model=ProjectManifest, status_code=status.HTTP_201_CREATED)
    def create_project(
        video: Annotated[UploadFile, File()],
        telemetry: Annotated[UploadFile, File()],
        name: Annotated[str, Form()],
        description: Annotated[str, Form()] = "",
        data_classification: Annotated[str, Form()] = "PUBLIC_DEMO",
        video_origin: Annotated[ProvenanceOrigin, Form()] = ProvenanceOrigin.UNKNOWN,
        telemetry_origin: Annotated[ProvenanceOrigin, Form()] = ProvenanceOrigin.UNKNOWN,
    ) -> ProjectManifest:
        try:
            return store.create_project(
                name=name,
                description=description,
                video_name=video.filename or "video.mp4",
                video=video.file,
                telemetry_name=telemetry.filename or "telemetry.csv",
                telemetry=telemetry.file,
                data_classification=data_classification,
                video_origin=video_origin,
                telemetry_origin=telemetry_origin,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/projects/{project_id}", response_model=ProjectManifest)
    def get_project(project_id: str) -> ProjectManifest:
        try:
            return store.get_project(project_id)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="Project not found") from exc

    @app.post("/api/projects/{project_id}/runs", response_model=RunRecord, status_code=202)
    def create_run(project_id: str, config: RunConfig) -> RunRecord:
        try:
            record = store.create_run(project_id, config)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="Project not found") from exc
        runner.submit(record.run_id)
        return record

    @app.get("/api/runs/{run_id}", response_model=RunRecord)
    def get_run(run_id: str) -> RunRecord:
        try:
            return store.get_run(run_id)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc

    @app.post("/api/runs/{run_id}/resume", response_model=RunRecord, status_code=202)
    def resume_run(run_id: str) -> RunRecord:
        try:
            current = store.get_run(run_id)
            if current.status == RunStatus.COMPLETED:
                raise HTTPException(status_code=409, detail="Completed runs cannot be resumed")
            record, submitted = runner.resume(run_id)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        if not submitted:
            raise HTTPException(status_code=409, detail="Run is already active")
        return record

    @app.post("/api/runs/{run_id}/rerun", response_model=RunRecord, status_code=202)
    def create_linked_rerun(run_id: str, config: RunConfig) -> RunRecord:
        """Create new evidence for changed configuration; never rewrite a completed run."""

        try:
            record = store.create_linked_run(run_id, config)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        runner.submit(record.run_id)
        return record

    @app.post("/api/runs/{run_id}/cancel", response_model=RunRecord, status_code=202)
    def cancel_run(run_id: str) -> RunRecord:
        try:
            record, accepted = runner.request_cancel(run_id)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        if not accepted:
            raise HTTPException(status_code=409, detail="Run is not active")
        return record

    @app.get("/api/runs/{run_id}/viewer-manifest")
    def get_viewer_manifest(run_id: str) -> dict[str, object]:
        try:
            record = store.get_run(run_id)
            return build_viewer_manifest(
                record,
                store.run_dir(record.project_id, record.run_id),
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        except ViewerManifestUnavailable as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}/artifacts/{artifact_path:path}")
    def get_artifact(run_id: str, artifact_path: str) -> FileResponse:
        try:
            path = store.resolve_declared_artifact(run_id, artifact_path)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="Declared artifact not found") from exc
        return FileResponse(path)

    @app.get("/api/runs/{run_id}/artifact-index")
    def artifact_index(run_id: str) -> dict[str, object]:
        try:
            record = store.get_run(run_id)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        return {
            "run_id": run_id,
            "artifacts": [
                item.model_dump(mode="json")
                for item in record.artifacts
            ],
        }

    @app.get("/api/runs/{run_id}/readiness")
    def run_readiness(run_id: str) -> dict[str, object]:
        try:
            record = store.get_run(run_id)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        artifacts = {item.relative_path for item in record.artifacts}
        sparse_required = {
            "sparse/sparse_local.ply",
            "camera_poses.csv",
            "keyframes.json",
        }
        dense_visuals = {
            item
            for item in artifacts
            if item.startswith("dense/") and item.lower().endswith((".ply", ".obj", ".glb"))
        }
        export_summary: dict[str, str] = {}
        if "export_readiness.json" in artifacts:
            try:
                export_path = store.resolve_declared_artifact(run_id, "export_readiness.json")
                export_payload = json.loads(export_path.read_text(encoding="utf-8"))
                export_summary = {
                    name: str(value.get("status", "UNKNOWN"))
                    for name, value in export_payload.get("formats", {}).items()
                    if isinstance(value, dict)
                }
            except (OSError, ValueError, json.JSONDecodeError):
                export_summary = {"REPORT": "INVALID"}
        return {
            "schema_version": "1.0",
            "run_id": run_id,
            "stage": record.stage,
            "status": record.status,
            "sparse_preview_ready": sparse_required <= artifacts,
            "sparse_preview_missing": sorted(sparse_required - artifacts),
            "quality_report_ready": "quality_report.json" in artifacts,
            "dense_status": (
                "AVAILABLE"
                if dense_visuals
                else "FAILED_OR_UNAVAILABLE"
                if "dense_report.json" in artifacts
                else "NOT_READY_OR_NOT_REQUESTED"
            ),
            "dense_visual_artifacts": sorted(dense_visuals),
            "measurement_geometry": (
                "sparse/sparse_local.ply"
                if "sparse/sparse_local.ply" in artifacts
                else None
            ),
            "inferred_geometry_measurement_eligible": False,
            "export_report_ready": "export_readiness.json" in artifacts,
            "export_report_url": (
                f"/api/runs/{run_id}/exports"
                if "export_readiness.json" in artifacts
                else None
            ),
            "export_status": export_summary,
        }

    @app.get("/api/runs/{run_id}/exports")
    def export_readiness(run_id: str) -> dict[str, object]:
        try:
            store.get_run(run_id)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        try:
            path = store.resolve_declared_artifact(run_id, "export_readiness.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail="Export report is not ready") from exc
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=409, detail="Export report is invalid") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=409, detail="Export report is invalid")
        return payload

    @app.post("/api/runs/{run_id}/exports", status_code=status.HTTP_201_CREATED)
    def create_exports(run_id: str) -> dict[str, object]:
        """Convert already-declared terminal-run geometry without reconstruction."""

        try:
            record = store.get_run(run_id)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        terminal = {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
        if record.status not in terminal:
            raise HTTPException(
                status_code=409,
                detail="Geometry export requires a completed, failed, or cancelled run",
            )
        with store.execution_lock(run_id) as execution_lock:
            if not execution_lock.acquired:
                raise HTTPException(status_code=409, detail="Run is active")
            record = store.get_run(run_id)
            if record.status not in terminal:
                raise HTTPException(
                    status_code=409,
                    detail="Geometry export requires a completed, failed, or cancelled run",
                )
            run_dir = store.run_dir(record.project_id, record.run_id)
            try:
                payload = build_export_readiness(record, run_dir)
                report_path = run_dir / "export_readiness.json"
                write_export_readiness(report_path, payload)
                generated = export_artifact_paths(payload, run_dir)
                store.register_artifacts(record, [report_path, *generated])
            except (OSError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        return payload

    @app.post(
        "/api/runs/{run_id}/measurements",
        response_model=MeasurementRecord,
        status_code=status.HTTP_201_CREATED,
    )
    def create_measurement(run_id: str, measurement: MeasurementCreate) -> MeasurementRecord:
        try:
            return store.add_measurement(run_id, measurement)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Run or geometry artifact not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}/measurements")
    def list_measurements(run_id: str) -> dict[str, object]:
        try:
            measurements = store.list_measurements(run_id)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        return {
            "schema_version": "1.0",
            "run_id": run_id,
            "measurements": [item.model_dump(mode="json") for item in measurements],
            "evaluation": measurement_evaluation(measurements),
        }

    return app


app = create_app()
