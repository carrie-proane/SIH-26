from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .models import ProjectManifest, RunConfig, RunRecord
from .pipeline import PipelineRunner
from .storage import ProjectStore


def create_app(data_root: str | Path | None = None) -> FastAPI:
    store = ProjectStore(data_root or os.getenv("SIH_DATA_ROOT", "data/projects"))
    runner = PipelineRunner(store)
    app = FastAPI(title="SIH26158 Reconstruction API", version="0.1.0")
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

    return app


app = create_app()
