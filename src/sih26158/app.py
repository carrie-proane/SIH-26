from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.formparsers import MultiPartException

from .models import ProjectManifest, ProvenanceOrigin, RunConfig, RunRecord
from .pipeline import PipelineRunner
from .storage import ProjectStore
from .viewer_manifest import ViewerManifestUnavailable, build_viewer_manifest


class UploadBudgetMiddleware:
    """Check headers and every body chunk before multipart parsing can spool it."""

    def __init__(self, app, store: ProjectStore, max_bytes: int, reserve_bytes: int):
        self.app, self.store = app, store
        self.max_bytes, self.reserve_bytes = max_bytes, reserve_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != "/api/projects":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        try:
            declared = int(headers.get(b"content-length", b"0"))
            if declared < 0:
                raise ValueError
        except ValueError:
            return await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(scope, receive, send)
        if declared > self.max_bytes:
            return await JSONResponse({"detail": "Upload exceeds configured maximum size"}, status_code=413)(scope, receive, send)
        try:
            self.store.check_upload_space(declared, self.reserve_bytes, include_spool=True)
        except ValueError as exc:
            return await JSONResponse({"detail": str(exc)}, status_code=507)(scope, receive, send)
        received = 0
        rejection_status = None

        async def limited_receive():
            nonlocal received, rejection_status
            message = await receive()
            if message["type"] == "http.request":
                chunk_size = len(message.get("body", b""))
                received += chunk_size
                if received > self.max_bytes:
                    rejection_status = 413
                    # Multipart's exception path closes any partially spooled files.
                    raise MultiPartException("Upload exceeds configured maximum size")
                try:
                    self.store.check_upload_space(chunk_size, self.reserve_bytes, include_spool=True)
                except ValueError as exc:
                    rejection_status = 507
                    raise MultiPartException(str(exc)) from exc
            return message

        async def budget_send(message):
            # Starlette translates MultipartException to 400. Preserve its detail and
            # cleanup while returning the precise resource-limit status to the client.
            if rejection_status is not None and message["type"] == "http.response.start":
                message = dict(message, status=rejection_status)
            await send(message)

        return await self.app(scope, limited_receive, budget_send)


def _configured_bind_host() -> str:
    for index, arg in enumerate(sys.argv):
        if arg.startswith("--host="):
            return arg.split("=", 1)[1]
        if arg == "--host" and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
    return os.getenv("UVICORN_HOST", "127.0.0.1")


def create_app(data_root: str | Path | None = None) -> FastAPI:
    store = ProjectStore(data_root or os.getenv("SIH_DATA_ROOT", "data/projects"))
    runner = PipelineRunner(store)
    max_upload_bytes = int(os.getenv("SIH_MAX_UPLOAD_BYTES", str(4 * 1024**3)))
    reserve_bytes = int(os.getenv("SIH_MIN_FREE_DISK_BYTES", str(1024**3)))
    if max_upload_bytes <= 0 or reserve_bytes < 0:
        raise ValueError("Upload size must be positive and free-disk reserve non-negative")
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runner.startup()
        host = _configured_bind_host()
        if host not in {"127.0.0.1", "localhost", "::1"}:
            logging.getLogger(__name__).warning(
                "Server bound to %s without access control. Restrict network exposure; authentication is not implemented.", host
            )
        try:
            yield
        finally:
            runner.shutdown()

    app = FastAPI(title="SIH26158 Reconstruction API", version="0.1.0", lifespan=lifespan)
    app.state.store = store
    app.state.runner = runner
    app.add_middleware(UploadBudgetMiddleware, store=store, max_bytes=max_upload_bytes,
                       reserve_bytes=reserve_bytes)
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
            store.check_upload_space((video.size or 0) + (telemetry.size or 0), reserve_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=507, detail=str(exc)) from exc
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

    return app


app = create_app()
