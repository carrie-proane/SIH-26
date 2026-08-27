"""Optional dense visual reconstruction that never weakens sparse evidence."""

from __future__ import annotations

import platform
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .geo import SimilarityTransform, transform_ply
from .storage import atomic_json, runtime_environment

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class DenseProviderError(RuntimeError):
    """A visual-only dense stage failed without invalidating sparse evidence."""


@dataclass(frozen=True)
class DenseContext:
    run_dir: Path
    frames_dir: Path
    sparse_model_dir: Path
    registered_images: int
    transform: SimilarityTransform


@dataclass
class DenseResult:
    provider: str
    status: str
    artifacts: list[Path] = field(default_factory=list)
    commands: list[list[str]] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)
    dense_point_count: int | None = None
    vertex_count: int | None = None
    face_count: int | None = None
    texture_coverage: float | None = None
    runtime_s: float = 0.0
    metric_alignment_preserved: bool = False
    measurement_eligible: bool = False


def ply_counts(path: Path) -> tuple[int, int]:
    """Read declared vertex/face counts without loading a potentially huge PLY body."""
    try:
        with path.open("rb") as stream:
            header = stream.read(256 * 1024).split(b"end_header", 1)[0].decode("ascii")
    except (OSError, UnicodeDecodeError) as exc:
        raise DenseProviderError(f"Dense PLY header is unreadable: {path.name}") from exc
    vertices = 0
    faces = 0
    for line in header.splitlines():
        fields = line.split()
        if fields[:2] == ["element", "vertex"]:
            vertices = int(fields[2])
        elif fields[:2] == ["element", "face"]:
            faces = int(fields[2])
    return vertices, faces


def _write_dense_metadata(
    context: DenseContext, result: DenseResult, report_path: Path, commands_path: Path
) -> list[Path]:
    environment = runtime_environment() | {"machine": platform.machine()}
    atomic_json(
        commands_path,
        {
            "schema_version": "1.0",
            "provider": result.provider,
            "commands": result.commands,
        },
    )
    atomic_json(
        report_path,
        {
            "schema_version": "1.0",
            "provider": result.provider,
            "status": result.status,
            "input_registered_images": context.registered_images,
            "dense_point_count": result.dense_point_count,
            "vertex_count": result.vertex_count,
            "face_count": result.face_count,
            "texture_coverage": result.texture_coverage,
            "runtime_s": result.runtime_s,
            "environment": environment,
            "warnings": result.warnings,
            "metric_alignment_preserved": result.metric_alignment_preserved,
            "sparse_evidence_preserved": True,
            "measurement_eligible": False,
            "measurement_statement": (
                "Visual dense/textured geometry is not used for verified measurement; "
                "measurements remain bound to sparse evidence geometry."
            ),
        },
    )
    return [report_path, commands_path]


class DenseReconstructionProvider(ABC):
    name: str

    @abstractmethod
    def availability(self) -> tuple[bool, str]:
        """Return whether this provider can execute locally and the evidence for that decision."""

    @abstractmethod
    def run(self, context: DenseContext) -> DenseResult:
        """Generate optional visual artifacts in local metric coordinates."""


class ColmapDenseProvider(DenseReconstructionProvider):
    name = "COLMAP_DENSE"

    def __init__(self, binary: str = "colmap", runner: CommandRunner = subprocess.run) -> None:
        self.binary = binary
        self.runner = runner

    def _help(self, command: str) -> str:
        completed = self.runner(
            [self.binary, command, "-h"], capture_output=True, text=True, check=False
        )
        return (completed.stdout or "") + (completed.stderr or "")

    def availability(self) -> tuple[bool, str]:
        if shutil.which(self.binary) is None:
            return False, "COLMAP executable is unavailable"
        completed = self.runner(
            [self.binary, "-h"], capture_output=True, text=True, check=False
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        if completed.returncode:
            return False, "COLMAP capability probe failed"
        if "without CUDA" in output:
            return False, "Installed COLMAP reports a non-CUDA build; PatchMatch is unavailable"
        required = {
            "image_undistorter": ("--image_path", "--input_path", "--output_path"),
            "patch_match_stereo": ("--workspace_path", "--PatchMatchStereo.gpu_index"),
            "stereo_fusion": ("--workspace_path", "--output_path"),
            "poisson_mesher": ("--input_path", "--output_path"),
        }
        for command, flags in required.items():
            help_text = self._help(command)
            missing = [flag for flag in flags if flag not in help_text]
            if missing:
                return False, f"COLMAP {command} lacks required flags: {', '.join(missing)}"
        return True, "Installed COLMAP exposes CUDA PatchMatch and required dense commands"

    def _optional_supported(self, command: str, required_flags: tuple[str, ...]) -> bool:
        help_text = self._help(command)
        return bool(help_text) and all(flag in help_text for flag in required_flags)

    def _execute(self, command: list[str], log_path: Path) -> None:
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write("$ " + " ".join(command) + "\n")
            completed = self.runner(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode:
            raise DenseProviderError(
                f"Dense command failed ({command[1]}, exit {completed.returncode}); inspect dense.log"
            )

    def run(self, context: DenseContext) -> DenseResult:
        available, reason = self.availability()
        if not available:
            raise DenseProviderError(reason)
        started = time.monotonic()
        dense_dir = context.run_dir / "dense"
        workspace = dense_dir / "workspace"
        raw_dir = dense_dir / "raw_sfm"
        textured_raw = raw_dir / "textured"
        textured_local = dense_dir / "textured"
        for directory in (workspace, raw_dir, textured_raw, textured_local):
            directory.mkdir(parents=True, exist_ok=True)
        log_path = context.run_dir / "logs" / "dense.log"
        log_path.write_text("", encoding="utf-8")
        fused_raw = raw_dir / "fused.ply"
        poisson_raw = raw_dir / "meshed-poisson.ply"
        simplified_raw = raw_dir / "meshed-poisson-simplified.ply"
        commands = [
            [
                self.binary,
                "image_undistorter",
                "--image_path",
                str(context.frames_dir),
                "--input_path",
                str(context.sparse_model_dir),
                "--output_path",
                str(workspace),
                "--output_type",
                "COLMAP",
            ],
            [
                self.binary,
                "patch_match_stereo",
                "--workspace_path",
                str(workspace),
                "--workspace_format",
                "COLMAP",
                "--PatchMatchStereo.geom_consistency",
                "1",
            ],
            [
                self.binary,
                "stereo_fusion",
                "--workspace_path",
                str(workspace),
                "--workspace_format",
                "COLMAP",
                "--input_type",
                "geometric",
                "--output_type",
                "PLY",
                "--output_path",
                str(fused_raw),
            ],
            [
                self.binary,
                "poisson_mesher",
                "--input_path",
                str(fused_raw),
                "--output_path",
                str(poisson_raw),
            ],
        ]
        simplifier = self._optional_supported(
            "mesh_simplifier",
            ("--input_path", "--output_path", "--MeshSimplification.target_face_ratio"),
        )
        if simplifier:
            commands.append(
                [
                    self.binary,
                    "mesh_simplifier",
                    "--input_path",
                    str(poisson_raw),
                    "--output_path",
                    str(simplified_raw),
                    "--MeshSimplification.target_face_ratio",
                    "0.25",
                ]
            )
        texturer = self._optional_supported(
            "mesh_texturer", ("--workspace_path", "--input_path", "--output_path")
        )
        texture_input = simplified_raw if simplifier else poisson_raw
        if texturer:
            commands.append(
                [
                    self.binary,
                    "mesh_texturer",
                    "--workspace_path",
                    str(workspace),
                    "--input_path",
                    str(texture_input),
                    "--output_path",
                    str(textured_raw),
                    "--output_type",
                    "TXT",
                ]
            )
        for command in commands:
            self._execute(command, log_path)
        for required in (fused_raw, poisson_raw):
            if not required.is_file():
                raise DenseProviderError(f"Dense command sequence did not produce {required.name}")

        fused_local = dense_dir / "fused.ply"
        poisson_local = dense_dir / "meshed-poisson.ply"
        transform_ply(fused_raw, fused_local, context.transform)
        transform_ply(poisson_raw, poisson_local, context.transform)
        artifacts = [fused_local, poisson_local, log_path]
        simplified_local: Path | None = None
        if simplifier and simplified_raw.is_file():
            simplified_local = dense_dir / "meshed-poisson-simplified.ply"
            transform_ply(simplified_raw, simplified_local, context.transform)
            artifacts.append(simplified_local)

        textured_mesh: Path | None = None
        if texturer:
            raw_meshes = sorted(textured_raw.glob("*.ply"))
            if raw_meshes:
                textured_mesh = textured_local / raw_meshes[0].name
                transform_ply(raw_meshes[0], textured_mesh, context.transform)
                artifacts.append(textured_mesh)
                for atlas in sorted(textured_raw.iterdir()):
                    if atlas.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                        destination = textured_local / atlas.name
                        shutil.copyfile(atlas, destination)
                        artifacts.append(destination)

        dense_points, _ = ply_counts(fused_local)
        mesh_for_counts = simplified_local or poisson_local
        vertices, faces = ply_counts(mesh_for_counts)
        warnings: list[dict[str, str]] = []
        if texturer and textured_mesh is None:
            warnings.append(
                {
                    "code": "TEXTURE_OUTPUT_UNAVAILABLE",
                    "message": "COLMAP texturing completed without a discoverable PLY mesh.",
                }
            )
        warnings.append(
            {
                "code": "TEXTURE_COVERAGE_NOT_MEASURABLE",
                "message": "This COLMAP output does not expose a reliable atlas coverage metric.",
            }
        )
        return DenseResult(
            provider=self.name,
            status="COMPLETED",
            artifacts=artifacts,
            commands=commands,
            warnings=warnings,
            dense_point_count=dense_points,
            vertex_count=vertices,
            face_count=faces,
            texture_coverage=None,
            runtime_s=time.monotonic() - started,
            metric_alignment_preserved=True,
            measurement_eligible=False,
        )


class OpenMVSProvider(DenseReconstructionProvider):
    name = "OPENMVS_EXTERNAL"
    tools = (
        "InterfaceCOLMAP",
        "DensifyPointCloud",
        "ReconstructMesh",
        "RefineMesh",
        "TextureMesh",
    )

    def __init__(
        self,
        runner: CommandRunner = subprocess.run,
        colmap_binary: str = "colmap",
    ) -> None:
        self.runner = runner
        self.colmap_binary = colmap_binary

    def availability(self) -> tuple[bool, str]:
        missing = [tool for tool in self.tools if shutil.which(tool) is None]
        if missing:
            return False, "OpenMVS tools unavailable: " + ", ".join(missing)
        return True, "OpenMVS external command suite is installed"

    def _execute(self, command: list[str], log_path: Path) -> None:
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write("$ " + " ".join(command) + "\n")
            completed = self.runner(
                command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode:
            raise DenseProviderError(
                f"OpenMVS command failed ({Path(command[0]).name}, exit {completed.returncode})"
            )

    def run(self, context: DenseContext) -> DenseResult:
        available, reason = self.availability()
        if not available:
            raise DenseProviderError(reason)
        started = time.monotonic()
        dense_dir = context.run_dir / "dense"
        raw_dir = dense_dir / "openmvs_sfm"
        raw_dir.mkdir(parents=True, exist_ok=True)
        # InterfaceCOLMAP expects a COLMAP project root with a `sparse/`
        # subdirectory. COLMAP's selected model is normally one level deeper,
        # so stage symlinks without duplicating the large source assets.
        input_root = raw_dir / "input"
        input_sparse = input_root / "sparse"
        input_images = input_root / "images"
        input_root.mkdir(exist_ok=True)
        log_path = context.run_dir / "logs" / "dense.log"
        log_path.write_text("", encoding="utf-8")
        scene = raw_dir / "scene.mvs"
        dense_scene = raw_dir / "scene_dense.mvs"
        dense_ply = raw_dir / "scene_dense.ply"
        mesh_ply = raw_dir / "scene_mesh.ply"
        refined_ply = raw_dir / "scene_mesh_refine.ply"
        textured_ply = raw_dir / "scene_mesh_texture.ply"
        commands: list[list[str]] = []
        if shutil.which(self.colmap_binary) is not None:
            commands.append(
                [
                    self.colmap_binary,
                    "image_undistorter",
                    "--image_path",
                    str(context.frames_dir),
                    "--input_path",
                    str(context.sparse_model_dir),
                    "--output_path",
                    str(input_root),
                    "--output_type",
                    "COLMAP",
                ]
            )
        else:
            for link, target in (
                (input_sparse, context.sparse_model_dir),
                (input_images, context.frames_dir),
            ):
                if link.exists() or link.is_symlink():
                    link.unlink()
                link.symlink_to(target, target_is_directory=True)
        commands.extend([
            [
                "InterfaceCOLMAP",
                "-w",
                str(raw_dir),
                "-i",
                str(input_root),
                "-o",
                str(scene),
                "--image-folder",
                "images",
            ],
            [
                "DensifyPointCloud",
                "-w",
                str(raw_dir),
                "-i",
                str(scene),
                "-o",
                str(dense_scene),
                "--resolution-level",
                "2",
            ],
            ["ReconstructMesh", "-w", str(raw_dir), "-i", str(dense_scene), "-o", str(mesh_ply)],
            [
                "RefineMesh",
                "-w",
                str(raw_dir),
                "-i",
                str(dense_scene),
                "-m",
                str(mesh_ply),
                "-o",
                str(refined_ply),
            ],
            [
                "TextureMesh",
                "-w",
                str(raw_dir),
                "-i",
                str(dense_scene),
                "-m",
                str(refined_ply),
                "-o",
                str(textured_ply),
                "--export-type",
                "ply",
            ],
        ])
        for command in commands:
            self._execute(command, log_path)
        raw_plys = sorted(raw_dir.glob("*.ply"))
        if not raw_plys:
            raise DenseProviderError("OpenMVS produced no discoverable PLY output")
        artifacts = [log_path]
        local_outputs: list[Path] = []
        for raw in raw_plys:
            destination = dense_dir / raw.name
            transform_ply(raw, destination, context.transform)
            artifacts.append(destination)
            local_outputs.append(destination)
        for atlas in sorted(raw_dir.iterdir()):
            if atlas.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                destination = dense_dir / atlas.name
                shutil.copyfile(atlas, destination)
                artifacts.append(destination)
        dense_local = next(
            (path for path in local_outputs if path.name == dense_ply.name), local_outputs[0]
        )
        mesh_local = next(
            (path for path in local_outputs if path.name == textured_ply.name), local_outputs[-1]
        )
        vertices, faces = ply_counts(mesh_local)
        dense_points, _ = ply_counts(dense_local)
        return DenseResult(
            provider=self.name,
            status="COMPLETED",
            artifacts=artifacts,
            commands=commands,
            warnings=[
                {
                    "code": "TEXTURE_COVERAGE_NOT_MEASURABLE",
                    "message": "OpenMVS output did not expose a reliable atlas coverage metric.",
                }
            ],
            dense_point_count=dense_points,
            vertex_count=vertices,
            face_count=faces,
            runtime_s=time.monotonic() - started,
            metric_alignment_preserved=True,
            measurement_eligible=False,
        )


class UnavailableProvider(DenseReconstructionProvider):
    name = "UNAVAILABLE"

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def availability(self) -> tuple[bool, str]:
        return False, self.reason

    def run(self, context: DenseContext) -> DenseResult:
        return DenseResult(
            provider=self.name,
            status="UNAVAILABLE",
            warnings=[{"code": "DENSE_PROVIDER_UNAVAILABLE", "message": self.reason}],
            metric_alignment_preserved=False,
            measurement_eligible=False,
        )


def select_dense_provider(preference: str = "auto") -> DenseReconstructionProvider:
    colmap = ColmapDenseProvider()
    openmvs = OpenMVSProvider()
    if preference in {"auto", "colmap"}:
        available, reason = colmap.availability()
        if available:
            return colmap
        if preference == "colmap":
            return UnavailableProvider(reason)
        colmap_reason = reason
    else:
        colmap_reason = "COLMAP dense provider was not selected"
    if preference in {"auto", "openmvs"}:
        available, reason = openmvs.availability()
        if available:
            return openmvs
        return UnavailableProvider(f"{colmap_reason}; {reason}")
    return UnavailableProvider(f"Unsupported dense provider preference: {preference}")


def run_dense_stage(
    context: DenseContext,
    provider: DenseReconstructionProvider,
) -> DenseResult:
    """Run a visual provider and always persist its honest report/log contract."""
    dense_dir = context.run_dir / "dense"
    dense_dir.mkdir(exist_ok=True)
    log_path = context.run_dir / "logs" / "dense.log"
    if not log_path.exists():
        log_path.write_text("", encoding="utf-8")
    started = time.monotonic()
    try:
        result = provider.run(context)
    except Exception as exc:  # noqa: BLE001 - this boundary must preserve sparse success
        result = DenseResult(
            provider=provider.name,
            status="FAILED_VISUAL_ONLY",
            warnings=[
                {
                    "code": "DENSE_RECONSTRUCTION_FAILED_SPARSE_PRESERVED",
                    "message": str(exc),
                }
            ],
            runtime_s=time.monotonic() - started,
            metric_alignment_preserved=False,
            measurement_eligible=False,
        )
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"Dense visual stage failed safely: {exc}\n")
    result.runtime_s = max(result.runtime_s, time.monotonic() - started)
    if log_path not in result.artifacts:
        result.artifacts.append(log_path)
    report_path = context.run_dir / "dense_report.json"
    commands_path = context.run_dir / "dense_commands.json"
    result.artifacts.extend(
        _write_dense_metadata(context, result, report_path, commands_path)
    )
    return result
