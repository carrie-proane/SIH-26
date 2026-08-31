"""Optional dense visual reconstruction that never weakens sparse evidence."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .geo import SimilarityTransform, transform_ply
from .storage import atomic_json, runtime_environment

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]

OPENMVS_EMPTY_TEXTURE_RGB = np.array([255, 127, 39], dtype=np.uint8)
OPENMVS_DECLARED_EMPTY_TEXTURE_RGB = np.array([255, 0, 255], dtype=np.uint8)
OPENMVS_DECLARED_EMPTY_TEXTURE_VALUE = "16711935"
OPENMVS_EMPTY_TEXTURE_COLORS = (
    OPENMVS_EMPTY_TEXTURE_RGB,
    OPENMVS_DECLARED_EMPTY_TEXTURE_RGB,
)
MIN_TEXTURE_COVERAGE = 0.15
MAX_PRIMARY_CLIP_FRACTION = 0.02
OPENMVS_BIN_ENV = "OPENMVS_BIN"


class DenseProviderError(RuntimeError):
    """A visual-only dense stage failed without invalidating sparse evidence."""


@dataclass(frozen=True)
class DenseContext:
    run_dir: Path
    frames_dir: Path
    sparse_model_dir: Path
    registered_images: int
    transform: SimilarityTransform
    mask_dir: Path | None = None
    reconstruction_target: str = "FULL_SCENE"
    scene_analysis: dict[str, object] = field(default_factory=dict)
    profile: str = "preview"


@dataclass(frozen=True)
class DenseProfileSettings:
    resolution_level: int
    number_views_fuse: int
    filter_point_cloud: int
    remove_spurious: int
    refine_resolution_level: int
    texture_resolution_level: int
    max_texture_size: int


def dense_profile_settings(profile: str) -> DenseProfileSettings:
    """Return deterministic quality settings for the declared run profile."""

    settings = {
        "smoke": DenseProfileSettings(3, 2, 0, 10, 2, 1, 4096),
        "preview": DenseProfileSettings(2, 2, 0, 20, 1, 0, 4096),
        "balanced": DenseProfileSettings(1, 3, 0, 25, 0, 0, 8192),
        "accurate": DenseProfileSettings(0, 3, 1, 30, 0, 0, 8192),
        "diagnostic": DenseProfileSettings(1, 3, 1, 30, 0, 0, 8192),
    }
    return settings.get(profile, settings["preview"])


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
    mesh_filtering: dict[str, object] = field(default_factory=dict)
    texture_validation: dict[str, object] = field(default_factory=dict)
    quality_profile: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TextureAtlasAssessment:
    status: str
    accepted: bool
    atlas_files: tuple[str, ...] = ()
    coverage: float | None = None
    empty_fraction: float | None = None
    primary_clip_fraction: float | None = None
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "accepted": self.accepted,
            "atlas_files": list(self.atlas_files),
            "atlas_nonempty_fraction": self.coverage,
            "empty_pixel_fraction": self.empty_fraction,
            "primary_clip_fraction": self.primary_clip_fraction,
            "thresholds": {
                "minimum_atlas_nonempty_fraction": MIN_TEXTURE_COVERAGE,
                "maximum_primary_clip_fraction": MAX_PRIMARY_CLIP_FRACTION,
            },
            "viewer_face_filter_contract": {
                "strategy": "ATLAS_EMPTY_COLOR",
                "empty_rgb": OPENMVS_DECLARED_EMPTY_TEXTURE_RGB.tolist(),
                "empty_tolerance": 2,
                "minimum_supported_samples": 1,
            },
            "reasons": list(self.reasons),
        }


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


def _ply_texture_atlases(mesh_path: Path) -> list[Path]:
    """Return only texture files explicitly referenced by a PLY mesh.

    OpenMVS writes one ``comment TextureFile`` entry per atlas. Older builds
    occasionally omit those comments, so a stem-scoped fallback is retained.
    Paths are reduced to basenames to prevent a malformed PLY from escaping
    the raw artifact directory.
    """
    try:
        with mesh_path.open("rb") as stream:
            header = stream.read(256 * 1024).split(b"end_header", 1)[0].decode("ascii")
    except (OSError, UnicodeDecodeError):
        return []
    names = [
        line.removeprefix("comment TextureFile ").strip()
        for line in header.splitlines()
        if line.startswith("comment TextureFile ")
    ]
    atlases = [mesh_path.parent / Path(name).name for name in names if name]
    if not atlases:
        for suffix in (".png", ".jpg", ".jpeg"):
            atlases.extend(sorted(mesh_path.parent.glob(f"{mesh_path.stem}*{suffix}")))
    return list(dict.fromkeys(atlases))


def assess_texture_atlas(mesh_path: Path) -> TextureAtlasAssessment:
    """Reject missing, mostly empty, or primary-colour-clipped texture atlases.

    OpenMVS uses RGB(255, 127, 39) for faces that no source view can texture.
    The second guard detects the characteristic exact RGB-cube clipping caused
    by pathological seam leveling while allowing naturally dark or white
    scenes (neutral black/white pixels are deliberately excluded).
    """
    atlases = _ply_texture_atlases(mesh_path)
    if not mesh_path.is_file() or not atlases:
        return TextureAtlasAssessment(
            status="REJECTED",
            accepted=False,
            reasons=("Textured mesh has no declared texture atlas.",),
        )

    total_pixels = 0
    empty_pixels = 0
    primary_clip_pixels = 0
    unreadable: list[str] = []
    for atlas in atlases:
        image = cv2.imread(str(atlas), cv2.IMREAD_COLOR)
        if image is None:
            unreadable.append(atlas.name)
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        total_pixels += int(rgb.shape[0] * rgb.shape[1])
        empty = np.zeros(rgb.shape[:2], dtype=bool)
        for empty_color in OPENMVS_EMPTY_TEXTURE_COLORS:
            empty |= np.all(rgb == empty_color, axis=2)
        empty_pixels += int(np.count_nonzero(empty))
        at_cube_corner = np.all((rgb <= 2) | (rgb >= 253), axis=2)
        neutral = np.ptp(rgb.astype(np.int16), axis=2) <= 2
        primary_clipped = at_cube_corner & ~neutral & ~empty
        primary_clip_pixels += int(np.count_nonzero(primary_clipped))

    reasons: list[str] = []
    if unreadable:
        reasons.append("Unreadable texture atlas files: " + ", ".join(unreadable))
    if total_pixels == 0:
        reasons.append("No readable texture pixels were produced.")
        return TextureAtlasAssessment(
            status="REJECTED",
            accepted=False,
            atlas_files=tuple(atlas.name for atlas in atlases),
            reasons=tuple(reasons),
        )

    empty_fraction = empty_pixels / total_pixels
    coverage = 1.0 - empty_fraction
    non_empty_pixels = max(total_pixels - empty_pixels, 1)
    primary_clip_fraction = primary_clip_pixels / non_empty_pixels
    if coverage < MIN_TEXTURE_COVERAGE:
        reasons.append(
            f"Atlas non-empty fraction {coverage:.1%} is below the "
            f"{MIN_TEXTURE_COVERAGE:.0%} acceptance floor."
        )
    if primary_clip_fraction > MAX_PRIMARY_CLIP_FRACTION:
        reasons.append(
            f"Primary-colour clipping {primary_clip_fraction:.1%} exceeds the "
            f"{MAX_PRIMARY_CLIP_FRACTION:.0%} acceptance ceiling."
        )
    if unreadable:
        accepted = False
    else:
        accepted = not reasons
    return TextureAtlasAssessment(
        status="ACCEPTED" if accepted else "REJECTED",
        accepted=accepted,
        atlas_files=tuple(atlas.name for atlas in atlases),
        coverage=coverage,
        empty_fraction=empty_fraction,
        primary_clip_fraction=primary_clip_fraction,
        reasons=tuple(reasons),
    )


def _archive_rejected_texture(mesh_path: Path, attempt: str) -> None:
    rejected_dir = mesh_path.parent / "texture_rejected" / attempt
    rejected_dir.mkdir(parents=True, exist_ok=True)
    for path in [mesh_path, *_ply_texture_atlases(mesh_path)]:
        if path.is_file():
            destination = rejected_dir / path.name
            if destination.exists():
                destination.unlink()
            shutil.move(str(path), destination)


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
            "reconstruction_target": context.reconstruction_target,
            "masking": {
                "status": "APPLIED" if context.mask_dir is not None else "NOT_APPLIED",
                "operational_mask_directory": (
                    str(context.mask_dir.relative_to(context.run_dir))
                    if context.mask_dir is not None
                    else None
                ),
                "consistent_dense_and_texture_use": context.mask_dir is not None,
            },
            "scene_analysis": context.scene_analysis,
            "mesh_filtering": result.mesh_filtering,
            "texture_validation": result.texture_validation,
            "quality_profile": result.quality_profile,
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
        completed = self.runner([self.binary, "-h"], capture_output=True, text=True, check=False)
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
        profile_settings = dense_profile_settings(context.profile)
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
        patch_help = self._help("patch_match_stereo")
        patch_max_size = {
            "smoke": "800",
            "preview": "1200",
            "balanced": "2000",
            "accurate": "-1",
            "diagnostic": "2000",
        }.get(context.profile, "1200")
        if "--PatchMatchStereo.max_image_size" in patch_help:
            commands[1].extend(["--PatchMatchStereo.max_image_size", patch_max_size])
        fusion_help = self._help("stereo_fusion")
        if "--StereoFusion.min_num_pixels" in fusion_help:
            commands[2].extend(
                ["--StereoFusion.min_num_pixels", str(profile_settings.number_views_fuse)]
            )
        if (
            context.profile in {"accurate", "diagnostic"}
            and "--StereoFusion.max_reproj_error" in fusion_help
        ):
            commands[2].extend(["--StereoFusion.max_reproj_error", "1.5"])
        if context.mask_dir is not None:
            patch_help = self._help("patch_match_stereo")
            mask_flag = "--PatchMatchStereo.mask_path"
            if mask_flag not in patch_help:
                raise DenseProviderError(
                    "Installed COLMAP dense PatchMatch cannot consume the declared masks; "
                    "sparse evidence is preserved and an OpenMVS provider is required."
                )
            commands[1].extend([mask_flag, str(context.mask_dir)])
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
            mesh_filtering={
                "status": "TOOL_DEFAULTS_ONLY",
                "method": "COLMAP_POISSON_AND_OPTIONAL_SIMPLIFIER",
                "adaptive_component_filter": False,
                "rationale": (
                    "This COLMAP build exposes no verified connected-component cleaning flag; "
                    "no unsupported post-processing was claimed."
                ),
            },
            quality_profile={
                "requested": context.profile,
                "provider": self.name,
                "patch_match_max_image_size": int(patch_max_size),
                "minimum_fusion_support": profile_settings.number_views_fuse,
                "support_aware_filtering": True,
            },
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
        openmvs_bin: str | Path | None = None,
    ) -> None:
        self.runner = runner
        self.colmap_binary = colmap_binary
        configured_bin = (
            openmvs_bin if openmvs_bin is not None else os.environ.get(OPENMVS_BIN_ENV)
        )
        self.openmvs_bin = Path(configured_bin).expanduser() if configured_bin else None

    def _tool(self, name: str) -> str:
        """Resolve an OpenMVS executable from OPENMVS_BIN or the process PATH."""

        if self.openmvs_bin is not None:
            return str(self.openmvs_bin / name)
        return name

    def _tool_available(self, name: str) -> bool:
        if self.openmvs_bin is not None:
            candidate = self.openmvs_bin / name
            return candidate.is_file() and os.access(candidate, os.X_OK)
        return shutil.which(name) is not None

    def availability(self) -> tuple[bool, str]:
        missing = [tool for tool in self.tools if not self._tool_available(tool)]
        if missing:
            location = (
                f"OPENMVS_BIN={self.openmvs_bin}"
                if self.openmvs_bin is not None
                else "PATH"
            )
            return False, f"OpenMVS tools unavailable in {location}: " + ", ".join(missing)
        location = (
            f"OPENMVS_BIN={self.openmvs_bin}"
            if self.openmvs_bin is not None
            else "PATH"
        )
        return True, f"OpenMVS external command suite is installed ({location})"

    def _execute(self, command: list[str], log_path: Path) -> None:
        resolved_command = [self._tool(command[0]), *command[1:]]
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write("$ " + " ".join(resolved_command) + "\n")
            completed = self.runner(
                resolved_command,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode:
            raise DenseProviderError(
                f"OpenMVS command failed ({Path(command[0]).name}, exit {completed.returncode})"
            )

    def _help(self, command: str) -> str:
        completed = self.runner(
            [self._tool(command), "-h"], capture_output=True, text=True, check=False
        )
        return (completed.stdout or "") + (completed.stderr or "")

    def _require_flags(self, command: str, flags: tuple[str, ...]) -> None:
        help_text = self._help(command)
        missing = [flag for flag in flags if flag not in help_text]
        if missing:
            raise DenseProviderError(
                f"Installed OpenMVS {command} cannot honor required policy flags: "
                + ", ".join(missing)
            )

    @staticmethod
    def export_existing_outputs(
        context: DenseContext,
        raw_dir: Path,
        texture_assessment: TextureAtlasAssessment | None = None,
    ) -> tuple[list[Path], int, int, int]:
        """Export completed OpenMVS outputs into the stable viewer contract.

        Keeping this step separate allows an interrupted/previously failed
        export to be recovered without repeating hours of dense processing.
        Raw OpenMVS files remain untouched as provenance-bearing originals.
        """
        dense_raw = raw_dir / "scene_dense.ply"
        mesh_raw = raw_dir / "scene_mesh.ply"
        refined_raw = raw_dir / "scene_mesh_refine.ply"
        textured_raw = raw_dir / "scene_mesh_texture.ply"
        if not dense_raw.is_file() or not mesh_raw.is_file():
            raise DenseProviderError(
                "OpenMVS export recovery requires scene_dense.ply and scene_mesh.ply"
            )

        dense_dir = context.run_dir / "dense"
        textured_dir = dense_dir / "textured"
        textured_dir.mkdir(parents=True, exist_ok=True)
        for stale in textured_dir.iterdir():
            if stale.is_file():
                stale.unlink()
        artifacts: list[Path] = []

        fused_local = dense_dir / "fused.ply"
        transform_ply(dense_raw, fused_local, context.transform)
        artifacts.append(fused_local)

        mesh_local = dense_dir / "meshed-openmvs.ply"
        transform_ply(mesh_raw, mesh_local, context.transform)
        artifacts.append(mesh_local)
        count_mesh = mesh_local

        if refined_raw.is_file():
            refined_local = dense_dir / "meshed-openmvs-refined.ply"
            transform_ply(refined_raw, refined_local, context.transform)
            artifacts.append(refined_local)
            count_mesh = refined_local

        if textured_raw.is_file():
            texture_assessment = texture_assessment or assess_texture_atlas(textured_raw)
        if textured_raw.is_file() and texture_assessment and texture_assessment.accepted:
            textured_local = textured_dir / "model.ply"
            transform_ply(textured_raw, textured_local, context.transform)
            artifacts.append(textured_local)
            count_mesh = textured_local
            for atlas in _ply_texture_atlases(textured_raw):
                destination = textured_dir / atlas.name
                shutil.copyfile(atlas, destination)
                artifacts.append(destination)

        dense_points, _ = ply_counts(fused_local)
        vertices, faces = ply_counts(count_mesh)
        return artifacts, dense_points, vertices, faces

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
        mesh_ply = raw_dir / "scene_mesh.ply"
        refined_ply = raw_dir / "scene_mesh_refine.ply"
        textured_ply = raw_dir / "scene_mesh_texture.ply"
        commands: list[list[str]] = []
        profile_settings = dense_profile_settings(context.profile)
        remove_spurious_value = profile_settings.remove_spurious
        if context.reconstruction_target == "PRIMARY_SUBJECT":
            remove_spurious_value += 20
        remove_spurious = str(remove_spurious_value)
        close_holes = "0" if context.reconstruction_target == "PRIMARY_SUBJECT" else "15"
        self._require_flags(
            "DensifyPointCloud",
            ("--resolution-level", "--number-views-fuse", "--filter-point-cloud"),
        )
        self._require_flags(
            "ReconstructMesh", ("--remove-spurious", "--close-holes", "--free-space-support")
        )
        texture_help = self._help("TextureMesh")
        self._require_flags(
            "TextureMesh",
            ("--resolution-level", "--max-texture-size", "--empty-color", "--sharpness-weight"),
        )
        conservative_texture_flags = ("--local-seam-leveling",)
        supports_conservative_retry = all(
            flag in texture_help for flag in conservative_texture_flags
        )
        if context.mask_dir is not None:
            self._require_flags("DensifyPointCloud", ("--mask-path", "--ignore-mask-label"))
            self._require_flags("TextureMesh", ("--ignore-mask-label", "--close-holes"))
        if shutil.which(self.colmap_binary) is not None:
            undistort_command = [
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
            commands.append(undistort_command)
            self._execute(undistort_command, log_path)
        else:
            for link, target in (
                (input_sparse, context.sparse_model_dir),
                (input_images, context.frames_dir),
            ):
                if link.exists() or link.is_symlink():
                    link.unlink()
                link.symlink_to(target, target_is_directory=True)
        densify_command = [
            "DensifyPointCloud",
            "-w",
            str(raw_dir),
            "-i",
            str(scene),
            "-o",
            str(dense_scene),
            "--resolution-level",
            str(profile_settings.resolution_level),
            "--number-views-fuse",
            str(profile_settings.number_views_fuse),
            "--filter-point-cloud",
            str(profile_settings.filter_point_cloud),
        ]
        if context.mask_dir is not None:
            densify_command.extend(["-m", str(context.mask_dir), "--ignore-mask-label", "0"])
        texture_command = [
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
            "--close-holes",
            close_holes,
            "--resolution-level",
            str(profile_settings.texture_resolution_level),
            "--max-texture-size",
            str(profile_settings.max_texture_size),
            "--empty-color",
            OPENMVS_DECLARED_EMPTY_TEXTURE_VALUE,
            "--sharpness-weight",
            "0.5",
        ]
        if context.mask_dir is not None:
            texture_command.extend(["--ignore-mask-label", "0"])
        core_commands = [
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
            densify_command,
            [
                "ReconstructMesh",
                "-w",
                str(raw_dir),
                "-i",
                str(dense_scene),
                "-o",
                str(mesh_ply),
                "--free-space-support",
                "0",
                "--remove-spurious",
                remove_spurious,
                "--close-holes",
                close_holes,
                "--remove-spikes",
                "1",
            ],
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
                "--resolution-level",
                str(profile_settings.refine_resolution_level),
            ],
        ]
        commands.extend(core_commands)
        for command in core_commands:
            self._execute(command, log_path)

        warnings: list[dict[str, str]] = []
        texture_assessment = TextureAtlasAssessment(
            status="NOT_PRODUCED",
            accepted=False,
            reasons=("OpenMVS did not produce an accepted photographic texture atlas.",),
        )
        commands.append(texture_command)
        try:
            self._execute(texture_command, log_path)
        except DenseProviderError as exc:
            warnings.append(
                {
                    "code": "TEXTURE_GENERATION_FAILED_DENSE_PRESERVED",
                    "message": str(exc),
                }
            )
        else:
            texture_assessment = assess_texture_atlas(textured_ply)

        if not texture_assessment.accepted:
            rejected_reason = "; ".join(texture_assessment.reasons) or "atlas validation failed"
            warnings.append(
                {
                    "code": "TEXTURE_ATLAS_REJECTED_INITIAL",
                    "message": rejected_reason,
                }
            )
            if textured_ply.is_file():
                _archive_rejected_texture(textured_ply, "initial")
            if supports_conservative_retry:
                safe_texture_command = [
                    *texture_command,
                    "--local-seam-leveling",
                    "0",
                ]
                commands.append(safe_texture_command)
                try:
                    self._execute(safe_texture_command, log_path)
                except DenseProviderError as exc:
                    texture_assessment = TextureAtlasAssessment(
                        status="REJECTED",
                        accepted=False,
                        reasons=(str(exc),),
                    )
                else:
                    texture_assessment = assess_texture_atlas(textured_ply)
                if texture_assessment.accepted:
                    warnings.append(
                        {
                            "code": "TEXTURE_ATLAS_CONSERVATIVE_RETRY_ACCEPTED",
                            "message": (
                                "The initial atlas failed validation; a local-seam-leveling-disabled "
                                "retry passed and is the only textured artifact declared."
                            ),
                        }
                    )
                else:
                    retry_reason = (
                        "; ".join(texture_assessment.reasons) or "atlas validation failed"
                    )
                    warnings.append(
                        {
                            "code": "TEXTURE_ATLAS_REJECTED_FINAL",
                            "message": (
                                "The conservative retry was also rejected; no Textured Model "
                                f"will be declared. {retry_reason}"
                            ),
                        }
                    )
                    if textured_ply.is_file():
                        _archive_rejected_texture(textured_ply, "conservative_retry")
            else:
                warnings.append(
                    {
                        "code": "TEXTURE_ATLAS_RETRY_UNSUPPORTED",
                        "message": (
                            "Installed OpenMVS lacks the documented conservative texturing flags; "
                            "the rejected texture is not declared."
                        ),
                    }
                )

        exported, dense_points, vertices, faces = self.export_existing_outputs(
            context, raw_dir, texture_assessment
        )
        artifacts = [log_path, *exported]
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
            mesh_filtering={
                "status": "APPLIED",
                "method": "OPENMVS_RECONSTRUCT_MESH_COMPONENT_AND_SPURIOUS_FACE_FILTER",
                "remove_spurious": int(remove_spurious),
                "close_holes": int(close_holes),
                "adaptive": True,
                "rationale": (
                    "PRIMARY_SUBJECT uses stronger component removal and preserves holes; "
                    "FULL_SCENE retains more disconnected static geometry."
                ),
            },
            texture_validation=texture_assessment.as_dict(),
            quality_profile={
                "requested": context.profile,
                "provider": self.name,
                "resolution_level": profile_settings.resolution_level,
                "minimum_fusion_support": profile_settings.number_views_fuse,
                "visibility_filter": profile_settings.filter_point_cloud,
                "refine_resolution_level": profile_settings.refine_resolution_level,
                "texture_resolution_level": profile_settings.texture_resolution_level,
                "max_texture_size": profile_settings.max_texture_size,
                "support_aware_filtering": True,
                "untextured_face_policy": "DECLARED_EMPTY_COLOR_FILTER",
                "empty_texture_rgb": OPENMVS_DECLARED_EMPTY_TEXTURE_RGB.tolist(),
            },
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


def select_dense_provider(
    preference: str = "auto", *, require_masks: bool = False
) -> DenseReconstructionProvider:
    colmap = ColmapDenseProvider()
    openmvs = OpenMVSProvider()
    if preference in {"auto", "colmap"}:
        available, reason = colmap.availability()
        if available and require_masks:
            mask_flag = "--PatchMatchStereo.mask_path"
            if mask_flag not in colmap._help("patch_match_stereo"):
                available = False
                reason = (
                    "Installed COLMAP dense PatchMatch does not expose a documented mask path; "
                    "masked dense reconstruction requires OpenMVS/external execution"
                )
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
    result.artifacts.extend(_write_dense_metadata(context, result, report_path, commands_path))
    return result
