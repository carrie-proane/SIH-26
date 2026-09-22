import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from pydantic import ValidationError

from sih26158.completion import boundary_id, complete_planar_gaps, load_observed_mesh
from sih26158.completion_contract import (
    CompletionParameters,
    CompletionProvenance,
    GapReview,
    MeshCoordinates,
    MeshSource,
)
from sih26158.dataset import DatasetAsset
from sih26158.storage import sha256_file


def review(image, decision="CONFIRMED_SMALL_GAP"):
    return GapReview(
        boundary_id=boundary_id(list(range(8))),
        decision=decision,
        reviewer="synthetic fixture",
        explanation="Removed patch in a known plane; not real-world evidence",
        source_images=(image,),
    )


def update_mesh(root, source, mesh):
    mesh.export(root / source.artifact.path)
    return source.model_copy(
        update={
            "artifact": DatasetAsset(
                path=source.artifact.path, sha256=sha256_file(root / source.artifact.path)
            )
        }
    )


def test_removed_patch_preserves_source_coordinates_and_inferred_face_identity(
    observed_ring, tmp_path
):
    root, source, image = observed_ring
    original = (root / source.artifact.path).read_bytes()
    report = complete_planar_gaps(root, tmp_path / "derived", source, reviews=(review(image),))
    assert report.status == "COMPLETED", report.model_dump()
    assert report.generated_face_count == 6
    assert report.measurement_eligible is False
    assert len(report.rejected_regions) == 1  # Outer scene boundary is retained.
    assert "AREA_LIMIT" in report.rejected_regions[0]["reasons"]
    output = tmp_path / "derived/completion"
    patch = trimesh.load(output / "generated_mesh.ply", process=False)
    assert np.max(np.abs(patch.vertices[:, 2])) == 0  # Removed-patch planar error.
    original_mesh = load_observed_mesh(root, source)
    assert np.array_equal(
        patch.vertices,
        original_mesh.vertices[report.accepted_regions[0]["output_vertex_source_ids"]],
    )
    assert np.all(patch.face_normals[:, 2] > 0.99)
    assert (root / source.artifact.path).read_bytes() == original
    provenance = CompletionProvenance.model_validate_json((output / "provenance.json").read_text())
    assert provenance.measurable is False and provenance.geometry_source == "AI_INFERRED"
    assert provenance.face_ids == list(range(len(patch.faces)))
    assert provenance.regions[0]["face_ids"] == provenance.face_ids
    assert provenance.output.sha256 == sha256_file(output / "generated_mesh.ply")
    assert provenance.source.coordinates == source.coordinates
    complete_planar_gaps(root, tmp_path / "repeat", source, reviews=(review(image),))
    assert (output / "generated_mesh.ply").read_bytes() == (
        tmp_path / "repeat/completion/generated_mesh.ply"
    ).read_bytes()


@pytest.mark.parametrize("decision", [None, "UNKNOWN", "STRUCTURAL_OPENING"])
def test_true_or_unreviewed_openings_are_never_filled(observed_ring, tmp_path, decision):
    root, source, image = observed_ring
    report = complete_planar_gaps(
        root,
        tmp_path / "derived",
        source,
        reviews=() if decision is None else (review(image, decision),),
    )
    assert report.status == "REJECTED"
    assert not (tmp_path / "derived/completion/generated_mesh.ply").exists()
    if decision == "STRUCTURAL_OPENING":
        assert "STRUCTURAL_OPENING" in report.rejected_regions[0]["reasons"]


@pytest.mark.parametrize("kind", ["large", "corner", "insufficient_support", "nonmanifold"])
def test_geometric_rejections_even_when_operator_confirms(observed_ring, tmp_path, kind):
    root, source, image = observed_ring
    mesh = load_observed_mesh(root, source)
    parameters = CompletionParameters()
    if kind == "large":
        mesh.vertices *= 4
    elif kind == "corner":
        mesh.vertices[8, 2] = 0.5
    elif kind == "insufficient_support":
        parameters = CompletionParameters(min_support_faces=20)
    else:
        mesh = trimesh.Trimesh(
            vertices=np.vstack((mesh.vertices, [[0, 0, 0.1]])),
            faces=np.vstack((mesh.faces, [[0, 8, 16]])),
            process=False,
        )
    source = update_mesh(root, source, mesh)
    report = complete_planar_gaps(
        root, tmp_path / "derived", source, parameters=parameters, reviews=(review(image),)
    )
    assert report.status in {"REJECTED", "UNAVAILABLE"}
    assert report.generated_face_count == 0


def test_missing_mesh_sparse_points_and_hash_mismatch_are_unavailable(observed_ring, tmp_path):
    root, source, _ = observed_ring
    assert complete_planar_gaps(root, tmp_path / "none", None).status == "UNAVAILABLE"
    (root / source.artifact.path).write_bytes(b"changed")
    report = complete_planar_gaps(root, tmp_path / "hash", source)
    assert report.status == "UNAVAILABLE" and "checksum" in report.reason
    cloud = trimesh.points.PointCloud([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    source = update_mesh(root, source, cloud)
    assert complete_planar_gaps(root, tmp_path / "cloud", source).status == "UNAVAILABLE"


def test_immutable_source_and_derived_directory_are_enforced(observed_ring, tmp_path):
    root, source, _ = observed_ring
    with pytest.raises(ValueError, match="immutable"):
        complete_planar_gaps(root, root / "completion", source)
    target = tmp_path / "derived"
    complete_planar_gaps(root, target, source)
    with pytest.raises(FileExistsError):
        complete_planar_gaps(root, target, source)
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="immutable"):
        complete_planar_gaps(root, alias / "completion", source)


def test_image_evidence_is_required_and_hash_bound(observed_ring, tmp_path):
    root, source, image = observed_ring
    no_image = review(image).model_copy(update={"source_images": ()})
    report = complete_planar_gaps(root, tmp_path / "empty", source, reviews=(no_image,))
    assert report.status == "REJECTED"
    (root / image.path).write_bytes(b"tampered")
    report = complete_planar_gaps(root, tmp_path / "tampered", source, reviews=(review(image),))
    assert report.status == "UNAVAILABLE"
    assert not (tmp_path / "tampered/completion/generated_mesh.ply").exists()


def test_coordinates_cannot_be_unknown_nonmetric_or_reoriented(observed_ring):
    _, source, _ = observed_ring
    for update in (
        {"units": "degrees"},
        {"source_frame": "GLTF"},
        {"alignment_verified": False},
        {"axis_convention": ["east", "up", "-north"]},
    ):
        with pytest.raises(ValidationError):
            MeshCoordinates.model_validate(source.coordinates.model_dump() | update)
    with pytest.raises(ValidationError):
        MeshSource.model_validate(source.model_dump() | {"geometry_source": "AI_INFERRED"})
    with pytest.raises(ValidationError):
        CompletionParameters(max_area_m2=float("nan"))


def test_translated_enu_coordinates_are_not_reoriented(observed_ring, tmp_path):
    root, source, image = observed_ring
    # Source parser itself stores float32; output must not introduce additional error.
    mesh = load_observed_mesh(root, source)
    mesh.vertices += [10, 20, 30]
    source = update_mesh(root, source, mesh)
    report = complete_planar_gaps(root, tmp_path / "translated", source, reviews=(review(image),))
    assert report.status == "COMPLETED", report.reason
    patch = trimesh.load(tmp_path / "translated/completion/generated_mesh.ply", process=False)
    assert np.all(patch.vertices[:, 2] == 30)
    assert np.allclose(patch.bounds.mean(axis=0), [10, 20, 30])


def test_published_json_schemas_validate_real_output(observed_ring, tmp_path):
    from sih26158.completion_contract import CompletionReport

    root, source, image = observed_ring
    complete_planar_gaps(root, tmp_path / "derived", source, reviews=(review(image),))
    report = json.loads((tmp_path / "derived/completion/report.json").read_text())
    assert CompletionReport.model_validate(report).schema_version == "1.0"
    schema_path = Path(__file__).parents[1] / "docs/schemas/completion-report.schema.json"
    assert json.loads(schema_path.read_text()) == CompletionReport.model_json_schema()
