from pathlib import Path

import cv2
import numpy as np
import pytest

from sih26158.completion_contract import MeshCoordinates, MeshSource
from sih26158.dataset import DatasetAsset
from sih26158.storage import sha256_file


@pytest.fixture
def observed_ring(tmp_path):
    """Synthetic removed-patch experiment; never real scene/accuracy evidence."""
    root = tmp_path / "source"
    root.mkdir()
    fixture = Path(__file__).parent / "fixtures/completion/planar_ring.ply"
    (root / "observed.ply").write_bytes(fixture.read_bytes())
    (root / "alignment.json").write_text('{"fixture": "SYNTHETIC_ENU"}')
    cv2.imwrite(str(root / "source.png"), np.zeros((12, 12, 3), np.uint8))

    def asset(name):
        return DatasetAsset(path=name, sha256=sha256_file(root / name))

    source = MeshSource(
        run_id="synthetic_removed_patch",
        artifact=asset("observed.ply"),
        coordinates=MeshCoordinates(
            alignment_verified=True,
            altitude_reference="SYNTHETIC_LOCAL_Z",
            alignment_evidence=asset("alignment.json"),
        ),
    )
    return root, source, asset("source.png")
