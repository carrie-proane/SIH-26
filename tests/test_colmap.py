import csv
from pathlib import Path

import numpy as np

from sih26158.colmap import ColmapRunner


def test_colmap_text_camera_pose_export(tmp_path: Path) -> None:
    images = tmp_path / "images.txt"
    images.write_text(
        "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"
        "1 1 0 0 0 -2 -3 -4 1 frame_0001.jpg\n"
        "1.0 2.0 -1\n",
        encoding="utf-8",
    )
    output = tmp_path / "poses.csv"
    ColmapRunner._export_camera_poses(images, output)
    with output.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["image_name"] == "frame_0001.jpg"
    assert np.allclose([float(row["sfm_x"]), float(row["sfm_y"]), float(row["sfm_z"])], [2, 3, 4])

