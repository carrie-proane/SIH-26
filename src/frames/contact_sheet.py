"""Render the complete frame decision set for fast human review."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def create_contact_sheet(rows: Sequence[dict[str, object]], output_path: str | Path, columns: int = 5, thumbnail_width: int = 240) -> Path:
    """Draw thumbnails with green selected or red rejected borders and scores."""

    if columns < 1 or thumbnail_width < 40:
        raise ValueError("Invalid contact sheet dimensions")
    thumb_height = round(thumbnail_width * 9 / 16)
    count = len(rows)
    sheet_rows = max(1, math.ceil(count / columns))
    sheet = np.full((sheet_rows * thumb_height, columns * thumbnail_width, 3), 28, np.uint8)
    for position, row in enumerate(rows):
        image = cv2.imread(str(row["frame_path"]))
        if image is None:
            continue
        image = cv2.resize(image, (thumbnail_width, thumb_height), interpolation=cv2.INTER_AREA)
        color = (0, 190, 0) if row["selected"] else (0, 0, 220)
        cv2.rectangle(image, (1, 1), (thumbnail_width - 2, thumb_height - 2), color, 4)
        label = f"#{row['frame_index']}  {float(row['composite_score']):.3f}"
        cv2.rectangle(image, (4, 5), (155, 29), (0, 0, 0), -1)
        cv2.putText(image, label, (9, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        y, x = divmod(position, columns)
        sheet[y * thumb_height:(y + 1) * thumb_height, x * thumbnail_width:(x + 1) * thumbnail_width] = image
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise OSError(f"Could not write contact sheet: {output_path}")
    return output_path
