"""Utilities for image conversion, drawing and simple metrics."""
from typing import Tuple, List
import sys
from pathlib import Path

# Ensure local src imports work when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np


def draw_box_label(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    score: float = None,
    depth_m: float = None,
    color=(0, 255, 0),
) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    lines = [f"{label} {score:.2f}" if score is not None else label]
    if depth_m is not None:
        lines.append(f"~{depth_m:.2f}m")

    line_height = 18
    total_height = line_height * len(lines) + 4
    max_width = max(cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] for text in lines)

    cv2.rectangle(frame, (x1, y1 - total_height), (x1 + max_width + 6, y1), color, -1)
    for idx, text in enumerate(lines):
        offset = y1 - 4 - (len(lines) - 1 - idx) * line_height
        cv2.putText(frame, text, (x1 + 3, offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)


def resize_keep_aspect(frame: np.ndarray, width: int = 640) -> Tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    if w == width:
        return frame, 1.0
    scale = width / float(w)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (width, new_h))
    return resized, scale


def box_scale_back(box: List[int], scale: float) -> List[int]:
    return [int(round(b / scale)) for b in box]
