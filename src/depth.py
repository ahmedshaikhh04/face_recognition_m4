"""Lightweight depth estimation utilities for monocular webcam feeds.

This module provides a pinhole-camera approximation to estimate the
distance to a face using the bounding box width. It intentionally avoids
external model downloads so it can run in constrained environments.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass
class DepthEstimate:
    """Container for depth metadata."""

    meters: float
    confidence: float


class DepthEstimator:
    """Estimate face depth from a 2D bounding box.

    The calculation uses a simple pinhole-camera model:

    depth = (known_face_width * focal_length_px) / box_pixel_width

    where `focal_length_px` is derived from an assumed horizontal field of
    view. This keeps the dependency surface small while still providing a
    useful, human-readable distance estimate for UI overlays and basic
    range filtering.
    """

    def __init__(self, face_width_m: float = 0.16, fov_deg: float = 60.0, max_depth_m: float = 4.0):
        self.face_width_m = face_width_m
        self.fov_deg = fov_deg
        self.max_depth_m = max_depth_m

    def _focal_length_px(self, frame_width: int) -> float:
        # Derive focal length in pixels from horizontal field of view.
        return (frame_width / 2.0) / math.tan(math.radians(self.fov_deg) / 2.0)

    def estimate(self, box: Sequence[int], frame_shape: Tuple[int, int, int]) -> Optional[DepthEstimate]:
        """Return an approximate depth estimate in meters.

        Args:
            box: (x1, y1, x2, y2) bounding box in pixel coordinates.
            frame_shape: Shape of the original frame (H, W, C).
        """

        if not box or len(box) != 4:
            return None

        x1, _, x2, _ = box
        box_width_px = max(1, x2 - x1)
        frame_width = frame_shape[1]

        focal_len_px = self._focal_length_px(frame_width)
        depth_m = (self.face_width_m * focal_len_px) / box_width_px

        # Normalize confidence based on how far the box width deviates from a reasonable range.
        confidence = max(0.0, min(1.0, 1.0 - (depth_m / self.max_depth_m)))
        return DepthEstimate(meters=depth_m, confidence=confidence)
