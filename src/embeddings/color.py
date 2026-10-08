"""
src/embeddings/color.py — Deterministic color estimation for object crops.

Uses HSV color space analysis to estimate dominant color.
Does NOT claim perfect color recognition — used as a hard filter hint.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# HSV color ranges: (lower_hsv, upper_hsv, name)
# Hue is 0-179 in OpenCV, Saturation 0-255, Value 0-255
COLOR_RANGES = [
    ((0, 100, 100),   (10, 255, 255),   "red"),
    ((160, 100, 100), (179, 255, 255),  "red"),   # red wraps around
    ((10, 100, 100),  (22, 255, 255),   "orange"),
    ((22, 100, 100),  (38, 255, 255),   "yellow"),
    ((38, 80, 80),    (85, 255, 255),   "green"),
    ((85, 80, 80),    (130, 255, 255),  "blue"),
    ((130, 80, 80),   (160, 255, 255),  "purple"),
    ((0, 0, 180),     (179, 30, 255),   "white"),
    ((0, 0, 0),       (179, 255, 50),   "black"),
    ((0, 0, 50),      (179, 50, 180),   "gray"),
    ((10, 100, 50),   (22, 200, 200),   "brown"),
]


def estimate_dominant_color(image: np.ndarray, top_k: int = 1) -> Optional[str]:
    """
    Estimate the dominant color of a BGR image crop.

    Args:
        image: BGR numpy array
        top_k: Return top_k most dominant colors (returns the first)

    Returns:
        Color name string or None if estimation fails.
    """
    if image is None or image.size == 0:
        return None

    try:
        # Resize for speed
        small = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        color_counts: dict = {}

        for lower, upper, name in COLOR_RANGES:
            lower_arr = np.array(lower, dtype=np.uint8)
            upper_arr = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_arr, upper_arr)
            count = int(mask.sum() / 255)
            if name in color_counts:
                color_counts[name] += count
            else:
                color_counts[name] = count

        if not color_counts:
            return None

        # Get dominant
        dominant = max(color_counts, key=color_counts.get)
        total_pixels = small.shape[0] * small.shape[1]

        # If no color has at least 10% coverage, label as "mixed"
        if color_counts[dominant] < total_pixels * 0.10:
            return None

        return dominant

    except Exception as e:
        logger.debug("Color estimation error: %s", e)
        return None


def color_similarity_score(estimated: Optional[str],
                            target_colors: list[str]) -> float:
    """
    Return a score [0, 1] for how well an estimated color matches target colors.

    Handles:
    - Exact match: 1.0
    - Related colors: 0.5 (e.g., "orange" vs "red")
    - No match: 0.0
    """
    if not estimated or not target_colors:
        return 0.5  # neutral when no color info

    estimated = estimated.lower()
    target_colors = [c.lower() for c in target_colors]

    if estimated in target_colors:
        return 1.0

    # Related color groups
    related_groups = [
        {"red", "orange", "brown"},
        {"blue", "purple"},
        {"white", "gray"},
        {"black", "gray"},
        {"green"},
        {"yellow", "orange"},
    ]

    for group in related_groups:
        if estimated in group:
            for tc in target_colors:
                if tc in group:
                    return 0.5

    return 0.0


def get_color_rgb(color_name: str) -> Tuple[int, int, int]:
    """Get approximate RGB for a color name (for UI display)."""
    COLOR_RGB = {
        "red": (220, 50, 50),
        "orange": (255, 140, 0),
        "yellow": (255, 220, 0),
        "green": (50, 180, 50),
        "blue": (50, 100, 220),
        "purple": (130, 50, 180),
        "white": (240, 240, 240),
        "black": (30, 30, 30),
        "gray": (150, 150, 150),
        "brown": (139, 90, 43),
    }
    return COLOR_RGB.get(color_name.lower(), (128, 128, 128))
