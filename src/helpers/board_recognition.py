"""Board recognition stub using OpenCV."""

from __future__ import annotations

import cv2  # type: ignore

ENABLE_INTEGRATION = False


def locate_board(image_path: str) -> None:
    """Locate board on the screen (stub)."""
    if not ENABLE_INTEGRATION:
        return
    # TODO: implement template matching recognition
    _ = cv2.imread(image_path)
    print("Board recognition called (stub)")
