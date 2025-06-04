"""Overlay stub for demonstration purposes."""

from __future__ import annotations

ENABLE_INTEGRATION = False  # Toggle for lab testing only


def attach_overlay() -> None:
    """Attach transparent overlay to the chess GUI (stub)."""
    if not ENABLE_INTEGRATION:
        return
    # TODO: implement platform-specific overlay (PyQt/Win32 GDI+) here
    print("Overlay attached (stub)")
