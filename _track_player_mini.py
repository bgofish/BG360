# SPDX-FileCopyrightText: 2025
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import math
from pathlib import Path


class TrackPlayer:
    """Loads a camera_track.json exported by 360_record and provides
    camera transforms for a given snapshot index."""

    def __init__(self, path: str):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        seg  = self._find_segment(data)
        if seg is None:
            raise ValueError("No usable segment found in camera_track.json")

        settings = data.get("settings", {})

        # Support both key-name variants written by different 360_record versions
        self.center    = tuple(seg.get("poi") or seg["center"])
        self.radius    = float(seg["radius"])
        self.elevation = float(seg.get("elevation", 0.0))
        self.arc_start = float(seg.get("start_angle", seg.get("arc_start", 0.0)))
        arc_range      = float(seg.get("arc_degrees", seg.get("arc_end", 360.0) - self.arc_start))
        self.arc_end   = self.arc_start + arc_range
        self.up_axis   = str(seg.get("orbit_axis", seg.get("up_axis", "y"))).lower()

        # FoV: prefer segment value, fall back to settings block
        self.fov = float(seg.get("fov", settings.get("fov", 45.0)))

    # ── public ────────────────────────────────────────────────────────────────

    def get_camera_at_snap(self, snap_index: int, arc_per_snap: float, loop: bool):
        """Return (eye, target, up, fov) for the given snapshot index."""
        arc_range = self.arc_end - self.arc_start
        offset    = snap_index * arc_per_snap

        if loop and arc_range > 0:
            angle = self.arc_start + (offset % arc_range)
        else:
            angle = min(self.arc_start + offset, self.arc_end)

        eye    = self._orbit_position(angle)
        target = self.center
        up     = (0.0, 1.0, 0.0) if self.up_axis == "y" else (0.0, 0.0, 1.0)
        return eye, target, up, self.fov

    @property
    def info(self) -> str:
        return (
            f"arc {self.arc_start:.0f}\u00b0\u2013{self.arc_end:.0f}\u00b0  "
            f"radius={self.radius:.3f}m  "
            f"elevation={self.elevation:.3f}m  "
            f"fov={self.fov:.1f}\u00b0"
        )

    # ── private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _find_segment(data: dict):
        """Return first segment regardless of type name."""
        segments = data.get("segments", [])
        return segments[0] if segments else None

    def _orbit_position(self, angle_deg: float):
        """Elevation is a metre offset along the up-axis from the POI."""
        theta = math.radians(angle_deg)
        cx, cy, cz = self.center
        r = self.radius
        e = self.elevation   # metres, not degrees

        if self.up_axis == "y":
            x = cx + r * math.cos(theta)
            y = cy + e
            z = cz + r * math.sin(theta)
        else:  # z-up
            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)
            z = cz + e

        return (x, y, z)
