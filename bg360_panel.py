# SPDX-FileCopyrightText: 2025
# SPDX-License-Identifier: GPL-3.0-or-later

"""360 Background Plugin for Lichtfeld Studio.

Loads an equirectangular (360) image and composites it behind the Gaussian splat
using a dual-render mask (black bg vs white bg) for clean alpha separation.
"""

from __future__ import annotations
import os
import math
import threading
from pathlib import Path

import lichtfeld as lf


# ── Module-level state ────────────────────────────────────────────────────────

_bg_tensor   = None   # [H, W, 3] CUDA float32 equirectangular image
_bg_path     = ""
_enabled     = False


def _load_equirect(path: str) -> str:
    """Load equirectangular image to a CUDA float32 tensor [H, W, 3]."""
    global _bg_tensor, _bg_path
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        _bg_tensor = lf.Tensor.from_numpy(arr).cuda()
        _bg_path   = path
        lf.log.info(f"360 BG: loaded {path} — {arr.shape[1]}x{arr.shape[0]}")
        return ""
    except Exception as e:
        lf.log.error(f"360 BG: load error – {e}")
        return str(e)


def _sample_equirect(eq, R, fov_x_deg: float, width: int, height: int):
    """
    Project equirectangular image to a perspective frustum.
    eq: [H_eq, W_eq, 3] CUDA tensor
    R:  [3, 3] numpy array  (cols = right, up, forward)
    Returns [height, width, 3] float32 numpy array.
    """
    import numpy as np
    H_eq, W_eq = eq.shape[0], eq.shape[1]
    fov_x = math.radians(fov_x_deg)
    fov_y = 2.0 * math.atan(math.tan(fov_x / 2.0) * height / width)
    tx    = math.tan(fov_x / 2.0)
    ty    = math.tan(fov_y / 2.0)

    xs = np.linspace(-1.0, 1.0, width,  dtype=np.float32)
    ys = np.linspace( 1.0,-1.0, height, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)

    dx = xg * tx
    dy = yg * ty
    dz = np.ones_like(dx)
    norm = np.sqrt(dx**2 + dy**2 + dz**2)
    dx /= norm;  dy /= norm;  dz /= norm

    rx = R[0,0]*dx + R[0,1]*dy + R[0,2]*dz
    ry = R[1,0]*dx + R[1,1]*dy + R[1,2]*dz
    rz = R[2,0]*dx + R[2,1]*dy + R[2,2]*dz

    lon = np.arctan2(rx, rz)
    lat = np.arcsin(np.clip(ry, -1.0, 1.0))

    u = (lon / (2.0 * math.pi) + 0.5) * (W_eq - 1)
    v = (0.5 - lat / math.pi)          * (H_eq - 1)

    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = np.clip(u0 + 1, 0, W_eq - 1)
    v1 = np.clip(v0 + 1, 0, H_eq - 1)
    u0 = np.clip(u0, 0, W_eq - 1)
    v0 = np.clip(v0, 0, H_eq - 1)

    fu = (u - np.floor(u)).astype(np.float32)[:, :, np.newaxis]
    fv = (v - np.floor(v)).astype(np.float32)[:, :, np.newaxis]

    eq_np = eq.cpu().numpy()
    c00 = eq_np[v0, u0]
    c10 = eq_np[v0, u1]
    c01 = eq_np[v1, u0]
    c11 = eq_np[v1, u1]

    return (c00*(1-fu)*(1-fv) + c10*fu*(1-fv) +
            c01*(1-fu)*fv     + c11*fu*fv).astype(np.float32)


def _build_rotation(eye: tuple, target: tuple, up: tuple) -> object:
    """Build [3,3] rotation matrix from eye/target/up."""
    import numpy as np
    fwd = np.array(target, dtype=np.float64) - np.array(eye, dtype=np.float64)
    fwd /= np.linalg.norm(fwd)
    up_v = np.array(up, dtype=np.float64)
    right = np.cross(fwd, up_v)
    if np.linalg.norm(right) < 1e-6:
        up_v = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, up_v)
    right   /= np.linalg.norm(right)
    true_up  = np.cross(right, fwd)
    true_up /= np.linalg.norm(true_up)
    return np.stack([right, true_up, fwd], axis=1).astype(np.float32)


def composite_render(width: int, height: int, fov_x: float,
                     eye: tuple, target: tuple,
                     up: tuple = (0.0, 1.0, 0.0),
                     threshold: float = 0.02):
    """
    Render splat with 360 background composited.
    Returns [H, W, 3] float32 numpy array or None.
    """
    import numpy as np
    global _bg_tensor
    if _bg_tensor is None:
        return None

    bg_black = lf.Tensor.zeros((3,), device='cuda', dtype='float32')
    bg_white = lf.Tensor.ones( (3,), device='cuda', dtype='float32')

    r_black = lf.render_at(eye, target, width, height, fov_x,
                           up=up, bg_color=bg_black)
    r_white = lf.render_at(eye, target, width, height, fov_x,
                           up=up, bg_color=bg_white)

    if r_black is None or r_white is None:
        lf.log.error("360 BG: render_at returned None")
        return None

    rb   = r_black.cpu().numpy()
    rw   = r_white.cpu().numpy()
    diff = np.abs(rw - rb).sum(axis=-1)
    is_bg = diff > threshold

    R      = _build_rotation(eye, target, up)
    bg_np  = _sample_equirect(_bg_tensor, R, fov_x, width, height)
    result = np.where(is_bg[:, :, np.newaxis], bg_np, rb)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


# ── Panel ─────────────────────────────────────────────────────────────────────

class BG360Panel(lf.ui.Panel):
    id    = "bg360.panel"
    label = "360 Background"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 300

    def __init__(self):
        self._status         = ""
        self._pending_path   = None
        self._pending_result = None   # numpy array ready to upload to texture
        self._threshold      = 0.02
        self._preview_tex    = None
        self._preview_w      = 512
        self._preview_h      = 512
        self._rendering      = False

    @classmethod
    def poll(cls, context) -> bool:
        return True

    def draw(self, ui):
        global _enabled, _bg_tensor, _bg_path

        ui.heading("360° Background")

        # ── Apply pending image load ──────────────────────────────────────────
        if self._pending_path is not None:
            err = _load_equirect(self._pending_path)
            self._status       = f"Error: {err}" if err else f"Loaded: {Path(self._pending_path).name}"
            self._pending_path = None

        # ── Upload pending render result to texture ───────────────────────────
        if self._pending_result is not None:
            import numpy as np
            arr = self._pending_result
            self._pending_result = None
            t = lf.Tensor.from_numpy(arr).cuda()
            if self._preview_tex is None:
                self._preview_tex = lf.ui.DynamicTexture(t)
            else:
                self._preview_tex.update(t)
            self._status    = "Preview rendered."
            self._rendering = False
            lf.ui.request_redraw()

        # ── Status ───────────────────────────────────────────────────────────
        if self._status:
            ui.label(self._status)

        # ── Enable toggle ─────────────────────────────────────────────────────
        changed, new_val = ui.checkbox("Enable 360 Background", _enabled)
        if changed:
            if new_val and _bg_tensor is None:
                self._status = "Load an image first."
            else:
                _enabled = new_val

        ui.separator()

        # ── Image file ───────────────────────────────────────────────────────
        ui.text_disabled(Path(_bg_path).name if _bg_path else "No image loaded")

        if ui.button("Browse Image"):
            initial = str(Path(_bg_path).parent) if _bg_path else os.path.expanduser("~")
            def _browse(initial=initial):
                try:
                    path = lf.ui.open_image_dialog(initial)
                    if path:
                        self._pending_path = path
                        lf.ui.request_redraw()
                except Exception as e:
                    self._status = f"Browse error: {e}"
            threading.Thread(target=_browse, daemon=True).start()

        ui.separator()

        # ── Threshold ─────────────────────────────────────────────────────────
        ui.label("Mask threshold")
        changed, new_t = ui.slider_float("##thresh", self._threshold, 0.001, 0.2)
        if changed:
            self._threshold = float(new_t)

        ui.separator()

        # ── Preview & Save ────────────────────────────────────────────────────
        if _bg_tensor is not None and lf.has_scene():

            if self._rendering:
                ui.text_disabled("Rendering…")
            else:
                if ui.button_styled("Render Preview", "primary"):
                    self._start_preview_thread()

            if self._preview_tex is not None:
                avail = ui.get_content_region_avail()
                w = int(avail[0])
                h = int(w * self._preview_h / max(1, self._preview_w))
                ui.image_texture(self._preview_tex, (w, h))

            ui.separator()

            if not self._rendering:
                if ui.button("Save Composite Render"):
                    self._start_save_thread()

    def _get_camera(self):
        """Return (pos, target, fov_x, width, height) from current view."""
        import numpy as np
        view = lf.get_current_view()
        rot  = view.rotation.cpu().numpy()
        fwd  = (float(rot[0,2]), float(rot[1,2]), float(rot[2,2]))
        pos  = view.position
        target = (pos[0]+fwd[0], pos[1]+fwd[1], pos[2]+fwd[2])
        return pos, target, view.fov_x, view.width, view.height

    def _start_preview_thread(self):
        self._rendering = True
        self._status    = "Rendering preview…"
        pos, target, fov_x, _, _ = self._get_camera()
        threshold = self._threshold

        def _run():
            try:
                result = composite_render(
                    512, 512, fov_x, pos, target,
                    threshold=threshold
                )
                if result is None:
                    self._status   = "Render failed — check log."
                    self._rendering = False
                else:
                    self._preview_w      = 512
                    self._preview_h      = 512
                    self._pending_result = result
                lf.ui.request_redraw()
            except Exception as e:
                import traceback
                self._status    = f"Error: {e}"
                self._rendering = False
                lf.log.error(f"360 BG preview: {traceback.format_exc()}")
                lf.ui.request_redraw()

        threading.Thread(target=_run, daemon=True).start()

    def _start_save_thread(self):
        self._rendering = True
        self._status    = "Rendering full resolution…"
        pos, target, fov_x, W, H = self._get_camera()
        threshold = self._threshold
        out_path  = str(Path(_bg_path).parent / "composite_render.png") if _bg_path else ""

        def _run():
            try:
                import numpy as np
                from PIL import Image

                result = composite_render(W, H, fov_x, pos, target,
                                          threshold=threshold)
                if result is None:
                    self._status    = "Render failed — check log."
                    self._rendering = False
                    lf.ui.request_redraw()
                    return

                arr = (result * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(arr).save(out_path)
                self._status    = f"Saved: {out_path}"
                self._rendering = False
                lf.log.info(f"360 BG: saved to {out_path}")
                lf.ui.request_redraw()
            except Exception as e:
                import traceback
                self._status    = f"Save error: {e}"
                self._rendering = False
                lf.log.error(f"360 BG save: {traceback.format_exc()}")
                lf.ui.request_redraw()

        threading.Thread(target=_run, daemon=True).start()
