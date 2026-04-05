# SPDX-FileCopyrightText: 2025
# SPDX-License-Identifier: GPL-3.0-or-later

"""360 Background Plugin for Lichtfeld Studio.

Loads an equirectangular (360) image and composites it behind the Gaussian splat
using a dual-render mask (black bg vs white bg) for clean alpha separation.

render_at() is called from the main thread via lf.on_frame() to avoid crashes.
"""

from __future__ import annotations
import os
import math
import threading
from pathlib import Path

import lichtfeld as lf


# ── Module-level state ────────────────────────────────────────────────────────

_bg_tensor   = None
_bg_path     = ""
_enabled     = False

# Render request/result queue (main-thread safe via on_frame)
_render_request  = None   # dict with render params, set by panel
_render_result   = None   # numpy array, set by on_frame callback
_render_error    = None   # str error message
_render_pending  = False  # True while waiting for on_frame to execute


def _on_frame_render():
    """Called from main thread via lf.on_frame() — safe to call render_at."""
    global _render_request, _render_result, _render_error, _render_pending
    if _render_request is None:
        return
    req = _render_request
    _render_request = None

    try:
        import numpy as np
        eye       = req['eye']
        target    = req['target']
        width     = req['width']
        height    = req['height']
        fov_x     = req['fov_x']
        up        = req['up']
        threshold = req['threshold']

        bg_black = lf.Tensor.zeros((3,), device='cuda', dtype='float32')
        bg_white = lf.Tensor.ones( (3,), device='cuda', dtype='float32')

        r_black = lf.render_at(eye, target, width, height, fov_x,
                               up=up, bg_color=bg_black)
        r_white = lf.render_at(eye, target, width, height, fov_x,
                               up=up, bg_color=bg_white)

        if r_black is None or r_white is None:
            _render_error   = "render_at returned None"
            _render_pending = False
            lf.ui.request_redraw()
            return

        rb   = r_black.cpu().numpy()
        rw   = r_white.cpu().numpy()
        diff = np.abs(rw - rb).sum(axis=-1)
        is_bg = diff > threshold

        R     = _build_rotation(eye, target, up)
        bg_np = _sample_equirect(_bg_tensor, R, fov_x, width, height)

        result = np.where(is_bg[:, :, np.newaxis], bg_np, rb)
        _render_result  = np.clip(result, 0.0, 1.0).astype(np.float32)
        _render_error   = None

    except Exception as e:
        import traceback
        _render_error  = str(e)
        _render_result = None
        lf.log.error(f"360 BG on_frame render: {traceback.format_exc()}")

    _render_pending = False
    lf.stop_animation()   # stop the per-frame callback
    lf.ui.request_redraw()


def _load_equirect(path: str) -> str:
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
    import numpy as np
    H_eq, W_eq = eq.shape[0], eq.shape[1]
    fov_x = math.radians(fov_x_deg)
    fov_y = 2.0 * math.atan(math.tan(fov_x / 2.0) * height / width)
    tx    = math.tan(fov_x / 2.0)
    ty    = math.tan(fov_y / 2.0)

    xs = np.linspace(-1.0, 1.0, width,  dtype=np.float32)
    ys = np.linspace( 1.0,-1.0, height, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    dx = xg * tx;  dy = yg * ty;  dz = np.ones_like(dx)
    norm = np.sqrt(dx**2 + dy**2 + dz**2)
    dx /= norm;  dy /= norm;  dz /= norm

    rx = R[0,0]*dx + R[0,1]*dy + R[0,2]*dz
    ry = R[1,0]*dx + R[1,1]*dy + R[1,2]*dz
    rz = R[2,0]*dx + R[2,1]*dy + R[2,2]*dz

    lon = np.arctan2(rx, rz)
    lat = np.arcsin(np.clip(ry, -1.0, 1.0))
    u   = (lon / (2.0 * math.pi) + 0.5) * (W_eq - 1)
    v   = (0.5 - lat / math.pi)          * (H_eq - 1)

    u0  = np.clip(np.floor(u).astype(np.int32), 0, W_eq-1)
    v0  = np.clip(np.floor(v).astype(np.int32), 0, H_eq-1)
    u1  = np.clip(u0 + 1, 0, W_eq-1)
    v1  = np.clip(v0 + 1, 0, H_eq-1)
    fu  = (u - np.floor(u)).astype(np.float32)[:,:,np.newaxis]
    fv  = (v - np.floor(v)).astype(np.float32)[:,:,np.newaxis]

    eq_np = eq.cpu().numpy()
    return (eq_np[v0,u0]*(1-fu)*(1-fv) + eq_np[v0,u1]*fu*(1-fv) +
            eq_np[v1,u0]*(1-fu)*fv     + eq_np[v1,u1]*fu*fv).astype(np.float32)


def _build_rotation(eye, target, up):
    import numpy as np
    fwd = np.array(target, dtype=np.float64) - np.array(eye, dtype=np.float64)
    fwd /= np.linalg.norm(fwd)
    up_v = np.array(up, dtype=np.float64)
    right = np.cross(fwd, up_v)
    if np.linalg.norm(right) < 1e-6:
        up_v  = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, up_v)
    right   /= np.linalg.norm(right)
    true_up  = np.cross(right, fwd)
    true_up /= np.linalg.norm(true_up)
    return np.stack([right, true_up, fwd], axis=1).astype(np.float32)


# ── Panel ─────────────────────────────────────────────────────────────────────

class BG360Panel(lf.ui.Panel):
    id    = "bg360.panel"
    label = "360 Background"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 300

    def __init__(self):
        self._status       = ""
        self._pending_path = None
        self._preview_tex  = None
        self._preview_w    = 512
        self._preview_h    = 512
        self._threshold    = 0.02
        self._save_on_next = False   # if True, save result instead of preview

    @classmethod
    def poll(cls, context) -> bool:
        return True

    def draw(self, ui):
        global _enabled, _bg_tensor, _bg_path
        global _render_pending, _render_result, _render_error

        ui.heading("360° Background")

        # ── Apply pending image load (from browse thread) ─────────────────────
        if self._pending_path is not None:
            err = _load_equirect(self._pending_path)
            self._status       = f"Error: {err}" if err else f"Loaded: {Path(self._pending_path).name}"
            self._pending_path = None

        # ── Pick up completed render result ───────────────────────────────────
        if _render_result is not None and not _render_pending:
            arr = _render_result
            _render_result = None
            if self._save_on_next:
                self._save_on_next = False
                self._save_array(arr)
            else:
                t = lf.Tensor.from_numpy(arr).cuda()
                if self._preview_tex is None:
                    self._preview_tex = lf.ui.DynamicTexture(t)
                else:
                    self._preview_tex.update(t)
                self._status = "Preview rendered."

        if _render_error is not None and not _render_pending:
            self._status = f"Render error: {_render_error}"
            _render_error = None

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
            if _render_pending:
                ui.text_disabled("Rendering…")
            else:
                if ui.button_styled("Render Preview", "primary"):
                    self._request_render(save=False, width=512, height=512)
                ui.same_line()
                if ui.button("Save Full Render"):
                    view = lf.get_current_view()
                    self._request_render(save=True,
                                         width=view.width, height=view.height)

            if self._preview_tex is not None:
                avail = ui.get_content_region_avail()
                w = int(avail[0])
                h = int(w * self._preview_h / max(1, self._preview_w))
                ui.image_texture(self._preview_tex, (w, h))

    def _request_render(self, save: bool, width: int, height: int):
        global _render_request, _render_pending, _render_result, _render_error
        view   = lf.get_current_view()
        rot    = view.rotation.cpu().numpy()
        fwd    = (float(rot[0,2]), float(rot[1,2]), float(rot[2,2]))
        pos    = view.position
        target = (pos[0]+fwd[0], pos[1]+fwd[1], pos[2]+fwd[2])

        _render_result   = None
        _render_error    = None
        _render_pending  = True
        self._save_on_next = save
        self._preview_w    = width
        self._preview_h    = height
        self._status       = "Rendering…"

        _render_request = {
            'eye': pos, 'target': target,
            'width': width, 'height': height,
            'fov_x': view.fov_x,
            'up': (0.0, 1.0, 0.0),
            'threshold': self._threshold,
        }
        # Kick off per-frame callback on main thread
        lf.on_frame(_on_frame_render)

    def _save_array(self, arr):
        try:
            import numpy as np
            from PIL import Image
            out = str(Path(_bg_path).parent / "composite_render.png") if _bg_path \
                  else str(Path.home() / "composite_render.png")
            Image.fromarray((arr * 255).clip(0,255).astype(np.uint8)).save(out)
            self._status = f"Saved: {out}"
            lf.log.info(f"360 BG: saved to {out}")
        except Exception as e:
            self._status = f"Save error: {e}"
            lf.log.error(f"360 BG save: {e}")
