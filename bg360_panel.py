# SPDX-FileCopyrightText: 2025
# SPDX-License-Identifier: GPL-3.0-or-later

"""360 Background Plugin for Lichtfeld Studio.

Loads an equirectangular (360) image and composites it behind the Gaussian splat.
Uses lf.run() to execute render_at in the main interpreter context where it works.
"""

from __future__ import annotations
import os
import math
import threading
import tempfile
import json
from pathlib import Path

import lichtfeld as lf


# ── Module-level state ────────────────────────────────────────────────────────

_bg_path    = ""
_enabled    = False
_result_path = ""   # path where the render script saves its output


def _load_equirect(path: str) -> str:
    """Validate the image is readable."""
    global _bg_path
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        w, h = img.size
        _bg_path = path
        lf.log.info(f"360 BG: image OK {path} — {w}x{h}")
        return ""
    except Exception as e:
        lf.log.error(f"360 BG: load error – {e}")
        return str(e)


def _write_render_script(params_path: str, out_path: str) -> str:
    """Write a self-contained Python script that render_at can run from."""
    script = f'''
import lichtfeld as lf
import numpy as np
import math
import json
from pathlib import Path

# Load params
with open({repr(params_path)}, "r") as f:
    p = json.load(f)

eye       = tuple(p["eye"])
target    = tuple(p["target"])
width     = p["width"]
height    = p["height"]
fov_x     = p["fov_x"]
up        = tuple(p["up"])
threshold = p["threshold"]
bg_path   = p["bg_path"]
out_path  = p["out_path"]

# Load equirectangular image
from PIL import Image
img    = Image.open(bg_path).convert("RGB")
eq_np  = np.array(img, dtype=np.float32) / 255.0
H_eq, W_eq = eq_np.shape[:2]

# Render splat with black and white backgrounds
bg_black = lf.Tensor.zeros((3,), device="cuda", dtype="float32")
bg_white = lf.Tensor.ones( (3,), device="cuda", dtype="float32")

r_black = lf.render_at(eye, target, width, height, fov_x, up=up, bg_color=bg_black)
r_white = lf.render_at(eye, target, width, height, fov_x, up=up, bg_color=bg_white)

if r_black is None or r_white is None:
    lf.log.error("360 BG script: render_at returned None")
else:
    rb   = r_black.cpu().numpy()
    rw   = r_white.cpu().numpy()
    diff = np.abs(rw - rb).sum(axis=-1)
    is_bg = diff > threshold

    # Build camera rotation matrix from eye/target/up
    fwd_v = np.array(target) - np.array(eye)
    fwd_v /= np.linalg.norm(fwd_v)
    up_v   = np.array(up, dtype=np.float64)
    right  = np.cross(fwd_v, up_v)
    if np.linalg.norm(right) < 1e-6:
        up_v  = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd_v, up_v)
    right   /= np.linalg.norm(right)
    true_up  = np.cross(right, fwd_v)
    true_up /= np.linalg.norm(true_up)
    R = np.stack([right, true_up, fwd_v], axis=1).astype(np.float32)

    # Project equirectangular to perspective frustum
    fov_x_r = math.radians(fov_x)
    fov_y_r = 2.0 * math.atan(math.tan(fov_x_r/2.0) * height / width)
    tx = math.tan(fov_x_r/2.0)
    ty = math.tan(fov_y_r/2.0)

    xs = np.linspace(-1.0, 1.0, width,  dtype=np.float32)
    ys = np.linspace( 1.0,-1.0, height, dtype=np.float32)
    xg, yg = np.meshgrid(xs, ys)
    dx = xg*tx;  dy = yg*ty;  dz = np.ones_like(dx)
    norm = np.sqrt(dx**2 + dy**2 + dz**2)
    dx /= norm;  dy /= norm;  dz /= norm

    rx = R[0,0]*dx + R[0,1]*dy + R[0,2]*dz
    ry = R[1,0]*dx + R[1,1]*dy + R[1,2]*dz
    rz = R[2,0]*dx + R[2,1]*dy + R[2,2]*dz

    lon = np.arctan2(rx, rz)
    lat = np.arcsin(np.clip(ry, -1.0, 1.0))
    u   = (lon / (2.0*math.pi) + 0.5) * (W_eq-1)
    v   = (0.5 - lat/math.pi)          * (H_eq-1)

    u0 = np.clip(np.floor(u).astype(np.int32), 0, W_eq-1)
    v0 = np.clip(np.floor(v).astype(np.int32), 0, H_eq-1)
    u1 = np.clip(u0+1, 0, W_eq-1)
    v1 = np.clip(v0+1, 0, H_eq-1)
    fu = (u - np.floor(u)).astype(np.float32)[:,:,np.newaxis]
    fv = (v - np.floor(v)).astype(np.float32)[:,:,np.newaxis]

    bg_proj = (eq_np[v0,u0]*(1-fu)*(1-fv) + eq_np[v0,u1]*fu*(1-fv) +
               eq_np[v1,u0]*(1-fu)*fv     + eq_np[v1,u1]*fu*fv).astype(np.float32)

    result = np.where(is_bg[:,:,np.newaxis], bg_proj, rb)
    result = np.clip(result, 0.0, 1.0)

    # Save result
    from PIL import Image as PILImage
    arr = (result * 255).astype(np.uint8)
    PILImage.fromarray(arr).save(out_path)
    lf.log.info(f"360 BG script: saved composite to {{out_path}}")
'''
    return script


# ── Panel ─────────────────────────────────────────────────────────────────────

class BG360Panel(lf.ui.Panel):
    id    = "bg360.panel"
    label = "360 Background"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 300

    def __init__(self):
        self._status         = ""
        self._pending_path   = None
        self._preview_tex    = None
        self._preview_w      = 512
        self._preview_h      = 512
        self._threshold      = 0.02
        self._rendering      = False
        self._pending_result = None   # path to saved PNG to load as preview
        self._script_path    = ""
        self._params_path    = ""
        self._out_path       = ""

    @classmethod
    def poll(cls, context) -> bool:
        return True

    def draw(self, ui):
        global _enabled, _bg_path

        ui.heading("360° Background")

        # ── Apply pending browse ──────────────────────────────────────────────
        if self._pending_path is not None:
            err = _load_equirect(self._pending_path)
            self._status       = f"Error: {err}" if err else f"Loaded: {Path(self._pending_path).name}"
            self._pending_path = None

        # ── Pick up completed render ──────────────────────────────────────────
        if self._pending_result is not None:
            result_png = self._pending_result
            self._pending_result = None
            if Path(result_png).exists():
                try:
                    from PIL import Image
                    import numpy as np
                    arr = np.array(Image.open(result_png).convert("RGB"),
                                   dtype=np.float32) / 255.0
                    t = lf.Tensor.from_numpy(arr).cuda()
                    if self._preview_tex is None:
                        self._preview_tex = lf.ui.DynamicTexture(t)
                    else:
                        self._preview_tex.update(t)
                    self._status    = "Preview rendered."
                    self._rendering = False
                except Exception as e:
                    self._status    = f"Load result error: {e}"
                    self._rendering = False
            else:
                self._status    = "Render failed — check log."
                self._rendering = False

        # ── Status ───────────────────────────────────────────────────────────
        if self._status:
            ui.label(self._status)

        # ── Enable toggle ─────────────────────────────────────────────────────
        changed, new_val = ui.checkbox("Enable 360 Background", _enabled)
        if changed:
            if new_val and not _bg_path:
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
        if _bg_path and lf.has_scene():
            if self._rendering:
                ui.text_disabled("Rendering…")
            else:
                if ui.button_styled("Render Preview", "primary"):
                    self._do_render(width=512, height=512, save=False)
                ui.same_line()
                if ui.button("Save Full Render"):
                    view = lf.get_current_view()
                    self._do_render(width=view.width, height=view.height, save=True)

            if self._preview_tex is not None:
                avail = ui.get_content_region_avail()
                w = int(avail[0])
                h = int(w * self._preview_h / max(1, self._preview_w))
                ui.image_texture(self._preview_tex, (w, h))

    def _do_render(self, width: int, height: int, save: bool):
        """Write params + script to temp files, run via lf.run(), poll for result."""
        global _bg_path
        try:
            view   = lf.get_current_view()
            rot    = view.rotation.cpu().numpy()
            fwd    = (float(rot[0,2]), float(rot[1,2]), float(rot[2,2]))
            pos    = view.position
            target = (pos[0]+fwd[0], pos[1]+fwd[1], pos[2]+fwd[2])

            tmp_dir = tempfile.gettempdir()
            params_path = os.path.join(tmp_dir, "bg360_params.json")
            script_path = os.path.join(tmp_dir, "bg360_render.py")
            out_path    = (str(Path(_bg_path).parent / "composite_render.png")
                           if save else
                           os.path.join(tmp_dir, "bg360_preview.png"))

            # Write params
            params = {
                "eye":       list(pos),
                "target":    list(target),
                "width":     width,
                "height":    height,
                "fov_x":     float(view.fov_x),
                "up":        [0.0, 1.0, 0.0],
                "threshold": self._threshold,
                "bg_path":   _bg_path,
                "out_path":  out_path,
            }
            with open(params_path, "w") as f:
                json.dump(params, f)

            # Write script
            script = _write_render_script(params_path, out_path)
            with open(script_path, "w") as f:
                f.write(script)

            self._rendering   = True
            self._status      = "Rendering…"
            self._preview_w   = width
            self._preview_h   = height

            # Run script in main interpreter, then poll for result in thread
            lf.run(script_path)

            # Poll for the output file in a background thread
            def _poll():
                import time
                for _ in range(60):   # wait up to 30s
                    if Path(out_path).exists():
                        self._pending_result = out_path
                        lf.ui.request_redraw()
                        return
                    time.sleep(0.5)
                self._status    = "Render timed out."
                self._rendering = False
                lf.ui.request_redraw()

            threading.Thread(target=_poll, daemon=True).start()

        except Exception as e:
            import traceback
            self._status    = f"Error: {e}"
            self._rendering = False
            lf.log.error(f"360 BG _do_render: {traceback.format_exc()}")
