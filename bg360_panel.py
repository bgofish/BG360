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

_bg_tensor:    object = None   # [H, W, 3] CUDA float32 equirectangular image
_bg_path:      str    = ""
_preview_tex:  object = None   # DynamicTexture for panel preview
_enabled:      bool   = False


def _load_equirect(path: str):
    """Load equirectangular image to a CUDA float32 tensor [H, W, 3]."""
    global _bg_tensor, _bg_path
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        t   = lf.Tensor.from_numpy(arr).cuda()
        _bg_tensor = t
        _bg_path   = path
        lf.log.info(f"360 BG: loaded {path} — {arr.shape[1]}x{arr.shape[0]}")
        return ""
    except Exception as e:
        lf.log.error(f"360 BG: load error – {e}")
        return str(e)


def _sample_equirect(eq: object, rot_mat, fov_x_deg: float,
                     width: int, height: int) -> object:
    """
    Project equirectangular image to a perspective frustum.
    eq:      [H_eq, W_eq, 3] CUDA tensor
    rot_mat: [3, 3] CUDA tensor  (camera rotation, cols = right/up/forward)
    Returns: [height, width, 3] CUDA tensor
    """
    import math

    H_eq, W_eq = eq.shape[0], eq.shape[1]
    fov_x = math.radians(fov_x_deg)
    fov_y = 2.0 * math.atan(math.tan(fov_x / 2.0) * height / width)

    xs = lf.Tensor.linspace(-1.0, 1.0, width,  device='cuda', dtype='float32')
    ys = lf.Tensor.linspace( 1.0,-1.0, height, device='cuda', dtype='float32')

    tx = math.tan(fov_x / 2.0)
    ty = math.tan(fov_y / 2.0)

    import numpy as np
    xs_np = xs.numpy()
    ys_np = ys.numpy()
    xg, yg = np.meshgrid(xs_np, ys_np)
    zg     = np.ones_like(xg)
    dx     = xg * tx
    dy     = yg * ty
    dz     = zg
    norm   = np.sqrt(dx**2 + dy**2 + dz**2)
    dx /= norm;  dy /= norm;  dz /= norm

    R = rot_mat.cpu().numpy()
    rx = R[0,0]*dx + R[0,1]*dy + R[0,2]*dz
    ry = R[1,0]*dx + R[1,1]*dy + R[1,2]*dz
    rz = R[2,0]*dx + R[2,1]*dy + R[2,2]*dz

    lon = np.arctan2(rx, rz)
    lat = np.arcsin(np.clip(ry, -1.0, 1.0))

    u = (lon / (2.0 * math.pi) + 0.5) * (W_eq - 1)
    v = (0.5 - lat / math.pi)          * (H_eq - 1)

    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = u0 + 1
    v1 = v0 + 1
    fu = (u - u0).astype(np.float32)
    fv = (v - v0).astype(np.float32)

    u0 = np.clip(u0, 0, W_eq - 1)
    u1 = np.clip(u1, 0, W_eq - 1)
    v0 = np.clip(v0, 0, H_eq - 1)
    v1 = np.clip(v1, 0, H_eq - 1)

    eq_np = eq.cpu().numpy()

    c00 = eq_np[v0, u0]
    c10 = eq_np[v0, u1]
    c01 = eq_np[v1, u0]
    c11 = eq_np[v1, u1]

    fu = fu[:, :, np.newaxis]
    fv = fv[:, :, np.newaxis]

    result = (c00 * (1-fu) * (1-fv) +
              c10 *    fu  * (1-fv) +
              c01 * (1-fu) *    fv  +
              c11 *    fu  *    fv)

    return lf.Tensor.from_numpy(result.astype(np.float32)).cuda()


def composite_render(width: int, height: int,
                     fov_x: float,
                     eye: tuple, target: tuple,
                     up: tuple = (0.0, 1.0, 0.0),
                     threshold: float = 0.02) -> object | None:
    """
    Render splat with 360 background composite.
    Returns [H, W, 3] CUDA tensor or None on failure.
    """
    global _bg_tensor, _enabled
    if not _enabled or _bg_tensor is None:
        lf.log.warning(f"360 BG: composite_render guard — enabled={_enabled} tensor={'set' if _bg_tensor is not None else 'None'}")
        return None

    try:
        lf.log.info(f"360 BG: starting render {width}x{height} fov={fov_x} eye={eye} target={target}")

        bg_black = lf.Tensor.zeros((3,), device='cuda', dtype='float32')
        bg_white = lf.Tensor.ones( (3,), device='cuda', dtype='float32')

        lf.log.info("360 BG: calling render_at black...")
        r_black = lf.render_at(eye, target, width, height, fov_x,
                               up=up, bg_color=bg_black)
        lf.log.info(f"360 BG: r_black={'ok' if r_black is not None else 'None'}")

        lf.log.info("360 BG: calling render_at white...")
        r_white = lf.render_at(eye, target, width, height, fov_x,
                               up=up, bg_color=bg_white)
        lf.log.info(f"360 BG: r_white={'ok' if r_white is not None else 'None'}")

        if r_black is None or r_white is None:
            lf.log.error("360 BG: render_at returned None")
            return None

        import numpy as np
        rb = r_black.cpu().numpy()
        rw = r_white.cpu().numpy()
        diff = np.abs(rw - rb).sum(axis=-1)
        is_bg = diff > threshold

        fwd = np.array(target) - np.array(eye)
        fwd /= np.linalg.norm(fwd)
        up_v = np.array(up)
        right = np.cross(fwd, up_v)
        if np.linalg.norm(right) < 1e-6:
            up_v = np.array([0.0, 0.0, 1.0])
            right = np.cross(fwd, up_v)
        right /= np.linalg.norm(right)
        true_up = np.cross(right, fwd)
        true_up /= np.linalg.norm(true_up)
        R = np.stack([right, true_up, fwd], axis=1).astype(np.float32)
        rot_t = lf.Tensor.from_numpy(R).cuda()

        lf.log.info("360 BG: sampling equirect...")
        bg_proj = _sample_equirect(_bg_tensor, rot_t, fov_x, width, height)
        bg_np   = bg_proj.cpu().numpy()

        result = np.where(is_bg[:, :, np.newaxis], bg_np, rb)
        result = np.clip(result, 0.0, 1.0).astype(np.float32)
        lf.log.info("360 BG: composite done")
        return lf.Tensor.from_numpy(result).cuda()

    except Exception as e:
        lf.log.error(f"360 BG: composite error – {e}")
        import traceback
        lf.log.error(traceback.format_exc())
        return None


# ── Panel ─────────────────────────────────────────────────────────────────────

class BG360Panel(lf.ui.Panel):
    id       = "bg360.panel"
    label    = "360 Background"
    space    = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order    = 300

    def __init__(self):
        self._status          = ""
        self._pending_path    = None
        self._threshold       = 0.02
        self._preview_tensor  = None
        self._preview_tex     = None
        self._width           = 512
        self._height          = 512

    @classmethod
    def poll(cls, context) -> bool:
        return True

    def draw(self, ui):
        global _enabled, _bg_tensor, _bg_path

        ui.heading("360° Background")

        if self._pending_path is not None:
            err = _load_equirect(self._pending_path)
            if err:
                self._status = f"Error: {err}"
            else:
                self._status = f"Loaded: {Path(self._pending_path).name}"
                self._preview_tensor = None
            self._pending_path = None

        if self._status:
            ui.label(self._status)

        changed, new_val = ui.checkbox("Enable 360 Background", _enabled)
        if changed:
            _enabled = new_val
            if _enabled and _bg_tensor is None:
                self._status = "Load an image first."
                _enabled = False

        ui.separator()

        if _bg_path:
            ui.text_disabled(Path(_bg_path).name)
        else:
            ui.text_disabled("No image loaded")

        if ui.button("Browse Image"):
            initial = str(Path(_bg_path).parent) if _bg_path else os.path.expanduser("~")
            def _browse(initial=initial):
                try:
                    path = lf.ui.open_image_dialog(initial)
                    if path:
                        self._pending_path = path
                except Exception as e:
                    self._status = f"Browse error: {e}"
            threading.Thread(target=_browse, daemon=True).start()

        ui.separator()

        ui.label("Mask threshold")
        changed, new_t = ui.slider_float("##thresh", self._threshold, 0.001, 0.2)
        if changed:
            self._threshold = float(new_t)

        ui.separator()

        if _bg_tensor is not None and lf.has_scene():
            if ui.button_styled("Render Preview", "primary"):
                self._do_preview(ui)

            if self._preview_tex is not None:
                avail = ui.get_content_region_avail()
                w = int(avail[0])
                h = int(w * self._height / max(1, self._width))
                ui.image_texture(self._preview_tex, (w, h))

        ui.separator()

        if _bg_tensor is not None and lf.has_scene():
            if ui.button("Save Composite Render"):
                self._do_save()

    def _do_preview(self, ui):
        """Render a small composite and show in panel."""
        lf.log.info("360 BG: _do_preview called")
        try:
            view = lf.get_current_view()
            lf.log.info(f"360 BG: got view — fov_x={view.fov_x}")
            rot     = view.rotation.cpu().numpy()
            import numpy as np
            fwd     = (float(rot[0,2]), float(rot[1,2]), float(rot[2,2]))
            raw_pos = view.position
            pos     = (float(raw_pos[0]), float(raw_pos[1]), float(raw_pos[2]))
            target  = (pos[0]+fwd[0], pos[1]+fwd[1], pos[2]+fwd[2])
            lf.log.info(f"360 BG: pos={pos} target={target} fwd={fwd}")

            W, H = 512, 512
            result = composite_render(W, H, view.fov_x, pos, target,
                                      threshold=self._threshold)
            if result is None:
                self._status = "Render failed — check log."
                lf.log.error("360 BG: composite_render returned None in preview")
                return

            self._width  = W
            self._height = H

            if self._preview_tex is None:
                self._preview_tex = lf.ui.DynamicTexture(result)
            else:
                self._preview_tex.update(result)

            self._status = "Preview rendered."
            lf.log.info("360 BG: preview rendered ok")
        except Exception as e:
            self._status = f"Preview error: {e}"
            lf.log.error(f"360 BG preview error: {e}")
            import traceback
            lf.log.error(traceback.format_exc())

    def _do_save(self):
        """Save full-resolution composite render to file."""
        lf.log.info("360 BG: _do_save called")
        try:
            import numpy as np
            from PIL import Image

            if not _bg_path:
                self._status = "No image loaded."
                return

            view    = lf.get_current_view()
            rot     = view.rotation.cpu().numpy()
            fwd     = (float(rot[0,2]), float(rot[1,2]), float(rot[2,2]))
            raw_pos = view.position
            pos     = (float(raw_pos[0]), float(raw_pos[1]), float(raw_pos[2]))
            target  = (pos[0]+fwd[0], pos[1]+fwd[1], pos[2]+fwd[2])
            lf.log.info(f"360 BG: save pos={pos} target={target}")

            result = composite_render(
                view.width, view.height, view.fov_x, pos, target,
                threshold=self._threshold
            )
            if result is None:
                self._status = "Render failed — check log."
                return

            arr  = (result.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            img  = Image.fromarray(arr)
            out  = str(Path(_bg_path).parent / "composite_render.png")
            img.save(out)
            self._status = f"Saved: {out}"
            lf.log.info(f"360 BG: saved composite to {out}")

        except Exception as e:
            self._status = f"Save error: {e}"
            lf.log.error(f"360 BG save error: {e}")
            import traceback
            lf.log.error(traceback.format_exc())
