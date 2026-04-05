# SPDX-FileCopyrightText: 2025
# SPDX-License-Identifier: GPL-3.0-or-later

"""360 Background Plugin for Lichtfeld Studio.

Loads an equirectangular (360) image and composites it behind the Gaussian splat
using a dual-render mask (black bg vs white bg) for clean alpha separation.
Supports still image output at multiple resolutions and video output via LFS path.
"""

from __future__ import annotations
import os
import math
import threading
import subprocess
import sys
from pathlib import Path

import lichtfeld as lf


# ── Constants ─────────────────────────────────────────────────────────────────

_RESOLUTIONS = [
    (512,  512,  "Preview 512²"),
    (1280,  720, "720p  (1280×720)"),
    (1920, 1080, "1080p (1920×1080)"),
    (2560, 1440, "2K    (2560×1440)"),
    (3840, 2160, "4K    (3840×2160)"),
]

# ── Module-level state ────────────────────────────────────────────────────────

_bg_tensor = None   # [H, W, 3] CUDA float32 equirectangular image
_bg_path   = ""
_enabled   = False


def _request_redraw():
    for fn in ("tag_redraw", "request_redraw", "redraw", "invalidate"):
        f = getattr(lf.ui, fn, None)
        if callable(f):
            try:
                f()
                return
            except Exception:
                pass


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


def _sample_equirect(eq, rot_mat, fov_x_deg: float, width: int, height: int):
    import numpy as np
    H_eq, W_eq = eq.shape[0], eq.shape[1]
    fov_x = math.radians(fov_x_deg)
    fov_y = 2.0 * math.atan(math.tan(fov_x / 2.0) * height / width)
    tx = math.tan(fov_x / 2.0)
    ty = math.tan(fov_y / 2.0)

    xs = lf.Tensor.linspace(-1.0, 1.0, width,  device='cuda', dtype='float32')
    ys = lf.Tensor.linspace( 1.0,-1.0, height, device='cuda', dtype='float32')
    xs_np = xs.numpy();  ys_np = ys.numpy()
    xg, yg = np.meshgrid(xs_np, ys_np)
    dx = xg*tx;  dy = yg*ty;  dz = np.ones_like(xg)
    norm = np.sqrt(dx**2 + dy**2 + dz**2)
    dx /= norm;  dy /= norm;  dz /= norm

    R = rot_mat.cpu().numpy()
    rx = R[0,0]*dx + R[0,1]*dy + R[0,2]*dz
    ry = R[1,0]*dx + R[1,1]*dy + R[1,2]*dz
    rz = R[2,0]*dx + R[2,1]*dy + R[2,2]*dz

    lon = np.arctan2(rx, rz)
    lat = np.arcsin(np.clip(ry, -1.0, 1.0))
    u   = (lon / (2.0*math.pi) + 0.5) * (W_eq - 1)
    v   = (0.5 - lat / math.pi)        * (H_eq - 1)

    u0 = np.clip(np.floor(u).astype(np.int32), 0, W_eq-1)
    v0 = np.clip(np.floor(v).astype(np.int32), 0, H_eq-1)
    u1 = np.clip(u0+1, 0, W_eq-1);  v1 = np.clip(v0+1, 0, H_eq-1)
    fu = (u - np.floor(u)).astype(np.float32)[:,:,np.newaxis]
    fv = (v - np.floor(v)).astype(np.float32)[:,:,np.newaxis]

    eq_np = eq.cpu().numpy()
    return lf.Tensor.from_numpy(
        (eq_np[v0,u0]*(1-fu)*(1-fv) + eq_np[v0,u1]*fu*(1-fv) +
         eq_np[v1,u0]*(1-fu)*fv     + eq_np[v1,u1]*fu*fv).astype(np.float32)
    ).cuda()


def _build_rot_tensor(eye, target, up=(0,1,0)):
    import numpy as np
    fwd = np.array(target, dtype=np.float64) - np.array(eye, dtype=np.float64)
    fwd /= np.linalg.norm(fwd)
    up_v = np.array(up, dtype=np.float64)
    right = np.cross(fwd, up_v)
    if np.linalg.norm(right) < 1e-6:
        up_v = np.array([0.0, 0.0, 1.0]); right = np.cross(fwd, up_v)
    right   /= np.linalg.norm(right)
    true_up  = np.cross(right, fwd); true_up /= np.linalg.norm(true_up)
    R = np.stack([right, true_up, fwd], axis=1).astype(np.float32)
    return lf.Tensor.from_numpy(R).cuda()


def composite_render(width, height, fov_x, eye, target,
                     up=(0.0, 1.0, 0.0), threshold=0.02):
    """Returns [H,W,3] CUDA tensor or None."""
    global _bg_tensor
    if _bg_tensor is None:
        return None
    try:
        import numpy as np
        bg_black = lf.Tensor.zeros((3,), device='cuda', dtype='float32')
        bg_white = lf.Tensor.ones( (3,), device='cuda', dtype='float32')
        r_black  = lf.render_at(eye, target, width, height, fov_x,
                                up=up, bg_color=bg_black)
        r_white  = lf.render_at(eye, target, width, height, fov_x,
                                up=up, bg_color=bg_white)
        if r_black is None or r_white is None:
            return None
        rb    = r_black.cpu().numpy()
        rw    = r_white.cpu().numpy()
        is_bg = np.abs(rw - rb).sum(axis=-1) > threshold
        rot_t = _build_rot_tensor(eye, target, up)
        bg_np = _sample_equirect(_bg_tensor, rot_t, fov_x, width, height).cpu().numpy()
        result = np.where(is_bg[:,:,np.newaxis], bg_np, rb)
        return lf.Tensor.from_numpy(np.clip(result, 0, 1).astype(np.float32)).cuda()
    except Exception as e:
        import traceback
        lf.log.error(f"360 BG composite_render: {traceback.format_exc()}")
        return None


def _collate_video_pyav(frames_dir: Path, out_path: str, fps: int, width: int, height: int) -> str:
    try:
        import av
        container = av.open(out_path, mode='w')
        stream    = container.add_stream('h264', rate=fps)
        stream.width  = width
        stream.height = height
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '18', 'preset': 'slow'}

        import numpy as np
        from PIL import Image as PILImage
        pngs = sorted(frames_dir.glob("frame_*.png"))
        for png in pngs:
            arr   = np.array(PILImage.open(png).convert("RGB"))
            frame = av.VideoFrame.from_ndarray(arr, format='rgb24')
            for pkt in stream.encode(frame):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
        container.close()
        return ""
    except Exception as e:
        return str(e)


def _collate_video_ffmpeg(frames_dir: Path, out_path: str, fps: int) -> str:
    try:
        pattern = str(frames_dir / "frame_%06d.png")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", pattern,
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "slow",
            "-pix_fmt", "yuv420p",
            out_path
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        if result.returncode != 0:
            return result.stderr[-500:]
        return ""
    except Exception as e:
        return str(e)


# ── Panel ─────────────────────────────────────────────────────────────────────

class BG360Panel(lf.ui.Panel):
    id    = "bg360.panel"
    label = "360 Background"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 300

    def __init__(self):
        self._status          = ""
        self._pending_path    = None
        self._pending_lfs_path = None
        self._threshold       = 0.02
        self._preview_tex     = None
        self._preview_w       = 512
        self._preview_h       = 512
        # Resolution
        self._res_idx         = 0    # index into _RESOLUTIONS
        # Path type: "lfs" or "circular"
        self._path_type       = "lfs"
        # LFS path
        self._lfs_path        = ""
        self._lfs_player      = None
        self._lfs_status      = ""
        # Circular track
        self._circ_path       = ""
        self._circ_player     = None
        self._circ_status     = ""
        self._arc_per_snap    = 1.0
        self._circ_loop       = True
        self._pending_circ_path = None
        # Video
        self._fps             = 24
        self._fps_str         = "24"
        self._encoder_idx     = 0    # 0=PyAV, 1=FFmpeg
        self._video_expanded  = False
        self._video_status    = ""
        self._video_thread    = None
        self._video_stop      = False
        self._video_progress  = ""

    @classmethod
    def poll(cls, context) -> bool:
        return True

    # ── helpers ───────────────────────────────────────────────────────────────

    def _current_res(self):
        return _RESOLUTIONS[self._res_idx]

    def _get_camera(self):
        """Return (pos, target, fov_x) from current viewport."""
        import numpy as np
        view = lf.get_current_view()
        rot  = view.rotation.cpu().numpy()
        fwd  = (float(rot[0,2]), float(rot[1,2]), float(rot[2,2]))
        pos  = view.position
        pos  = (float(pos[0]), float(pos[1]), float(pos[2]))
        target = (pos[0]+fwd[0], pos[1]+fwd[1], pos[2]+fwd[2])
        return pos, target, float(view.fov_x)

    # ── draw ──────────────────────────────────────────────────────────────────

    def draw(self, ui):
        global _enabled, _bg_tensor, _bg_path

        ui.heading("360° Background")

        # Apply pending image load
        if self._pending_path is not None:
            err = _load_equirect(self._pending_path)
            self._status       = f"Error: {err}" if err else f"Loaded: {Path(self._pending_path).name}"
            self._pending_path = None

        # Apply pending LFS path load
        if self._pending_lfs_path is not None:
            self._load_lfs_path(self._pending_lfs_path)
            self._pending_lfs_path = None

        # Apply pending circular track load
        if self._pending_circ_path is not None:
            self._load_circ_path(self._pending_circ_path)
            self._pending_circ_path = None

        if self._status:
            ui.label(self._status)

        # Enable
        changed, new_val = ui.checkbox("Enable 360 Background", _enabled)
        if changed:
            if new_val and _bg_tensor is None:
                self._status = "Load an image first."
            else:
                _enabled = new_val

        ui.separator()

        # ── Image ─────────────────────────────────────────────────────────────
        ui.label("360° Image")
        ui.text_disabled(Path(_bg_path).name if _bg_path else "No image loaded")
        if ui.button("Browse Image##bg"):
            initial = str(Path(_bg_path).parent) if _bg_path else os.path.expanduser("~")
            def _browse(initial=initial):
                try:
                    path = lf.ui.open_image_dialog(initial)
                    if path:
                        self._pending_path = path
                        _request_redraw()
                except Exception as e:
                    self._status = f"Browse error: {e}"
            threading.Thread(target=_browse, daemon=True).start()

        ui.separator()

        # ── Mask threshold ────────────────────────────────────────────────────
        ui.label("Mask threshold")
        ui.same_line()
        changed, new_t = ui.slider_float("##thresh", self._threshold, 0.02, 2.0)
        if changed:
            self._threshold = float(new_t)

        ui.separator()

        # ── Resolution ────────────────────────────────────────────────────────
        ui.label("Output resolution")
        res_labels = [r[2] for r in _RESOLUTIONS]
        changed, new_idx = ui.combo("##res", self._res_idx, res_labels)
        if changed:
            self._res_idx = new_idx

        ui.separator()

        # ── Still render ──────────────────────────────────────────────────────
        if _bg_tensor is not None and lf.has_scene():
            W, H, _ = self._current_res()
            if ui.button_styled("Render Preview", "primary"):
                self._do_still(preview=True)
            ui.same_line()
            if ui.button(f"Save {W}×{H}##still"):
                self._do_still(preview=False)

            if self._preview_tex is not None:
                avail = ui.get_content_region_avail()
                pw = int(avail[0])
                ph = int(pw * self._preview_h / max(1, self._preview_w))
                ui.image_texture(self._preview_tex, (pw, ph))

        ui.separator()

        # ── Video section ─────────────────────────────────────────────────────
        arrow = "▼" if self._video_expanded else "▶"
        if ui.button(f"{arrow} Video Output##vtoggle"):
            self._video_expanded = not self._video_expanded

        if self._video_expanded:
            self._draw_video_section(ui)

    def _draw_video_section(self, ui):
        ui.separator()

        # Path type selector
        ui.label("Camera path type")
        changed, new_idx = ui.combo("##pathtype", 0 if self._path_type == "lfs" else 1,
                                    ["LFS Camera Path", "Circular Track"])
        if changed:
            self._path_type = "lfs" if new_idx == 0 else "circular"

        ui.separator()

        if self._path_type == "lfs":
            # LFS path
            ui.label("LFS Camera Path")
            ui.text_disabled(Path(self._lfs_path).name if self._lfs_path else "No path loaded")
            if ui.button("Browse Path##lfs"):
                def _browse():
                    try:
                        path = lf.ui.open_json_file_dialog()
                        if path:
                            self._pending_lfs_path = path
                            _request_redraw()
                    except Exception as e:
                        self._lfs_status = f"Browse error: {e}"
                threading.Thread(target=_browse, daemon=True).start()
            if self._lfs_status:
                ui.label(self._lfs_status)

        else:
            # Circular track
            ui.label("Circular Track JSON")
            ui.text_disabled(Path(self._circ_path).name if self._circ_path else "No track loaded")
            if ui.button("Browse Track##circ"):
                def _browse():
                    try:
                        path = lf.ui.open_json_file_dialog()
                        if path:
                            self._pending_circ_path = path
                            _request_redraw()
                    except Exception as e:
                        self._circ_status = f"Browse error: {e}"
                threading.Thread(target=_browse, daemon=True).start()
            if self._circ_status:
                ui.label(self._circ_status)

            # Arc per snap
            ui.label("Arc per frame (°)")
            ui.same_line()
            changed, new_arc = ui.slider_float("##arc", self._arc_per_snap, 0.02, 2.0)
            if changed:
                self._arc_per_snap = max(0.02, float(new_arc))

            # Loop
            changed, new_loop = ui.checkbox("Loop track##circloop", self._circ_loop)
            if changed:
                self._circ_loop = new_loop

        ui.separator()

        # FPS
        ui.label("Frame rate (fps)")
        changed, new_fps = ui.input_int("##fps", self._fps, 1, 10)
        if changed:
            self._fps = max(1, min(120, new_fps))

        # Encoder
        ui.label("Encoder")
        changed, new_enc = ui.combo("##enc", self._encoder_idx,
                                    ["PyAV (built-in) — MKV", "FFmpeg (external) — MP4"])
        if changed:
            self._encoder_idx = new_enc

        ui.separator()

        # Video status / progress
        if self._video_progress:
            ui.label(self._video_progress)
        if self._video_status:
            ui.label(self._video_status)

        # Work out active player and frame count
        active_player = None
        total_frames  = 0
        W, H, _       = self._current_res()

        if self._path_type == "lfs" and self._lfs_player is not None:
            active_player = self._lfs_player
            dur = self._lfs_player.total_duration
            total_frames = int(dur * self._fps)
            ui.text_disabled(f"~{total_frames} frames @ {self._fps}fps  ({dur:.1f}s)")
        elif self._path_type == "circular" and self._circ_player is not None:
            active_player = self._circ_player
            arc_range = abs(self._circ_player.arc_end - self._circ_player.arc_start)
            total_frames = int(arc_range / max(0.001, self._arc_per_snap))
            ui.text_disabled(f"~{total_frames} frames  ({arc_range:.0f}° arc)")

        if _bg_tensor is not None and active_player is not None and lf.has_scene():
            if self._video_thread is not None and self._video_thread.is_alive():
                if ui.button_styled("Stop Video##vstop", "error"):
                    self._video_stop = True
            else:
                if ui.button_styled(f"Render Video {W}×{H}##vrender", "primary"):
                    self._do_video(W, H, active_player, total_frames)

    # ── Still render ──────────────────────────────────────────────────────────

    def _do_still(self, preview: bool):
        try:
            import numpy as np
            from PIL import Image

            W, H, _ = self._current_res()
            pos, target, fov_x = self._get_camera()

            result = composite_render(W, H, fov_x, pos, target,
                                      threshold=self._threshold)
            if result is None:
                self._status = "Render failed — check log."
                return

            arr = (result.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

            if preview:
                t = lf.Tensor.from_numpy(arr.astype(np.float32) / 255.0).cuda()
                if self._preview_tex is None:
                    self._preview_tex = lf.ui.DynamicTexture(t)
                else:
                    self._preview_tex.update(t)
                self._preview_w = W
                self._preview_h = H
                self._status = f"Preview rendered ({W}×{H})."
            else:
                out = str(Path(_bg_path).parent / f"composite_{W}x{H}.png")
                Image.fromarray(arr).save(out)
                self._status = f"Saved: {out}"
                lf.log.info(f"360 BG: saved {out}")

        except Exception as e:
            import traceback
            self._status = f"Error: {e}"
            lf.log.error(f"360 BG still: {traceback.format_exc()}")
        _request_redraw()

    # ── LFS path load ─────────────────────────────────────────────────────────

    def _load_lfs_path(self, path: str):
        try:
            import sys, os
            plugin_dir = os.path.dirname(os.path.abspath(__file__))
            # Try importing LFSPathPlayer from the training render plugin
            # Fall back to bundled minimal player
            try:
                sys.path.insert(0, plugin_dir)
                from lfs_path_player import LFSPathPlayer
            except ImportError:
                from _lfs_player_mini import LFSPathPlayer

            self._lfs_player = LFSPathPlayer(path)
            self._lfs_path   = path
            self._lfs_status = (f"✓ Loaded — {self._lfs_player.n_keyframes} keyframes "
                                f"| {self._lfs_player.total_duration:.1f}s")
            lf.log.info(f"360 BG: LFS path loaded {path}")
        except Exception as e:
            self._lfs_player = None
            self._lfs_status = f"✗ Load error: {e}"
            lf.log.error(f"360 BG LFS load: {e}")

    def _load_circ_path(self, path: str):
        try:
            import sys
            plugin_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, plugin_dir)
            try:
                from track_player import TrackPlayer
            except ImportError:
                from _track_player_mini import TrackPlayer
            self._circ_player = TrackPlayer(path)
            self._circ_path   = path
            self._circ_status = f"✓ Loaded — {self._circ_player.info}"
            lf.log.info(f"360 BG: circular track loaded {path}")
        except Exception as e:
            self._circ_player = None
            self._circ_status = f"✗ Load error: {e}"
            lf.log.error(f"360 BG circular track load: {e}")

    # ── Video render ──────────────────────────────────────────────────────────

    def _do_video(self, width: int, height: int, player, total_frames: int):
        import tempfile
        self._video_stop     = False
        self._video_status   = ""
        self._video_progress = "Starting…"

        fps        = self._fps
        threshold  = self._threshold
        encoder    = self._encoder_idx
        path_type  = self._path_type
        arc_per_snap = self._arc_per_snap
        circ_loop  = self._circ_loop
        out_dir    = Path(_bg_path).parent if _bg_path else Path.home()
        ext        = "mkv" if encoder == 0 else "mp4"
        out_video  = str(out_dir / f"bg360_video_{width}x{height}.{ext}")
        frames_dir = Path(tempfile.mkdtemp(prefix="bg360_frames_"))

        def _run():
            try:
                import numpy as np
                from PIL import Image

                for i in range(total_frames):
                    if self._video_stop:
                        self._video_progress = "Stopped."
                        _request_redraw()
                        return

                    if path_type == "lfs":
                        secs_per_frame = 1.0 / fps
                        pos, target_pt, up_vec, fov = player.get_camera_at_snap(
                            i, secs_per_frame, loop=False
                        )
                    else:  # circular
                        pos, target_pt, up_vec, fov = player.get_camera_at_snap(
                            i, arc_per_snap, loop=circ_loop
                        )

                    # fov from LFSPathPlayer is in degrees already
                    # fov from TrackPlayer is also degrees
                    result = composite_render(
                        width, height, float(fov),
                        pos, target_pt, up=up_vec,
                        threshold=threshold
                    )

                    if result is None:
                        self._video_progress = f"Frame {i} failed — skipping."
                        continue

                    arr = (result.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    Image.fromarray(arr).save(frames_dir / f"frame_{i:06d}.png")

                    pct = int((i + 1) / total_frames * 100)
                    self._video_progress = f"Rendering frame {i+1}/{total_frames} ({pct}%)"
                    _request_redraw()

                # Collate
                self._video_progress = "Collating video…"
                _request_redraw()

                if encoder == 0:
                    err = _collate_video_pyav(frames_dir, out_video, fps, width, height)
                else:
                    err = _collate_video_ffmpeg(frames_dir, out_video, fps)

                if err:
                    self._video_status   = f"✗ Collate error: {err}"
                else:
                    self._video_status   = f"✓ Saved: {out_video}"
                    lf.log.info(f"360 BG: video saved to {out_video}")

                self._video_progress = ""

                try:
                    import shutil
                    shutil.rmtree(frames_dir)
                except Exception:
                    pass

            except Exception as e:
                import traceback
                self._video_status   = f"✗ Error: {e}"
                self._video_progress = ""
                lf.log.error(f"360 BG video: {traceback.format_exc()}")

            _request_redraw()

        self._video_thread = threading.Thread(target=_run, daemon=True)
        self._video_thread.start()# SPDX-FileCopyrightText: 2025
# SPDX-License-Identifier: GPL-3.0-or-later

"""360 Background Plugin for Lichtfeld Studio.

Loads an equirectangular (360) image and composites it behind the Gaussian splat
using a dual-render mask (black bg vs white bg) for clean alpha separation.
Supports still image output at multiple resolutions and video output via LFS path.
"""

from __future__ import annotations
import os
import math
import threading
import subprocess
import sys
from pathlib import Path

import lichtfeld as lf


# ── Constants ─────────────────────────────────────────────────────────────────

_RESOLUTIONS = [
    (512,  512,  "Preview 512²"),
    (1280,  720, "720p  (1280×720)"),
    (1920, 1080, "1080p (1920×1080)"),
    (2560, 1440, "2K    (2560×1440)"),
    (3840, 2160, "4K    (3840×2160)"),
]

# ── Module-level state ────────────────────────────────────────────────────────

_bg_tensor = None   # [H, W, 3] CUDA float32 equirectangular image
_bg_path   = ""
_enabled   = False


def _request_redraw():
    for fn in ("tag_redraw", "request_redraw", "redraw", "invalidate"):
        f = getattr(lf.ui, fn, None)
        if callable(f):
            try:
                f()
                return
            except Exception:
                pass


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


def _sample_equirect(eq, rot_mat, fov_x_deg: float, width: int, height: int):
    import numpy as np
    H_eq, W_eq = eq.shape[0], eq.shape[1]
    fov_x = math.radians(fov_x_deg)
    fov_y = 2.0 * math.atan(math.tan(fov_x / 2.0) * height / width)
    tx = math.tan(fov_x / 2.0)
    ty = math.tan(fov_y / 2.0)

    xs = lf.Tensor.linspace(-1.0, 1.0, width,  device='cuda', dtype='float32')
    ys = lf.Tensor.linspace( 1.0,-1.0, height, device='cuda', dtype='float32')
    xs_np = xs.numpy();  ys_np = ys.numpy()
    xg, yg = np.meshgrid(xs_np, ys_np)
    dx = xg*tx;  dy = yg*ty;  dz = np.ones_like(xg)
    norm = np.sqrt(dx**2 + dy**2 + dz**2)
    dx /= norm;  dy /= norm;  dz /= norm

    R = rot_mat.cpu().numpy()
    rx = R[0,0]*dx + R[0,1]*dy + R[0,2]*dz
    ry = R[1,0]*dx + R[1,1]*dy + R[1,2]*dz
    rz = R[2,0]*dx + R[2,1]*dy + R[2,2]*dz

    lon = np.arctan2(rx, rz)
    lat = np.arcsin(np.clip(ry, -1.0, 1.0))
    u   = (lon / (2.0*math.pi) + 0.5) * (W_eq - 1)
    v   = (0.5 - lat / math.pi)        * (H_eq - 1)

    u0 = np.clip(np.floor(u).astype(np.int32), 0, W_eq-1)
    v0 = np.clip(np.floor(v).astype(np.int32), 0, H_eq-1)
    u1 = np.clip(u0+1, 0, W_eq-1);  v1 = np.clip(v0+1, 0, H_eq-1)
    fu = (u - np.floor(u)).astype(np.float32)[:,:,np.newaxis]
    fv = (v - np.floor(v)).astype(np.float32)[:,:,np.newaxis]

    eq_np = eq.cpu().numpy()
    return lf.Tensor.from_numpy(
        (eq_np[v0,u0]*(1-fu)*(1-fv) + eq_np[v0,u1]*fu*(1-fv) +
         eq_np[v1,u0]*(1-fu)*fv     + eq_np[v1,u1]*fu*fv).astype(np.float32)
    ).cuda()


def _build_rot_tensor(eye, target, up=(0,1,0)):
    import numpy as np
    fwd = np.array(target, dtype=np.float64) - np.array(eye, dtype=np.float64)
    fwd /= np.linalg.norm(fwd)
    up_v = np.array(up, dtype=np.float64)
    right = np.cross(fwd, up_v)
    if np.linalg.norm(right) < 1e-6:
        up_v = np.array([0.0, 0.0, 1.0]); right = np.cross(fwd, up_v)
    right   /= np.linalg.norm(right)
    true_up  = np.cross(right, fwd); true_up /= np.linalg.norm(true_up)
    R = np.stack([right, true_up, fwd], axis=1).astype(np.float32)
    return lf.Tensor.from_numpy(R).cuda()


def composite_render(width, height, fov_x, eye, target,
                     up=(0.0, 1.0, 0.0), threshold=0.02):
    """Returns [H,W,3] CUDA tensor or None."""
    global _bg_tensor
    if _bg_tensor is None:
        return None
    try:
        import numpy as np
        bg_black = lf.Tensor.zeros((3,), device='cuda', dtype='float32')
        bg_white = lf.Tensor.ones( (3,), device='cuda', dtype='float32')
        r_black  = lf.render_at(eye, target, width, height, fov_x,
                                up=up, bg_color=bg_black)
        r_white  = lf.render_at(eye, target, width, height, fov_x,
                                up=up, bg_color=bg_white)
        if r_black is None or r_white is None:
            return None
        rb    = r_black.cpu().numpy()
        rw    = r_white.cpu().numpy()
        is_bg = np.abs(rw - rb).sum(axis=-1) > threshold
        rot_t = _build_rot_tensor(eye, target, up)
        bg_np = _sample_equirect(_bg_tensor, rot_t, fov_x, width, height).cpu().numpy()
        result = np.where(is_bg[:,:,np.newaxis], bg_np, rb)
        return lf.Tensor.from_numpy(np.clip(result, 0, 1).astype(np.float32)).cuda()
    except Exception as e:
        import traceback
        lf.log.error(f"360 BG composite_render: {traceback.format_exc()}")
        return None


def _collate_video_pyav(frames_dir: Path, out_path: str, fps: int, width: int, height: int) -> str:
    try:
        import av
        container = av.open(out_path, mode='w')
        stream    = container.add_stream('h264', rate=fps)
        stream.width  = width
        stream.height = height
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '18', 'preset': 'slow'}

        import numpy as np
        from PIL import Image as PILImage
        pngs = sorted(frames_dir.glob("frame_*.png"))
        for png in pngs:
            arr   = np.array(PILImage.open(png).convert("RGB"))
            frame = av.VideoFrame.from_ndarray(arr, format='rgb24')
            for pkt in stream.encode(frame):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
        container.close()
        return ""
    except Exception as e:
        return str(e)


def _collate_video_ffmpeg(frames_dir: Path, out_path: str, fps: int) -> str:
    try:
        pattern = str(frames_dir / "frame_%06d.png")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", pattern,
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "slow",
            "-pix_fmt", "yuv420p",
            out_path
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        if result.returncode != 0:
            return result.stderr[-500:]
        return ""
    except Exception as e:
        return str(e)


# ── Panel ─────────────────────────────────────────────────────────────────────

class BG360Panel(lf.ui.Panel):
    id    = "bg360.panel"
    label = "360 Background"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 300

    def __init__(self):
        self._status          = ""
        self._pending_path    = None
        self._pending_lfs_path = None
        self._threshold       = 0.02
        self._preview_tex     = None
        self._preview_w       = 512
        self._preview_h       = 512
        # Resolution
        self._res_idx         = 0    # index into _RESOLUTIONS
        # Path type: "lfs" or "circular"
        self._path_type       = "lfs"
        # LFS path
        self._lfs_path        = ""
        self._lfs_player      = None
        self._lfs_status      = ""
        # Circular track
        self._circ_path       = ""
        self._circ_player     = None
        self._circ_status     = ""
        self._arc_per_snap    = 1.0
        self._circ_loop       = True
        self._pending_circ_path = None
        # Video
        self._fps             = 24
        self._fps_str         = "24"
        self._encoder_idx     = 0    # 0=PyAV, 1=FFmpeg
        self._video_expanded  = False
        self._video_status    = ""
        self._video_thread    = None
        self._video_stop      = False
        self._video_progress  = ""

    @classmethod
    def poll(cls, context) -> bool:
        return True

    # ── helpers ───────────────────────────────────────────────────────────────

    def _current_res(self):
        return _RESOLUTIONS[self._res_idx]

    def _get_camera(self):
        """Return (pos, target, fov_x) from current viewport."""
        import numpy as np
        view = lf.get_current_view()
        rot  = view.rotation.cpu().numpy()
        fwd  = (float(rot[0,2]), float(rot[1,2]), float(rot[2,2]))
        pos  = view.position
        pos  = (float(pos[0]), float(pos[1]), float(pos[2]))
        target = (pos[0]+fwd[0], pos[1]+fwd[1], pos[2]+fwd[2])
        return pos, target, float(view.fov_x)

    # ── draw ──────────────────────────────────────────────────────────────────

    def draw(self, ui):
        global _enabled, _bg_tensor, _bg_path

        ui.heading("360° Background")

        # Apply pending image load
        if self._pending_path is not None:
            err = _load_equirect(self._pending_path)
            self._status       = f"Error: {err}" if err else f"Loaded: {Path(self._pending_path).name}"
            self._pending_path = None

        # Apply pending LFS path load
        if self._pending_lfs_path is not None:
            self._load_lfs_path(self._pending_lfs_path)
            self._pending_lfs_path = None

        # Apply pending circular track load
        if self._pending_circ_path is not None:
            self._load_circ_path(self._pending_circ_path)
            self._pending_circ_path = None

        if self._status:
            ui.label(self._status)

        # Enable
        changed, new_val = ui.checkbox("Enable 360 Background", _enabled)
        if changed:
            if new_val and _bg_tensor is None:
                self._status = "Load an image first."
            else:
                _enabled = new_val

        ui.separator()

        # ── Image ─────────────────────────────────────────────────────────────
        ui.label("360° Image")
        ui.text_disabled(Path(_bg_path).name if _bg_path else "No image loaded")
        if ui.button("Browse Image##bg"):
            initial = str(Path(_bg_path).parent) if _bg_path else os.path.expanduser("~")
            def _browse(initial=initial):
                try:
                    path = lf.ui.open_image_dialog(initial)
                    if path:
                        self._pending_path = path
                        _request_redraw()
                except Exception as e:
                    self._status = f"Browse error: {e}"
            threading.Thread(target=_browse, daemon=True).start()

        ui.separator()

        # ── Mask threshold ────────────────────────────────────────────────────
        ui.label("Mask threshold")
        changed, new_t = ui.slider_float("##thresh", self._threshold, 0.001, 0.2)
        if changed:
            self._threshold = float(new_t)

        ui.separator()

        # ── Resolution ────────────────────────────────────────────────────────
        ui.label("Output resolution")
        res_labels = [r[2] for r in _RESOLUTIONS]
        changed, new_idx = ui.combo("##res", self._res_idx, res_labels)
        if changed:
            self._res_idx = new_idx

        ui.separator()

        # ── Still render ──────────────────────────────────────────────────────
        if _bg_tensor is not None and lf.has_scene():
            W, H, _ = self._current_res()
            if ui.button_styled("Render Preview", "primary"):
                self._do_still(preview=True)
            ui.same_line()
            if ui.button(f"Save {W}×{H}##still"):
                self._do_still(preview=False)

            if self._preview_tex is not None:
                avail = ui.get_content_region_avail()
                pw = int(avail[0])
                ph = int(pw * self._preview_h / max(1, self._preview_w))
                ui.image_texture(self._preview_tex, (pw, ph))

        ui.separator()

        # ── Video section ─────────────────────────────────────────────────────
        arrow = "▼" if self._video_expanded else "▶"
        if ui.button(f"{arrow} Video Output##vtoggle"):
            self._video_expanded = not self._video_expanded

        if self._video_expanded:
            self._draw_video_section(ui)

    def _draw_video_section(self, ui):
        ui.separator()

        # Path type selector
        ui.label("Camera path type")
        changed, new_idx = ui.combo("##pathtype", 0 if self._path_type == "lfs" else 1,
                                    ["LFS Camera Path", "Circular Track"])
        if changed:
            self._path_type = "lfs" if new_idx == 0 else "circular"

        ui.separator()

        if self._path_type == "lfs":
            # LFS path
            ui.label("LFS Camera Path")
            ui.text_disabled(Path(self._lfs_path).name if self._lfs_path else "No path loaded")
            if ui.button("Browse Path##lfs"):
                def _browse():
                    try:
                        path = lf.ui.open_json_file_dialog()
                        if path:
                            self._pending_lfs_path = path
                            _request_redraw()
                    except Exception as e:
                        self._lfs_status = f"Browse error: {e}"
                threading.Thread(target=_browse, daemon=True).start()
            if self._lfs_status:
                ui.label(self._lfs_status)

        else:
            # Circular track
            ui.label("Circular Track JSON")
            ui.text_disabled(Path(self._circ_path).name if self._circ_path else "No track loaded")
            if ui.button("Browse Track##circ"):
                def _browse():
                    try:
                        path = lf.ui.open_json_file_dialog()
                        if path:
                            self._pending_circ_path = path
                            _request_redraw()
                    except Exception as e:
                        self._circ_status = f"Browse error: {e}"
                threading.Thread(target=_browse, daemon=True).start()
            if self._circ_status:
                ui.label(self._circ_status)

            # Arc per snap
            ui.label("Arc per frame (°)")
            changed, new_arc = ui.drag_float("##arc", self._arc_per_snap, 0.1, 0.01, 360.0)
            if changed:
                self._arc_per_snap = max(0.01, float(new_arc))

            # Loop
            changed, new_loop = ui.checkbox("Loop track##circloop", self._circ_loop)
            if changed:
                self._circ_loop = new_loop

        ui.separator()

        # FPS
        ui.label("Frame rate (fps)")
        changed, new_fps = ui.input_int("##fps", self._fps, 1, 10)
        if changed:
            self._fps = max(1, min(120, new_fps))

        # Encoder
        ui.label("Encoder")
        changed, new_enc = ui.combo("##enc", self._encoder_idx,
                                    ["PyAV (built-in) — MKV", "FFmpeg (external) — MP4"])
        if changed:
            self._encoder_idx = new_enc

        ui.separator()

        # Video status / progress
        if self._video_progress:
            ui.label(self._video_progress)
        if self._video_status:
            ui.label(self._video_status)

        # Work out active player and frame count
        active_player = None
        total_frames  = 0
        W, H, _       = self._current_res()

        if self._path_type == "lfs" and self._lfs_player is not None:
            active_player = self._lfs_player
            dur = self._lfs_player.total_duration
            total_frames = int(dur * self._fps)
            ui.text_disabled(f"~{total_frames} frames @ {self._fps}fps  ({dur:.1f}s)")
        elif self._path_type == "circular" and self._circ_player is not None:
            active_player = self._circ_player
            arc_range = abs(self._circ_player.arc_end - self._circ_player.arc_start)
            total_frames = int(arc_range / max(0.001, self._arc_per_snap))
            ui.text_disabled(f"~{total_frames} frames  ({arc_range:.0f}° arc)")

        if _bg_tensor is not None and active_player is not None and lf.has_scene():
            if self._video_thread is not None and self._video_thread.is_alive():
                if ui.button_styled("Stop Video##vstop", "error"):
                    self._video_stop = True
            else:
                if ui.button_styled(f"Render Video {W}×{H}##vrender", "primary"):
                    self._do_video(W, H, active_player, total_frames)

    # ── Still render ──────────────────────────────────────────────────────────

    def _do_still(self, preview: bool):
        try:
            import numpy as np
            from PIL import Image

            W, H, _ = self._current_res()
            pos, target, fov_x = self._get_camera()

            result = composite_render(W, H, fov_x, pos, target,
                                      threshold=self._threshold)
            if result is None:
                self._status = "Render failed — check log."
                return

            arr = (result.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

            if preview:
                t = lf.Tensor.from_numpy(arr.astype(np.float32) / 255.0).cuda()
                if self._preview_tex is None:
                    self._preview_tex = lf.ui.DynamicTexture(t)
                else:
                    self._preview_tex.update(t)
                self._preview_w = W
                self._preview_h = H
                self._status = f"Preview rendered ({W}×{H})."
            else:
                out = str(Path(_bg_path).parent / f"composite_{W}x{H}.png")
                Image.fromarray(arr).save(out)
                self._status = f"Saved: {out}"
                lf.log.info(f"360 BG: saved {out}")

        except Exception as e:
            import traceback
            self._status = f"Error: {e}"
            lf.log.error(f"360 BG still: {traceback.format_exc()}")
        _request_redraw()

    # ── LFS path load ─────────────────────────────────────────────────────────

    def _load_lfs_path(self, path: str):
        try:
            import sys, os
            plugin_dir = os.path.dirname(os.path.abspath(__file__))
            # Try importing LFSPathPlayer from the training render plugin
            # Fall back to bundled minimal player
            try:
                sys.path.insert(0, plugin_dir)
                from lfs_path_player import LFSPathPlayer
            except ImportError:
                from _lfs_player_mini import LFSPathPlayer

            self._lfs_player = LFSPathPlayer(path)
            self._lfs_path   = path
            self._lfs_status = (f"✓ Loaded — {self._lfs_player.n_keyframes} keyframes "
                                f"| {self._lfs_player.total_duration:.1f}s")
            lf.log.info(f"360 BG: LFS path loaded {path}")
        except Exception as e:
            self._lfs_player = None
            self._lfs_status = f"✗ Load error: {e}"
            lf.log.error(f"360 BG LFS load: {e}")

    def _load_circ_path(self, path: str):
        try:
            import sys
            plugin_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, plugin_dir)
            try:
                from track_player import TrackPlayer
            except ImportError:
                from _track_player_mini import TrackPlayer
            self._circ_player = TrackPlayer(path)
            self._circ_path   = path
            self._circ_status = f"✓ Loaded — {self._circ_player.info}"
            lf.log.info(f"360 BG: circular track loaded {path}")
        except Exception as e:
            self._circ_player = None
            self._circ_status = f"✗ Load error: {e}"
            lf.log.error(f"360 BG circular track load: {e}")

    # ── Video render ──────────────────────────────────────────────────────────

    def _do_video(self, width: int, height: int, player, total_frames: int):
        import tempfile
        self._video_stop     = False
        self._video_status   = ""
        self._video_progress = "Starting…"

        fps        = self._fps
        threshold  = self._threshold
        encoder    = self._encoder_idx
        path_type  = self._path_type
        arc_per_snap = self._arc_per_snap
        circ_loop  = self._circ_loop
        out_dir    = Path(_bg_path).parent if _bg_path else Path.home()
        ext        = "mkv" if encoder == 0 else "mp4"
        out_video  = str(out_dir / f"bg360_video_{width}x{height}.{ext}")
        frames_dir = Path(tempfile.mkdtemp(prefix="bg360_frames_"))

        def _run():
            try:
                import numpy as np
                from PIL import Image

                for i in range(total_frames):
                    if self._video_stop:
                        self._video_progress = "Stopped."
                        _request_redraw()
                        return

                    if path_type == "lfs":
                        secs_per_frame = 1.0 / fps
                        pos, target_pt, up_vec, fov = player.get_camera_at_snap(
                            i, secs_per_frame, loop=False
                        )
                    else:  # circular
                        pos, target_pt, up_vec, fov = player.get_camera_at_snap(
                            i, arc_per_snap, loop=circ_loop
                        )

                    # fov from LFSPathPlayer is in degrees already
                    # fov from TrackPlayer is also degrees
                    result = composite_render(
                        width, height, float(fov),
                        pos, target_pt, up=up_vec,
                        threshold=threshold
                    )

                    if result is None:
                        self._video_progress = f"Frame {i} failed — skipping."
                        continue

                    arr = (result.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    Image.fromarray(arr).save(frames_dir / f"frame_{i:06d}.png")

                    pct = int((i + 1) / total_frames * 100)
                    self._video_progress = f"Rendering frame {i+1}/{total_frames} ({pct}%)"
                    _request_redraw()

                # Collate
                self._video_progress = "Collating video…"
                _request_redraw()

                if encoder == 0:
                    err = _collate_video_pyav(frames_dir, out_video, fps, width, height)
                else:
                    err = _collate_video_ffmpeg(frames_dir, out_video, fps)

                if err:
                    self._video_status   = f"✗ Collate error: {err}"
                else:
                    self._video_status   = f"✓ Saved: {out_video}"
                    lf.log.info(f"360 BG: video saved to {out_video}")

                self._video_progress = ""

                try:
                    import shutil
                    shutil.rmtree(frames_dir)
                except Exception:
                    pass

            except Exception as e:
                import traceback
                self._video_status   = f"✗ Error: {e}"
                self._video_progress = ""
                lf.log.error(f"360 BG video: {traceback.format_exc()}")

            _request_redraw()

        self._video_thread = threading.Thread(target=_run, daemon=True)
        self._video_thread.start()
