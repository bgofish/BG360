# Minimal LFS camera path player — bundled with BG360 plugin
import json, math
from pathlib import Path


def _lerp(a, b, t):
    return a + (b - a) * t

def _slerp(q1, q2, t):
    dot = sum(a*b for a,b in zip(q1,q2))
    if dot < 0:
        q2 = tuple(-x for x in q2); dot = -dot
    if dot > 0.9995:
        r = tuple(a + t*(b-a) for a,b in zip(q1,q2))
        n = math.sqrt(sum(x*x for x in r))
        return tuple(x/n for x in r)
    th0 = math.acos(max(-1,min(1,dot)))
    th  = th0 * t
    s1  = math.cos(th) - dot * math.sin(th) / math.sin(th0)
    s2  = math.sin(th) / math.sin(th0)
    return tuple(s1*a + s2*b for a,b in zip(q1,q2))

def _focal_to_fov(mm, sensor=24.0):
    return math.degrees(2*math.atan(sensor/(2*mm))) if mm > 0 else 60.0

def _quat_rotate(q, v):
    qw,qx,qy,qz = q; vx,vy,vz = v
    tx = 2*(qy*vz - qz*vy); ty = 2*(qz*vx - qx*vz); tz = 2*(qx*vy - qy*vx)
    return (vx+qw*tx+(qy*tz-qz*ty), vy+qw*ty+(qz*tx-qx*tz), vz+qw*tz+(qx*ty-qy*tx))


class LFSPathPlayer:
    def __init__(self, path: str):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        raw  = data.get("keyframes", [])
        if not raw:
            raise ValueError("No keyframes in file")
        self._kfs = sorted([{
            "time": float(k["time"]),
            "pos":  tuple(float(v) for v in k["position"]),
            "rot":  tuple(float(v) for v in k["rotation"]),
            "fov":  _focal_to_fov(float(k.get("focal_length_mm", 35))),
        } for k in raw], key=lambda k: k["time"])
        self.total_duration = self._kfs[-1]["time"]
        self.n_keyframes    = len(self._kfs)

    def get_camera_at_snap(self, snap_index, secs_per_snap, loop=False):
        t = snap_index * secs_per_snap
        if self.total_duration > 0:
            t = t % self.total_duration if loop else min(t, self.total_duration)
        kfs = self._kfs
        n   = self.n_keyframes
        if t <= kfs[0]["time"]:
            pos, rot, fov = kfs[0]["pos"], kfs[0]["rot"], kfs[0]["fov"]
        elif t >= kfs[-1]["time"]:
            pos, rot, fov = kfs[-1]["pos"], kfs[-1]["rot"], kfs[-1]["fov"]
        else:
            for i in range(n-1):
                a, b = kfs[i], kfs[i+1]
                if a["time"] <= t <= b["time"]:
                    span  = b["time"] - a["time"]
                    alpha = (t - a["time"]) / span if span > 0 else 0
                    pos   = tuple(_lerp(a["pos"][j], b["pos"][j], alpha) for j in range(3))
                    rot   = _slerp(a["rot"], b["rot"], alpha)
                    fov   = _lerp(a["fov"], b["fov"], alpha)
                    break
        fwd    = _quat_rotate(rot, (0,0,1))
        up_vec = _quat_rotate(rot, (0,1,0))
        target = (pos[0]+fwd[0], pos[1]+fwd[1], pos[2]+fwd[2])
        return pos, target, up_vec, fov
