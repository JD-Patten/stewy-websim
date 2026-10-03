"""The desk: JD's standing-desk frame (desk_base.stl, Onshape export in metres, z up, at full
height) as a black mesh, centred on the world origin. The oak top (6 ft x 25 in x 1 in) sits on
the frame; the page adds it as a MuJoCo box (see DESK in app.js) and draws it with a wood texture.

    python build_desk.py        ->  web/meshes/stewy_desk.json + .bin (same layout as stewy_meshes)
"""
import json
import os
import struct

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "..", "web")
BLACK = [0.06, 0.06, 0.065]

d = open(os.path.join(HERE, "desk_base.stl"), "rb").read()
n = struct.unpack("<I", d[80:84])[0]
t = np.frombuffer(d[84:], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]), count=n)["v"]
P = t.reshape(-1, 3).astype(np.float64)
lo, hi = P.min(0), P.max(0)
P[:, :2] -= (lo[:2] + hi[:2]) / 2                      # frame centred on the origin, feet on z = 0
P[:, 2] -= lo[2]
print(f"[desk] frame {np.round((hi - lo) * 1000)} mm, top of frame at {(hi[2] - lo[2]) * 1000:.1f} mm")
# the telescoping legs: the bottom stage stays on the floor, the middle stage (from the 543 mm
# joint) travels half as far as the top stage (from the 867 mm joint, with the frame and desk
# top). The page moves them as MuJoCo mocap bodies "desk_mid" and "desk"
STAGES = [("world", 0.0, 0.543), ("desk_mid", 0.543, 0.867), ("desk", 0.867, 9.0)]
tri = P.reshape(-1, 3, 3)
zc = tri[:, :, 2].mean(1)
blob, man = bytearray(), []
for body, z0, z1 in STAGES:
    Pt = tri[(zc >= z0) & (zc < z1)].reshape(-1, 3)
    cell = np.round(Pt / 0.001).astype(np.int64)            # 1 mm vertex clustering
    _, first, inv = np.unique(cell, axis=0, return_index=True, return_inverse=True)
    V = Pt[first]
    T = inv.reshape(-1)[np.arange(len(Pt)).reshape(-1, 3)]
    T = T[(T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])]
    vo = len(blob)
    blob += V.astype(np.float32).tobytes()
    io = len(blob)
    blob += T.astype(np.uint32).tobytes()
    man.append(dict(body=body, color=BLACK, vertex_offset=vo, vertex_count=int(len(V)), index_offset=io, index_count=int(T.size)))
    print(f"[desk] {body}: {len(T)} triangles")
print(f"[desk] {len(blob) / 1e6:.2f} MB")
open(os.path.join(WEB, "meshes", "stewy_desk.bin"), "wb").write(bytes(blob))
json.dump(man, open(os.path.join(WEB, "meshes", "stewy_desk.json"), "w"), indent=1)
