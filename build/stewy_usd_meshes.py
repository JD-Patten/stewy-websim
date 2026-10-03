"""Export Stewy's visual meshes per rigid body, in each body's own frame, for three.js.

Every Mesh under a body is triangulated, moved into the body frame, merged, and thinned by
vertex clustering (--grid mm), then written as one binary: per body a Float32 xyz block
and a Uint32 index block, described by a JSON manifest (body name, offsets, counts, colour).

    isaaclab.bat -p scripts/stewy_usd_meshes.py --out_dir <web>/meshes [--grid 0.6]
"""
import argparse
import json
import os

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--usd", default=r"C:\Users\JD\Documents\stewy\isaacsim\stewy_robot_v2.usd")
ap.add_argument("--out_dir", required=True)
ap.add_argument("--grid", type=float, default=0.6, help="vertex clustering cell, mm (0 = keep all)")
ap.add_argument("--palette", default=None,
                help="JSON file {part-name substring: [r,g,b] or null to leave it out}, checked in order before the defaults")
ap.add_argument("--list", action="store_true", help="print every part with the colour it gets")
AppLauncher.add_app_launcher_args(ap)
a, _ = ap.parse_known_args()
a.headless = True
app = AppLauncher(a).app

import numpy as np  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

stage = Usd.Stage.Open(a.usd)
xc = UsdGeom.XformCache()
bodies = [p for p in stage.Traverse(Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.RigidBodyAPI)]


PALETTE = list(json.load(open(a.palette)).items()) if a.palette else []


def colour(prim, part):
    """a --palette match first, then displayColor if authored, else a palette by part name."""
    for key, col in PALETTE:
        if key.lower() in part.lower():
            return None if col is None else [float(x) for x in col]
    try:
        c = UsdGeom.Mesh(prim).GetDisplayColorAttr().Get()
        if c and len(c) and tuple(c[0]) != (0.5, 0.5, 0.5):
            return [float(x) for x in c[0]]
    except Exception:  # noqa: BLE001
        pass
    p = part.lower()
    for key, col in (("rubber", [0.12, 0.12, 0.13]), ("carbon", [0.08, 0.08, 0.09]), ("tube", [0.08, 0.08, 0.09]),
                     ("magnet", [0.72, 0.74, 0.78]), ("bearing", [0.72, 0.74, 0.78]), ("nut", [0.72, 0.74, 0.78]),
                     ("clevis", [0.72, 0.74, 0.78]), ("m3", [0.72, 0.74, 0.78]), ("servo", [0.15, 0.3, 0.75]),
                     ("mg90", [0.15, 0.3, 0.75]), ("pcb", [0.1, 0.45, 0.25]), ("jst", [0.9, 0.9, 0.85]),
                     ("arm", [0.96, 0.55, 0.16]), ("base", [0.93, 0.93, 0.9]), ("plate", [0.93, 0.93, 0.9])):
        if key in p:
            return col
    return [0.85, 0.85, 0.82]


os.makedirs(a.out_dir, exist_ok=True)
blob, manifest, total = bytearray(), [], 0
for body in bodies:
    bw = np.array(xc.GetLocalToWorldTransform(body), dtype=np.float64)          # row-vector convention
    inv = np.linalg.inv(bw)
    parts = {}
    for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh) or prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        # skip meshes that belong to a nested rigid body
        q = prim.GetParent()
        nested = False
        while q and q != body:
            if q.HasAPI(UsdPhysics.RigidBodyAPI):
                nested = True
                break
            q = q.GetParent()
        if nested:
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = np.array(mesh.GetPointsAttr().Get() or [], dtype=np.float64)
        counts = np.array(mesh.GetFaceVertexCountsAttr().Get() or [], dtype=np.int64)
        idx = np.array(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int64)
        if not len(pts):
            continue
        m = np.array(xc.GetLocalToWorldTransform(prim), dtype=np.float64) @ inv
        p = np.c_[pts, np.ones(len(pts))] @ m
        tris, o = [], 0
        for c in counts:
            for k in range(1, c - 1):
                tris.append((idx[o], idx[o + k], idx[o + k + 1]))
            o += c
        # the part name is the visual's folder under the body (e.g. Rubber_Foot, Left_Arm)
        rel = str(prim.GetPath())[len(str(body.GetPath())):].split("/")
        part = rel[2] if len(rel) > 2 else rel[-1]
        col = colour(prim, part)
        if col is None:                                     # palette null: leave the part out
            continue
        col = tuple(round(x, 3) for x in col)
        if a.list:
            print(f"[part] {body.GetName():>14s} {part:<45s} {col}", flush=True)
        parts.setdefault(col, []).append((p[:, :3], np.array(tris, dtype=np.int64)))
    for col, lst in parts.items():
        P, T, off = [], [], 0
        for p, t in lst:
            P.append(p)
            T.append(t + off)
            off += len(p)
        P, T = np.concatenate(P), np.concatenate(T)
        if a.grid > 0:
            cell = np.round(P / (a.grid / 1000.0)).astype(np.int64)
            _, first, inv_map = np.unique(cell, axis=0, return_index=True, return_inverse=True)
            P = P[first]
            T = inv_map.reshape(-1)[T]
            T = T[(T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])]
        pos_off = len(blob)
        blob += P.astype(np.float32).tobytes()
        idx_off = len(blob)
        blob += T.astype(np.uint32).tobytes()
        manifest.append(dict(body=body.GetName(), color=list(col), vertex_offset=pos_off, vertex_count=int(len(P)),
                             index_offset=idx_off, index_count=int(T.size)))
        total += len(T)
open(os.path.join(a.out_dir, "stewy_meshes.bin"), "wb").write(bytes(blob))
json.dump(manifest, open(os.path.join(a.out_dir, "stewy_meshes.json"), "w"), indent=1)
print(f"[meshes] {len(manifest)} pieces over {len(bodies)} bodies, {total} triangles, "
      f"{len(blob)/1e6:.1f} MB", flush=True)
