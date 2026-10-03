"""Add parts from the Onshape URDF export to the sim's visual meshes: the head (a toggleable
group on the top plate) and the ball bearings (arm balls and top-plate bearings).

The URDF is in its CAD pose, which differs from the sim's default pose, so each part is put
into a sim body's own frame by matching geometry the two share:
- top plate: the six top-bearing centres in the URDF against the six plate hinge anchors in
  stewy.xml, paired leg by leg through the arm each leg hangs from (the plate is 3-fold
  symmetric, so the pairing is what fixes its rotation);
- arms: the URDF arm link frame, turned about the servo axis so the arm ball lands on the
  sim's ball-joint anchor.

    python build_urdf_parts.py --urdf "<...>/Main Assembly (4)/main_assembly" [--grid 0.6]

Writes web/meshes/stewy_extra.json + .bin (same piece layout as stewy_meshes); build_web.py
merges them into the mesh pack.
"""
import argparse
import collections
import json
import math
import os
import re
import struct
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "..", "web")
ap = argparse.ArgumentParser()
ap.add_argument("--urdf", required=True, help="the exported main_assembly folder (urdf/ and meshes/)")
ap.add_argument("--grid", type=float, default=0.6, help="vertex clustering cell, mm")
a = ap.parse_args()

LIGHT_BLUE, SILVER = [0.55, 0.78, 0.95], [0.76, 0.78, 0.81]
# head parts that are 3D printed take the printed-part colour; hidden hardware is left out
PRINTED = ("Head_Bottom_Half", "Head_Top_Half", "Distance_Sensor_Stop", "Joystick_Standoff")
SKIP = ("_4mm_Heat_Set_Insert", "Socket_button_head_screw", "M3_Lock_Nut")
DEFAULT_COL = {"Platine_step": [0.1, 0.3, 0.7], "XH_2Y": [0.91, 0.91, 0.91], "JST___XH_MALE_5_pin": [0.91, 0.91, 0.91]}


def T(o):
    if o is None:
        return np.eye(4)
    xyz = [float(x) for x in (o.get("xyz") or "0 0 0").split()]
    rr, pp, yy = [float(x) for x in (o.get("rpy") or "0 0 0").split()]
    cr, sr, cp, sp, cy, sy = math.cos(rr), math.sin(rr), math.cos(pp), math.sin(pp), math.cos(yy), math.sin(yy)
    R = (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
         @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, xyz
    return M


root = ET.parse(os.path.join(a.urdf, "urdf", "main_assembly.urdf")).getroot()
links = {l.get("name"): l for l in root.findall("link")}
par, kids = {}, collections.defaultdict(list)
for j in root.findall("joint"):
    c, p = j.find("child").get("link"), j.find("parent").get("link")
    par[c] = (p, T(j.find("origin")))
    kids[p].append(c)


def W(l):
    M = np.eye(4)
    while l in par:
        p, t = par[l]
        M = t @ M
        l = p
    return M


def sub(l):
    out = [l]
    for c in kids[l]:
        out += sub(c)
    return out


def stl(fn):
    d = open(fn, "rb").read()
    n = struct.unpack("<I", d[80:84])[0]
    if 84 + 50 * n == len(d):
        v = np.frombuffer(d[84:], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]), count=n)["v"]
        return v.reshape(-1, 3, 3).astype(float)
    vs = [list(map(float, ln.split()[1:4])) for ln in d.decode(errors="ignore").splitlines()
          if ln.strip().startswith("vertex")]
    return np.array(vs).reshape(-1, 3, 3)


def visuals(l):
    """(part name, triangles in URDF world, rgb or None) per visual of link l"""
    out = []
    for v in links[l].findall("visual"):
        m = v.find("geometry/mesh")
        if m is None:
            continue
        name = m.get("filename").split("/")[-1][:-4]
        s = np.array([float(x) for x in (m.get("scale") or "1 1 1").split()])
        M = W(l) @ T(v.find("origin"))
        t = stl(os.path.join(a.urdf, "meshes", name + ".stl")) * s
        c = v.find("material/color")
        out.append((re.sub(r"(_\d+)+$", "", name), t @ M[:3, :3].T + M[:3, 3],
                    [float(x) for x in c.get("rgba").split()[:3]] if c is not None else None))
    return out


def centre(t):
    p = t.reshape(-1, 3)
    return (p.max(0) + p.min(0)) / 2


# ---- the sim model at its default pose -------------------------------------------------
m = mujoco.MjModel.from_xml_path(os.path.join(WEB, "stewy.xml"))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
LEG_ARM = ["left_arm", "left_arm_1", "right_arm", "right_arm_1", "right_arm_2", "left_arm_2"]
ARMS = sorted(set(LEG_ARM))
ARM_ANCHOR = {n: np.array([(1 if n.startswith("left") else -1) * 0.04975042, 0.000931254, 0.004000013]) for n in ARMS}

every = [(l, *v) for l in links for v in visuals(l)]
balls = {l: centre(t) for l, n, t, _ in every if n == "Tapped_Ball_Bearing"}
tops = [centre(t) for _, n, t, _ in every if n == "Bearing"]

# top plate: Kabsch fit URDF world -> sim plate frame from the six legs
pi = m.body("top_plate").id
pR, pp = d.xmat[pi].reshape(3, 3), d.xpos[pi]
P, Q = [], []
for k, arm in enumerate(LEG_ARM):
    P.append(pR.T @ (d.xanchor[m.joint(f"plate_rev_leg{k}").id] - pp))
    wa = W(arm)[:3, 3]
    b = min(balls.values(), key=lambda x: np.linalg.norm(x - wa))
    Q.append(min(tops, key=lambda x: abs(np.linalg.norm(x - b) - 0.11377)))
P, Q = np.array(P), np.array(Q)
cq, cpn = Q.mean(0), P.mean(0)
U, S, Vt = np.linalg.svd((Q - cq).T @ (P - cpn))
R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
plate_T = np.eye(4)
plate_T[:3, :3], plate_T[:3, 3] = R, cpn - R @ cq
res = np.abs(Q @ R.T + plate_T[:3, 3] - P).max() * 1000
print(f"[parts] top plate fit: worst leg {res:.3f} mm")
assert res < 1.0, "the URDF's legs do not match the sim's - check the export"

# arms: URDF link frame, turned about z so the ball sits on the sim anchor
arm_T = {}
for n in ARMS:
    Wi = np.linalg.inv(W(n))
    b = min(balls.values(), key=lambda x: np.linalg.norm(x - W(n)[:3, 3]))
    loc = (Wi @ np.r_[b, 1])[:3]
    an = ARM_ANCHOR[n]
    th = math.atan2(an[1], an[0]) - math.atan2(loc[1], loc[0])
    Rz = np.eye(4)
    Rz[:2, :2] = [[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]
    arm_T[n] = Rz @ Wi
    print(f"[parts] {n}: turned {math.degrees(th):+.2f} deg, ball {np.linalg.norm((Rz @ Wi @ np.r_[b, 1])[:3] - an) * 1000:.2f} mm off")

# ---- collect the pieces ----------------------------------------------------------------
plate_set = set(sub("top_plate"))
pieces = collections.defaultdict(list)          # (body, colour, group) -> [triangles]
for l, name, t, rgb in every:
    if name == "Tapped_Ball_Bearing":
        arm = min(ARMS, key=lambda n: np.linalg.norm(W(n)[:3, 3] - centre(t)))
        M = arm_T[arm]
        pieces[(arm, tuple(SILVER), "")].append(t @ M[:3, :3].T + M[:3, 3])
    elif name == "Bearing":
        pieces[("top_plate", tuple(SILVER), "")].append(t @ plate_T[:3, :3].T + plate_T[:3, 3])
    elif l not in plate_set and centre(t)[2] > 0.08 and not name.startswith(SKIP):
        col = LIGHT_BLUE if any(k in name for k in PRINTED) else (rgb or DEFAULT_COL.get(name, [0.6, 0.6, 0.6]))
        pieces[("top_plate", tuple(round(c, 3) for c in col), "head")].append(t @ plate_T[:3, :3].T + plate_T[:3, 3])

blob, man, total = bytearray(), [], 0
for (body, col, group), lst in pieces.items():
    tri = np.concatenate(lst).reshape(-1, 3)
    Tidx = np.arange(len(tri)).reshape(-1, 3)
    cell = np.round(tri / (a.grid / 1000.0)).astype(np.int64)
    _, first, inv = np.unique(cell, axis=0, return_index=True, return_inverse=True)
    Pv = tri[first]
    Ti = inv.reshape(-1)[Tidx]
    Ti = Ti[(Ti[:, 0] != Ti[:, 1]) & (Ti[:, 1] != Ti[:, 2]) & (Ti[:, 0] != Ti[:, 2])]
    po = len(blob)
    blob += Pv.astype(np.float32).tobytes()
    io = len(blob)
    blob += Ti.astype(np.uint32).tobytes()
    man.append(dict(body=body, color=list(col), vertex_offset=po, vertex_count=int(len(Pv)),
                    index_offset=io, index_count=int(Ti.size), **({"group": group} if group else {})))
    total += len(Ti)
open(os.path.join(WEB, "meshes", "stewy_extra.bin"), "wb").write(bytes(blob))
json.dump(man, open(os.path.join(WEB, "meshes", "stewy_extra.json"), "w"), indent=1)
print(f"[parts] {len(man)} pieces, {total} triangles, {len(blob) / 1e6:.2f} MB "
      f"(head {sum(p['index_count'] for p in man if p.get('group') == 'head') // 3} triangles)")
