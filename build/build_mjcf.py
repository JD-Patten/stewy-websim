"""Build a MuJoCo MJCF of Stewy from the Isaac USD dump (stewy_usd.json).

Same articulation as PhysX: a tree rooted at the base (free joint) through servo arm 0,
its rod, clevis and the top plate, then out to the other five clevises and rods, with
the five excluded ball joints closed as `connect` equality constraints. Masses, centres
of mass and principal inertias are the USD's. Only the 12 contact spheres collide, with
the friction the Isaac pull test calibrated to (feet 0.51, arm tips 0.28 - the real
robot's measured values). Servos are plain torque motors: the page computes the
measured MG90S law itself every physics step, like Isaac's explicit actuator.

    python build_mjcf.py            -> ../web/stewy.xml, ../web/stewy_meta.json
"""
import json
import math
import os
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "..", "web")
os.makedirs(WEB, exist_ok=True)
D = json.load(open(os.path.join(HERE, "stewy_usd.json")))

SIM_JOINT_ORDER = ["servo_revolute_left_arm", "servo_revolute_left_arm_1", "servo_revolute_right_arm",
                   "servo_revolute_right_arm_1", "servo_revolute_right_arm_2", "servo_revolute_left_arm_2"]
FRICTION = {"foot": 0.51, "arm": 0.28}
SERVO_ARMATURE = 1.6316666666666663e-4


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qrot(q, v):
    return qmul(qmul(q, np.array([0.0, *v])), qconj(q))[1:]


def name(p):
    return p.split("/")[-1]


bodies = {name(b["path"]): b for b in D["bodies"]}
for b in bodies.values():
    b["pos"] = np.array(b["pos"])
    b["quat"] = np.array(b["quat"]) / np.linalg.norm(b["quat"])

# ---- spanning tree from the non-excluded joints, rooted at the base ------------------
adj = {}
for j in D["joints"]:
    if j["exclude"]:
        continue
    a, c = name(j["body0"][0]), name(j["body1"][0])
    adj.setdefault(a, []).append((c, j))
    adj.setdefault(c, []).append((a, j))
parent, pjoint, order = {"root": None}, {}, ["root"]
dq = deque(["root"])
while dq:
    u = dq.popleft()
    for v, j in adj.get(u, []):
        if v not in parent:
            parent[v], pjoint[v] = u, j
            order.append(v)
            dq.append(v)
assert len(order) == len(bodies), (len(order), len(bodies))
children = {b: [c for c in order if parent.get(c) == b] for b in order}


def joint_in_child(j, child):
    """(pos, axis) of joint j expressed in the child body's frame."""
    side = "1" if name(j["body1"][0]) == child else "0"
    pos = np.array(j["pos" + side])
    rot = np.array(j["rot" + side] or [1, 0, 0, 0])
    ax = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}[str(j.get("axis") or "X")]
    return pos, qrot(rot, ax)


def fmt(v, p=7):
    return " ".join(f"{x:.{p}g}" for x in v)


# ---- contact spheres, per body, in body frames ----------------------------------------
spheres = {}
for s in D["shapes"]:
    if s["type"] != "Sphere":
        continue
    bname = s["path"].split("/")[-4]
    b = bodies[bname]
    local = qrot(qconj(b["quat"]), np.array(s["pos"]) - b["pos"])
    kind = "foot" if bname == "root" else "arm"
    spheres.setdefault(bname, []).append((local, s["radius"], kind))

# lowest point at rest -> spawn height
low = min((b["pos"] + qrot(b["quat"], p))[2] - r for bn, lst in spheres.items() for p, r, _ in lst
          for b in [bodies[bn]])

lines = []


def emit_body(bn, depth):
    b = bodies[bn]
    ind = "  " * depth
    if parent[bn] is None:
        pos, quat = b["pos"] - np.array([0, 0, low - 0.0005]), b["quat"]
    else:
        pb = bodies[parent[bn]]
        pos = qrot(qconj(pb["quat"]), b["pos"] - pb["pos"])
        quat = qmul(qconj(pb["quat"]), b["quat"])
    lines.append(f'{ind}<body name="{bn}" pos="{fmt(pos)}" quat="{fmt(quat)}">')
    pa = b.get("principal_axes") or [1, 0, 0, 0]
    lines.append(f'{ind}  <inertial pos="{fmt(b["com"])}" quat="{fmt(pa)}" mass="{b["mass"]:.7g}" '
                 f'diaginertia="{fmt(b["diag_inertia"])}"/>')
    if parent[bn] is None:
        lines.append(f'{ind}  <freejoint name="root"/>')
    else:
        j = pjoint[bn]
        jp, ax = joint_in_child(j, bn)
        jn = name(j["path"])
        if j["type"] == "PhysicsSphericalJoint":
            lines.append(f'{ind}  <joint name="{jn}" type="ball" pos="{fmt(jp)}" damping="1e-6"/>')
        else:
            servo = jn.startswith("servo_revolute")
            extra = (f' armature="{SERVO_ARMATURE:.6g}" range="-1.5707 1.5707" limited="true"' if servo
                     else ' damping="1e-6"')
            lines.append(f'{ind}  <joint name="{jn}" type="hinge" pos="{fmt(jp)}" axis="{fmt(ax)}"{extra}/>')
    for i, (p, r, kind) in enumerate(spheres.get(bn, [])):
        lines.append(f'{ind}  <geom name="{bn}_{kind}{i}" class="contact" type="sphere" pos="{fmt(p)}" '
                     f'size="{r:.5g}" friction="{FRICTION[kind]} 0.005 0.0001"/>')
    # debug sticks from this body's joint to each child joint (not collided, massless)
    here = joint_in_child(pjoint[bn], bn)[0] if parent[bn] else np.array(b["com"])
    for c in children[bn]:
        cj = pjoint[c]
        side = "0" if name(cj["body0"][0]) == bn else "1"
        p2 = np.array(cj["pos" + side])
        if np.linalg.norm(p2 - here) > 1e-4:
            lines.append(f'{ind}  <geom class="stick" type="capsule" fromto="{fmt(here)} {fmt(p2)}"/>')
    for c in children[bn]:
        emit_body(c, depth + 1)
    lines.append(f"{ind}</body>")


emit_body("root", 2)

loops = []
for j in D["joints"]:
    if j["exclude"]:
        a, c = name(j["body0"][0]), name(j["body1"][0])
        loops.append(f'    <connect name="{name(j["path"])}" body1="{a}" body2="{c}" anchor="{fmt(j["pos0"])}" '
                     f'solref="0.004 1"/>')
        # a stick on the arm out to its ball, for the debug view
acts = "\n".join(f'    <motor name="{n}" joint="{n}" gear="1" ctrllimited="false"/>' for n in SIM_JOINT_ORDER)

xml = f"""<mujoco model="stewy">
  <compiler angle="radian" inertiafromgeom="false" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81" cone="pyramidal"/>
  <visual><headlight ambient="0.4 0.4 0.4"/></visual>
  <default>
    <geom contype="0" conaffinity="0"/>
    <default class="contact"><geom contype="1" conaffinity="2" condim="3" priority="1" solref="0.004 1"
      rgba="0.9 0.4 0.2 1"/></default>
    <default class="stick"><geom size="0.0025" rgba="0.6 0.65 0.7 1" group="1"/></default>
  </default>
  <worldbody>
    <light pos="0 0 1" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="2 2 0.01" contype="2" conaffinity="1" friction="1 0.005 0.0001"
      rgba="0.8 0.8 0.8 1"/>
{chr(10).join(lines)}
  </worldbody>
  <equality>
{chr(10).join(loops)}
  </equality>
  <actuator>
{acts}
  </actuator>
</mujoco>
"""
open(os.path.join(WEB, "stewy.xml"), "w").write(xml)
meta = dict(servo_joints=SIM_JOINT_ORDER, spawn_lift=float(-(low - 0.0005)), body_order=order,
            loops=[name(j["path"]) for j in D["joints"] if j["exclude"]],
            total_mass=float(sum(b["mass"] for b in bodies.values())))
json.dump(meta, open(os.path.join(WEB, "stewy_meta.json"), "w"), indent=1)
print(f"[mjcf] {len(order)} bodies, {len(loops)} loop constraints, {sum(len(v) for v in spheres.values())} spheres, "
      f"mass {meta['total_mass']*1000:.1f} g, lowest point {low*1000:.2f} mm")
