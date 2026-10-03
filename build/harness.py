"""Run exported Stewy policies in MuJoCo exactly as the browser page will - same servo
law, same observation/action path - and measure what they do.

    python harness.py walk62 [--seconds 8] [--dir 0]      walk policy (dir = remap rotation 0/1/2)
    python harness.py spin_ccw [--mirror]                 turn policy (mirror -> clockwise)
"""
import argparse
import json
import math
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "..", "web")
MG = dict(kp=0.815225204826973, kd=0.04367582236525295, sat=0.07508598256940573, effort=0.050994580000000005,
          vlim=10.723671891424365, break_away=0.03132643496906681, hold=0.007945146044038913,
          floor=0.025497290000000002, delay_s=0.006666643753051759)

# ---- firmware IK (port of stewy_ik.py) -------------------------------------------------
ARM, ROD = 50.0, 100.0
PERM, SIGNS = [2, 5, 3, 0, 4, 1], [-1.0] * 6
A0 = np.radians([0.0552, -0.0552, 0.0552, -0.0552, 0.0552, -0.0552])
LIMIT = 1.53


def rot(axis, a):
    c, s = math.cos(a), math.sin(a)
    return {"z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]])}[axis]


def conn(o1, o2, zh):
    d1, d2 = (o1 - o2) / 6.0, (o1 - o2) / 3.0
    c = o1 - 2 * d1
    r = math.sqrt(c * c + d2 * d2 - 2 * c * d2 * math.cos(math.pi / 3))
    n1 = np.array([-math.sqrt(r * r - (o1 / 2) ** 2), o1 / 2, zh])
    n6 = np.array([n1[0], -n1[1], zh])
    r120, r240 = rot("z", 2 * math.pi / 3), rot("z", 4 * math.pi / 3)
    return np.stack([n1, r240 @ n6, r240 @ n1, r120 @ n6, r120 @ n1, n6])


PLATE, SERVO_TR = conn(40.0, 20.0, -10.0), conn(73.0, 15.5, 0.0)
SERVO_ROT = np.stack([rot("x", math.radians(xa)) @ rot("y", math.radians(15.0)) @ rot("z", math.radians(za))
                      for za, xa in zip([-120, -120, 0, 0, 120, 120], [0, -180, 0, -180, 0, -180])])


def ik(pose):
    t = pose[:3]
    R = rot("z", pose[5]) @ rot("y", pose[4]) @ rot("x", pose[3])
    p = PLATE @ R.T + t - SERVO_TR
    p = np.einsum("kij,kj->ki", SERVO_ROT, p)
    px, py, pz = p[:, 0], p[:, 1], p[:, 2]
    val = np.clip((ROD ** 2 - px * px - py * py - pz * pz - ARM ** 2) / (-2 * ARM * np.sqrt(py * py + pz * pz)), -1, 1)
    ang = np.arctan2(pz, py) + np.where(pz < 0, np.arccos(val), -np.arccos(val))
    out = np.zeros(6)
    out[PERM] = np.array(SIGNS) * (ang - A0)
    return np.clip(out, -LIMIT, LIMIT)


def to_fw(sim):
    return sim[PERM] * np.array(SIGNS) + A0


def from_fw(fw):
    out = np.zeros(6)
    out[PERM] = np.array(SIGNS) * (fw - A0)
    return out


ROT_IDX = [[0, 1, 2, 3, 4, 5], [2, 3, 4, 5, 0, 1], [4, 5, 0, 1, 2, 3]]


def sym(sim, k=0, mirror=False):
    fw = to_fw(sim)[ROT_IDX[k]]
    if mirror:
        fw = -fw[[5, 4, 3, 2, 1, 0]]
    return from_fw(fw)


def _round_even(x):
    """torch.round: halves go to the even integer (Python's round does the same)."""
    return round(x)


def canon(P, vx, vy, w=0.0):
    """The env's _canonicalize: a base-frame omni command (m/s, m/s, rad/s) -> the
    policy's canonical command plus the (k, mirror) that map its output back."""
    S = P["symmetry"]
    mode, v = S["mode"], math.hypot(vx, vy)
    th_b, phi = math.atan2(vy, vx), math.radians(S["frame_deg"])
    th = (phi - th_b) if S["left_handed"] else (th_b - phi)          # IK frame
    sector, m = 2 * math.pi / 3, False
    if mode == "rot3_mirror":
        m = w < 0
    elif mode == "rot3_mirror_dir":
        m = (th - _round_even(th / sector) * sector) < 0 and v > 0
    if m:
        th, w = -th, -w
    k = 0
    if mode in ("rot3", "rot3_mirror", "rot3_mirror_dir") and v > 0:
        k = _round_even(th / sector) % 3
        th -= k * sector
    return (v * math.cos(th), v * math.sin(th), w), k, m


# ---- policy ------------------------------------------------------------------------
class Policy:
    def __init__(self, path):
        self.p = json.load(open(path))
        self.W = [(np.array(l["W"]), np.array(l["b"])) for l in self.p["layers"]]
        n = self.p["normalizer"]
        self.mean, self.std, self.eps = (np.array(n["mean"]), np.array(n["std"]), n["eps"]) if n else (0, 1, 0)

    def __call__(self, obs):
        x = (np.asarray(obs) - self.mean) / (self.std + self.eps)
        for i, (W, b) in enumerate(self.W):
            x = W @ x + b
            if i < len(self.W) - 1:
                x = np.where(x > 0, x, np.expm1(x))          # ELU
        return x


class Servos:
    """The measured MG90S: delay, stick-slip, PD, DC-motor torque-speed clip."""

    def __init__(self, dt, q0):
        self.n = max(0, round(MG["delay_s"] / dt))
        self.buf = [q0.copy() for _ in range(self.n + 1)]
        self.driving = np.zeros(6, bool)
        self.latch = q0.copy()

    def torque(self, des, q, qd):
        self.buf.append(des.copy())
        des = self.buf.pop(0)
        err = des - q
        mag = np.abs(err)
        was = self.driving
        self.driving = np.where(was, mag >= MG["hold"], mag > MG["break_away"])
        self.latch = np.where(was | self.driving, q, self.latch)
        drive = q + np.sign(err) * np.maximum(MG["kp"] * mag, MG["floor"]) / MG["kp"]
        des = np.where(self.driving, drive, self.latch)
        tau = MG["kp"] * (des - q) - MG["kd"] * qd
        hi = np.clip(MG["sat"] * (1.0 - qd / MG["vlim"]), 0, MG["effort"])
        lo = np.clip(MG["sat"] * (-1.0 - qd / MG["vlim"]), -MG["effort"], 0)
        return np.clip(tau, lo, hi)


def run(name, seconds=8.0, k=0, mirror=False, cmd=None, verbose=True):
    pol = Policy(os.path.join(WEB, "policies", name + ".json"))
    P = pol.p
    m = mujoco.MjModel.from_xml_path(os.path.join(WEB, "stewy.xml"))
    d = mujoco.MjData(m)
    jadr = [m.jnt_qposadr[m.joint(n).id] for n in json.load(open(os.path.join(WEB, "stewy_meta.json")))["servo_joints"]]
    vadr = [m.jnt_dofadr[m.joint(n).id] for n in json.load(open(os.path.join(WEB, "stewy_meta.json")))["servo_joints"]]
    mujoco.mj_forward(m, d)
    servo = Servos(m.opt.timestep, d.qpos[jadr].copy())
    sub = round(P["control_dt"] / m.opt.timestep)
    ob = P["obs"]
    periods = ob["clock_periods"]
    if cmd is None:
        th = math.radians(ob["forward_direction_deg"])
        cmd = (ob["forward_speed"] * math.cos(th), ob["forward_speed"] * math.sin(th)) \
            if ob["command_mode"] == "forward" else (0.0, 0.0)
    act = P["action"]
    table = np.array(act["table"]) if act["kind"] == "ga_residual" else None
    filt = np.zeros(6)
    phase, dt = 0.0, P["control_dt"]
    target = d.qpos[jadr].copy()
    omni = ob["command_mode"] == "omni"
    if omni:
        c3, k, mirror = canon(P, *cmd)
        cmd_obs = [c3[0] / ob["omni_obs_v"], c3[1] / ob["omni_obs_v"], c3[2] / ob["omni_obs_w"]]
    log, pacc = [], []
    plate = m.body("top_plate").id
    vel6, v_prev = np.zeros(6), None
    for step in range(int(seconds / dt)):
        clock = []
        for per in periods:
            a = 2 * math.pi * phase / per
            clock += [math.sin(a), math.cos(a)]
        obs = (cmd_obs if omni else [cmd[0] * ob["command_scale"], cmd[1] * ob["command_scale"]]) + clock
        a = np.clip(pol(obs), -1, 1)
        if table is not None:
            idx = int(round(phase / dt)) % len(table)
            if act.get("filter_alpha"):
                filt = filt + act["filter_alpha"] * (a - filt)     # low-passed corrections
                a = filt
            target = np.clip(table[idx] + a * np.array(act["residual_rad"]), -act["servo_limit"], act["servo_limit"])
        else:
            filt = filt + act["filter_alpha"] * (a - filt) if act["filter_alpha"] else a
            pose = np.array(act["pose_center"]) + filt * np.array(act["pose_scale"])
            target = ik(pose)
        target = sym(target, k, mirror)
        phase = (phase + dt) % ob["gait_period_s"]
        for _ in range(sub):
            d.ctrl[:] = servo.torque(target, d.qpos[jadr], d.qvel[vadr])
            mujoco.mj_step(m, d)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, plate, vel6, 0)
        if v_prev is not None:
            pacc.append(np.linalg.norm(vel6[3:] - v_prev) / dt)       # plate COM, control-rate differences
        v_prev = vel6[3:].copy()
        q = d.qpos[3:7]
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        log.append((d.qpos[0], d.qpos[1], yaw, d.qpos[2]))
    L = np.array(log)
    s = 25
    dur = (len(L) - 1 - s) * dt
    dx, dy = L[-1, 0] - L[s, 0], L[-1, 1] - L[s, 1]
    turned = np.unwrap(L[s:, 2])
    # robot-frame direction: each step's displacement turned into the start heading's frame,
    # so a walker that drifts in yaw is still judged on where it walks relative to its body
    ddx, ddy, rel = np.diff(L[s:, 0]), np.diff(L[s:, 1]), turned[:-1] - turned[0]
    bx = np.sum(ddx * np.cos(rel) + ddy * np.sin(rel))
    by = np.sum(-ddx * np.sin(rel) + ddy * np.cos(rel))
    res = dict(speed=math.hypot(dx, dy) / dur * 1000, dir=math.degrees(math.atan2(dy, dx)),
               body_speed=math.hypot(bx, by) / dur * 1000, body_dir=math.degrees(math.atan2(by, bx)),
               yaw_rate=math.degrees((turned[-1] - turned[0]) / dur), z_min=L[:, 3].min() * 1000,
               upright=bool(L[-1, 3] > -0.05), plate_acc=float(np.sqrt(np.mean(np.square(pacc[s:])))))
    if verbose:
        print(f"[{name} k={k} mirror={mirror}] {res['speed']:.1f} mm/s toward {res['dir']:.0f} deg, "
              f"turn {res['yaw_rate']:+.1f} deg/s, plate {res['plate_acc']:.1f} m/s2, base z min {res['z_min']:.1f} mm")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--dir", type=int, default=0)
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--check", action="store_true", help="compare the MLP with the exported torch output")
    ap.add_argument("--omni", type=float, default=None, help="omni: walk toward this base-frame angle (deg)")
    ap.add_argument("--sweep", action="store_true", help="omni: the lab's 12-direction check (every 30 deg)")
    a = ap.parse_args()
    if a.sweep or a.omni is not None:
        P = json.load(open(os.path.join(WEB, "policies", a.policy + ".json")))
        v = P["obs"]["omni_v_max"]
        angs = [a.omni] if a.omni is not None else [30.0 * i for i in range(12)]
        for deg in angs:
            t = math.radians(deg)
            r = run(a.policy, a.seconds, cmd=(v * math.cos(t), v * math.sin(t), 0.0), verbose=False)
            err = math.hypot(r["speed"] * math.cos(math.radians(r["dir"])) - v * 1000 * math.cos(t),
                             r["speed"] * math.sin(math.radians(r["dir"])) - v * 1000 * math.sin(t)) / (v * 1000)
            off = (r["dir"] - deg + 180) % 360 - 180
            print(f"walk {deg:5.0f} deg: {r['speed']:5.1f} mm/s toward {r['dir']:6.0f} ({off:+4.0f} off), "
                  f"turn {r['yaw_rate']:+5.1f} deg/s, rel err {err:.2f}, upright {r['upright']}")
        raise SystemExit
    if a.check:
        pol = Policy(os.path.join(WEB, "policies", a.policy + ".json"))
        got = pol(pol.p["test"]["obs"])
        print("MLP max |diff| vs torch:", np.abs(got - np.array(pol.p["test"]["action"])).max())
    run(a.policy, a.seconds, a.dir, a.mirror)
