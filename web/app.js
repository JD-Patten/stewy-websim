// Stewy Live: MuJoCo (WASM) physics + the blind policies trained in Isaac Lab + three.js.
// Everything between the network and the motors mirrors the training env and the
// Python harness (build/harness.py): the same observation, the GA gait table or the
// firmware IK, Stewy's exact symmetry, and the MG90S servo law measured on the bench.
import loadMujoco from "./vendor/mujoco.js";

const $ = (s) => document.querySelector(s);
const overlay = $("#overlay");
const say = (msg, error = false) => {
  overlay.hidden = !msg;
  overlay.classList.toggle("error", error);
  $("#overlay-text").textContent = msg || "";
};

// ---- measured MG90S (servo_test_bench/isaac_model/params/mg90s_fleet.json) --------------
const MG = { kp: 0.815225204826973, kd: 0.04367582236525295, sat: 0.07508598256940573,
  effort: 0.050994580000000005, vlim: 10.723671891424365, breakAway: 0.03132643496906681,
  hold: 0.007945146044038913, floor: 0.025497290000000002, delay: 0.006666643753051759 };

class Servos {
  constructor(dt, q0) {
    this.n = Math.max(0, Math.round(MG.delay / dt));
    this.buf = Array.from({ length: this.n + 1 }, () => Float64Array.from(q0));
    this.driving = new Uint8Array(6);
    this.latch = Float64Array.from(q0);
    this.tau = new Float64Array(6);
  }
  torque(target, q, qd) {
    this.buf.push(Float64Array.from(target));
    const des = this.buf.shift();
    for (let i = 0; i < 6; i++) {
      const err = des[i] - q[i], mag = Math.abs(err), was = this.driving[i];
      const drive = was ? mag >= MG.hold : mag > MG.breakAway;
      this.driving[i] = drive ? 1 : 0;
      if (was || drive) this.latch[i] = q[i];
      const d = drive ? q[i] + Math.sign(err) * Math.max(MG.kp * mag, MG.floor) / MG.kp : this.latch[i];
      const t = MG.kp * (d - q[i]) - MG.kd * qd[i];
      const hi = Math.min(Math.max(MG.sat * (1 - qd[i] / MG.vlim), 0), MG.effort);
      const lo = Math.max(Math.min(MG.sat * (-1 - qd[i] / MG.vlim), 0), -MG.effort);
      this.tau[i] = Math.min(Math.max(t, lo), hi);
    }
    return this.tau;
  }
}

// ---- firmware IK (port of stewy_ik.py) --------------------------------------------------
const ARM = 50, ROD = 100, PERM = [2, 5, 3, 0, 4, 1], SIGNS = [-1, -1, -1, -1, -1, -1], LIMIT = 1.53;
const A0 = [0.0552, -0.0552, 0.0552, -0.0552, 0.0552, -0.0552].map((d) => (d * Math.PI) / 180);
const rot = (axis, a) => {
  const c = Math.cos(a), s = Math.sin(a);
  if (axis === "z") return [[c, -s, 0], [s, c, 0], [0, 0, 1]];
  if (axis === "y") return [[c, 0, s], [0, 1, 0], [-s, 0, c]];
  return [[1, 0, 0], [0, c, -s], [0, s, c]];
};
const mm = (A, B) => A.map((r) => [0, 1, 2].map((j) => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j]));
const mv = (A, v) => A.map((r) => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]);
function conn(o1, o2, zh) {
  const d1 = (o1 - o2) / 6, d2 = (o1 - o2) / 3, c = o1 - 2 * d1;
  const r = Math.sqrt(c * c + d2 * d2 - 2 * c * d2 * Math.cos(Math.PI / 3));
  const n1 = [-Math.sqrt(r * r - (o1 / 2) ** 2), o1 / 2, zh], n6 = [n1[0], -n1[1], zh];
  const r120 = rot("z", (2 * Math.PI) / 3), r240 = rot("z", (4 * Math.PI) / 3);
  return [n1, mv(r240, n6), mv(r240, n1), mv(r120, n6), mv(r120, n1), n6];
}
const PLATE = conn(40, 20, -10), SERVO_TR = conn(73, 15.5, 0);
const SERVO_ROT = [-120, -120, 0, 0, 120, 120].map((za, i) =>
  mm(mm(rot("x", ([0, -180, 0, -180, 0, -180][i] * Math.PI) / 180), rot("y", (15 * Math.PI) / 180)),
    rot("z", (za * Math.PI) / 180)));
function ik(pose) {
  const R = mm(mm(rot("z", pose[5]), rot("y", pose[4])), rot("x", pose[3]));
  const out = new Float64Array(6);
  for (let k = 0; k < 6; k++) {
    const pr = mv(R, PLATE[k]);
    const p = mv(SERVO_ROT[k], [pr[0] + pose[0] - SERVO_TR[k][0], pr[1] + pose[1] - SERVO_TR[k][1],
      pr[2] + pose[2] - SERVO_TR[k][2]]);
    const [px, py, pz] = p;
    let v = (ROD * ROD - px * px - py * py - pz * pz - ARM * ARM) / (-2 * ARM * Math.sqrt(py * py + pz * pz));
    v = Math.min(1, Math.max(-1, v));
    const ang = Math.atan2(pz, py) + (pz < 0 ? Math.acos(v) : -Math.acos(v));
    out[PERM[k]] = Math.min(LIMIT, Math.max(-LIMIT, SIGNS[k] * (ang - A0[k])));
  }
  return out;
}

// ---- Stewy's exact symmetry (checked in the IK and in physics, RL Lab 2026-09-28) --------
const ROT_IDX = [[0, 1, 2, 3, 4, 5], [2, 3, 4, 5, 0, 1], [4, 5, 0, 1, 2, 3]];
const MIRROR = [5, 4, 3, 2, 1, 0];
function sym(sim, k, mirror) {
  if (!k && !mirror) return sim;
  const fw = PERM.map((p, i) => sim[p] * SIGNS[i] + A0[i]);
  let f2 = ROT_IDX[k].map((j) => fw[j]);
  if (mirror) f2 = MIRROR.map((j) => -f2[j]);
  const out = new Float64Array(6);
  PERM.forEach((p, i) => { out[p] = SIGNS[i] * (f2[i] - A0[i]); });
  return out;
}

// the env's _canonicalize: a base-frame walk command -> the policy's canonical command
// and the (k, mirror) that map its servo targets back. Halves round to even, as torch.round.
const roundEven = (x) => { const r = Math.round(x); return Math.abs(x % 1) === 0.5 && r % 2 ? r - 1 : r; };
function canon(S, vx, vy, w) {
  const v = Math.hypot(vx, vy), thB = Math.atan2(vy, vx), phi = (S.frame_deg * Math.PI) / 180;
  let th = S.left_handed ? phi - thB : thB - phi, m = false, k = 0;
  const sector = (2 * Math.PI) / 3;
  if (S.mode === "rot3_mirror") m = w < 0;
  else if (S.mode === "rot3_mirror_dir") m = th - roundEven(th / sector) * sector < 0 && v > 0;
  if (m) { th = -th; w = -w; }
  if (S.mode !== "none" && v > 0) { k = ((roundEven(th / sector) % 3) + 3) % 3; th -= k * sector; }
  return { cmd: [v * Math.cos(th), v * Math.sin(th), w], k, m };
}

// ---- the exported actor ------------------------------------------------------------------
class Policy {
  constructor(p) {
    this.p = p;
    this.layers = p.layers.map((l) => ({ W: Float32Array.from(l.W.flat()), b: Float32Array.from(l.b),
      nIn: l.W[0].length, nOut: l.W.length }));
    const n = p.normalizer;
    this.mean = n ? n.mean : null; this.std = n ? n.std : null; this.eps = n ? n.eps : 0;
  }
  act(obs) {
    let x = Float32Array.from(obs, (v, i) => (this.mean ? (v - this.mean[i]) / (this.std[i] + this.eps) : v));
    this.layers.forEach((L, li) => {
      const y = new Float32Array(L.nOut);
      for (let o = 0; o < L.nOut; o++) {
        let s = L.b[o];
        const row = o * L.nIn;
        for (let i = 0; i < L.nIn; i++) s += L.W[row + i] * x[i];
        y[o] = li < this.layers.length - 1 ? (s > 0 ? s : Math.expm1(s)) : s;
      }
      x = y;
    });
    return Array.from(x, (v) => Math.min(1, Math.max(-1, v)));
  }
}

// ---- terrain: a MuJoCo heightfield filled with layered Perlin noise ------------------------
// The patch (1.5 m square, 5 mm cells) follows the robot: when it strays 25 cm from the
// patch centre the heightfield is moved under it and refilled. The noise is sampled at world
// coordinates on a cell-aligned grid, so the ground under the robot is identical after a move.
// The last 12 cm of the patch taper to zero height to meet the flat backdrop.
const TER = { n: 301, half: 0.75, zmax: 0.08, taper: 0.12, recentre: 0.25 };
TER.cell = (2 * TER.half) / (TER.n - 1);
// height = peak-to-peak mm of each layer, size = its feature size (noise period) in cm
const LEVELS = [
  { label: "Hills", h: 0, hMax: 40, s: 40, sMin: 15, sMax: 100 },
  { label: "Bumps", h: 0, hMax: 20, s: 8, sMin: 3, sMax: 20 },
  { label: "Grit", h: 0, hMax: 6, s: 3, sMin: 2, sMax: 6 },
];
function terrainXml(xml) {
  const m = xml.match(/<geom name="floor"[^>]*\/>/);
  if (!m) throw new Error("stewy.xml: no floor geom to replace with terrain");
  const geom = m[0].replace(/type="plane"\s+size="[^"]*"/,
    `type="hfield" hfield="terrain" pos="0 0 ${-TER.zmax / 2}"`);
  const asset = `<asset><hfield name="terrain" nrow="${TER.n}" ncol="${TER.n}" ` +
    `size="${TER.half} ${TER.half} ${TER.zmax} 0.01"/></asset>\n  `;
  return xml.replace(m[0], geom).replace("<worldbody>", asset + "<worldbody>");
}
function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function perlin(seed) {                                  // Ken Perlin's improved noise, 2D, ~-1..1
  const r = mulberry32(seed), perm = Array.from({ length: 256 }, (_, i) => i);
  for (let i = 255; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
  const p = new Uint8Array(512);
  for (let i = 0; i < 512; i++) p[i] = perm[i & 255];
  const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10);
  const lerp = (a, b, t) => a + t * (b - a);
  const grad = (h, x, y) => {
    switch (h & 7) {
      case 0: return x + y; case 1: return -x + y; case 2: return x - y; case 3: return -x - y;
      case 4: return x; case 5: return -x; case 6: return y; default: return -y;
    }
  };
  return (x, y) => {
    const X = Math.floor(x), Y = Math.floor(y), xf = x - X, yf = y - Y, xi = X & 255, yi = Y & 255;
    const u = fade(xf), v = fade(yf);
    const aa = p[p[xi] + yi], ab = p[p[xi] + yi + 1], ba = p[p[xi + 1] + yi], bb = p[p[xi + 1] + yi + 1];
    return lerp(lerp(grad(aa, xf, yf), grad(ba, xf - 1, yf), u),
      lerp(grad(ab, xf, yf - 1), grad(bb, xf - 1, yf - 1), u), v);
  };
}

// ---- boot ---------------------------------------------------------------------------------
async function fetchOk(url, kind) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return kind === "json" ? r.json() : kind === "text" ? r.text() : r.arrayBuffer();
}

async function main() {
  if (!window.THREE) throw new Error("three.js did not load (the CDN script was blocked or offline).");
  const [wasmBinary, xml, meta, meshPack, manifest] = await Promise.all([
    fetchOk("vendor/mujoco.wasm", "bin"), fetchOk("stewy.xml", "text"), fetchOk("stewy_meta.json", "json"),
    fetchOk("meshes/stewy_mesh_pack.json", "json"),
    fetchOk("policies/manifest.json", "json")]);
  // every GA-residual walker in the manifest can be picked; walk_smoother (the robot's walker) first
  // any-direction (omni, pose IK) walkers join the list after them. The group B and C walkers
  // only walk as part of the three-walker set, which follows the robot's walker
  const walkMs = manifest.filter((m) => m.action === "ga_residual" || m.mode === "omni");
  const pol = Object.fromEntries(await Promise.all(
    walkMs.map(async (m) => [m.name, await fetchOk(`policies/${m.name}.json`, "json")])));
  const groupMs = walkMs.filter((m) => m.group).sort((a, b) => a.group.localeCompare(b.group));
  // one walker on the page: the three-walker set, which picks a group walker per direction
  if (groupMs.length !== 3) throw new Error("the three-walker set needs walkers A, B and C");
  const walkers = [{ name: "three_walkers", combo: true, base: groupMs[0].name }];
  const walkPs = walkers.map((m) => pol[m.combo ? m.base : m.name]);
  // turn-in-place policies (pose IK); the robot's own turn first
  const turners = manifest.filter((m) => m.action === "pose_ik" && m.mode !== "omni")
    .sort((a, b) => (b.name === "spin_calm_legs") - (a.name === "spin_calm_legs"));
  const spinPs = await Promise.all(turners.map((m) => fetchOk(`policies/${m.name}.json`, "json")));
  const meshMan = meshPack.pieces;
  const meshBin = Uint8Array.from(atob(meshPack.data), (c) => c.charCodeAt(0)).buffer;
  say("Starting MuJoCo…");
  let mujoco;
  try {
    mujoco = await loadMujoco({ wasmBinary });
  } catch (e) {
    throw new Error("This viewer blocked the WebAssembly physics engine. Open the page from the local " +
      "stewy-web-sim folder or a Hugging Face Space instead. (" + e.message + ")");
  }
  const model = mujoco.MjModel.from_xml_string(terrainXml(xml));
  model.hfield_data.fill(0.5);                         // 0.5 of zmax, lifted by -zmax/2 = flat at z 0
  const data = new mujoco.MjData(model);
  mujoco.mj_forward(model, data);

  const JOINT = 3, BODY = 1, GEOM = 5;
  const floorId = mujoco.mj_name2id(model, GEOM, "floor");
  const jq = meta.servo_joints.map((n) => model.jnt_qposadr[mujoco.mj_name2id(model, JOINT, n)]);
  const jv = meta.servo_joints.map((n) => model.jnt_dofadr[mujoco.mj_name2id(model, JOINT, n)]);
  const physDt = model.opt.timestep;
  const walkNets = walkPs.map((p) => new Policy(p)), spinNets = spinPs.map((p) => new Policy(p));
  let spinP = spinPs[0], spin = spinNets[0];
  let wi = 0, walkP = walkPs[0], walk = walkNets[0];
  const isOmni = () => walkP.obs.command_mode === "omni";
  const isCombo = () => !!walkers[wi].combo;
  // the three-walker set: every rotation (k) and mirror of the A, B and C walkers, keyed by the
  // direction it walks. Rotation k turns a walk by -120k deg; the mirror reflects it about the
  // 30 deg line (sym_frame_deg -150), after which k turns it by +120k. C's six are all distinct
  const comboMap = new Map();
  for (const gm of groupMs) {
    const P = pol[gm.name], net = new Policy(P), base = P.obs.forward_direction_deg;
    for (const m of [false, true]) for (let k = 0; k < 3; k++) {
      const d = ((Math.round(m ? 60 - base + 120 * k : base - 120 * k) + 540) % 360) - 180;
      if (!comboMap.has(d)) comboMap.set(d, { P, net, k, m, group: gm.group });
    }
  }
  const ctrlDt = walkP.control_dt;
  if (Object.values(pol).some((p) => Math.abs(p.control_dt - ctrlDt) > 1e-9)) throw new Error("walkers differ in control_dt");
  const sub = Math.round(ctrlDt / physDt);

  // ---- controller state ----
  let mode = { kind: "stand" }, phase = 0, filt = new Float64Array(6), simTime = 0;
  let servos, target = new Float64Array(6);
  const loads = new Float64Array(6);
  const readQ = () => { const q = data.qpos; return jq.map((a) => q[a]); };
  const readQd = () => { const v = data.qvel; return jv.map((a) => v[a]); };
  function resetRobot() {
    mujoco.mj_resetData(model, data);
    moveTerrain(0, 0);
    data.qpos[2] += Math.max(0, footprintTop(0, 0)) + 0.002;   // drop onto the ground, never into it
    mujoco.mj_forward(model, data);
    servos = new Servos(physDt, readQ());
    target = new Float64Array(6); phase = 0; filt = new Float64Array(6); simTime = 0;
    history.length = 0; trail.count = 0;
  }
  function setMode(m) {
    mode = m; phase = 0; filt = new Float64Array(6);
    renderPad(); updateButtons();
  }

  function clocks(periods) {
    const out = [];
    for (const per of periods) { const a = (2 * Math.PI * phase) / per; out.push(Math.sin(a), Math.cos(a)); }
    return out;
  }
  function controlStep() {
    if (mode.kind === "walk" && isOmni()) {
      // any-direction walker: fold the command into the policy's wedge, map its targets back
      const P = walkP, ob = P.obs, th = (dirs()[mode.k] * Math.PI) / 180, v = ob.omni_v_max;
      const c = canon(P.symmetry, v * Math.cos(th), v * Math.sin(th), 0);
      const a = walk.act([c.cmd[0] / ob.omni_obs_v, c.cmd[1] / ob.omni_obs_v, c.cmd[2] / ob.omni_obs_w,
        ...clocks(ob.clock_periods)]);
      const al = P.action.filter_alpha;
      for (let i = 0; i < 6; i++) filt[i] = al ? filt[i] + al * (a[i] - filt[i]) : a[i];
      const pose = P.action.pose_center.map((c0, i) => c0 + filt[i] * P.action.pose_scale[i]);
      target = sym(ik(pose), c.k, c.m);
      phase = (phase + ctrlDt) % ob.gait_period_s;
    } else if (mode.kind === "walk") {
      // one fixed-direction walker, or the three-walker set's walker for this direction
      const c = isCombo() ? comboMap.get(dirs()[mode.k]) : { P: walkP, net: walk, k: mode.k, m: false };
      const P = c.P, ob = P.obs, th = (ob.forward_direction_deg * Math.PI) / 180;
      const obs = [ob.forward_speed * Math.cos(th) * ob.command_scale, ob.forward_speed * Math.sin(th) * ob.command_scale,
        ...clocks(ob.clock_periods)];
      let a = c.net.act(obs);
      if (P.action.filter_alpha) {           // low-passed corrections (smooth-walking recipe)
        for (let i = 0; i < 6; i++) filt[i] += P.action.filter_alpha * (a[i] - filt[i]);
        a = Array.from(filt);
      }
      const tab = P.action.table, idx = Math.round(phase / ctrlDt) % tab.length;
      const res = P.action.residual_rad, lim = P.action.servo_limit;
      const t = tab[idx].map((v, i) => Math.min(lim, Math.max(-lim, v + a[i] * (Array.isArray(res) ? res[i] : res))));
      target = sym(t, c.k, c.m);
      phase = (phase + ctrlDt) % ob.gait_period_s;
    } else if (mode.kind === "turn") {
      const P = spinP, ob = P.obs;
      const a = spin.act([0, 0, ...clocks(ob.clock_periods)]);
      const al = P.action.filter_alpha;
      for (let i = 0; i < 6; i++) filt[i] = al ? filt[i] + al * (a[i] - filt[i]) : a[i];
      const pose = P.action.pose_center.map((c, i) => c + filt[i] * P.action.pose_scale[i]);
      target = sym(ik(pose), 0, mode.dir < 0);
      phase = (phase + ctrlDt) % ob.gait_period_s;
    } else {
      target = new Float64Array(6);
    }
    loads.fill(0);
    for (let s = 0; s < sub; s++) {
      const tau = servos.torque(target, readQ(), readQd());
      const ctrl = data.ctrl;
      for (let i = 0; i < 6; i++) { ctrl[i] = tau[i]; loads[i] += Math.abs(tau[i]) / sub; }
      mujoco.mj_step(model, data);
    }
    simTime += ctrlDt;
    const q = data.qpos;
    const yaw = Math.atan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] * q[5] + q[6] * q[6]));
    history.push([simTime, q[0], q[1], yaw]);
    while (history.length > 60) history.shift();
    trail.push(q[0], q[1]);
    if (Math.max(Math.abs(q[0] - tc[0]), Math.abs(q[1] - tc[1])) > TER.recentre) moveTerrain(q[0], q[1]);
    if (q[2] < -0.2 || !Number.isFinite(q[2])) resetRobot();
  }

  // ---- three.js scene (MuJoCo is z-up) ----
  const stage = $("#stage");
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.outputEncoding = THREE.sRGBEncoding;   // colours are authored as sRGB hex; light in linear
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  stage.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.005, 20);
  camera.up.set(0, 0, 1);
  camera.position.set(0.32, -0.38, 0.24);
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0.05);
  controls.enableDamping = true;
  controls.minDistance = 0.12; controls.maxDistance = 2.5;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x8899aa, 0.75));
  const sun = new THREE.DirectionalLight(0xffffff, 0.8);
  sun.position.set(0.4, -0.3, 0.9);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -0.3, right: 0.3, top: 0.3, bottom: -0.3, near: 0.05, far: 3 });
  sun.shadow.bias = -0.0005;
  scene.add(sun, sun.target);
  // ground: the terrain patch (same grid as the heightfield) inside a flat backdrop with a
  // square hole, both wearing one world-aligned grid texture (2 cm minor / 10 cm major squares)
  const gridCanvas = document.createElement("canvas");
  gridCanvas.width = gridCanvas.height = 512;
  const gridTex = (repeat) => {
    const t = new THREE.CanvasTexture(gridCanvas);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.encoding = THREE.sRGBEncoding;
    t.anisotropy = renderer.capabilities.getMaxAnisotropy();
    t.repeat.set(repeat, repeat);
    return t;
  };
  const TILE = 0.1;                                    // one texture tile = 10 cm
  const terrainTex = gridTex((2 * TER.half) / TILE), backTex = gridTex(1 / TILE);
  const terrainGeo = new THREE.PlaneGeometry(2 * TER.half, 2 * TER.half, TER.n - 1, TER.n - 1);
  const terrainMesh = new THREE.Mesh(terrainGeo, new THREE.MeshStandardMaterial({ map: terrainTex, roughness: 1 }));
  terrainMesh.receiveShadow = terrainMesh.castShadow = true;
  const backShape = new THREE.Shape();
  backShape.moveTo(-4, -4); backShape.lineTo(4, -4); backShape.lineTo(4, 4); backShape.lineTo(-4, 4);
  const hole = new THREE.Path(), hh = TER.half - 1e-4;
  hole.moveTo(-hh, -hh); hole.lineTo(-hh, hh); hole.lineTo(hh, hh); hole.lineTo(hh, -hh);
  backShape.holes.push(hole);
  const backdrop = new THREE.Mesh(new THREE.ShapeGeometry(backShape), new THREE.MeshStandardMaterial({ map: backTex, roughness: 1 }));
  backdrop.receiveShadow = true;
  scene.add(terrainMesh, backdrop);
  function drawGrid(bg, minor, major) {
    const g = gridCanvas.getContext("2d"), s = gridCanvas.width;
    g.fillStyle = bg; g.fillRect(0, 0, s, s);
    g.strokeStyle = minor; g.lineWidth = 4;
    for (let i = 1; i < 5; i++) {
      const p = (i * s) / 5;
      g.beginPath(); g.moveTo(p, 0); g.lineTo(p, s); g.moveTo(0, p); g.lineTo(s, p); g.stroke();
    }
    g.strokeStyle = major; g.lineWidth = 8;
    g.strokeRect(0, 0, s, s);
    terrainTex.needsUpdate = backTex.needsUpdate = true;
  }

  // terrain state: centre (world, cell-aligned), seed, and the height function
  let tc = [0, 0], seed = 1, noises = [];
  const reseed = () => { noises = LEVELS.map((_, i) => perlin(seed * 7919 + i * 104729)); };
  reseed();
  const flat = () => LEVELS.every((L) => L.h === 0);
  function groundRaw(x, y) {                           // metres, before the edge taper
    let z = 0;
    LEVELS.forEach((L, i) => { if (L.h) z += (L.h / 2000) * noises[i](x / (L.s / 100), y / (L.s / 100)); });
    return Math.min(TER.zmax / 2, Math.max(-TER.zmax / 2, z));
  }
  function taper(x, y) {
    const e = TER.half - Math.max(Math.abs(x - tc[0]), Math.abs(y - tc[1]));
    const t = Math.min(1, Math.max(0, e / TER.taper));
    return t * t * (3 - 2 * t);
  }
  const groundAt = (x, y) => (flat() ? 0 : groundRaw(x, y) * taper(x, y));
  function footprintTop(x, y) {                        // highest ground within 9 cm of (x, y)
    if (flat()) return 0;
    let top = -Infinity;
    for (let i = -8; i <= 8; i++) for (let j = -8; j <= 8; j++) {
      if (i * i + j * j <= 64) top = Math.max(top, groundAt(x + i * 0.011, y + j * 0.011));
    }
    return top;
  }
  function fillTerrain() {
    const n = TER.n, hd = model.hfield_data, pos = terrainGeo.attributes.position.array, isFlat = flat();
    for (let r = 0; r < n; r++) {
      const dy = (2 * r / (n - 1) - 1) * TER.half, y = tc[1] + dy;
      for (let c = 0; c < n; c++) {
        const dx = (2 * c / (n - 1) - 1) * TER.half;
        const z = isFlat ? 0 : groundRaw(tc[0] + dx, y) * taper(tc[0] + dx, y);
        hd[r * n + c] = (z + TER.zmax / 2) / TER.zmax;   // MuJoCo row r is y; three.js rows run +y down
        pos[3 * ((n - 1 - r) * n + c) + 2] = z;
      }
    }
    terrainGeo.attributes.position.needsUpdate = true;
    terrainGeo.computeVertexNormals();
  }
  // move the heightfield (and its mesh) to be centred near (x, y), then refill it. A static
  // geom's world position is only recomputed by mj_setConst, which also overwrites the state,
  // so the state is saved around it
  function moveTerrain(x, y) {
    const nx = Math.round(x / TER.cell) * TER.cell, ny = Math.round(y / TER.cell) * TER.cell;
    const moved = nx !== tc[0] || ny !== tc[1];
    tc = [nx, ny];
    if (moved) {
      const gp = model.geom_pos;
      gp[3 * floorId] = nx; gp[3 * floorId + 1] = ny;
      const keep = { qpos: Float64Array.from(data.qpos), qvel: Float64Array.from(data.qvel),
        warm: Float64Array.from(data.qacc_warmstart), ctrl: Float64Array.from(data.ctrl), time: data.time };
      mujoco.mj_setConst(model, data);
      data.qpos.set(keep.qpos); data.qvel.set(keep.qvel); data.qacc_warmstart.set(keep.warm);
      data.ctrl.set(keep.ctrl); data.time = keep.time;
    }
    fillTerrain();
    mujoco.mj_forward(model, data);
    terrainMesh.position.set(nx, ny, 0);
    backdrop.position.set(nx, ny, 0);
    const off = (v) => (v - TER.half) / TILE;
    terrainTex.offset.set(off(nx), off(ny));
    backTex.offset.set(nx / TILE, ny / TILE);
  }
  // a slider moved: refill in place, and lift the robot if the ground rose under it
  let pendingTerrain = false;
  function terrainChanged(before) {
    if (pendingTerrain) return;
    pendingTerrain = true;
    requestAnimationFrame(() => {
      pendingTerrain = false;
      const q = data.qpos, rise = footprintTop(q[0], q[1]) - before;
      fillTerrain();
      if (rise > 0) q[2] += rise + 0.001;
      mujoco.mj_forward(model, data);
    });
  }

  // robot: one group per MuJoCo body, meshes in the body's own frame
  const groups = new Map(), headMeshes = [];
  for (const piece of meshMan) {
    const pos = new Float32Array(meshBin, piece.vertex_offset, piece.vertex_count * 3);
    const idx = new Uint32Array(meshBin, piece.index_offset, piece.index_count);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    g.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({ color: new THREE.Color(...piece.color).convertSRGBToLinear(), roughness: 0.62,
      metalness: 0.05, flatShading: true });
    const mesh = new THREE.Mesh(g, mat);
    mesh.castShadow = true;
    if (piece.group === "head") headMeshes.push(mesh);
    if (!groups.has(piece.body)) {
      const grp = new THREE.Group();
      grp.userData.id = mujoco.mj_name2id(model, BODY, piece.body);
      groups.set(piece.body, grp);
      scene.add(grp);
    }
    groups.get(piece.body).add(mesh);
  }
  const chkHead = $("#chk-head");
  const showHead = () => headMeshes.forEach((h) => { h.visible = chkHead.checked; });
  chkHead.addEventListener("change", showHead);
  showHead();
  // path trail and a ground arrow for the walking command
  const trail = {
    max: 3000, count: 0, buf: new Float32Array(3000 * 3),
    push(x, y) {
      if (this.count === this.max) { this.buf.copyWithin(0, 3); this.count--; }
      this.buf.set([x, y, groundAt(x, y) + 0.0008], this.count * 3); this.count++;
    },
  };
  const trailGeo = new THREE.BufferGeometry();
  trailGeo.setAttribute("position", new THREE.BufferAttribute(trail.buf, 3));
  const trailLine = new THREE.Line(trailGeo, new THREE.LineBasicMaterial());
  scene.add(trailLine);
  const arrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 0.12, 0xe27a12, 0.03, 0.02);
  scene.add(arrow);

  function applyTheme() {
    const cs = getComputedStyle(document.documentElement);
    const c = (n) => new THREE.Color(cs.getPropertyValue(n).trim() || "#888888").convertSRGBToLinear();
    scene.background = c("--scene");
    drawGrid(cs.getPropertyValue("--floor").trim() || "#035772", cs.getPropertyValue("--grid-minor").trim() || "#5d8a9a",
      cs.getPropertyValue("--grid-major").trim() || "#f6f9f5");
    trailLine.material.color = c("--trail");
    arrow.setColor(c("--arrow"));
  }
  applyTheme();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyTheme);
  new MutationObserver(applyTheme).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  function resize() {
    const r = stage.getBoundingClientRect();
    renderer.setSize(Math.max(1, r.width), Math.max(1, r.height), false);
    renderer.domElement.style.width = "100%"; renderer.domElement.style.height = "100%";
    camera.aspect = Math.max(1, r.width) / Math.max(1, r.height);
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(stage);
  resize();

  const q3 = new THREE.Quaternion();
  let lastRoot = null;
  function draw() {
    const xp = data.xpos, xq = data.xquat;
    for (const grp of groups.values()) {
      const id = grp.userData.id;
      grp.position.set(xp[3 * id], xp[3 * id + 1], xp[3 * id + 2]);
      grp.quaternion.set(xq[4 * id + 1], xq[4 * id + 2], xq[4 * id + 3], xq[4 * id]);
    }
    const q = data.qpos;
    const root = new THREE.Vector3(q[0], q[1], 0.05);
    if ($("#sel-cam").value === "follow") {
      if (lastRoot) { const d = root.clone().sub(lastRoot); camera.position.add(d); controls.target.add(d); }
    }
    lastRoot = root;
    sun.position.set(q[0] + 0.4, q[1] - 0.3, 0.9); sun.target.position.set(q[0], q[1], 0);
    trailGeo.setDrawRange(0, trail.count);
    trailGeo.attributes.position.needsUpdate = true;
    if (mode.kind === "walk") {
      const yaw = Math.atan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] * q[5] + q[6] * q[6]));
      const th = yaw + (dirs()[mode.k] * Math.PI) / 180;
      arrow.visible = true;
      arrow.position.set(q[0], q[1], groundAt(q[0], q[1]) + 0.002);
      arrow.setDirection(new THREE.Vector3(Math.cos(th), Math.sin(th), 0));
    } else arrow.visible = false;
    controls.update();
    renderer.render(scene, camera);
  }

  // ---- readouts ----
  const history = [];
  const servoEls = meta.servo_joints.map((n, i) => {
    const row = document.createElement("div");
    row.className = "servo";
    row.innerHTML = `<span class="n">S${i + 1}</span><span class="bar"><i></i></span><span class="a">0°</span>`;
    $("#servos").append(row);
    return { bar: row.querySelector("i"), ang: row.querySelector(".a") };
  });
  function readouts() {
    $("#hud-time").textContent = simTime.toFixed(1) + " s";
    if (history.length > 10) {
      const a = history[0], b = history[history.length - 1], dt = b[0] - a[0];
      let turn = 0;
      for (let i = 1; i < history.length; i++) {
        let d = history[i][3] - history[i - 1][3];
        d = Math.atan2(Math.sin(d), Math.cos(d)); turn += d;
      }
      $("#ro-speed").textContent = (Math.hypot(b[1] - a[1], b[2] - a[2]) / dt * 1000).toFixed(0);
      $("#ro-turn").textContent = ((turn / dt) * 180 / Math.PI).toFixed(0);
    }
    const q = readQ();
    servoEls.forEach((s, i) => {
      const f = Math.min(1, loads[i] / MG.effort);
      s.bar.style.width = (f * 100).toFixed(0) + "%";
      s.bar.classList.toggle("hot", f > 0.9);
      s.ang.textContent = ((q[i] * 180) / Math.PI).toFixed(0) + "°";
    });
  }

  // ---- direction pad (top view, head up). Walk directions measured in MuJoCo, base frame ----
  const DIRS = [-90, 150, 30];                        // remap k = 0, 1, 2 (measured -88, 144, 36)
  const KEYS = ["W", "E", "Q"];
  // any-direction walkers: 12 commanded directions, every 30 deg from the head (the lab's check set)
  const OMNI_DIRS = Array.from({ length: 12 }, (_, i) => ((-90 + 30 * i + 540) % 360) - 180);
  const dirs = () => (isOmni() || isCombo() ? OMNI_DIRS : DIRS);
  const nearest = (deg) => {                          // index of the closest direction in dirs()
    const d = dirs().map((x) => Math.abs(((x - deg + 540) % 360) - 180));
    return d.indexOf(Math.min(...d));
  };
  const keyOf = (deg) => { const i = DIRS.indexOf(deg); return i < 0 ? "" : KEYS[i]; };
  const fromHead = (deg) => {                         // degrees clockwise from the head on the pad
    const [x, y] = toPad(deg); return Math.round((Math.atan2(x, -y) * 180 / Math.PI + 360) % 360) % 360;
  };
  const pad = $("#pad");
  const toPad = (deg) => { const t = (deg * Math.PI) / 180; return [-Math.cos(t), Math.sin(t)]; };
  function wedge(deg, w1 = 20, w2 = 32) {
    const [ux, uy] = toPad(deg), px = -uy, py = ux;
    const P = (r, w) => `${(ux * r + px * w).toFixed(1)},${(uy * r + py * w).toFixed(1)}`;
    return `M${P(56, -w1)} L${P(84, -w1)} L${P(84, -w2)} L${P(112, 0)} L${P(84, w2)} L${P(84, w1)} L${P(56, w1)} Z`;
  }
  function arc(sign) {
    const r1 = 60, r2 = 76, a0 = sign > 0 ? 200 : -20, a1 = sign > 0 ? 250 : -70;
    const pt = (r, a) => `${(r * Math.cos((a * Math.PI) / 180)).toFixed(1)},${(r * Math.sin((a * Math.PI) / 180)).toFixed(1)}`;
    const sw = sign > 0 ? 1 : 0;
    return `M${pt(r1, a0)} A${r1},${r1} 0 0 ${sw} ${pt(r1, a1)} L${pt(r1 - 8, a1)} L${pt((r1 + r2) / 2, a1 + (sign > 0 ? 16 : -16))} ` +
      `L${pt(r2 + 8, a1)} L${pt(r2, a1)} A${r2},${r2} 0 0 ${1 - sw} ${pt(r2, a0)} Z`;
  }
  function renderPad() {
    const hex = Array.from({ length: 6 }, (_, i) => {
      const a = ((60 * i + 30) * Math.PI) / 180; return `${(40 * Math.cos(a)).toFixed(1)},${(40 * Math.sin(a)).toFixed(1)}`;
    }).join(" ");
    let s = `<polygon class="body" points="${hex}"/><circle class="head" cx="0" cy="-22" r="6"/>` +
      `<text class="cap" x="0" y="6">HEAD</text><text class="cap" x="0" y="18">UP</text>`;
    const omni = isOmni() || isCombo();
    dirs().forEach((d, k) => {
      const [ux, uy] = toPad(d), on = mode.kind === "walk" && mode.k === k, key = keyOf(d);
      const name = omni ? (fromHead(d) ? `${fromHead(d)} degrees clockwise from the head` : "toward the head")
        : k === 0 ? "toward the head" : "direction " + (k + 1);
      s += `<g class="dir${on ? " on" : ""}" tabindex="0" role="button" aria-pressed="${on}" data-walk="${k}"
        aria-label="Walk ${name}${key ? ` (key ${key})` : ""}">
        <path d="${omni ? wedge(d, 8, 14) : wedge(d)}"/>${key ? `<text x="${(ux * 70).toFixed(1)}" y="${(uy * 70 + 4).toFixed(1)}" text-anchor="middle">${key}</text>` : ""}</g>`;
    });
    [[1, "A", "counter-clockwise"], [-1, "D", "clockwise"]].forEach(([sg, key, name]) => {
      const on = mode.kind === "turn" && mode.dir === sg, a = sg > 0 ? 225 : -45;
      const tx = 90 * Math.cos((a * Math.PI) / 180), ty = 90 * Math.sin((a * Math.PI) / 180);
      s += `<g class="dir${on ? " on" : ""}" tabindex="0" role="button" aria-pressed="${on}" data-turn="${sg}"
        aria-label="Turn ${name} (key ${key})"><path d="${arc(sg)}"/>
        <text x="${tx.toFixed(1)}" y="${(ty + 4).toFixed(1)}" text-anchor="middle">${key}</text></g>`;
    });
    pad.innerHTML = s;
    $("#hud-policy").textContent = mode.kind === "walk"
      ? (fromHead(dirs()[mode.k]) ? `walk ${fromHead(dirs()[mode.k])}° from head` : "walk toward head")
      : mode.kind === "turn" ? (mode.dir > 0 ? "turn ccw" : "turn cw") : "stand";
  }
  function updateButtons() { $("#btn-stop").classList.toggle("on", mode.kind === "stand"); }
  const pick = (el) => {
    const g = el.closest(".dir");
    if (!g) return;
    if (g.dataset.walk !== undefined) setMode({ kind: "walk", k: +g.dataset.walk });
    else setMode({ kind: "turn", dir: +g.dataset.turn });
  };
  pad.addEventListener("click", (e) => pick(e.target));
  pad.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(e.target); } });
  $("#btn-stop").addEventListener("click", () => setMode({ kind: "stand" }));
  $("#btn-reset").addEventListener("click", () => { resetRobot(); setMode({ kind: "stand" }); });
  // terrain sliders: height (peak-to-peak mm) and feature size (cm) per noise layer
  const terr = $("#terrain");
  const fmt = { h: (v) => `${v} mm`, s: (v) => `${v} cm` };
  terr.innerHTML = LEVELS.map((L, i) => `<div class="lvl"><span class="lvl-name">${L.label}</span>
      <label class="sl"><span>height</span><input type="range" min="0" max="${L.hMax}" step="${L.hMax > 10 ? 1 : 0.5}"
        value="${L.h}" data-i="${i}" data-k="h" aria-label="${L.label} height in mm"><output>${fmt.h(L.h)}</output></label>
      <label class="sl"><span>size</span><input type="range" min="${L.sMin}" max="${L.sMax}" step="1"
        value="${L.s}" data-i="${i}" data-k="s" aria-label="${L.label} feature size in cm"><output>${fmt.s(L.s)}</output></label>
    </div>`).join("");
  const groundUnderRobot = () => footprintTop(data.qpos[0], data.qpos[1]);
  terr.addEventListener("input", (e) => {
    const el = e.target;
    if (el.type !== "range") return;
    const before = groundUnderRobot();
    LEVELS[+el.dataset.i][el.dataset.k] = +el.value;
    el.nextElementSibling.textContent = fmt[el.dataset.k](+el.value);
    terrainChanged(before);
  });
  $("#btn-seed").addEventListener("click", () => { const before = groundUnderRobot(); seed++; reseed(); terrainChanged(before); });
  $("#btn-flat").addEventListener("click", () => {
    const before = groundUnderRobot();
    LEVELS.forEach((L) => { L.h = 0; });
    terr.querySelectorAll('input[data-k="h"]').forEach((el) => { el.value = 0; el.nextElementSibling.textContent = fmt.h(0); });
    terrainChanged(before);
  });
  let rate = 1;
  $("#sel-rate").addEventListener("change", (e) => { rate = +e.target.value; $("#hud-rate").textContent = `${rate}×`; });
  window.addEventListener("keydown", (e) => {
    if (e.target.closest && e.target.closest("select, input")) return;
    const k = e.key.toLowerCase();
    const map = { w: { kind: "walk", k: nearest(DIRS[0]) }, e: { kind: "walk", k: nearest(DIRS[1]) },
      q: { kind: "walk", k: nearest(DIRS[2]) },
      a: { kind: "turn", dir: 1 }, d: { kind: "turn", dir: -1 }, s: { kind: "stand" } };
    if (map[k]) setMode(map[k]);
    else if (k === "r") { resetRobot(); setMode({ kind: "stand" }); }
  });

  resetRobot();
  renderPad(); updateButtons();
  say("");
  let acc = 0, last = performance.now(), tick = 0;
  function frame(now) {
    acc += Math.min(0.1, (now - last) / 1000) * rate;
    last = now;
    let n = 0;
    while (acc >= ctrlDt && n < 8) { controlStep(); acc -= ctrlDt; n++; }
    if (n === 8) acc = 0;
    draw();
    if (++tick % 6 === 0) readouts();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  window.stewy = { model, data, mujoco, setMode, get mode() { return mode; }, get simTime() { return simTime; },
    history, controlStep, LEVELS, terrainChanged, groundAt, get terrainCentre() { return tc; } };
}

main().catch((e) => { console.error(e); say(e.message || String(e), true); });
