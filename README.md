# Stewy web sim

**[Try it in your browser →](https://jd-patten.github.io/stewy-websim/web/)**

Stewy is a small six-servo Stewart-platform walking robot with no sensors. This is a physics simulation of it that runs entirely in your browser: [MuJoCo](https://mujoco.org) compiled to WebAssembly, drawn with three.js, and driven by the same neural-network policies that were trained in NVIDIA Isaac Lab for the real robot.

The robot's firmware and hardware live in [JD-Patten/stewy](https://github.com/JD-Patten/stewy).

## What you can do

- **Walk in 12 directions**, every 30°. Three walking policies cover them all: Stewy is exactly 3-fold and mirror symmetric, so each policy is reused by rotating its servo commands 120° and mirroring them.
  - **A** walks toward the head. It's the smoothest walker, and the one on the real robot.
  - **B** walks away from the head, on a time-reversed firmware gait.
  - **C** walks 60° off the head axis, on a steered firmware gait. Mirroring gives it six directions.
- **Turn in place** either way. Clockwise is the mirror image of counter-clockwise.
- **Show or hide the head** with the Head checkbox. It's drawn only; the simulated robot's mass doesn't include it.
- **Change the ground**: hills, bumps and grit sliders. Every policy was trained on flat ground, blind, so this shows how far that carries.

| Input | Action |
|---|---|
| Direction pad, or <kbd>W</kbd> <kbd>Q</kbd> <kbd>E</kbd> | Walk |
| <kbd>A</kbd> <kbd>D</kbd> | Turn in place |
| <kbd>S</kbd> | Stand |
| <kbd>R</kbd> | Reset |

## How it works

- **Model** (`web/stewy.xml`): built from the Isaac Lab USD articulation, with 20 bodies, the 5 closed leg loops as `connect` constraints, and 12 contact spheres. The foot and arm friction come from pull tests on the real robot.
- **Servos:** a model of the MG90S measured on a test bench runs every 2 ms physics step: its torque–speed limit, stick-slip deadband and 6.7 ms delay.
- **Policies** (`web/policies/*.json`): small MLPs that take a gait clock and a command, and output corrections to the firmware gait table (walkers) or a top-plate pose solved through the firmware's inverse kinematics (turners). The page reproduces the training environment's observation and action path exactly. Each exported network is checked against PyTorch to about 1e-5.
- **Transfer:** the policies were never trained in MuJoCo. They walk about 40–45 mm/s here, against about 42–43 mm/s in Isaac Lab, and turn at about 40–50°/s.

## Running it locally

The page has to be served over HTTP; it can't be opened from disk.

```bash
cd web
python -m http.server 8733
```

Then open http://localhost:8733.

## Rebuilding

`build/` regenerates everything in `web/`:

- `build_mjcf.py`: `stewy_usd.json` (dumped from the Isaac USD) → `web/stewy.xml` and `web/stewy_meta.json`.
- `build_web.py`: `build/policies/*.json` (full-precision exports from Isaac Lab, including earlier walkers and turners kept for comparison) → compact `web/policies/` for the policies the page ships (`SHIP`), the manifest, the mesh pack and `web/index.html`.
- `stewy_usd_meshes.py` (runs in Isaac Lab): the robot's visual meshes from the USD, coloured by part name with `palette.json` (`--palette build/palette.json --out_dir web/meshes`).
- `build_urdf_parts.py`: parts from the Onshape URDF export that the Isaac USD lacks: the head (shown or hidden by the page's Head checkbox) and the ball bearings. The URDF is in its CAD pose, so each part is placed in a sim body's frame by matching shared geometry (the six top bearings against the plate's hinge anchors, leg by leg). `python build_urdf_parts.py --urdf "<...>/main_assembly"`.
- `harness.py`: runs any policy in Python MuJoCo with the same servo law and observation path as the page, to measure it outside the browser (`python harness.py walk_smoother`).

## Credits

- [MuJoCo](https://github.com/google-deepmind/mujoco) 3.14 WebAssembly build (`web/vendor/`), Apache License 2.0.
- [three.js](https://threejs.org) r128, MIT License, loaded from a CDN.
- Everything else © James Patten, MIT License (see `LICENSE`).
