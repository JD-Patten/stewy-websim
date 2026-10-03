---
title: Stewy Live
emoji: 🦿
colorFrom: gray
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
short_description: Stewy, a blind 6-servo Stewart-platform walker, in MuJoCo WASM
---

# Stewy Live

Stewy is a six-servo Stewart-platform walker with no sensors. This page runs it in **MuJoCo 3.14 (WebAssembly)** in the browser, driven by the same neural policies trained for the real robot in Isaac Lab.

## Controls

| Input | What it does |
|---|---|
| Direction pad, or <kbd>W</kbd> <kbd>Q</kbd> <kbd>E</kbd> | Walk. 12 directions, every 30°. <kbd>W</kbd> walks toward the head; <kbd>Q</kbd> and <kbd>E</kbd> are 120° either side. |
| <kbd>A</kbd> <kbd>D</kbd> | Turn in place |
| <kbd>S</kbd> | Stand |
| <kbd>R</kbd> | Reset |

## What's inside

**Model:** `stewy.xml` is built from the Isaac USD articulation: 20 bodies, the 5 closed kinematic loops as `connect` constraints, and 12 contact spheres. The friction values come from pull tests on the real robot.

**Servos:** the measured MG90S model runs every 2 ms physics step: the torque–speed curve, stick-slip and a 6.7 ms delay.

**Policies:** walking in 12 directions comes from three walkers (A toward the head, B away from it on a reversed firmware gait, C 60° off-axis on a steered gait), each reused by Stewy's exact 3-fold and mirror symmetry. They walk about 40–45 mm/s here (42–43 in Isaac Lab). Turning in place uses the calm-legs turner (42 °/s in Isaac Lab, about 41 here); clockwise is its mirror image. Earlier walkers and turners are in the menus for comparison.

**Rebuilding:** everything is regenerated from `../build` (see `build_mjcf.py`, `build_web.py`).
