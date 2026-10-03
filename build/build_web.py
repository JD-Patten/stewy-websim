"""Assemble the web folder: compact the exported policies (6 significant digits), write
the policy manifest, and wrap app.html (the artifact page body) into a standalone
index.html for local serving / a Hugging Face Space.

    python build_web.py
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(HERE, "..", "web")
os.makedirs(os.path.join(WEB, "policies"), exist_ok=True)


def rnd(x):
    if isinstance(x, float):
        return float(f"{x:.6g}")
    if isinstance(x, list):
        return [rnd(v) for v in x]
    if isinstance(x, dict):
        return {k: rnd(v) for k, v in x.items()}
    return x


# walker menu names for the page (label, short HUD name); others fall back to the export label
MENU = {"walk62": ("62 mm/s: fastest, shakiest", "walk62"),
        "walk_smooth": ("51 mm/s: smooth, 1.25 s cycle", "smooth"),
        "walk_smoother": ("43 mm/s: smoothest, on the robot", "smoothest"),
        "omni_walk60": ("25 mm/s: any direction (60° wedge)", "omni60"),
        "omni_walk60_s7": ("47 mm/s: any direction, drifts in yaw (newest omni)", "omni60s7"),
        "dir_b_away": ("42 mm/s: away from the head (group B)", "B"),
        "dir_c_steer": ("42 mm/s: -60° steered firmware gait (group C)", "C"),
        "spin_ccw": ("48°/s: first robot turner, wobbly", "robot"),
        "spin_steady": ("50°/s: steady rate", "steady"),
        "spin_plate": ("49°/s: calm plate", "calm"),
        "spin_both": ("51°/s: steady + calm", "both"),
        "spin_calm_legs": ("42°/s: calm legs, on the robot", "calmlegs")}

# the three-walker set (RL Lab 2026-09-30, three directions): each group's walker covers its
# directions by rotating (and for C, mirroring) the legs; the page combines them into 12
GROUP = {"walk_smoother": "A", "dir_b_away": "B", "dir_c_steer": "C"}

manifest = []
for p in sorted(glob.glob(os.path.join(HERE, "policies", "*.json"))):
    d = json.load(open(p))
    name = os.path.splitext(os.path.basename(p))[0]
    d.pop("checkpoint", None)
    out = os.path.join(WEB, "policies", name + ".json")
    json.dump(rnd(d), open(out, "w"), separators=(",", ":"))
    label, short = MENU.get(name, (d.get("label", name), name))
    manifest.append(dict(name=name, label=label, short=short, mode=d["obs"]["command_mode"],
                         action=d["action"]["kind"], **({"group": GROUP[name]} if name in GROUP else {})))
    print(f"{name}: {os.path.getsize(out)/1024:.0f} KB")
json.dump(manifest, open(os.path.join(WEB, "policies", "manifest.json"), "w"), indent=1)

# meshes: one JSON with the binary inlined as base64 (artifact hosts do not serve .bin)
import base64
man = json.load(open(os.path.join(WEB, "meshes", "stewy_meshes.json")))
bin_ = open(os.path.join(WEB, "meshes", "stewy_meshes.bin"), "rb").read()
json.dump(dict(pieces=man, data=base64.b64encode(bin_).decode()), open(os.path.join(WEB, "meshes", "stewy_mesh_pack.json"), "w"))
print(f"mesh pack: {os.path.getsize(os.path.join(WEB, 'meshes', 'stewy_mesh_pack.json'))/1024:.0f} KB")

body = open(os.path.join(WEB, "app.html"), encoding="utf-8").read()
open(os.path.join(WEB, "index.html"), "w", encoding="utf-8").write(
    '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    '</head>\n<body>\n' + body + "\n</body>\n</html>\n")
print("index.html written")
