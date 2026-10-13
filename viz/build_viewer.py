"""Inline viewer_data.json and sim.js into the viewer page.

    python viz/build_viewer.py   (after viz_export.py)
"""
import os
here = os.path.dirname(os.path.abspath(__file__))
tpl = open(os.path.join(here, "viewer_template.html"), encoding="utf-8").read()
data = open(os.path.join(here, "viewer_data.json"), encoding="utf-8").read().replace("</", "<" + "\\" + "/")
sim = open(os.path.join(here, "sim.js"), encoding="utf-8").read()
sim = sim.replace('if (typeof module !== "undefined") module.exports = { mulberry32, buildAdj, buildDist, runAttack };', "")
assert tpl.count("/*DATA*/") == 1 and tpl.count("/*SIM*/") == 1
out = tpl.replace("/*DATA*/", data).replace("/*SIM*/", sim)
path = os.path.join(here, "backbone_breach_viewer.html")
open(path, "w", encoding="utf-8").write(out)
print(f"wrote {path} ({len(out) / 1024:.0f} KB)")
