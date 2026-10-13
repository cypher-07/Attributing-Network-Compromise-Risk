"""Validate the EPSS-driven model against the real Aarnet topology.

    python src/check.py
"""
import json, sys
import numpy as np
import networkx as nx
from simulate import prepare, simulate, percolation_threshold
from scoring import build_table, P_FLOOR
from paths import CATALOG, EPSS, NETWORKS, CONFIGS
import os

tbl, epss, cat = build_table(CATALOG, EPSS)
if json.load(open(EPSS)).get("_fetched") == "MOCK-DO-NOT-USE":
    print("!! using MOCK EPSS -- numbers below are a pipeline test, not results\n")

G = nx.Graph(nx.read_graphml(os.path.join(NETWORKS, "Aarnet.graphml")))
nodes = sorted(G.nodes(), key=int)
n = len(nodes)
A, D, idx = prepare(G, nodes)
print(f"Aarnet: n={n}, edges={G.number_of_edges()}, "
      f"Molloy-Reed threshold pc={percolation_threshold(G):.3f}\n")

cfg = json.load(open(os.path.join(CONFIGS, "Aarnet_d0.2_s1.json")))
p_real = np.array([tbl[cfg["nodes"][v]["software"]] for v in nodes])
entry = [idx[v] for v, d in cfg["nodes"].items() if d["is_entry_point"]][0]

print("=== 1. neighbour effect: P(fall) = 1 - (1-p)^k ===")
print(f"  {'k':>3s} {'p=0.03 patched':>16s} {'p=0.83 vsftpd':>16s}")
for k in (1, 2, 4, 6):
    print(f"  {k:3d} {1-(1-0.03)**k:16.3f} {1-(1-0.826)**k:16.3f}")

print("\n=== 2. your config file, three reachability modes ===")
for mode in ("hop", "decay", "flat"):
    r = simulate(A, D, p_real, entry, 4000, mode=mode, seed=1)["frac"]
    print(f"  {mode:6s} mean {r.mean():.3f}  sd {r.std():.3f}  "
          f"p10 {np.quantile(r,.1):.3f}  p90 {np.quantile(r,.9):.3f}")

print("\n=== 3. across density, random entry ===")
rng = np.random.default_rng(5)
p_vuln = tbl["Apache HTTP Server 2.4.49"]
print(f"{'density':>8s} {'d/pc':>6s}  {'hop mean [sd]':>18s} {'decay mean [sd]':>18s}")
pc = percolation_threshold(G)
for d in (0.2, 0.4, 0.6, 0.8):
    acc = {"hop": [], "decay": []}
    for rep in range(40):
        vuln = rng.random(n) < d
        p = np.where(vuln, p_vuln, P_FLOOR)
        e = rng.integers(n, size=200)
        for m in acc:
            acc[m].append(simulate(A, D, p, e, 200, mode=m, seed=rep)["frac"])
    cols = []
    for m in ("hop", "decay"):
        a = np.concatenate(acc[m])
        cols.append(f"{a.mean():12.3f} [{a.std():.3f}]")
    print(f"{d:8.1f} {d/pc:6.2f}  {cols[0]:>18s} {cols[1]:>18s}")
