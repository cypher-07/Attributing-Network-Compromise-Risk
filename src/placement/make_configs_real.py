import csv
import json
import os
import networkx as nx
from assign_config_v3 import assign_configuration

os.makedirs("configs_real", exist_ok=True)

DENSITIES = [0.2, 0.4, 0.6, 0.8]
SEEDS = [1, 2, 3, 4, 5]
ENTRY_POINTS = 1

manifest = []
for filename in sorted(os.listdir("real_networks")):
    if not filename.endswith(".graphml"):
        continue
    name = filename.replace(".graphml", "")
    G = nx.read_graphml(os.path.join("real_networks", filename))

    if not nx.is_connected(nx.Graph(G)):
        biggest = max(nx.connected_components(nx.Graph(G)), key=len)
        G = G.subgraph(biggest).copy()

    for density in DENSITIES:
        for seed in SEEDS:
            cfg = assign_configuration(G, vulnerability_density=density,
                                       num_entry_points=ENTRY_POINTS, seed=seed)
            out = {
                "topology": name,
                "vulnerability_density": density,
                "num_entry_points": ENTRY_POINTS,
                "seed": seed,
                "nodes": cfg,
            }
            fname = f"{name}_d{density}_s{seed}.json"
            with open(os.path.join("configs_real", fname), "w") as f:
                json.dump(out, f, indent=2)
            n_vuln = sum(1 for v in cfg.values() if v["n_cves"] > 0)
            manifest.append({"file": fname, "topology": name, "density": density,
                             "seed": seed, "machines": len(cfg),
                             "vulnerable_machines": n_vuln})
    print(f"Done: {name}")

with open("configs_real_manifest.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
    writer.writeheader()
    writer.writerows(manifest)

print(f"Wrote {len(manifest)} config files to configs_real/")