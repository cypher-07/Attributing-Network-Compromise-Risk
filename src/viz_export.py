"""Export networks, layouts and configs for the propagation viewer page.

    python src/viz_export.py

Layout: Topology Zoo latitude/longitude where at least 80% of a network's
nodes have it (equirectangular, x scaled by cos(mean latitude)); the rest are
placed by a spring layout with the geolocated nodes held fixed. Networks with
sparser coordinates (Airtel, Marwan) use a Kamada-Kawai layout instead.
Coincident points (two POPs in one city) are nudged apart.

Each config is stored as the software index per node (in load_topology order)
plus the entry index; p comes from scoring.py through the package table, so
the page and the simulator read the same numbers.
"""
import json, os, sys
import numpy as np
import networkx as nx
import pandas as pd
from scoring import build_table, attack_cves, P_FLOOR, P_CEIL
from topology import load_topology, topology_names
from catalog_fetch import ORIGINAL
from paths import CONFIGS, CATALOG, EPSS, SWEEP, TOPOLOGY_FEATURES, VIZ, rel

CONFIG_DIR = CONFIGS
EPSS_FILE = EPSS
DENSITIES = (0.2, 0.4, 0.6, 0.8)
SEEDS = (1, 2, 3, 4, 5)
PATCHED = "Fully patched (no known CVEs)"


def layout(G, nodes):
    geo = {v: (G.nodes[v]["Longitude"], G.nodes[v]["Latitude"]) for v in nodes
           if "Longitude" in G.nodes[v] and "Latitude" in G.nodes[v]}
    if len(geo) >= 0.8 * len(nodes):
        lat0 = np.radians(np.mean([la for _, la in geo.values()]))
        pos = {v: np.array([lo * np.cos(lat0), -la]) for v, (lo, la) in geo.items()}
        pts = np.array(list(pos.values()))
        span = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1e-6)
        pos = {v: (p - pts.min(axis=0)) / span for v, p in pos.items()}
        if len(pos) < len(nodes):
            pos = nx.spring_layout(G, pos=pos, fixed=list(pos), seed=7,
                                   k=0.6 / np.sqrt(len(nodes)), iterations=200)
        kind = "geographic"
    else:
        pos = nx.kamada_kawai_layout(G)
        kind = "schematic"
    xy = np.array([pos[v] for v in nodes], dtype=float)
    xy = (xy - xy.min(axis=0)) / max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 1e-9)
    # nudge coincident nodes apart (same city)
    rng = np.random.default_rng(3)
    for _ in range(40):
        moved = False
        for i in range(len(xy)):
            d = np.hypot(*(xy - xy[i]).T)
            close = np.where((d < 0.045) & (np.arange(len(xy)) != i))[0]
            if len(close):
                xy[i] += rng.normal(0, 0.012, 2)
                moved = True
        if not moved:
            break
    xy = (xy - xy.min(axis=0)) / max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 1e-9)
    return xy, kind


def main(out):
    table, epss, catalog = build_table(CATALOG, EPSS_FILE)
    packages = [PATCHED] + sorted(catalog)
    pkg_index = {s: i for i, s in enumerate(packages)}
    pkg_rows = []
    for s in packages:
        if s == PATCHED:
            pkg_rows.append({"name": s, "p": table[s], "cve": None, "epss": None,
                             "group": "patched"})
            continue
        net = attack_cves(catalog[s])
        best = max(net, key=lambda v: epss[v["cve"]]) if net else None
        pkg_rows.append({"name": s, "p": table[s], "cve": best and best["cve"],
                         "epss": best and round(epss[best["cve"]], 4),
                         "group": "original" if s in ORIGINAL else "patched_version"})

    sweep = pd.read_csv(SWEEP)
    sweep = sweep[(sweep["mode"] == "decay") & np.isclose(sweep["floor"], P_FLOOR)]
    feats = pd.read_csv(TOPOLOGY_FEATURES).set_index("topology")

    nets = []
    for topo in topology_names():
        G, nodes, _ = load_topology(topo)
        idx = {v: i for i, v in enumerate(nodes)}
        xy, kind = layout(G, nodes)
        cfgs = {}
        for d in DENSITIES:
            for s in SEEDS:
                c = json.load(open(os.path.join(CONFIG_DIR, f"{topo}_d{d}_s{s}.json")))["nodes"]
                row = sweep[(sweep.topo_id == topo) & np.isclose(sweep.density, d)
                            & (sweep.config_seed == s)].iloc[0]
                cfgs[f"{d}_{s}"] = {
                    "sw": [pkg_index[c[v]["software"]] for v in nodes],
                    "entry": idx[next(v for v in nodes if c[v]["is_entry_point"])],
                    "sweep_mean": round(float(row.compromise_fraction), 4),
                }
        nets.append({
            "name": topo, "layout": kind, "pc": round(float(feats.loc[topo, "pc"]), 3),
            "labels": [str(G.nodes[v].get("label", v)) for v in nodes],
            "xy": [[round(float(a), 4), round(float(b), 4)] for a, b in xy],
            "edges": [[idx[u], idx[v]] for u, v in G.edges()],
            "configs": cfgs,
        })
    data = {"floor": P_FLOOR, "ceil": P_CEIL, "packages": pkg_rows, "networks": nets,
            "densities": DENSITIES, "seeds": SEEDS, "max_rounds": 60}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(data, open(out, "w"), separators=(",", ":"))
    print(f"wrote {rel(out)} ({os.path.getsize(out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main(os.path.join(VIZ, "viewer_data.json"))
