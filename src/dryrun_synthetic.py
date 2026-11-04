"""TEST DATA ONLY. Fake synthetic families so the Q2 pipeline can be run end to
end before the team's real synthetic networks exist.

    python src/dryrun_synthetic.py <out_dir>

Writes <out_dir>/networks/*.graphml and <out_dir>/configs/*.json in exactly the
format of the real data (config names {topology}_d{density}_s{seed}.json,
densities 0.2-0.8, seeds 1-5, 24-package catalog, is_entry_point and
is_privileged flags; the highest-degree node is marked privileged, as in the real
configs). Four generic families, five networks each, 40 nodes. Results from
these networks mean nothing for the project; they only prove the code runs.
Replace with the real synthetic families when they arrive.
"""
import json, os, sys
import numpy as np
import networkx as nx
from scoring import build_table
from paths import CATALOG, EPSS
from sweep import DENSITIES, SEEDS, PATCHED

N = 40


def families():
    out = {}
    for i in range(1, 6):
        out[f"scale_free_{i}"] = nx.barabasi_albert_graph(N, 3, seed=100 + i)
        g = nx.gnp_random_graph(N, 0.09, seed=200 + i)
        out[f"random_{i}"] = g
        out[f"small_world_{i}"] = nx.connected_watts_strogatz_graph(N, 4, 0.1, seed=300 + i)
        t = nx.random_labeled_tree(N, seed=400 + i)
        rng = np.random.default_rng(400 + i)
        for _ in range(4):                       # a few cross-links so it is not a pure tree
            u, v = rng.choice(N, 2, replace=False)
            t.add_edge(int(u), int(v))
        out[f"tree_like_{i}"] = t
    return out


def main(out_dir):
    table = build_table(CATALOG, EPSS)[0]
    packages = sorted(k for k in table if k != PATCHED)
    os.makedirs(os.path.join(out_dir, "networks"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "configs"), exist_ok=True)
    for name, g in families().items():
        comps = sorted(nx.connected_components(g), key=len, reverse=True)
        g = nx.convert_node_labels_to_integers(g.subgraph(comps[0]).copy())
        nx.write_graphml(g, os.path.join(out_dir, "networks", f"{name}.graphml"))
        n = g.number_of_nodes()
        priv = max(g.nodes, key=lambda v: (g.degree(v), -v))
        for d in DENSITIES:
            for s in SEEDS:
                rng = np.random.default_rng([n, int(round(d * 100)), s, abs(hash(name)) % 997])
                k = int(round(d * n))
                vuln = set(rng.choice(n, k, replace=False).tolist())
                others = [v for v in range(n) if v != priv]
                entry = int(rng.choice(others))
                nodes = {}
                for v in range(n):
                    sw = packages[rng.integers(len(packages))] if v in vuln else PATCHED
                    nodes[str(v)] = {"software": sw, "p": table[sw],
                                     "is_entry_point": v == entry,
                                     "is_privileged": v == priv}
                cfg = {"topology": name, "vulnerability_density": d, "num_entry_points": 1,
                       "seed": s, "adversary": "assume_breach_random_entry", "nodes": nodes}
                json.dump(cfg, open(os.path.join(out_dir, "configs", f"{name}_d{d}_s{s}.json"), "w"))
    print(f"wrote dry-run networks and configs to {out_dir}")


if __name__ == "__main__":
    main(sys.argv[1])
