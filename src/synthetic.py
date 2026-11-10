"""Synthetic network families for Q2 (cross-family transfer).

    python src/synthetic.py

Writes, without touching anything for the 20 Zoo networks:
  data/networks_synthetic/<family>_<i>.graphml   20 graphs, 5 per family
  data/configs_synthetic/<name>_d<density>_s<seed>.json   400 configs
  data/configs_synthetic_manifest.csv
  data/synthetic_features.csv                    10 features + n, edges, pc

Families (5 graphs each)
------------------------
  scale_free   extended Barabasi-Albert, m=1, p=0.45 (preferential attachment, hubs)
  random       random tree + uniformly random extra edges (1.4n edges total)
  small_world  Watts-Strogatz core (k=4, p=0.1) on 75% of nodes + dead-end leaves
  fat_tree     k-ary fat-tree switches (+ optional hosts); deterministic

Design choices (so the families differ in STRUCTURE, not in size or density)
---------------------------------------------------------------------------
  * scale_free, random, small_world share the same five sizes (20, 30, 45, 65,
    90 nodes) and the same mean degree (~4), so any difference between them is
    degree distribution / clustering / path length, not size or edge count.
    Zoo networks are sparser (mean degree ~2.8); that is a stated limitation.
  * fat_tree is fixed by its parameters: (k, hosts per edge switch) =
    (4,0) 20 nodes, (4,2) 36, (6,0) 45, (6,1) 63, (8,0) 80. Its five graphs
    differ in size only, with no randomness, so it has the least variety.
  * Graph seeds are fixed (SEED_BASE + index) so the files regenerate exactly.

Configs reuse the real-network rules, unchanged:
  entry point  one node, uniform at random, seeded on the config seed
  privileged   highest-degree node(s): 1 per network (1% of nodes, minimum 1)
  vulnerable   each node independently with probability = density
  software     redrawn by config_gen.regenerate from (n, density, seed) only,
               never from structure (rule 5), 24-package catalog
"""
import json, os, random, sys
import networkx as nx
import numpy as np
import pandas as pd
from paths import (NETWORKS_SYN, CONFIGS_SYN, CONFIGS_SYN_MANIFEST,
                   SYN_FEATURES, CATALOG, EPSS, rel)

SEED_BASE = 610
SIZES = (15, 20, 25, 30, 40, 50, 60, 70, 85, 100)
FAT_TREE_PARAMS = ((4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
                   (6, 1), (6, 2), (6, 3), (6, 4))
DENSITIES = (0.2, 0.4, 0.6, 0.8)
SEEDS = (1, 2, 3, 4, 5)
FAMILIES = ("scale_free", "random", "small_world", "fat_tree")
PATCHED = "Fully patched (no known CVEs)"


def fat_tree(k, hosts_per_edge=0):
    """k-ary fat-tree: (k/2)^2 core, k pods of k/2 aggregation + k/2 edge
    switches, optional hosts under each edge switch. Nodes carry a 'role'."""
    h = k // 2
    G = nx.Graph()
    core = [("c", i) for i in range(h * h)]
    for c in core:
        G.add_node(c, role="core")
    for p in range(k):
        agg = [("a", p, i) for i in range(h)]
        edg = [("e", p, i) for i in range(h)]
        for a in agg:
            G.add_node(a, role="aggregation")
        for e in edg:
            G.add_node(e, role="edge")
        for a in agg:
            for e in edg:
                G.add_edge(a, e)
        for i, a in enumerate(agg):             # agg i links to core group i
            for j in range(h):
                G.add_edge(a, core[i * h + j])
        for e in edg:
            for x in range(hosts_per_edge):
                host = ("h",) + e[1:] + (x,)
                G.add_node(host, role="host")
                G.add_edge(e, host)
    return nx.convert_node_labels_to_integers(G, label_attribute="label")


def connected_or_redraw(make, tries=100):
    for t in range(tries):
        G = make(t)
        if nx.is_connected(G):
            return G
    raise RuntimeError("could not draw a connected graph")


def random_graph(n, seed, edge_ratio=1.4):
    """Connected random graph with edge_ratio*n edges: a uniformly random
    labelled tree plus extra edges drawn uniformly at random. (Plain G(n,m) at
    this sparsity is almost never connected.)"""
    G = nx.random_labeled_tree(n, seed=seed)
    rng = random.Random(seed)
    while G.number_of_edges() < round(edge_ratio * n):
        u, v = rng.sample(range(n), 2)
        G.add_edge(u, v)
    return G


def small_world_with_leaves(n, seed, leaf_share=0.25):
    """Watts-Strogatz core (k=4, p=0.1) on 75% of the nodes; the rest hang off
    random core nodes as dead-end machines, as access routers do."""
    n_core = n - round(leaf_share * n)
    rng = random.Random(seed)
    G = nx.watts_strogatz_graph(n_core, 4, 0.1, seed=seed)
    for v in range(n_core, n):
        G.add_edge(v, rng.randrange(n_core))
    return G


def build_networks():
    """Returns {name: graph}, graphs relabelled to ints 0..n-1 with a 'family'
    attr. Every family has dead-end (degree-1) machines, like the Zoo
    networks, because the external attacker starts at one."""
    out = {}
    for i, n in enumerate(SIZES):
        seed = SEED_BASE + i
        out[f"scale_free_{i+1}"] = connected_or_redraw(
            lambda t: nx.extended_barabasi_albert_graph(n, 1, 0.45, 0, seed=1000 * (t + 1) + seed))
        out[f"random_{i+1}"] = random_graph(n, seed)
        out[f"small_world_{i+1}"] = connected_or_redraw(
            lambda t: small_world_with_leaves(n, 1000 * (t + 1) + seed))
    for i, (k, hosts) in enumerate(FAT_TREE_PARAMS):
        out[f"fat_tree_{i+1}"] = fat_tree(k, hosts)
    for name, G in out.items():
        G = nx.Graph(G)
        G.remove_edges_from(list(nx.selfloop_edges(G)))
        G.graph["family"] = name.rsplit("_", 1)[0]
        out[name] = G
        for v in G.nodes:
            G.nodes[v].pop("label", None)       # tuples are not graphml-safe
    return out


def assign_old_style(G, density, seed):
    """Entry point, privileged node(s) and vulnerable set, with the same rules
    as placement/assign_config_v3.py. Software is a placeholder here;
    config_gen.regenerate redraws it."""
    rng = random.Random(seed)
    nodes = list(G.nodes())
    entry = {str(n) for n in rng.sample(nodes, k=1)}
    k = max(1, round(0.01 * len(nodes)))
    ranked = sorted(nodes, key=lambda n: (-G.degree(n), str(n)))
    priv = {str(n) for n in ranked[:k]}
    cfg = {}
    for v in nodes:
        sw = "placeholder" if rng.random() < density else PATCHED
        cfg[str(v)] = {"software": sw, "is_entry_point": str(v) in entry,
                       "is_privileged": str(v) in priv}
    return {"vulnerability_density": density, "num_entry_points": 1,
            "seed": seed, "nodes": cfg}


def main():
    from scoring import build_table
    from config_gen import regenerate
    from topology import features, load_topology
    from simulate import percolation_threshold

    os.makedirs(NETWORKS_SYN, exist_ok=True)
    os.makedirs(CONFIGS_SYN, exist_ok=True)
    nets = build_networks()
    for name, G in nets.items():
        nx.write_graphml(G, os.path.join(NETWORKS_SYN, f"{name}.graphml"))

    table, _, catalog = build_table(CATALOG, EPSS)
    packages = sorted(catalog)
    man, feats = [], []
    for name in nets:
        G, nodes, _ = load_topology(name, NETWORKS_SYN)   # same loader as Zoo
        assert G.number_of_nodes() == nets[name].number_of_nodes(), name
        f = features(G)
        feats.append({"topology": name, "family": nets[name].graph["family"],
                      "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
                      "pc": percolation_threshold(G), **f})
        for d in DENSITIES:
            for s in SEEDS:
                old = assign_old_style(G, d, s)
                old["topology"] = name
                cfg = regenerate(old, catalog, table, packages)
                fn = f"{name}_d{d}_s{s}.json"
                json.dump(cfg, open(os.path.join(CONFIGS_SYN, fn), "w"), indent=1)
                ps = [x["p"] for x in cfg["nodes"].values() if x["software"] != PATCHED]
                man.append({"file": fn, "topology": name, "family": nets[name].graph["family"],
                            "density": d, "seed": s, "machines": len(cfg["nodes"]),
                            "vulnerable_machines": len(ps),
                            "mean_p_vulnerable": round(float(np.mean(ps)), 4) if ps else None})
    fd = pd.DataFrame(feats)
    bad = [c for c in fd.columns if fd[c].dtype.kind == "f" and not np.isfinite(fd[c]).all()]
    if bad:
        sys.exit(f"non-finite features {bad}")
    fd.to_csv(SYN_FEATURES, index=False, float_format="%.6f")
    pd.DataFrame(man).to_csv(CONFIGS_SYN_MANIFEST, index=False)
    print(f"wrote {len(nets)} networks to {rel(NETWORKS_SYN)}/, {len(man)} configs to "
          f"{rel(CONFIGS_SYN)}/, {rel(SYN_FEATURES)}")
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", "{:.3f}".format):
        print(fd.drop(columns="pc").set_index("topology"))
        print("\nfamily means:\n", fd.groupby("family").mean(numeric_only=True).round(3))


if __name__ == "__main__":
    main()
