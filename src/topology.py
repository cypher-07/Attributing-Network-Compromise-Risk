"""Topology Zoo ingestion and structural features.

    python src/topology.py

Ingestion
---------
Every network is reduced to a simple undirected graph on its largest connected
component. Parallel links (Topology Zoo MultiGraphs record several physical
circuits between the same POPs) are collapsed, self-loops dropped, and
everything removed is logged -- that log goes in threats to validity. Use
load_topology() everywhere so the simulator and the features see the same graph.

Features
--------
Ten, as specified in CLAUDE.md, all size-normalised so the model cannot learn
"big network = more risk". Node count and edge count are covariates, never
features. Spearman with n on the 20 Zoo networks in brackets.

  mean_degree          <k>. Intensive for sparse graphs; edge density is not
                       (it falls as 1/n), so density is deliberately not used.
  degree_gini          Gini of the degree sequence.
  betweenness_central  Freeman centralisation of normalised betweenness:
                       sum(max - b_i) / (n - 1). 1 for a star, 0 for a cycle.
  eigvec_gini          Gini of eigenvector centrality. Size-confounded [0.88]
                       -- eigenvector centrality localises on hubs in large
                       sparse graphs. PageRank Gini is not an alternative: it
                       duplicates degree_gini [0.99].
  norm_diameter        diameter / sqrt(n).
  norm_avg_path        average shortest path length / sqrt(n).
                       Distances in a roughly planar backbone grow like
                       sqrt(n), not n: dividing by (n - 1) over-corrects
                       [-0.86 / -0.94, against 0.01 / -0.32 with sqrt(n)].
  algebraic_conn       second-smallest eigenvalue of the NORMALISED Laplacian,
                       bounded in [0, 2]. Still falls with n [-0.83] under
                       every normalisation tried: larger sparse backbones
                       genuinely are less well connected.
  kcore_ratio          fraction of nodes in the main (maximum-k) core.
  avg_clustering       mean local clustering coefficient.
  assortativity        degree assortativity (Pearson).

Also written, not features: n_nodes, n_edges, pc (Molloy-Reed threshold, for
the d/pc anchor plot) and the ingestion log columns.
"""
import glob, os, sys
import numpy as np
import networkx as nx
import pandas as pd
from simulate import percolation_threshold
from paths import NETWORKS, TOPOLOGY_FEATURES, INGESTION_LOG, rel

NET_DIR = NETWORKS

FEATURES = ["mean_degree", "degree_gini", "betweenness_central", "eigvec_gini",
            "norm_diameter", "norm_avg_path", "algebraic_conn", "kcore_ratio",
            "avg_clustering", "assortativity"]


def load_topology(name, net_dir=NET_DIR):
    """Returns (G, nodes, log). G: simple undirected graph on the largest
    connected component. nodes: node ids in canonical order (numeric where the
    ids allow), which the simulator and config lookup must share."""
    raw = nx.read_graphml(os.path.join(net_dir, f"{name}.graphml"))
    G = nx.Graph(raw)                       # collapses parallel edges
    loops = nx.number_of_selfloops(G)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    dropped = sorted(set().union(*comps[1:])) if len(comps) > 1 else []
    G = G.subgraph(comps[0]).copy()
    try:
        nodes = sorted(G.nodes(), key=int)
    except ValueError:
        nodes = sorted(G.nodes())
    log = {
        "topology": name,
        "raw_nodes": raw.number_of_nodes(),
        "raw_edges": raw.number_of_edges(),
        "multigraph": raw.is_multigraph(),
        "parallel_edges_collapsed": raw.number_of_edges() - nx.Graph(raw).number_of_edges(),
        "self_loops_dropped": loops,
        "components": len(comps),
        "nodes_dropped": len(dropped),
        "dropped_ids": " ".join(map(str, dropped)),
    }
    return G, nodes, log


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def features(G):
    n = G.number_of_nodes()
    deg = np.array([d for _, d in G.degree()], dtype=float)
    bc = np.array(list(nx.betweenness_centrality(G, normalized=True).values()))
    ev = np.array(list(nx.eigenvector_centrality_numpy(G).values()))
    core = nx.core_number(G)
    kmax = max(core.values())
    lap = np.sort(nx.normalized_laplacian_spectrum(G))
    return {
        "mean_degree": deg.mean(),
        "degree_gini": gini(deg),
        "betweenness_central": float((bc.max() - bc).sum() / (n - 1)),
        "eigvec_gini": gini(np.abs(ev)),
        "norm_diameter": nx.diameter(G) / np.sqrt(n),
        "norm_avg_path": nx.average_shortest_path_length(G) / np.sqrt(n),
        "algebraic_conn": float(lap[1]),
        "kcore_ratio": sum(c == kmax for c in core.values()) / n,
        "avg_clustering": nx.average_clustering(G),
        "assortativity": nx.degree_assortativity_coefficient(G),
    }


def topology_names(net_dir=NET_DIR):
    return sorted(os.path.splitext(os.path.basename(f))[0]
                  for f in glob.glob(os.path.join(net_dir, "*.graphml")))


if __name__ == "__main__":
    net_dir, out = NET_DIR, TOPOLOGY_FEATURES
    feat_rows, log_rows = [], []
    for name in topology_names(net_dir):
        G, nodes, log = load_topology(name, net_dir)
        f = features(G)
        bad = [k for k, v in f.items() if not np.isfinite(v)]
        if bad:
            sys.exit(f"{name}: non-finite features {bad} -- refusing to write")
        feat_rows.append({"topology": name, "n_nodes": G.number_of_nodes(),
                          "n_edges": G.number_of_edges(),
                          "pc": percolation_threshold(G), **f})
        log_rows.append(log)

    feats = pd.DataFrame(feat_rows)
    feats.to_csv(out, index=False, float_format="%.6f")
    logs = pd.DataFrame(log_rows)
    logs.to_csv(INGESTION_LOG, index=False)
    print(f"wrote {rel(out)} ({len(feats)} topologies) and {rel(INGESTION_LOG)}\n")

    changed = logs[(logs.nodes_dropped > 0) | (logs.parallel_edges_collapsed > 0)
                   | (logs.self_loops_dropped > 0)]
    print("ingestion changes (threats to validity):")
    for _, r in changed.iterrows():
        print(f"  {r.topology:18s} parallel edges collapsed {r.parallel_edges_collapsed:3d}"
              f"  nodes dropped {r.nodes_dropped:2d} {r.dropped_ids}")

    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", "{:.3f}".format):
        print("\n", feats.set_index("topology"))
        corr = feats[FEATURES].corr().abs()
        pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(FEATURES)
                 for b in FEATURES[i + 1:] if corr.loc[a, b] > 0.8]
        print("\nfeature pairs with |r| > 0.8 (n=20):")
        for a, b, r in sorted(pairs, key=lambda t: -t[2]):
            print(f"  {a:20s} {b:20s} {r:.2f}")
        print("\ncorrelation with n_nodes (should be weak):")
        print(pd.DataFrame({
            "pearson": feats[FEATURES].corrwith(feats.n_nodes),
            "spearman": feats[FEATURES].corrwith(feats.n_nodes, method="spearman"),
        }).round(2).to_string())
