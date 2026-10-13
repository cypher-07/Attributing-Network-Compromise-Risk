"""Degree-preserving rewiring: is topology's effect carried by the degree
distribution, or by how the links are arranged?

    python src/rewiring.py [n_trials]

For every network, REPLICATES randomised copies are made at two rungs:

  degree      nx.connected_double_edge_swap: every node keeps its degree (so
              pc is unchanged); clustering, assortativity and the spatial
              chain structure are destroyed.
  joint       swaps only between edges whose swapped endpoints have equal
              degree: keeps the joint degree distribution (so assortativity
              too) and connectivity; clustering and long-range arrangement
              are still destroyed.

Each copy keeps node labels, so every config (software, entry point) applies
unchanged -- configuration is held fixed exactly, rule 5 holds. Every config
of the network is simulated on every copy with the primary model (decay,
shared exploits), and compared with the same config on the original graph.

Writes results/rewiring/rewiring.csv (one row per network x rung x replicate x
config) and results/rewiring/SUMMARY.md. A prediction was recorded in
CLAUDE.md before the first run.
"""
import json, os, sys, time
import numpy as np
import networkx as nx
import pandas as pd
from simulate import prepare, simulate
from scoring import build_table, P_FLOOR, P_CEIL
from topology import load_topology, topology_names, features
from paths import CONFIGS, CATALOG, EPSS, RESULTS, SWEEP, TOPOLOGY_FEATURES, rel

REPLICATES = 20
DENSITIES = (0.2, 0.4, 0.6, 0.8)
SEEDS = (1, 2, 3, 4, 5)
SWAPS_PER_EDGE = 10
OUT_DIR = os.path.join(RESULTS, "rewiring")
PATCHED = "Fully patched (no known CVEs)"


def rewire_degree(G, rng):
    H = G.copy()
    m = H.number_of_edges()
    nx.connected_double_edge_swap(H, nswap=SWAPS_PER_EDGE * m, seed=int(rng.integers(2 ** 31)))
    return H


def rewire_joint(G, rng):
    """Swap (a,b),(c,d) -> (a,d),(c,b) only when deg(b) == deg(d): every edge
    keeps the degrees at both ends, so the joint degree matrix is unchanged.
    Rejects swaps that disconnect the graph or create loops/multi-edges."""
    H = G.copy()
    deg = dict(H.degree())
    target = SWAPS_PER_EDGE * H.number_of_edges()
    done = tries = 0
    while done < target and tries < 50 * target:
        tries += 1
        edges = list(H.edges())
        (a, b), (c, d) = (edges[i] for i in rng.choice(len(edges), 2, replace=False))
        if rng.random() < 0.5:
            c, d = d, c
        if deg[b] != deg[d] or len({a, b, c, d}) < 4 or H.has_edge(a, d) or H.has_edge(c, b):
            continue
        H.remove_edges_from([(a, b), (c, d)])
        H.add_edges_from([(a, d), (c, b)])
        if nx.is_connected(H):
            done += 1
        else:
            H.remove_edges_from([(a, d), (c, b)])
            H.add_edges_from([(a, b), (c, d)])
    return H, done / target


def configs_for(topo, nodes, table, pkg_idx, q):
    out = []
    for d in DENSITIES:
        for s in SEEDS:
            cn = json.load(open(os.path.join(CONFIGS, f"{topo}_d{d}_s{s}.json")))["nodes"]
            sw = [cn[v]["software"] for v in nodes]
            p = np.array([table[x] for x in sw])
            entry = next(i for i, v in enumerate(nodes) if cn[v]["is_entry_point"])
            out.append((d, s, p, entry, (np.array([pkg_idx[x] for x in sw]), q, P_FLOOR, P_CEIL)))
    return out


def run(G, nodes, cfgs, n_trials, seed0):
    A, D, _ = prepare(G, nodes)
    return [float(simulate(A, D, p, e, n_trials, mode="decay", seed=seed0 + k, shared=sh)["frac"].mean())
            for k, (d, s, p, e, sh) in enumerate(cfgs)]


def main(n_trials):
    os.makedirs(OUT_DIR, exist_ok=True)
    table = build_table(CATALOG, EPSS)[0]
    packages = sorted(table)
    pkg_idx = {s: i for i, s in enumerate(packages)}
    q = np.array([(table[s] - P_FLOOR) / (P_CEIL - P_FLOOR) for s in packages])
    rows, struct = [], []
    t0 = time.time()
    for t_idx, topo in enumerate(topology_names()):
        G, nodes, _ = load_topology(topo)
        cfgs = configs_for(topo, nodes, table, pkg_idx, q)
        rng = np.random.default_rng([7, t_idx])
        graphs = [("original", 0, G, 1.0)]
        for r in range(REPLICATES):
            graphs.append(("degree", r + 1, rewire_degree(G, rng), 1.0))
            H, frac_done = rewire_joint(G, rng)
            graphs.append(("joint", r + 1, H, frac_done))
        for rung, rep, H, frac_done in graphs:
            f = features(H)
            struct.append({"topo_id": topo, "rung": rung, "replicate": rep, "swaps_done": frac_done,
                           **{k: f[k] for k in ("avg_clustering", "assortativity", "norm_avg_path",
                                                "norm_diameter", "degree_gini")}})
            seed0 = 1_000_000 * (["original", "degree", "joint"].index(rung) + 1) + 10_000 * t_idx + 100 * rep
            risks = run(H, nodes, cfgs, n_trials, seed0)
            for (d, s, *_), risk in zip(cfgs, risks):
                rows.append({"topo_id": topo, "n_nodes": len(nodes), "rung": rung, "replicate": rep,
                             "density": d, "config_seed": s, "compromise_fraction": risk,
                             "n_trials": n_trials, "seed0": seed0})
        print(f"  {topo:18s} {time.time() - t0:6.1f}s", file=sys.stderr)
    df = pd.DataFrame(rows)
    st = pd.DataFrame(struct)
    df.to_csv(os.path.join(OUT_DIR, "rewiring.csv"), index=False, float_format="%.5f")
    st.to_csv(os.path.join(OUT_DIR, "rewired_structure.csv"), index=False, float_format="%.4f")
    print(f"wrote {rel(OUT_DIR)}/rewiring.csv ({len(df)} rows) and rewired_structure.csv "
          f"in {time.time() - t0:.0f}s", file=sys.stderr)


def summarize():
    """Read rewiring.csv and write results/rewiring/SUMMARY.md + network table."""
    from scipy.stats import spearmanr, wilcoxon
    from q1_anova import logit, anova, components
    df = pd.read_csv(os.path.join(OUT_DIR, "rewiring.csv"))
    st = pd.read_csv(os.path.join(OUT_DIR, "rewired_structure.csv"))
    k = len(SEEDS)

    # per config: original vs mean over replicates, per rung
    cfg = df.groupby(["topo_id", "rung", "density", "config_seed"]).compromise_fraction.mean().unstack("rung")
    net = cfg.groupby("topo_id").mean()
    net["delta_degree"] = net.degree - net.original
    net["delta_joint"] = net.joint - net.original
    s = st.groupby(["topo_id", "rung"])[["avg_clustering", "assortativity", "norm_avg_path"]].mean().unstack("rung")
    for col in ("avg_clustering", "assortativity", "norm_avg_path"):
        net[f"{col}_orig"] = s[(col, "original")]
        net[f"{col}_degree"] = s[(col, "degree")]
    net["path_change"] = net.norm_avg_path_degree - net.norm_avg_path_orig
    net = net.sort_values("delta_degree")
    net.round(4).to_csv(os.path.join(OUT_DIR, "network_rewiring.csv"))

    # topology share when each network is replaced by its rewired copies
    shares = {}
    for rung in ("original", "degree", "joint"):
        d = cfg[rung].rename("compromise_fraction").reset_index()
        d["n_nodes"] = d.topo_id.map(df.groupby("topo_id").n_nodes.first())
        d["y"] = logit(d.compromise_fraction, d.n_nodes)
        shares[rung] = components(anova(d, "y ~ C(topo_id) * C(density)"), k)["topology"]
    rho_deg = spearmanr(net.original, net.degree)[0]
    rho_joint = spearmanr(net.original, net.joint)[0]
    var_kept = net.degree.var() / net.original.var()
    p_deg = wilcoxon(net.degree, net.original).pvalue
    rho_path = spearmanr(net.path_change, net.delta_degree)[0]
    swaps = st[st.rung == "joint"].swaps_done.mean()

    lines = [
        "# Rewiring experiment: is topology's effect the degree distribution?",
        "",
        f"Generated by `src/rewiring.py` from `{rel(os.path.join(OUT_DIR, 'rewiring.csv'))}`. "
        "Do not edit by hand.",
        "",
        f"Every network was rewired {REPLICATES} times at two rungs, keeping each node's degree "
        "(degree) or the joint degree distribution as well (joint), with configurations held "
        f"fixed. {int(df.n_trials.iloc[0]):,} attacks per configuration per graph. Joint-rung swaps "
        f"completed: {swaps:.0%} of the target on average.",
        "",
        "| | Original | Degree kept | Degree + joint kept |",
        "|---|---|---|---|",
        f"| Mean compromised | {net.original.mean():.1%} | {net.degree.mean():.1%} | {net.joint.mean():.1%} |",
        f"| Topology share (variance components) | {shares['original']:.1%} | {shares['degree']:.1%} | {shares['joint']:.1%} |",
        f"| Network ranking vs original (Spearman) | 1 | {rho_deg:+.2f} | {rho_joint:+.2f} |",
        f"| Clustering (mean) | {net.avg_clustering_orig.mean():.3f} | {net.avg_clustering_degree.mean():.3f} | |",
        f"| Assortativity (mean) | {net.assortativity_orig.mean():+.3f} | {net.assortativity_degree.mean():+.3f} | |",
        f"| Normalised path length (mean) | {net.norm_avg_path_orig.mean():.3f} | {net.norm_avg_path_degree.mean():.3f} | |",
        "",
        f"Between-network variance kept after degree-preserving rewiring: {var_kept:.0%}. "
        f"Change in network risk (degree rung) vs change in path length: Spearman {rho_path:+.2f}. "
        f"Original vs degree rung, paired over networks: Wilcoxon p = {p_deg:.2g}.",
        "",
        "## Per network (degree rung), sorted by change",
        "",
        "| Network | Original | Rewired | Change | Path length orig -> rewired |",
        "|---|---|---|---|---|",
    ]
    for t, r in net.iterrows():
        lines.append(f"| {t} | {r.original:.1%} | {r.degree:.1%} | {r.delta_degree * 100:+.1f} pp | "
                     f"{r.norm_avg_path_orig:.2f} -> {r.norm_avg_path_degree:.2f} |")
    open(os.path.join(OUT_DIR, "SUMMARY.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarize()
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
        summarize()
