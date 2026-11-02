"""Which machines should you patch first? A tested defence, not a guess.

    python src/patching.py [n_trials]       (about 5 minutes for 2,000 trials)

Take each of the 400 real configurations. Give the defender a budget: fix 25% of
the vulnerable machines (rounded up, at least 1). "Fix" means the machine now runs
fully patched software. Compare ways of choosing which machines:

  random         pick at random (the do-nothing-clever baseline)
  exploit_first  the machines that are easiest to break into (highest p)
  hub_first      the best-connected machines (most links)
  combined       highest p x links
  package_first  every machine running the single most dangerous software package,
                 then the next, until the budget is used (attacks the shared-exploit effect)

Then rerun the identical attacks (same seeds, so the comparison is paired) on the
patched network, for all three attackers, and measure how much the share
compromised falls. The question the project asks of the recommendations: does the
best strategy depend on the network's structure, or does one rule win everywhere?

Writes results/patching/patching.csv (one row per config x attacker x strategy),
summary_by_strategy.csv and structure_dependence.csv.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from simulate import prepare, simulate, percolation_threshold
from scoring import build_table, P_FLOOR, P_CEIL
from topology import load_topology, topology_names
from attackers import ADVERSARIES, entry_set, entries_per_trial
from sweep import sim_seed, DENSITIES, SEEDS, PATCHED
from paths import CONFIGS, CATALOG, EPSS, NETWORKS, RESULTS

OUT = os.path.join(RESULTS, "patching")
STRATEGIES = ("random", "exploit_first", "hub_first", "combined", "package_first")
BUDGET = 0.25


def budget_size(n_vulnerable, frac=BUDGET):
    return max(1, int(np.ceil(frac * n_vulnerable))) if n_vulnerable else 0


def choose(strategy, sw, p, deg, q_pkg, k, rng):
    """Indices of the k machines to patch. Only vulnerable machines are eligible."""
    vul = np.array([i for i, x in enumerate(sw) if x != PATCHED])
    k = min(k, len(vul))
    if k == 0:
        return np.array([], dtype=int)
    tie = rng.random(len(vul)) * 1e-9                       # random tie-breaks, reproducible
    if strategy == "random":
        return vul[rng.permutation(len(vul))[:k]]
    if strategy == "exploit_first":
        key = p[vul] + tie
    elif strategy == "hub_first":
        key = deg[vul] + tie
    elif strategy == "combined":
        key = p[vul] * deg[vul] + tie
    elif strategy == "package_first":
        # whole packages in order of danger; the last package is cut at the budget
        key = np.array([q_pkg[sw[i]] for i in vul]) + tie * 0   # same package -> same key
        order = np.lexsort((tie, -key))
        return vul[order[:k]]
    else:
        raise ValueError(strategy)
    return vul[np.argsort(-key)[:k]]


def apply_patch(sw, idx):
    out = list(sw)
    for i in idx:
        out[i] = PATCHED
    return out


def run(n_trials=2000, net_dir=NETWORKS, cfg_dir=CONFIGS, densities=DENSITIES, seeds=SEEDS,
        budget=BUDGET, verbose=True):
    table = build_table(CATALOG, EPSS, floor=P_FLOOR)[0]
    packages = sorted(table); pkg_idx = {s: i for i, s in enumerate(packages)}
    q = np.array([(table[s] - P_FLOOR) / (P_CEIL - P_FLOOR) for s in packages])
    q_pkg = {s: q[pkg_idx[s]] for s in packages}
    rows, t0 = [], time.time()
    for t_idx, topo in enumerate(topology_names(net_dir)):
        G, nodes, _ = load_topology(topo, net_dir)
        A, D, idx = prepare(G, nodes)
        deg = np.array([G.degree(v) for v in nodes], dtype=float)
        pc = percolation_threshold(G)
        for d in densities:
            for s in seeds:
                cn = json.load(open(os.path.join(cfg_dir, f"{topo}_d{d}_s{s}.json")))["nodes"]
                sw0 = [cn[v]["software"] for v in nodes]
                p0 = np.array([table[x] for x in sw0])
                n_vuln = sum(x != PATCHED for x in sw0)
                k = budget_size(n_vuln, budget)
                for strat in (None,) + STRATEGIES:
                    if strat is None:
                        sw = sw0
                    else:
                        rng = np.random.default_rng([t_idx, int(d * 10), s, STRATEGIES.index(strat)])
                        sw = apply_patch(sw0, choose(strat, sw0, p0, deg, q_pkg, k, rng))
                    p = np.array([table[x] for x in sw])
                    shared = (np.array([pkg_idx[x] for x in sw]), q, P_FLOOR, P_CEIL)
                    for a_idx, adv in enumerate(ADVERSARIES):
                        ents = entry_set(adv, G, nodes, cn)
                        seed = sim_seed(t_idx, d, s, 0) + 10_000_000 * a_idx      # same for every strategy
                        entry = entries_per_trial(adv, ents, idx, n_trials, seed + 7919)
                        frac = simulate(A, D, p, entry, n_trials, mode="decay", seed=seed,
                                        shared=shared)["frac"].mean()
                        rows.append({"topo_id": topo, "n_nodes": len(nodes), "pc": pc, "density": d,
                                     "config_seed": s, "adversary": adv,
                                     "strategy": "none" if strat is None else strat,
                                     "patched": 0 if strat is None else k,
                                     "vulnerable": n_vuln, "risk": float(frac)})
        if verbose:
            print(f"  {topo:18s} {time.time() - t0:6.0f}s", file=sys.stderr, flush=True)
    return pd.DataFrame(rows)


def analyse(df):
    key = ["topo_id", "density", "config_seed", "adversary"]
    base = df[df.strategy == "none"].set_index(key).risk.rename("base")
    d = df[df.strategy != "none"].join(base, on=key)
    d["reduction"] = d.base - d.risk                               # percentage points of the network
    d["relative"] = np.where(d.base > 0, d.reduction / d.base, 0.0)
    return d


def boot_ci(per_net, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = per_net.to_numpy(); m = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


def summary(d):
    rows = []
    for strat, g in d.groupby("strategy"):
        pn = g.groupby("topo_id").reduction.mean()
        lo, hi = boot_ci(pn)
        rows.append({"strategy": strat, "mean_reduction_pts": 100 * pn.mean(), "ci_low": 100 * lo,
                     "ci_high": 100 * hi, "mean_relative_reduction": g.relative.mean(),
                     **{f"d{dd}": 100 * x for dd, x in g.groupby("density").reduction.mean().items()},
                     **{a.split("_")[0]: 100 * x for a, x in g.groupby("adversary").reduction.mean().items()}})
    return pd.DataFrame(rows).sort_values("mean_reduction_pts", ascending=False)


def structure_dependence(d):
    """Per network, which strategy wins; and does the hub advantage track structure?"""
    pn = d.groupby(["topo_id", "strategy"]).reduction.mean().unstack()
    net = d.groupby("topo_id").agg(pc=("pc", "first"), n=("n_nodes", "first"))
    pn["best"] = pn[list(STRATEGIES)].idxmax(axis=1)
    out = pn.join(net)
    out["hub_minus_exploit"] = out.hub_first - out.exploit_first
    return out


def main(n_trials=2000):
    os.makedirs(OUT, exist_ok=True)
    df = run(n_trials)
    df.to_csv(os.path.join(OUT, "patching.csv"), index=False, float_format="%.6f")
    d = analyse(df)
    s = summary(d); s.to_csv(os.path.join(OUT, "summary_by_strategy.csv"), index=False, float_format="%.3f")
    sd = structure_dependence(d); sd.to_csv(os.path.join(OUT, "structure_dependence.csv"), float_format="%.4f")
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        print("\nFall in share compromised (percentage points of the network), 25% of vulnerable machines fixed")
        print(s.to_string(index=False))
        print("\nBest strategy per network:\n", sd.best.value_counts().to_string())
        print("\nhub_first minus exploit_first vs percolation threshold (Spearman):",
              round(spearmanr(sd.pc, sd.hub_minus_exploit)[0], 2))
    return df, s, sd


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
