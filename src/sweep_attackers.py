"""Sweep with the three adversaries (see attackers.py).

    python src/sweep_attackers.py [n_trials] [networks_dir] [configs_dir] [out_csv]

Same model as sweep.py (decay 0.5, floor 0.05, shared exploits); the only
difference is where the attack starts. For assume_breach_random_entry the rows
reproduce results/sweep.csv exactly (same seeds), which is the check that this
script did not change the model. The other two adversaries get their own seed
block (seed + 10_000_000 * adversary index).

Output: results/sweep_attackers.csv, one row per (config, adversary). Columns
match sweep.csv, plus `adversary` already present (now one of three values),
`n_entry_nodes` and `family`.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
from simulate import prepare, simulate
from scoring import build_table, P_FLOOR, P_CEIL
from topology import load_topology, topology_names
from attackers import ADVERSARIES, entry_set, entries_per_trial
from sweep import sim_seed, DENSITIES, SEEDS, PATCHED
from paths import CONFIGS, CATALOG, EPSS, NETWORKS, RESULTS, rel

OUT = os.path.join(RESULTS, "sweep_attackers.csv")


def run(n_trials=2000, net_dir=NETWORKS, cfg_dir=CONFIGS, out=OUT, family="real",
        densities=DENSITIES, seeds=SEEDS, verbose=True):
    table = build_table(CATALOG, EPSS, floor=P_FLOOR)[0]
    packages = sorted(table)
    pkg_idx = {s: i for i, s in enumerate(packages)}
    q = np.array([(table[s] - P_FLOOR) / (P_CEIL - P_FLOOR) for s in packages])
    rows, t0 = [], time.time()
    for t_idx, topo in enumerate(topology_names(net_dir)):
        G, nodes, _ = load_topology(topo, net_dir)
        A, D, idx = prepare(G, nodes)
        n = len(nodes)
        for d in densities:
            for s in seeds:
                cfg = json.load(open(os.path.join(cfg_dir, f"{topo}_d{d}_s{s}.json")))
                cn = cfg["nodes"]
                if set(cn) != set(nodes):
                    sys.exit(f"{topo} d{d} s{s}: config nodes differ from the graph")
                sw = [cn[v]["software"] for v in nodes]
                p = np.array([table[x] for x in sw])
                shared = (np.array([pkg_idx[x] for x in sw]), q, P_FLOOR, P_CEIL)
                n_vuln = sum(x != PATCHED for x in sw)
                for a_idx, adv in enumerate(ADVERSARIES):
                    ents = entry_set(adv, G, nodes, cn)
                    seed = sim_seed(t_idx, d, s, 0) + 10_000_000 * a_idx
                    entry = entries_per_trial(adv, ents, idx, n_trials, seed + 7919)
                    frac = simulate(A, D, p, entry, n_trials, mode="decay",
                                    seed=seed, shared=shared)["frac"]
                    rows.append({
                        "topo_id": topo, "family": family, "n_nodes": n, "density": d,
                        "config_seed": s, "adversary": adv, "mode": "decay",
                        "floor": P_FLOOR, "exploits": "shared",
                        "compromise_fraction": float(frac.mean()),
                        "frac_sd": float(frac.std()),
                        "frac_p10": float(np.quantile(frac, 0.1)),
                        "frac_p90": float(np.quantile(frac, 0.9)),
                        "vulnerable_machines": n_vuln, "mean_p": float(p.mean()),
                        "n_entry_nodes": len(ents),
                        "entry_vulnerable": bool(np.mean([cn[v]["software"] != PATCHED
                                                          for v in ents]) > 0.5),
                        "n_trials": n_trials, "seed": seed,
                    })
        if verbose:
            print(f"  {topo:18s} n={n:3d}  {time.time() - t0:6.1f}s", file=sys.stderr)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False, float_format="%.6f")
    if verbose:
        print(f"wrote {rel(out)}: {len(df)} rows in {time.time() - t0:.1f}s", file=sys.stderr)
    return df


if __name__ == "__main__":
    a = sys.argv
    run(int(a[1]) if len(a) > 1 else 2000,
        a[2] if len(a) > 2 else NETWORKS,
        a[3] if len(a) > 3 else CONFIGS,
        a[4] if len(a) > 4 else OUT)
