"""Full sweep: every config in data/configs/ through the simulator, one row per
scenario. All analysis reads results/sweep.csv and nothing else.

    python src/sweep.py [n_trials] [shared|independent]

Each scenario = (topology, density, config seed, mode, floor). The attacker is
assume-breach from the config's single entry point; every trial starts there
and differs only in which exploit attempts succeed.

Runs
----
  decay, floor 0.05 (P_FLOOR), ceiling 0.95, shared exploits -- the only run.

Decided 2026-10-12: decay only (hop dropped), one fixed floor (no floor
sensitivity runs), and correlated exploits (simulate.py): per trial the
attacker holds an exploit for package s with probability q_s = max EPSS over
its attack CVEs, and it then works on every node running s. Passing
"independent" reruns the old per-attempt model into
results/sensitivity/independent_exploits/sweep.csv for comparison; analysis reads sweep.csv only.

Seeds: the simulation seed is derived from (topology index, density, config
seed, run index) and written into every row, so any row reproduces alone.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
from simulate import prepare, simulate
from scoring import build_table, P_FLOOR, P_CEIL
from topology import load_topology, topology_names
from paths import CONFIGS, CATALOG, EPSS, SWEEP, SENSITIVITY, rel

CONFIG_DIR = CONFIGS
OUT = {"shared": SWEEP,
       "independent": os.path.join(SENSITIVITY, "independent_exploits", "sweep.csv")}
DENSITIES = (0.2, 0.4, 0.6, 0.8)
SEEDS = (1, 2, 3, 4, 5)
RUNS = [("decay", P_FLOOR)]
PATCHED = "Fully patched (no known CVEs)"


def sim_seed(t_idx, density, cfg_seed, run_idx):
    return 1_000_000 * (run_idx + 1) + 10_000 * t_idx + 100 * int(round(density * 10)) + cfg_seed


def main(n_trials, exploits="shared"):
    if exploits not in OUT:
        sys.exit(f"exploits must be one of {list(OUT)}")
    tables = {fl: build_table(CATALOG, EPSS, floor=fl)[0] for fl in {f for _, f in RUNS}}
    packages = sorted(tables[P_FLOOR])
    pkg_idx = {s: i for i, s in enumerate(packages)}
    # chance the attacker holds an exploit, recovered from p (exact inverse of
    # scoring.py's linear map; 0 for fully patched and no-foothold packages)
    q = np.array([(tables[P_FLOOR][s] - P_FLOOR) / (P_CEIL - P_FLOOR) for s in packages])
    rows = []
    t0 = time.time()
    for t_idx, topo in enumerate(topology_names()):
        G, nodes, _ = load_topology(topo)
        A, D, idx = prepare(G, nodes)
        n = len(nodes)
        for d in DENSITIES:
            for s in SEEDS:
                cfg = json.load(open(os.path.join(CONFIG_DIR, f"{topo}_d{d}_s{s}.json")))
                cn = cfg["nodes"]
                if set(cn) != set(nodes):
                    sys.exit(f"{topo} d{d} s{s}: config nodes differ from the "
                             "loaded graph -- regenerate configs")
                entries = [v for v in nodes if cn[v]["is_entry_point"]]
                if len(entries) != 1:
                    sys.exit(f"{topo} d{d} s{s}: expected 1 entry point, got {len(entries)}")
                entry = idx[entries[0]]
                sw = [cn[v]["software"] for v in nodes]
                n_vuln = sum(x["software"] != PATCHED for x in cn.values())
                for r_idx, (mode, fl) in enumerate(RUNS):
                    p = np.array([tables[fl][x] for x in sw])
                    if fl == P_FLOOR and not np.allclose(p, [cn[v]["p"] for v in nodes]):
                        sys.exit(f"{topo} d{d} s{s}: p in config disagrees with "
                                 "scoring.py -- regenerate configs")
                    seed = sim_seed(t_idx, d, s, r_idx)
                    shared = ((np.array([pkg_idx[x] for x in sw]), q, fl, P_CEIL)
                              if exploits == "shared" else None)
                    frac = simulate(A, D, p, entry, n_trials, mode=mode, seed=seed,
                                    shared=shared)["frac"]
                    rows.append({
                        "topo_id": topo, "n_nodes": n, "density": d,
                        "config_seed": s, "adversary": cfg["adversary"],
                        "mode": mode, "floor": fl, "exploits": exploits,
                        "compromise_fraction": float(frac.mean()),
                        "frac_sd": float(frac.std()),
                        "frac_p10": float(np.quantile(frac, 0.1)),
                        "frac_p90": float(np.quantile(frac, 0.9)),
                        "vulnerable_machines": n_vuln,
                        "mean_p": float(p.mean()),
                        "entry_vulnerable": cn[entries[0]]["software"] != PATCHED,
                        "n_trials": n_trials, "seed": seed,
                    })
        print(f"  {topo:18s} n={n:3d}  {time.time() - t0:6.1f}s", file=sys.stderr)

    os.makedirs(os.path.dirname(OUT[exploits]), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT[exploits], index=False, float_format="%.6f")
    print(f"wrote {rel(OUT[exploits])}: {len(df)} rows in {time.time() - t0:.1f}s", file=sys.stderr)
    return df


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000,
         sys.argv[2] if len(sys.argv) > 2 else "shared")
