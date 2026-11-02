"""The attack reproduction number: one plain-language number per scenario.

    python src/reproduction.py

R = the number of NEW machines one compromised machine takes over, on average,
in its own first round of attempts. It is the same idea as the R number of an
epidemic: R above 1 means an attack tends to grow, below 1 means it tends to
fizzle. It is built from the same two ingredients as the simulator (how
exploitable each machine is, and how the network lets an attacker reach it) and
uses no fitted parameters.

  M[u, v] = decay**(distance(u, v) - 1) * p[v]     chance u breaks into v
  R_mean  = average over machines u of  sum_v M[u, v]     (any machine, a typical foothold)
  R_entry = same sum, for the machines the attacker actually starts at

Reading: R_mean says how contagious this network + software setup is;
R_entry says how contagious the attacker's starting point is.

Writes results/reproduction/scenario_R.csv and prints how well R predicts risk,
alone and against the percolation-threshold and gradient-boosting predictors of
q2_predict.py (leave-one-network-out, real networks).
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import simulate as S
from simulate import prepare
from scoring import build_table, P_FLOOR
from topology import load_topology, topology_names
from attackers import ADVERSARIES, entry_set
from sweep import DENSITIES, SEEDS
import q2_predict as Q
from paths import CONFIGS, CATALOG, EPSS, NETWORKS, RESULTS, TOPOLOGY_FEATURES

OUT = os.path.join(RESULTS, "reproduction")


def reproduction(D, p, entry_idx, decay=None):
    """R_mean and R_entry for one configuration. D: distance matrix (-1 = unreachable)."""
    decay = S.DECAY if decay is None else decay
    reach = D > 0
    M = np.where(reach, decay ** np.where(reach, D - 1, 0), 0.0) * p[None, :]
    row = M.sum(axis=1)
    return float(row.mean()), float(row[list(entry_idx)].mean())


def build(net_dir=NETWORKS, cfg_dir=CONFIGS, densities=DENSITIES, seeds=SEEDS, family="real"):
    table = build_table(CATALOG, EPSS, floor=P_FLOOR)[0]
    rows = []
    for t in topology_names(net_dir):
        G, nodes, _ = load_topology(t, net_dir)
        A, D, idx = prepare(G, nodes)
        for d in densities:
            for s in seeds:
                cn = json.load(open(os.path.join(cfg_dir, f"{t}_d{d}_s{s}.json")))["nodes"]
                p = np.array([table[cn[v]["software"]] for v in nodes])
                for adv in ADVERSARIES:
                    ents = [idx[v] for v in entry_set(adv, G, nodes, cn)]
                    rm, re = reproduction(D, p, ents)
                    rows.append({"topo_id": t, "family": family, "density": d,
                                 "config_seed": s, "adversary": adv, "R_mean": rm, "R_entry": re})
    return pd.DataFrame(rows)


def r_models(df, cfg):
    """Models that use R. 'r_only' has no structure term at all."""
    return {"r_only": ["R_mean", "R_entry"] + [c for c in cfg if c.startswith("adv_")],
            "r_plus_config": ["R_mean", "R_entry"] + cfg}


def main():
    R = build()
    os.makedirs(OUT, exist_ok=True)
    R.to_csv(os.path.join(OUT, "scenario_R.csv"), index=False, float_format="%.5f")
    df, cfg = Q.build_table(os.path.join(RESULTS, "sweep_attackers.csv"), TOPOLOGY_FEATURES)
    df = df.merge(R, on=["topo_id", "family", "density", "config_seed", "adversary"], how="left")
    assert df.R_mean.notna().all()
    print("Spearman of R with the simulated share compromised (all 1,200 scenarios):")
    for c in ("R_mean", "R_entry", "mean_p", "pc"):
        print(f"  {c:8s} {spearmanr(df[c], df.compromise_fraction)[0]:+.2f}")
    nets = df.groupby("topo_id").agg(r=("compromise_fraction", "mean"), R=("R_mean", "mean"), pc=("pc", "first"))
    print(f"across the 20 networks: R_mean vs risk {spearmanr(nets.R, nets.r)[0]:+.2f}, "
          f"percolation vs risk {spearmanr(nets.pc, nets.r)[0]:+.2f}")
    print("share compromised by R_mean band:")
    df["R_band"] = pd.cut(df.R_mean, [0, 0.5, 1, 2, 4, 100], labels=["<0.5", "0.5-1", "1-2", "2-4", ">4"])
    print(df.groupby("R_band", observed=True).compromise_fraction.agg(["mean", "count"]).round(3).to_string())

    # leave-one-network-out, same protocol as q2_predict (real family only)
    fsets = Q.feature_sets(cfg); fsets.update(r_models(df, cfg))
    fam = df.copy(); nets_l = sorted(fam.topo_id.unique()); rows = []
    for mdl in ["config_only", "pc_linear", "structure_gb", "r_only", "r_plus_config"]:
        pred = pd.Series(index=fam.index, dtype=float)
        for held in nets_l:
            tr, te = fam[fam.topo_id != held], fam[fam.topo_id == held]
            kind = "pc_linear" if mdl in ("pc_linear", "r_only", "r_plus_config") else mdl   # ridge vs boosting
            pred[te.index] = Q.fit_predict(kind, fsets[mdl], tr, te)
        r2, rho, rn = Q.score(fam, pred.to_numpy())
        rows.append({"model": mdl, "r2": r2, "spearman_scenarios": rho, "spearman_networks": rn})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "r_vs_structure.csv"), index=False, float_format="%.4f")
    print("\nLeave-one-network-out, real networks:")
    print(res.round(2).to_string(index=False))
    return res


if __name__ == "__main__":
    main()
