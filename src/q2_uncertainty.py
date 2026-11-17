"""Does "structure transfers across network families" survive the two assumed settings?

    python src/q2_uncertainty.py [n_draws] [n_trials]      (32 draws x 1,000 attacks: about 20 min)

For each (floor, decay) drawn evenly from the same box as uncertainty.py (floor 0.02-0.10,
decay 0.3-0.7), rerun the assume-breach sweep on all 60 networks (real + synthetic), rebuild
the five-family transfer matrix, and record the average over the 20 train-on-one,
test-on-another cells for three predictors: software only, percolation threshold + software,
and gradient boosting on the structural features + software.

Writes results/uncertainty/q2_draws.csv and q2_summary.csv. The same sweeps also give, per draw, the
topology share of variation inside each family (the Q1 question), recorded in the same files.
"""
import os, sys, time, tempfile
import numpy as np
import pandas as pd
from scipy.stats import qmc
import simulate as S
import uncertainty as U
import q2_predict as Q
import q1_adversary as Q1
from paths import RESULTS, DATA, TOPOLOGY_FEATURES, NETWORKS, CONFIGS

NET_SYN, CFG_SYN = os.path.join(DATA, "networks_synthetic"), os.path.join(DATA, "configs_synthetic")
MODELS = ["config_only", "pc_linear", "structure_gb"]


def family_of(name):
    head, _, tail = name.rpartition("_")
    return head if head and tail.isdigit() else "real"


def load_all():
    return U.load_world(NETWORKS, CONFIGS) + U.load_world(NET_SYN, CFG_SYN)


def features():
    fr, fs = pd.read_csv(TOPOLOGY_FEATURES), pd.read_csv(os.path.join(DATA, "synthetic_features.csv"))
    cols = [c for c in fr.columns if c in fs.columns]
    return pd.concat([fr[cols], fs[cols]], ignore_index=True)


def one_draw(world, feat, floor, decay, trials, tmp):
    d = U.sweep(world, floor, decay, trials)
    s = pd.DataFrame({"topo_id": d.topo_id, "family": d.topo_id.map(family_of), "density": d.density,
                      "config_seed": d.config_seed, "adversary": "assume_breach_random_entry",
                      "n_nodes": d.n_nodes, "mean_p": d.mean_p, "compromise_fraction": d.f})
    sp, fp = os.path.join(tmp, "s.csv"), os.path.join(tmp, "f.csv")
    s.to_csv(sp, index=False); feat.to_csv(fp, index=False)
    out_split = {}
    for fam in sorted(s.family.unique()) + ["ALL"]:
        g = s if fam == "ALL" else s[s.family == fam]
        out_split[f"topology_share_{fam}"] = Q1.components(Q1.cube(g, [Q1.BASE])[0])["topology"]
    df, cfg = Q.build_table(sp, fp)
    res = Q.evaluate(df, cfg)
    off = res[res.train != res.test]
    out = dict(out_split)
    for m in MODELS:
        o = off[off.model == m]
        out[f"{m}_r2"] = o.r2.mean()
        out[f"{m}_rank"] = o.spearman_networks.mean()
    w = off.pivot_table(index=["train", "test"], columns="model", values="r2")
    out["gb_beats_config_pairs"] = int((w.structure_gb > w.config_only).sum())
    return out


def main(n_draws=32, trials=1000):
    world, feat = load_all(), features()
    pts = qmc.Sobol(d=2, scramble=True, seed=1).random(n_draws)
    lo, hi = np.array([U.FLOOR_RANGE[0], U.DECAY_RANGE[0]]), np.array([U.FLOOR_RANGE[1], U.DECAY_RANGE[1]])
    pts = lo + pts * (hi - lo)
    rows, t0 = [], time.time()
    with tempfile.TemporaryDirectory() as tmp:
        for k, (fl, dc) in enumerate(pts):
            r = one_draw(world, feat, float(fl), float(dc), trials, tmp)
            rows.append({"draw": k, "floor": fl, "decay": dc, **r})
            print(f"draw {k + 1:3d}/{n_draws} floor {fl:.3f} decay {dc:.2f}  R2 software {r['config_only_r2']:.2f} "
                  f"structure {r['structure_gb_r2']:.2f}  rank {r['config_only_rank']:.2f} / {r['structure_gb_rank']:.2f}"
                  f"  [{time.time() - t0:5.0f}s]", flush=True)
    S.DECAY = 0.5
    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, "uncertainty"); os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "q2_draws.csv"), index=False, float_format="%.4f")
    items = []
    for m in MODELS:
        items += [(f"{m}: R2, median", df[f"{m}_r2"].median()), (f"{m}: R2, worst draw", df[f"{m}_r2"].min()),
                  (f"{m}: rank of unseen networks, median", df[f"{m}_rank"].median()),
                  (f"{m}: rank, worst draw", df[f"{m}_rank"].min())]
    items += [("draws where structure_gb beats software-only on R2 (average)", (df.structure_gb_r2 > df.config_only_r2).mean()),
              ("draws where structure_gb ranks unseen networks better", (df.structure_gb_rank > df.config_only_rank).mean()),
              ("fewest pairs (of 20) where structure_gb beats software-only", df.gb_beats_config_pairs.min())]
    for c in [c for c in df.columns if c.startswith("topology_share_")]:
        fam = c[len("topology_share_"):]
        items += [(f"topology share, {fam}: median", df[c].median()), (f"topology share, {fam}: minimum", df[c].min()),
                  (f"topology share, {fam}: maximum", df[c].max()),
                  (f"draws where configuration beats topology, {fam}", (df[c] < 0.5).mean())]
    sm = pd.DataFrame(items, columns=["quantity", "value"])
    sm.to_csv(os.path.join(out, "q2_summary.csv"), index=False, float_format="%.4f")
    print("\n", sm.to_string(index=False))
    return df, sm


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 32, int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
