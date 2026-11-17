"""Q2 with uncertainty: every cell of the transfer matrix gets an interval, and a
leave-one-family-out test trains on ALL the other families.

    python src/q2_lofo.py [sweep_attackers_all.csv] [features_all.csv] [n_boot]

Why: the 5 x 5 matrix in q2_predict.py gives one number per cell. The test networks in a cell
are only a sample (10-30 networks), so each number has sampling error. Here the test networks
are resampled with replacement (n_boot times) and R^2 and the ranking of networks are
recomputed; the 2.5% and 97.5% points are the interval. The trained model is not refit, so
the intervals answer "how much would this cell move with a different sample of test networks",
not "with different training data".

Leave-one-family-out (LOFO): hold out one family, train on the other four families together
(all their networks), test on the held-out one. This is the realistic use: train on everything
you have, apply to a kind of network you have not seen.

Writes results/q2_all/q2_transfer_ci.csv (every cell) and q2_lofo.csv (LOFO).
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import q2_predict as Q
from paths import RESULTS

OUT = os.path.join(RESULTS, "q2_all")
MODELS = ["config_only", "pc_linear", "structure_gb"]


def _network_table(test, pred_logit):
    """Per network: sums needed for R^2 on the share scale, and mean true / predicted share."""
    d = pd.DataFrame({"n": test.topo_id.to_numpy(), "t": test.compromise_fraction.to_numpy(),
                      "p": Q.expit(pred_logit)})
    d["e2"] = (d.t - d.p) ** 2
    d["t2"] = d.t ** 2
    g = d.groupby("n")
    return pd.DataFrame({"cnt": g.size(), "sse": g.e2.sum(), "st": g.t.sum(), "st2": g.t2.sum(),
                         "mt": g.t.mean(), "mp": g.p.mean()})


def _scores(w, T):
    """R^2 and network-level Spearman for a vector of network weights (bootstrap counts)."""
    n = (w * T.cnt).sum()
    sst = (w * T.st2).sum() - (w * T.st).sum() ** 2 / n
    r2 = 1 - (w * T.sse).sum() / sst
    sel = np.repeat(np.arange(len(T)), w.astype(int))
    rho = spearmanr(T.mt.to_numpy()[sel], T.mp.to_numpy()[sel])[0] if np.ptp(T.mp.to_numpy()[sel]) > 0 else np.nan
    return r2, rho


def with_interval(test, pred_logit, n_boot, rng):
    T = _network_table(test, pred_logit)
    r2, rho = _scores(np.ones(len(T)), T)
    boots = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(len(T), size=len(T)), minlength=len(T)).astype(float)
        boots.append(_scores(w, T))
    b = np.array(boots, dtype=float)
    lo, hi = np.nanpercentile(b, [2.5, 97.5], axis=0)          # each: [R^2, rank]
    return {"r2": r2, "r2_lo": lo[0], "r2_hi": hi[0], "rank": rho, "rank_lo": lo[1], "rank_hi": hi[1],
            "n_networks": len(T)}


def main(scen=os.path.join(RESULTS, "sweep_attackers_all.csv"), feat=os.path.join(RESULTS, "features_all.csv"),
         n_boot=300):
    df, cfg = Q.build_table(scen, feat)
    fsets = Q.feature_sets(cfg)
    fams = sorted(df.family.unique())
    rng = np.random.default_rng(0)
    cells, lofo = [], []
    for m in MODELS:
        cols = fsets[m]
        for i in fams:
            for j in fams:
                te = df[df.family == j]
                if i != j:
                    pred = Q.fit_predict(m, cols, df[df.family == i], te)
                else:
                    pred = pd.Series(index=te.index, dtype=float)
                    for held in sorted(te.topo_id.unique()):
                        a, b = te[te.topo_id != held], te[te.topo_id == held]
                        pred[b.index] = Q.fit_predict(m, cols, a, b)
                    pred = pred.to_numpy()
                cells.append({"model": m, "train": i, "test": j, **with_interval(te, pred, n_boot, rng)})
        for f in fams:
            tr, te = df[df.family != f], df[df.family == f]
            lofo.append({"model": m, "held_out_family": f, **with_interval(te, Q.fit_predict(m, cols, tr, te), n_boot, rng)})
    os.makedirs(OUT, exist_ok=True)
    c, l = pd.DataFrame(cells), pd.DataFrame(lofo)
    c.to_csv(os.path.join(OUT, "q2_transfer_ci.csv"), index=False, float_format="%.4f")
    l.to_csv(os.path.join(OUT, "q2_lofo.csv"), index=False, float_format="%.4f")
    t = l.copy()
    t["R2"] = t.apply(lambda r: f"{r.r2:5.2f} ({r.r2_lo:5.2f} to {r.r2_hi:5.2f})", axis=1)
    t["rank"] = t.apply(lambda r: f"{r['rank']:5.2f} ({r.rank_lo:5.2f} to {r.rank_hi:5.2f})", axis=1)
    with pd.option_context("display.width", 220):
        print("Leave-one-family-out (train on the other four families):")
        print(t.pivot(index="held_out_family", columns="model", values="R2").to_string())
        print()
        print(t.pivot(index="held_out_family", columns="model", values="rank").to_string())
    return c, l


if __name__ == "__main__":
    a = sys.argv
    main(*(a[1:3]), n_boot=int(a[3]) if len(a) > 3 else 300)
