"""Q1 with the attacker added: how much of the difference in damage comes from
the network, the software setup, or the attacker?

    python src/q1_adversary.py [sweep_attackers.csv]

Same idea as q1_anova.py (variance components on the logit scale, compared
within a density level, shares sum to 100%), extended to three attackers that
run on the SAME configs. Balanced design at one density:

    y[t, c, a] = mu + T[t] + C[c in t] + A[a] + TA[t, a] + e[t, c, a]

    T   network (topology)               random, nested config C within it
    C   configuration                    5 draws per network
    A   attacker                         3 models, crossed with everything
    TA  network x attacker               "the attacker matters more on some networks"
    e   attacker x configuration + Monte Carlo noise

Method of moments from the classical ANOVA mean squares (t networks, k configs
per network, a attackers):

    sigma2_e  = MS_e
    sigma2_C  = (MS_C  - MS_e) / a
    sigma2_TA = (MS_TA - MS_e) / k
    sigma2_A  = (MS_A  - MS_TA) / (t k)
    sigma2_T  = (MS_T - MS_C - MS_TA + MS_e) / (k a)

Negative estimates are set to zero. Mean squares are averaged over the four
density levels first (the same pooling q1_anova.py uses), so density itself is
not a source of variance. Intervals resample whole networks, keeping each
network's configurations together.

Check built in: with the assume-breach attacker alone (a = 1) the same code
gives the midterm split, topology 25.8%, configuration 74.2%.
"""
import os, sys
import numpy as np
import pandas as pd
from paths import RESULTS

SWEEP = os.path.join(RESULTS, "sweep_attackers.csv")
BASE = "assume_breach_random_entry"


def logit(x, n):
    eps = 0.5 / n
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))


def cube(df, advs):
    """y[density, topology, seed, adversary] as an array, plus topology order."""
    df = df[df.adversary.isin(advs)].copy()
    df["y"] = logit(df.compromise_fraction, df.n_nodes)
    topos = sorted(df.topo_id.unique())
    dens = sorted(df.density.unique())
    seeds = sorted(df.config_seed.unique())
    Y = np.empty((len(dens), len(topos), len(seeds), len(advs)))
    ix = df.set_index(["density", "topo_id", "config_seed", "adversary"]).y
    for i, d in enumerate(dens):
        for j, t in enumerate(topos):
            for k, s in enumerate(seeds):
                for m, a in enumerate(advs):
                    Y[i, j, k, m] = ix[(d, t, s, a)]
    return Y, topos, dens


def mean_squares(Y):
    """Per-density mean squares for the design above, averaged over densities."""
    D, t, k, a = Y.shape
    out = {key: 0.0 for key in ("T", "C", "A", "TA", "e")}
    for y in Y:                                   # one density
        g = y.mean()
        mt = y.mean(axis=(1, 2))                  # network means
        mc = y.mean(axis=2)                       # config means (t, k)
        ma = y.mean(axis=(0, 1))                  # attacker means
        mta = y.mean(axis=1)                      # network x attacker (t, a)
        ss_t = k * a * ((mt - g) ** 2).sum()
        ss_c = a * ((mc - mt[:, None]) ** 2).sum()
        ss_a = t * k * ((ma - g) ** 2).sum() if a > 1 else 0.0
        ss_ta = (k * ((mta - mt[:, None] - ma[None, :] + g) ** 2).sum()) if a > 1 else 0.0
        ss_e = ((y - mc[:, :, None] - mta[:, None, :] + mt[:, None, None]) ** 2).sum()
        df_e = t * (k - 1) * (a - 1) if a > 1 else None
        out["T"] += ss_t / (t - 1)
        out["C"] += ss_c / (t * (k - 1))
        if a > 1:
            out["A"] += ss_a / (a - 1)
            out["TA"] += ss_ta / ((t - 1) * (a - 1))
            out["e"] += ss_e / df_e
    return {key: v / D for key, v in out.items()}


def components(Y):
    D, t, k, a = Y.shape
    ms = mean_squares(Y)
    if a == 1:                                    # midterm design: C is the residual
        s_c = ms["C"]
        s_t = max((ms["T"] - ms["C"]) / k, 0.0)
        parts = {"topology": s_t, "configuration": s_c}
    else:
        s_e = ms["e"]
        parts = {
            "topology": max((ms["T"] - ms["C"] - ms["TA"] + ms["e"]) / (k * a), 0.0),
            "configuration": max((ms["C"] - ms["e"]) / a, 0.0),
            "attacker": max((ms["A"] - ms["TA"]) / (t * k), 0.0),
            "network_x_attacker": max((ms["TA"] - ms["e"]) / k, 0.0),
            "residual": s_e,
        }
    tot = sum(parts.values())
    return {k_: v / tot for k_, v in parts.items()}


def boot(Y, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    t = Y.shape[1]
    res = [components(Y[:, rng.integers(t, size=t)]) for _ in range(B)]
    return pd.DataFrame(res)


def main(path=SWEEP):
    df = pd.read_csv(path)
    advs = sorted(df.adversary.unique(), key=lambda s: (s != BASE, s))
    rows = []
    # check: midterm design, assume-breach only
    Yb, topos, dens = cube(df, [BASE])
    chk = components(Yb)
    print(f"check, assume-breach only: topology {chk['topology']:.1%}, "
          f"configuration {chk['configuration']:.1%}  (midterm: 25.8% / 74.2%)")
    rows.append({"design": "assume_breach_only", "density": "all", **chk})
    Y, topos, dens = cube(df, advs)
    allc = components(Y)
    bt = boot(Y)
    lo, hi = bt.quantile(0.025), bt.quantile(0.975)
    rows.append({"design": "three_attackers", "density": "all", **allc})
    for i, d in enumerate(dens):
        rows.append({"design": "three_attackers", "density": f"{d:.0%}",
                     **components(Y[i:i + 1])})
    out = pd.DataFrame(rows)
    ci = pd.DataFrame({"component": lo.index, "ci_low": lo.values, "ci_high": hi.values})
    out.to_csv(os.path.join(os.path.dirname(path), "q1_adversary_split.csv"),
               index=False, float_format="%.4f")
    ci.to_csv(os.path.join(os.path.dirname(path), "q1_adversary_split_ci.csv"),
              index=False, float_format="%.4f")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(out.to_string(index=False))
        print("\n95% network-bootstrap interval, all densities:")
        print(ci.to_string(index=False))
        m = df.groupby("adversary").compromise_fraction.mean()
        print("\nmean share compromised:\n", m.round(4).to_string())
    return out, ci


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else SWEEP)
