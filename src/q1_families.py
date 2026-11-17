"""Does "configuration matters more than topology" hold in every family of networks?

    python src/q1_families.py [sweep_attackers_all.csv]

Runs the same variance split as q1_adversary.py separately inside each family
(real ISP networks and each synthetic family) and for all 60 networks pooled.
Two designs per family: the midterm attacker alone (comparable to the 25.8% / 74.2%
split), and all three attackers. Intervals resample whole networks.
With 10 networks per synthetic family the intervals are wide: read them.

Writes results/q1_families.csv.
"""
import os, sys
import numpy as np
import pandas as pd
import q1_adversary as Q
from paths import RESULTS

ALL = os.path.join(RESULTS, "sweep_attackers_all.csv")


def split(df, advs, B=1000):
    Y, _, _ = Q.cube(df, advs)
    c = Q.components(Y)
    b = Q.boot(Y, B=B)
    return c, b.quantile(.025), b.quantile(.975), Y.shape[1]


def main(path=ALL):
    df = pd.read_csv(path)
    advs3 = sorted(df.adversary.unique(), key=lambda s: (s != Q.BASE, s))
    groups = {f: df[df.family == f] for f in sorted(df.family.unique())}
    groups["ALL (pooled)"] = df
    rows = []
    for name, g in groups.items():
        for design, advs in (("assume_breach_only", [Q.BASE]), ("three_attackers", advs3)):
            c, lo, hi, n = split(g, advs)
            for k, v in c.items():
                rows.append({"family": name, "design": design, "networks": n, "component": k,
                             "share": v, "ci_low": lo[k], "ci_high": hi[k]})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "q1_families.csv"), index=False, float_format="%.4f")
    for design in ("assume_breach_only", "three_attackers"):
        t = out[out.design == design].copy()
        t["txt"] = t.apply(lambda r: f"{r.share:5.1%} ({r.ci_low:4.0%}-{r.ci_high:4.0%})", axis=1)
        print(f"\n== {design}: share of variation, 95% interval over networks")
        print(t.pivot(index="family", columns="component", values="txt").to_string())
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ALL)
