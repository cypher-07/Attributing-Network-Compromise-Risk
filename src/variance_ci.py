"""Bootstrap confidence intervals for the topology / configuration split.

    python src/variance_ci.py            (a few seconds; needs matplotlib)

Resamples the 20 networks with replacement (2,000 times) and recomputes the
topology share, overall and within each density level. Writes
  results/q1_variance_split_ci.csv
  figures/fig1b_variance_split_ci.pdf   (bars with 95% intervals, no title)
Reads results/sweep.csv only; nothing existing is changed.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sensitivity import share, add_logit
from paths import SWEEP, RESULTS, FIGURES

B = 2000

if __name__ == "__main__":
    df = add_logit(pd.read_csv(SWEEP).rename(columns={"compromise_fraction": "f"}), "f")
    tops = df.topo_id.unique()
    groups = {t: df[df.topo_id == t] for t in tops}
    rng = np.random.default_rng(0)
    levels = ["all"] + sorted(df.density.unique())
    boot = {l: [] for l in levels}
    for _ in range(B):
        pick = rng.choice(tops, len(tops))
        parts = []
        for i, t in enumerate(pick):
            g = groups[t].copy(); g["topo_id"] = f"{t}_{i}"; parts.append(g)
        d = pd.concat(parts)
        boot["all"].append(share(d))
        for dens in levels[1:]:
            boot[dens].append(share(d[d.density == dens], by_density=True))
    rows = []
    for l in levels:
        pt = share(df) if l == "all" else share(df[df.density == l], by_density=True)
        lo, hi = np.percentile(boot[l], [2.5, 97.5])
        rows.append({"density": l if l == "all" else f"{l:.0%}", "topology": pt,
                     "ci_low": lo, "ci_high": hi, "configuration": 1 - pt})
        print(f"{rows[-1]['density']:>4s}: topology {pt:.1%}  (95% interval {lo:.1%} to {hi:.1%})")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "q1_variance_split_ci.csv"), index=False, float_format="%.4f")

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    x = np.arange(len(out))
    err = np.vstack([out.topology - out.ci_low, out.ci_high - out.topology])
    ax.bar(x, out.topology, color="#2a6fb0", width=0.6)
    ax.errorbar(x, out.topology, yerr=err, fmt="none", ecolor="#0b0b0b", capsize=3, lw=1)
    ax.set_xticks(x, ["all" if d == "all" else d for d in out.density])
    ax.set_xlabel("share of nodes vulnerable"); ax.set_ylabel("share of variance from topology")
    ax.yaxis.set_major_formatter(PercentFormatter(1, 0)); ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, "fig1b_variance_split_ci.pdf"))
    print("wrote results/q1_variance_split_ci.csv and figures/fig1b_variance_split_ci.pdf")
