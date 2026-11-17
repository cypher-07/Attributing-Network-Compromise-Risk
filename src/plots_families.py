import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paths import RESULTS, FIGURES

ORDER = ["small_world", "random", "real", "fat_tree", "scale_free", "ALL (pooled)"]
LABEL = {"small_world": "small-world", "random": "random", "real": "real ISP",
         "fat_tree": "fat-tree", "scale_free": "scale-free", "ALL (pooled)": "all 140 pooled"}
COL = {"topology": "#2a6f97", "configuration": "#e09f3e", "attacker": "#9b2226", "other": "#b8b8b8"}


def fig_families():
    q = pd.read_csv(os.path.join(RESULTS, "q1_families.csv"))
    q = q[q.design == "three_attackers"]
    p = q.pivot(index="family", columns="component", values="share").loc[ORDER]
    ci = q[q.component == "topology"].set_index("family").loc[ORDER]
    n = ci["networks"]
    p["other"] = p["network_x_attacker"] + p["residual"]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(ORDER)); bottom = np.zeros(len(ORDER))
    for c, name in [("topology", "topology (network shape)"), ("configuration", "software configuration"),
                    ("attacker", "attacker's starting point"), ("other", "noise and interactions")]:
        v = p[c].values * 100
        ax.bar(x, v, bottom=bottom, color=COL[c], label=name, width=0.65)
        for i, (b, h) in enumerate(zip(bottom, v)):
            if h >= 8:
                ax.text(i - (0.17 if c == "topology" else 0), b + h / 2, f"{h:.0f}%", ha="center", va="center", color="white", fontsize=9)
        bottom += v
    lo, hi = ci.ci_low.values * 100, ci.ci_high.values * 100; mid = ci.share.values * 100
    ax.errorbar(x, mid, yerr=[mid - lo, hi - mid], fmt="none", ecolor="black", capsize=3, lw=1)
    ax.set_xticks(x, [f"{LABEL[f]}\n({int(n[f])} networks)" for f in ORDER], fontsize=9)
    ax.set_ylabel("share of variation in compromise (%)")
    ax.set_ylim(0, 100)
    ax.set_title("What drives compromise depends on the kind of network", loc="left", fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.005, "Black whiskers: 95% interval on the topology share (bootstrap over networks). Floor 0.05, decay 0.5.", fontsize=7.5, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(FIGURES, "fig_q1_families." + e), dpi=200)
    plt.close(fig)


def fig_settings():
    d = pd.read_csv(os.path.join(RESULTS, "uncertainty", "q2_draws.csv"))
    fams = ["small_world", "random", "real", "fat_tree", "scale_free", "ALL"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    for i, f in enumerate(fams):
        v = d["topology_share_" + f] * 100
        a.bar(i, v.max() - v.min(), bottom=v.min(), color=COL["topology"], alpha=0.35, width=0.6)
        a.scatter(np.full(len(v), i) + np.linspace(-0.18, 0.18, len(v)), v, s=9, color=COL["topology"], zorder=3)
        a.hlines(v.median(), i - 0.3, i + 0.3, color="black", lw=2, zorder=4)
    a.axhline(50, color="#999", lw=0.8, ls="--")
    a.set_xticks(range(len(fams)), [LABEL.get(f if f != "ALL" else "ALL (pooled)", f) for f in fams], fontsize=8.5)
    a.set_ylabel("topology share of variation (%)")
    a.set_title("Topology share across 24 settings of floor and decay", loc="left", fontsize=11)
    a.text(-0.4, 51, "dashed line: topology = configuration", fontsize=7.5, color="#666", ha="left", va="bottom")
    a.spines[["top", "right"]].set_visible(False)
    for col, name, c in [("config_only_r2", "software only", COL["configuration"]),
                         ("pc_linear_r2", "percolation threshold + software", "#888888"),
                         ("structure_gb_r2", "structure-aware model", COL["topology"])]:
        b.scatter(d.floor, d[col], s=22, color=c, label=name)
    b.axhline(0, color="#999", lw=0.8)
    b.set_xlabel("floor (chance of breaking into a patched machine)")
    b.set_ylabel("R$^2$ on unseen families (average of 20 pairs)")
    b.set_title("Q2 accuracy across the same 24 settings", loc="left", fontsize=11)
    b.legend(frameon=False, fontsize=8, loc="lower left")
    b.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(FIGURES, "fig_settings_check." + e), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    fig_families(); fig_settings()
