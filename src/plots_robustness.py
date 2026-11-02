"""Three plain figures for the report. Reads the CSVs written by uncertainty.py,
reproduction.py and patching.py; changes nothing.

    python src/plots_robustness.py        -> figures/fig_uncertainty.png/.pdf, fig_reproduction..., fig_patching...
No titles inside the images: put the sentence in the caption.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from paths import RESULTS, FIGURES

BLUE, GREY, ORANGE = "#2a6fb0", "#8a8f98", "#d9822b"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGURES, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def uncertainty():
    d = pd.read_csv(os.path.join(RESULTS, "uncertainty", "draws.csv"))
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    sc = ax.scatter(d.floor, d.topology_share, c=d.decay, cmap="viridis", s=34, edgecolor="white", linewidth=.5)
    ax.axhline(0.5, color="#444", ls="--", lw=1)
    box = dict(fc="none", ec="none")
    ax.text(0.0205, 0.515, "above: network layout matters more", fontsize=8.5, color="#444", bbox=box)
    ax.text(0.0205, 0.485, "below: software setup matters more", fontsize=8.5, color="#444", va="top", bbox=box)
    ax.plot(0.05, 0.258, marker="*", ms=15, color=ORANGE, mec="black", mew=.6, ls="", label="midterm setting (25.8%)")
    ax.yaxis.set_major_formatter(PercentFormatter(1, 0)); ax.set_ylim(0, 0.95)
    ax.set_xlabel("chance of breaking into a fully patched machine (the floor)")
    ax.set_ylabel("share of variation explained by layout")
    cb = fig.colorbar(sc, ax=ax, pad=.02); cb.set_label("how fast an attack weakens per extra step")
    ax.legend(loc="upper left", frameon=False)
    save(fig, "fig_uncertainty")


def reproduction():
    R = pd.read_csv(os.path.join(RESULTS, "reproduction", "scenario_R.csv"))
    s = pd.read_csv(os.path.join(RESULTS, "sweep_attackers.csv"))
    m = R.merge(s, on=["topo_id", "density", "config_seed", "adversary"])
    m["band"] = pd.cut(m.R_mean, [0, 1, 2, 4, 100], labels=["under 1", "1 to 2", "2 to 4", "over 4"])
    g = m.groupby("band", observed=True).compromise_fraction.agg(["mean", "count"])
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    b = ax.bar(g.index.astype(str), g["mean"], color=BLUE, width=.62)
    for r, (mean, n) in zip(b, zip(g["mean"], g["count"])):
        ax.text(r.get_x() + r.get_width() / 2, mean + .012, f"{mean:.0%}\n({n} setups)", ha="center", fontsize=8.5)
    ax.yaxis.set_major_formatter(PercentFormatter(1, 0)); ax.set_ylim(0, .72)
    ax.set_xlabel("new machines one taken-over machine breaks into (R)")
    ax.set_ylabel("average share of network compromised")
    save(fig, "fig_reproduction")


def patching():
    s = pd.read_csv(os.path.join(RESULTS, "patching", "summary_by_strategy.csv"))
    labels = {"combined": "easiest to break\n and best connected", "package_first": "whole software\n package at a time",
              "exploit_first": "easiest to break", "hub_first": "best connected", "random": "random"}
    s = s.sort_values("mean_reduction_pts")
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    y = np.arange(len(s))
    cols = [GREY if k == "random" else BLUE for k in s.strategy]
    ax.barh(y, s.mean_reduction_pts, color=cols, height=.62,
            xerr=[s.mean_reduction_pts - s.ci_low, s.ci_high - s.mean_reduction_pts],
            error_kw=dict(ecolor="#222", lw=1, capsize=3))
    for i, v in enumerate(s.mean_reduction_pts):
        ax.text(s.ci_high.iloc[i] + .3, i, f"{v:.1f}", va="center", fontsize=9)
    ax.set_yticks(y, [labels[k] for k in s.strategy])
    ax.set_xlim(0, 20.5)
    ax.set_xlabel("drop in share of network compromised (percentage points)")
    save(fig, "fig_patching")


if __name__ == "__main__":
    uncertainty(); reproduction(); patching()
    print("wrote figures/fig_uncertainty, fig_reproduction, fig_patching (.png and .pdf)")
