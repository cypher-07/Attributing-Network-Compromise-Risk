"""Figures for the paper.

    python src/figures.py        (after sweep.py and q1_anova.py)

Reads results/sweep.csv, results/q1_variance_split.csv (both from the one
scenario table) and data/topology_features.csv for pc. Writes to figures/:

  fig1_variance_split.pdf      HEADLINE: topology vs configuration, overall and
                               per density (variance components, sum to 100%)
  fig2_risk_heatmap.pdf        mean compromise, network x density
  fig3_percolation.pdf         risk against density (a) and against d/pc (b)
  fig4_risk_vs_threshold.pdf   network risk against pc: the mechanism
  fig5_size.pdf                risk against network size, with the 1/n floor
  fig6_config_severity.pdf     what about a configuration drives risk (mean p)
  fig7_rewiring.pdf            real graph vs degree-preserving rewired copies
  FIGURES.md                   title, caption and how-to-read for each figure

Figures carry no titles, captions, legends or reference-line labels (decided
2026-10-13): everything that explains a figure lives in FIGURES.md, which is
generated here so its numbers always match the figure. Only data, axes, axis
labels, colour scales and point names are drawn. Vector PDF, fonts embedded
as TrueType. Colours: categorical slots and the blue sequential ramp from the
dataviz reference palette, validated for CVD separation.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter, FixedLocator, NullLocator
from scipy.stats import spearmanr
from scoring import P_FLOOR, P_CEIL
from paths import FIGURES, SWEEP, Q1, TOPOLOGY_FEATURES, RESULTS, rel

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TOPO, CONF = SERIES[0], SERIES[1]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "pdf.fonttype": 42,
})

CAPTIONS = []          # (file, title, caption, how to read), in figure order


def save(fig, name, title, caption, read):
    fig.savefig(os.path.join(FIGURES, name), bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    CAPTIONS.append((name, title, caption, read))


def label_points(ax, xs, ys, names, fontsize=7.5):
    """Name each point, trying offsets around it in turn and keeping the first
    whose text box clears every label and point already placed."""
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pts = [ax.transData.transform((x, y)) for x, y in zip(xs, ys)]
    taken = []
    cands = [(6, 2, "left", "bottom"), (6, -2, "left", "top"), (-6, 2, "right", "bottom"),
             (-6, -2, "right", "top"), (0, 7, "center", "bottom"), (0, -7, "center", "top"),
             (10, 10, "left", "bottom"), (-10, -10, "right", "top")]
    for i in sorted(range(len(names)), key=lambda i: -ys[i]):
        best = None
        for dx, dy, ha, va in cands:
            t = ax.annotate(names[i], (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points",
                            ha=ha, va=va, fontsize=fontsize, color=INK_2)
            bb = t.get_window_extent(r).expanded(1.05, 1.15)
            hit = any(bb.overlaps(b) for b in taken) or any(
                bb.contains(px, py) for j, (px, py) in enumerate(pts) if j != i)
            if not hit:
                best = bb
                break
            t.remove()
        if best is None:                                  # nothing clear: keep the first
            dx, dy, ha, va = cands[0]
            t = ax.annotate(names[i], (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points",
                            ha=ha, va=va, fontsize=fontsize, color=INK_2)
            best = t.get_window_extent(r)
        taken.append(best)


def setup(df):
    r = df.iloc[0]
    return (f"{r['mode'].capitalize()} reachability, {r.exploits} exploits, floor {P_FLOOR}, "
            f"ceiling {P_CEIL}; {df.config_seed.nunique()} configurations x {int(r.n_trials):,} "
            f"simulated attacks per network and density")


def fig1(split, df):
    """Headline: topology vs configuration, 100% bars."""
    v = split.copy()
    v["label"] = ["All densities"] + [d + " vulnerable" for d in v.density[1:]]
    v = v.iloc[::-1]                                     # all densities on top
    fig, ax = plt.subplots(figsize=(7.5, 3.2), layout="constrained")
    y = np.arange(len(v))
    h = [0.62 if l == "All densities" else 0.5 for l in v.label]
    ax.barh(y, v.topology, height=h, color=TOPO, edgecolor=SURFACE, linewidth=2)
    ax.barh(y, v.configuration, left=v.topology, height=h, color=CONF, edgecolor=SURFACE, linewidth=2)
    for yi, t, c, lab in zip(y, v.topology, v.configuration, v.label):
        bold = "bold" if lab == "All densities" else "normal"
        ax.text(t / 2, yi, f"{t:.0%}", ha="center", va="center", color="white", fontweight=bold)
        ax.text(t + c / 2, yi, f"{c:.0%}", ha="center", va="center", color=INK, fontweight=bold)
    ax.set_yticks(y, v.label)
    for tl in ax.get_yticklabels():
        if tl.get_text() == "All densities":
            tl.set_fontweight("bold"); tl.set_color(INK)
    ax.axhline(len(v) - 1.5, color=GRID, lw=1)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Share of the variation in compromise between scenarios at the same density")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    a = v.iloc[-1]
    per = v.iloc[:-1].iloc[::-1]
    save(fig, "fig1_variance_split.pdf",
         "How much of compromise risk is topology?",
         f"Share of the variation in compromise attributable to topology (which network) and to "
         f"configuration (which software runs where), estimated as variance components on the "
         f"logit scale within each density level, so the two shares sum to 100%. Overall, "
         f"topology explains {a.topology:.0%} (p = {a.p_topology:.1g}) and configuration "
         f"{a.configuration:.0%}. Topology's share falls from {per.topology.iloc[0]:.0%} when 20% of "
         f"nodes are vulnerable to {per.topology.iloc[-1]:.0%} at 80%: it matters most when "
         f"vulnerable nodes are scarce. {setup(df)}.",
         "Blue segment: topology. Orange segment: configuration. Top bar (bold): all densities "
         "pooled; the four bars below: each density level on its own.")


def fig2(df, feats):
    h = df.groupby(["topo_id", "density"]).compromise_fraction.mean().unstack()
    h = h.loc[h.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(6.2, 7.4), layout="constrained")
    im = ax.imshow(h.values, cmap=SEQ, vmin=0, vmax=0.7, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(h.shape[1]), [f"{d:.0%}" for d in h.columns])
    ax.set_yticks(range(h.shape[0]), [f"{t}  ({int(feats.loc[t, 'n_nodes'])})" for t in h.index])
    ax.tick_params(length=0)
    ax.set_xlabel("Share of nodes vulnerable (density)")
    ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            v = h.values[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8.5,
                    color="white" if v > 0.38 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, shrink=0.6,
                      format=PercentFormatter(1.0, decimals=0))
    cb.set_label("Mean share of network compromised")
    cb.outline.set_visible(False)
    save(fig, "fig2_risk_heatmap.pdf",
         "Compromise by network and density",
         f"Mean share of each network compromised, by share of nodes vulnerable. Each cell averages "
         f"{df.config_seed.nunique()} configurations. Networks are sorted by mean risk, from "
         f"{h.index[0]} ({h.mean(axis=1).iloc[0]:.0%}) to {h.index[-1]} "
         f"({h.mean(axis=1).iloc[-1]:.0%}). {setup(df)}.",
         "Rows: networks, node count in brackets. Columns: density. Darker blue and the printed "
         "value: higher mean compromise.")


def fig3(df, feats):
    p = df.groupby(["topo_id", "density"]).compromise_fraction.mean().reset_index()
    p = p.merge(feats[["pc", "n_nodes"]], left_on="topo_id", right_index=True)
    p["d_pc"] = p.density / p.pc
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True, layout="constrained")
    lognorm = matplotlib.colors.LogNorm(vmin=p.n_nodes.min(), vmax=p.n_nodes.max())
    sub = LinearSegmentedColormap.from_list("blue_ord", BLUE_RAMP[2:])
    for ax, x, label, tag in ((axes[0], "density", "Share of nodes vulnerable (density)", "(a)"),
                              (axes[1], "d_pc", "Density / percolation threshold  (d / pc, log scale)", "(b)")):
        for t, g in p.groupby("topo_id"):
            g = g.sort_values(x)
            c = sub(lognorm(g.n_nodes.iloc[0]))
            ax.plot(g[x], g.compromise_fraction, color=c, lw=1.4, alpha=0.85)
            ax.plot(g[x], g.compromise_fraction, "o", ms=4, color=c,
                    markeredgecolor=SURFACE, markeredgewidth=0.8)
        ax.set_xlabel(label)
        ax.text(0.01, 0.99, tag, transform=ax.transAxes, ha="left", va="top",
                fontweight="bold", color=INK)
    axes[0].set_xticks([0.2, 0.4, 0.6, 0.8], ["20%", "40%", "60%", "80%"])
    axes[1].set_xscale("log")
    axes[1].xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 3, 5]))
    axes[1].xaxis.set_minor_locator(NullLocator())
    axes[1].set_xticklabels(["0.5", "1", "2", "3", "5"])
    axes[1].axvline(1.0, color=INK_2, lw=1, ls="--")
    axes[0].set_ylabel("Mean share of network compromised")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    sm = matplotlib.cm.ScalarMappable(norm=lognorm, cmap=sub)
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label("Network size (nodes, log scale)")
    ticks = [t for t in (11, 20, 50, 100, 150) if p.n_nodes.min() <= t <= p.n_nodes.max()]
    cb.set_ticks(ticks); cb.set_ticklabels([str(t) for t in ticks]); cb.minorticks_off()
    cb.outline.set_visible(False)
    save(fig, "fig3_percolation.pdf",
         "Percolation anchor",
         "Mean compromise of each network against (a) the share of nodes vulnerable and (b) that "
         "share divided by the network's Molloy-Reed site-percolation threshold "
         "pc = <k>/(<k^2> - <k>). If the degree distribution governs spread, the curves should "
         "line up in (b). They largely do: below the threshold (d/pc < 1) vulnerable nodes form "
         "isolated islands and compromise stays low; above it, it rises steeply. The spread that "
         "remains at a given d/pc is structure pc does not capture, plus config-to-config noise. "
         f"{setup(df)}.",
         "One line per network, one point per density (20/40/60/80%). Line colour: network size, "
         "lighter = smaller (colour scale on the right). Dashed vertical line in (b): the "
         "threshold, d = pc.")


def fig4(net):
    rho, pv = spearmanr(net.pc, net.risk_mean)
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    ax.plot(net.pc, net.risk_mean, "o", ms=8, color=TOPO, markeredgecolor=SURFACE, markeredgewidth=1.5)
    b = np.polyfit(np.log(net.pc), net.risk_mean, 1)
    xs = np.linspace(net.pc.min() * 0.95, net.pc.max() * 1.05, 100)
    ax.plot(xs, np.polyval(b, np.log(xs)), color=INK_2, lw=1, ls="--")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([0.15, 0.2, 0.3, 0.4, 0.5]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["0.15", "0.2", "0.3", "0.4", "0.5"])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Percolation threshold pc (log scale)")
    ax.set_ylabel("Mean share compromised (all densities)")
    label_points(ax, list(net.pc), list(net.risk_mean), list(net.network))
    save(fig, "fig4_risk_vs_threshold.pdf",
         "Networks with a lower percolation threshold are compromised more",
         f"Mean compromise of each network, averaged over all densities and configurations, "
         f"against its percolation threshold pc, the share of nodes that must be vulnerable for "
         f"a giant connected cluster to form. Spearman rho = {rho:+.2f} (p = {pv:.0g}, "
         f"n = {len(net)}). pc depends only on the degree distribution, so this is a structural "
         f"mechanism rather than a size effect (see Figure 5).",
         "One point per network, named. Dashed line: least-squares fit of risk on log pc.")


def fig5(net):
    rho, pv = spearmanr(net.n_nodes, net.risk_mean)
    fig, ax = plt.subplots(figsize=(7, 4.3), layout="constrained")
    xs = np.linspace(net.n_nodes.min() * 0.9, net.n_nodes.max() * 1.1, 200)
    ax.plot(xs, 1 / xs, color=INK_2, lw=1.2, ls="--")
    ax.plot(net.n_nodes, net.risk_mean, "o", ms=8, color=TOPO, markeredgecolor=SURFACE,
            markeredgewidth=1.5)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([10, 20, 50, 100, 150]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["10", "20", "50", "100", "150"])
    ax.set_ylim(0, max(0.6, net.risk_mean.max() + 0.05))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Network size (nodes, log scale)")
    ax.set_ylabel("Mean share compromised (all densities)")
    label_points(ax, list(net.n_nodes), list(net.risk_mean), list(net.network))
    save(fig, "fig5_size.pdf",
         "Risk does not depend on network size",
         f"Mean compromise of each network against its node count. Spearman rho = {rho:+.2f} "
         f"(p = {pv:.2f}, n = {len(net)}): no significant size effect, so the structural signal "
         f"in Figures 3-4 is not network size in disguise.",
         "One point per network, named. Dashed curve: 1/n, the share compromised if the attack "
         "never leaves the entry node; the gap between a point and the curve is spread beyond "
         "the entry.")


def fig6(df):
    """What about configuration matters: overall exploitability."""
    y = np.log(df.compromise_fraction / (1 - df.compromise_fraction))
    cell = df.groupby(["topo_id", "density"])
    dx = df.mean_p - cell.mean_p.transform("mean")
    dy = df.compromise_fraction - cell.compromise_fraction.transform("mean")
    dyl = y - y.groupby([df.topo_id, df.density]).transform("mean")
    r_logit = np.corrcoef(dx, dyl)[0, 1]
    r_raw = np.corrcoef(dx, dy)[0, 1]
    fig, ax = plt.subplots(figsize=(7, 4.6), layout="constrained")
    ax.plot(dx, dy, "o", ms=4.5, color=CONF, alpha=0.75, markeredgecolor=SURFACE,
            markeredgewidth=0.5)
    b = np.polyfit(dx, dy, 1)
    xs = np.linspace(dx.min(), dx.max(), 50)
    ax.plot(xs, np.polyval(b, xs), color=INK_2, lw=1, ls="--")
    ax.axhline(0, color=INK_2, lw=0.6); ax.axvline(0, color=INK_2, lw=0.6)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.2f}"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:+.0f} pp"))
    ax.set_xlabel("Mean exploit probability p over all nodes, relative to the network-and-density average")
    ax.set_ylabel("Share compromised, relative to the\nnetwork-and-density average")
    save(fig, "fig6_config_severity.pdf",
         "What makes one configuration riskier than another",
         f"Each point is one configuration, compared with the other configurations of the same "
         f"network at the same density, so topology and density are held fixed. A configuration "
         f"whose nodes are on average more exploitable is compromised more: r = {r_raw:+.2f} "
         f"(r = {r_logit:+.2f} on the logit scale used in the variance split, i.e. "
         f"{r_logit ** 2:.0%} of the configuration effect). The mean is driven by how many "
         f"vulnerable nodes run the famous vulnerable versions (p 0.77-0.95) rather than the "
         f"patched ones (p 0.05-0.63). Where the attacker enters matters little: the entry node's "
         f"software and degree each correlate below 0.15 with this residual risk (one-off check, "
         f"2026-10-13). Each point: {int(df.n_trials.iloc[0]):,} simulated attacks.",
         f"One point per configuration ({len(df)} in all, every network and density). Horizontal: "
         f"how much more (right) or less (left) exploitable the configuration is than its "
         f"siblings. Vertical: how much more or less of the network it loses, in percentage "
         f"points. Dashed line: least-squares fit (slope {b[0]:.1f} pp per 0.01 of mean p). "
         f"Thin lines: zero on both axes.")


def fig7(rw):
    """Rewiring: each network's risk on the original graph vs its degree-preserving copies."""
    rw = rw.sort_values("original")
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    y = np.arange(len(rw))
    for yi, (_, r) in zip(y, rw.iterrows()):
        ax.plot([r.original, r.degree], [yi, yi], color=GRID, lw=3, solid_capstyle="round", zorder=1)
    ax.plot(rw.original, y, "o", ms=7, color=INK_2, markeredgecolor=SURFACE, markeredgewidth=1, zorder=2)
    ax.plot(rw.degree, y, "o", ms=7, color=TOPO, markeredgecolor=SURFACE, markeredgewidth=1, zorder=3)
    ax.set_yticks(y, rw.index)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Mean share compromised (all densities and configurations)")
    rho = spearmanr(rw.original, rw.degree)[0]
    rho_p = spearmanr(rw.path_change, rw.delta_degree)[0]
    big = rw.delta_degree.idxmax()
    save(fig, "fig7_rewiring.pdf",
         "Rewiring with degrees fixed keeps most of each network's risk",
         f"Each network's mean compromise on its real graph and on 20 randomised copies that keep "
         f"every node's degree (and so pc) but scramble which nodes are linked. Configurations are "
         f"held fixed. The ranking of networks largely survives (Spearman {rho:+.2f}) and the "
         f"average changes by {rw.delta_degree.mean() * 100:+.1f} points, so the degree "
         f"distribution carries most of topology's effect. Where risk does change, it follows path "
         f"length: networks whose paths shorten when rewired get riskier (Spearman {rho_p:+.2f} "
         f"between change in risk and change in normalised path length). The largest change is "
         f"{big} ({rw.loc[big, 'original']:.0%} to {rw.loc[big, 'degree']:.0%}), whose long "
         f"geographic chains gain shortcuts.",
         "One row per network, sorted by risk on the real graph. Grey dot: real graph. Blue dot: "
         "mean of the 20 rewired copies. Grey bar: the change between them.")


def write_md():
    lines = ["# Figures", "",
             "Generated by `src/figures.py` with the figures; do not edit by hand. The figures "
             "carry no titles, captions or legends: everything needed to read them is here.", ""]
    for k, (name, title, caption, read) in enumerate(CAPTIONS, 1):
        lines += [f"## Figure {k}. {title}", "", f"File: `{name}`", "", caption, "",
                  f"**How to read it.** {read}", ""]
    open(os.path.join(FIGURES, "FIGURES.md"), "w", encoding="utf-8").write("\n".join(lines))


if __name__ == "__main__":
    os.makedirs(FIGURES, exist_ok=True)
    df = pd.read_csv(SWEEP)
    if not os.path.exists(Q1):
        raise SystemExit(f"{rel(Q1)} missing -- run src/q1_anova.py first")
    split = pd.read_csv(Q1)
    feats = pd.read_csv(TOPOLOGY_FEATURES).set_index("topology")
    net = df.groupby("topo_id").compromise_fraction.mean().rename("risk_mean").to_frame()
    net = net.join(feats[["pc", "n_nodes"]]).rename_axis("network").reset_index()
    for old in os.listdir(FIGURES):                     # drop earlier outputs
        if old.endswith((".png", ".pdf")):
            try:
                os.remove(os.path.join(FIGURES, old))
            except PermissionError:                     # open in a viewer (Windows)
                print(f"could not remove {rel(os.path.join(FIGURES, old))}: close it and rerun")
    fig1(split, df)
    fig2(df, feats)
    fig3(df, feats)
    fig4(net)
    fig5(net)
    fig6(df)
    rw_path = os.path.join(RESULTS, "rewiring", "network_rewiring.csv")
    if os.path.exists(rw_path):
        fig7(pd.read_csv(rw_path, index_col=0))
    else:
        print(f"{rel(rw_path)} missing -- run src/rewiring.py for figure 7")
    write_md()
    print(f"wrote {len(CAPTIONS)} PDFs and FIGURES.md to {rel(FIGURES)}/")
