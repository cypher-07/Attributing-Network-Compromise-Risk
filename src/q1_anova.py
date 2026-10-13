"""Q1: behind attack propagation, how much is topology and how much configuration?

    python src/q1_anova.py [path/to/sweep.csv]

Reads results/sweep.csv only (or another sweep, e.g. a sensitivity run) and
writes three files NEXT TO the sweep it read:
  q1_variance_split.csv  headline split: one row for all densities, one per level
  network_risk.csv       mean compromise per network and density, with pc
  SUMMARY.md             the numbers in plain language
One adversary this round (assume-breach).

HEADLINE (2026-10-12): variance components, which sum to 100%. Within a
density level, two scenarios differ because they sit on different networks
(topology) or have different configs (configuration). Classical ANOVA on the
logit scale, balanced (CLAUDE.md):

    logit(compromise_fraction) ~ C(topology) * C(density)

  sigma2_config   = MS_resid
  sigma2_topology = (MS_between - MS_resid) / k,   MS_between pooling
                    topology and topology x density (76 df), k = configs/cell
  topology share  = sigma2_topology / (sigma2_topology + sigma2_config)

Both components are chance-corrected and the two shares sum to 1.

Density is a scenario setting, not a source of risk to explain (decided
2026-10-12). It drops out because every comparison is made at one density
level; how a network's effect changes with density counts as topology.

Config seed is nested in topology x density (seed 3 on Aarnet and seed 3 on
Colt are unrelated draws), so configuration variance is the residual: which
nodes are vulnerable, what they run, where the entry point is (plus a small
Monte Carlo part, estimated and reported). A seed main effect is fitted once
as a check; it should be ~0.

Check: omega^2 with density's SS removed from the total,
  omega^2_effect = (SS_effect - df_effect * MS_resid) / (SS_total - SS_density + MS_resid)
  configuration  = SS_resid / (SS_total - SS_density)
It does not sum to 1 (omega^2 corrects the effects for chance; the residual
is a raw share), which is why it is not the headline. Also run on the raw
fraction and per density level. (An above-threshold-only check was run once
-- topology 26.5% vs 28.5% pooled -- and removed on 2026-10-12.)
"""
import os, sys
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from paths import SWEEP, TOPOLOGY_FEATURES, rel

T, D, TD, R = "C(topo_id)", "C(density)", "C(topo_id):C(density)", "Residual"


def logit(x, n):
    # fractions are means over trials and never hit 0 or 1 here; clip by half a
    # node anyway so a future edge case cannot produce +-inf
    eps = 0.5 / n
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))


def anova(df, formula):
    return anova_lm(smf.ols(formula, data=df).fit(), typ=1)


def ms(table, k):
    return table.loc[k, "sum_sq"] / table.loc[k, "df"]


def components(table, k):
    """Topology / configuration shares from variance components; sum to 1."""
    terms = [x for x in (T, TD) if x in table.index]
    ms_b = sum(table.loc[x, "sum_sq"] for x in terms) / sum(table.loc[x, "df"] for x in terms)
    v_t = max((ms_b - ms(table, R)) / k, 0.0)
    return {"topology": v_t / (v_t + ms(table, R)),
            "configuration": ms(table, R) / (v_t + ms(table, R)),
            "p_topology": table.loc[T, "PR(>F)"]}


def omega2(table):
    """omega^2 shares with density's SS removed; do not sum to 1."""
    rest = table["sum_sq"].sum() - (table.loc[D, "sum_sq"] if D in table.index else 0.0)
    om = lambda k: (table.loc[k, "sum_sq"] - table.loc[k, "df"] * ms(table, R)) / (rest + ms(table, R))
    out = {"topology": om(T), "configuration": table.loc[R, "sum_sq"] / rest,
           "p_topology": table.loc[T, "PR(>F)"]}
    if TD in table.index:
        out["topology_x_density"] = om(TD)
        out["p_topology_x_density"] = table.loc[TD, "PR(>F)"]
    return out


def main(sweep_path):
    out_dir = os.path.dirname(sweep_path)
    df = pd.read_csv(sweep_path)
    if any(df[c].nunique() != 1 for c in ("adversary", "mode", "floor", "exploits")):
        sys.exit("sweep mixes adversaries, modes, floors or exploit models -- Q1 expects one of each")
    cells = df.groupby(["topo_id", "density"]).size()
    if cells.nunique() != 1:
        sys.exit(f"design is unbalanced: {cells.value_counts().to_dict()}")
    k = int(cells.iloc[0])
    df["y"] = logit(df.compromise_fraction, df.n_nodes)
    by_d = dict(list(df.groupby("density")))

    full = anova(df, "y ~ C(topo_id) * C(density)")
    rows = [{"density": "all", "mean_compromised": df.compromise_fraction.mean(),
             **components(full, k)}]
    rows += [{"density": f"{d:.0%}", "mean_compromised": g.compromise_fraction.mean(),
              **components(anova(g, "y ~ C(topo_id)"), k)} for d, g in by_d.items()]
    split = pd.DataFrame(rows)[["density", "topology", "configuration", "p_topology",
                                "mean_compromised"]]
    split.to_csv(os.path.join(out_dir, "q1_variance_split.csv"), index=False, float_format="%.4g")
    # checks, reported in SUMMARY.md only
    om = pd.Series(omega2(full))
    omr = pd.Series(omega2(anova(df.assign(y=df.compromise_fraction), "y ~ C(topo_id) * C(density)")))

    # diagnostics
    density_share = full.loc[D, "sum_sq"] / full.sum_sq.sum()
    m = df.compromise_fraction.clip(1e-6, 1 - 1e-6)
    mc = ((df.frac_sd ** 2 / df.n_trials) / (m * (1 - m)) ** 2).mean() / ms(full, R)
    S = "C(config_seed)"
    seed = anova(df, f"y ~ C(topo_id) * C(density) + {S}")
    seed_om = (seed.loc[S, "sum_sq"] - seed.loc[S, "df"] * ms(seed, R)) / (seed.sum_sq.sum() + ms(seed, R))

    # one row per network
    feats = pd.read_csv(TOPOLOGY_FEATURES).set_index("topology")
    net = df.pivot_table(index="topo_id", columns="density", values="compromise_fraction")
    net.columns = [f"risk_d{d:g}" for d in net.columns]
    net.insert(0, "risk_mean", net.mean(axis=1))
    net.insert(0, "pc", feats.loc[net.index, "pc"])
    net.insert(0, "n_nodes", feats.loc[net.index, "n_nodes"])
    net = net.sort_values("risk_mean", ascending=False).rename_axis("network").reset_index()
    net.to_csv(os.path.join(out_dir, "network_risk.csv"), index=False, float_format="%.4f")

    # plain-language summary
    h = split.iloc[0]
    per = split.iloc[1:]
    mean_d = df.groupby("density").compromise_fraction.mean()
    pct = lambda x: f"{x:.1%}"
    r = df.iloc[0]
    top, bot = net.iloc[0], net.iloc[-1]
    lines = [
        "# Q1 results: topology vs configuration",
        "",
        f"Generated by `src/q1_anova.py` from `{rel(sweep_path)}`. Do not edit by hand.",
        "",
        f"Setup: {df.topo_id.nunique()} Topology Zoo networks x {df.density.nunique()} densities x "
        f"{k} configs = {len(df)} scenarios, {int(r.n_trials):,} simulated attacks each. "
        f"Assume-breach attacker, {r['mode']} reachability, {r.exploits} exploits, floor {r.floor:g}.",
        "",
        "## Headline",
        "",
        "Two attacks at the same density differ in how far they spread because they are on",
        "different networks (topology) or because the software on the network differs (configuration).",
        "",
        "| | Share of risk variance |",
        "|---|---|",
        f"| **Topology** | **{pct(h.topology)}** (p = {h.p_topology:.1g}) |",
        f"| **Configuration** | **{pct(h.configuration)}** |",
        "| Total | 100% |",
        "",
        "## By density (share of nodes vulnerable)",
        "",
        "| Density | Mean compromised | Topology | Configuration |",
        "|---|---|---|---|",
    ]
    for d, (_, p) in zip(by_d, per.iterrows()):
        lines.append(f"| {d:.0%} | {mean_d[d]:.0%} | {p.topology:.0%} | {p.configuration:.0%} |")
    lines += [
        "",
        "## Networks",
        "",
        f"Riskiest: {top.network} ({top.risk_mean:.0%} mean compromised); "
        f"least risky: {bot.network} ({bot.risk_mean:.0%}). Full table: `network_risk.csv`.",
        f"Network risk vs percolation threshold pc: Spearman "
        f"{net.risk_mean.corr(net.pc, method='spearman'):+.2f}; vs size: "
        f"{net.risk_mean.corr(net.n_nodes, method='spearman'):+.2f}.",
        "",
        "## Checks",
        "",
        f"- omega^2 split (chance-corrected effects, so it does not sum to 100%): topology "
        f"{pct(om.topology)}, configuration {pct(om.configuration)}, topology x density "
        f"{pct(om.topology_x_density)} (p = {om.p_topology_x_density:.2g}).",
        f"- Same on the raw fraction instead of logit: topology {pct(omr.topology)}, "
        f"configuration {pct(omr.configuration)}.",
        f"- Density itself is {pct(density_share)} of all variance; it is a scenario setting, "
        "excluded from the shares by comparing within each level.",
        f"- Monte Carlo noise is {pct(mc)} of the configuration term.",
        f"- Config seed label as a factor: omega^2 = {seed_om:.3f}, "
        f"p = {seed.loc[S, 'PR(>F)']:.2f} (expected ~0).",
    ]
    if max(h.topology, h.configuration) > 0.9:
        lines += ["", "**WARNING: one source takes > 90% -- check for an independence violation "
                  "before believing it (CLAUDE.md).**"]
    open(os.path.join(out_dir, "SUMMARY.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print("\nwrote " + ", ".join(rel(os.path.join(out_dir, f)) for f in
                                 ("q1_variance_split.csv", "network_risk.csv", "SUMMARY.md")))


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else SWEEP)
