# How much of network compromise risk is topology?

CYSE 610 course project (George Mason University). We simulate assume-breach
attacks on 20 real ISP backbones from the Internet Topology Zoo and measure how
much of the variation in how far an attack spreads comes from the network's
**topology**, and how much from its **software configuration**.

**Current result** ([`results/q1_variance_split.csv`](results/q1_variance_split.csv)): at a fixed
share of vulnerable nodes, **topology explains 26%** of the variation in
compromise and **configuration 74%**. Topology matters most when vulnerable
nodes are scarce (38% at 20% density) and least when they are common (13% at
80%).

## Layout

```
src/                     all code (run from the project root, see below)
  placement/             original configuration generator and 12-package catalog
data/
  networks/              20 Topology Zoo .graphml files
  catalog/               software catalog (NVD) and EPSS scores, cached
  configs/               one JSON per scenario: software, p, entry point per node
  configs_manifest.csv   index of the configs
  topology_features.csv  10 structural features + percolation threshold pc
  ingestion_log.csv      what loading dropped (largest component, parallel links)
results/
  q1_variance_split.csv  topology vs configuration, overall and per density
  network_risk.csv       mean compromise per network and density
  sweep.csv              one row per scenario; all analysis reads only this
  rewiring/              degree-preserving rewiring experiment
  sensitivity/independent_exploits/   same outputs under the old exploit model
figures/                 7 PDF figures + FIGURES.md (title, caption, how to read each)
viz/                     interactive propagation viewer (single HTML page)
```

## Run order

From the project root. The first two steps hit NVD and FIRST; their outputs are
cached in `data/catalog/`, so skip them unless you want a fresh pull.

```bash
python src/catalog_fetch.py      # NVD -> data/catalog/vulnerability_catalog.json
python src/epss_fetch.py         # FIRST -> data/catalog/epss_scores.json
python src/scoring.py            # print p per package and EPSS coverage
python src/topology.py           # features -> data/topology_features.csv
python src/config_gen.py         # redraw software in data/configs (in place)
python src/sweep.py              # -> results/sweep.csv (~6 s)
python src/q1_anova.py           # -> results/SUMMARY.md, q1_variance_split.csv, network_risk.csv
python src/rewiring.py           # -> results/rewiring/ (~3 min; degree-preserving rewiring)
python src/figures.py            # -> figures/*.pdf + figures/FIGURES.md
python src/viz_export.py && python viz/build_viewer.py   # -> viz/backbone_breach_viewer.html
python src/check.py              # validation on Aarnet
```

Sensitivity run (old independent-exploit model):

```bash
python src/sweep.py 2000 independent
python src/q1_anova.py results/sensitivity/independent_exploits/sweep.csv
```

All paths live in `src/paths.py`.

## The model

**Attacker.** Assume-breach: one entry node per config, chosen at random
independently of the graph, already compromised.

**Software and susceptibility.** Each vulnerable node runs one of 24 packages:
12 famous vulnerable versions (Apache 2.4.49, vsftpd 2.3.4, ...) and 12
well-patched versions of the same products. Other nodes are fully patched.
For a package,

    q = max EPSS over its foothold CVEs      (chance the attacker has a working exploit)
    p = 0.05 + 0.90 × q                      (average per-attempt success)

- **EPSS, not CVSS.** EPSS is a probability of exploitation in the wild; CVSS
  is a severity score (FIRST: CVSS 7+ prioritisation ~4% efficient vs ~65% for
  EPSS 0.1). <https://www.first.org/epss/model>, arXiv:2302.14172,
  arXiv:1908.04856.
- **Max, never count or sum.** The attacker needs one working exploit.
- **Foothold CVEs only:** network attack vector and an impact that gives access
  (CVSS v3 confidentiality and integrity HIGH; v2 both at least PARTIAL, or a
  privilege-gain flag). DoS, disclosure and user enumeration don't count.
- **Floor 0.05**, even on patched nodes: credential reuse doesn't care about
  patch level. **Ceiling 0.95**: exploits still fail sometimes.

**Shared exploits.** At the start of each attack, the attacker either holds a
working exploit for each package or not (probability q). A held exploit works
at 0.95 on every node running that package; otherwise attempts succeed at the
0.05 floor. Identical software falls together, as with real worms. Each node's
average success rate is still exactly p.

**Spread (independent cascade, decay reachability).** A newly compromised node
makes one attempt on every node it can reach; an attempt over d links succeeds
with `p × 0.5^(d − 1)`. Each attacker–target pair fires at most once, so the
outcome measures structure rather than how many rounds were run.

**Analysis.** For each scenario, the mean share of nodes compromised over 2,000
attacks. Variance components on the logit scale, compared within each density
level, split that variation into topology and configuration; the two shares
add to 100%. Density (share of nodes vulnerable: 20/40/60/80%) is a scenario
setting, not something to explain.

## Known limitations

Topology Zoo nodes are points of presence, not hosts; all 20 networks are ISP
backbones; the decay constant 0.5 has no empirical source; EPSS is a 30-day
in-the-wild probability used as an exploit-availability chance; CVSS v2's
"partial" is a looser foothold bar than v3's "high".
