"""Sensitivity of the topology share to the floor, the decay constant and the EPSS mapping,
plus a bootstrap interval (over networks) for the baseline share.

    python src/sensitivity.py            # about 2-3 minutes

Writes results/sensitivity/parameter_sensitivity.csv and prints the table.
Does not change any existing file.
"""
import json, os
import numpy as np, pandas as pd
import simulate as S
from simulate import prepare, simulate
from scoring import build_table, P_CEIL
from topology import load_topology, topology_names
from paths import CONFIGS, CATALOG, EPSS, SWEEP, SENSITIVITY

TRIALS = 2000
DENS = (0.2, 0.4, 0.6, 0.8)


def share(d, by_density=False):
    """Topology share of variance (logit scale), within density level."""
    if by_density:
        g = d.groupby("topo_id").y; n = d.topo_id.nunique(); k = len(d) / n
        msw = (d.y - g.transform("mean")).pow(2).sum() / (len(d) - n)
        msb = k * ((g.mean() - d.y.mean()) ** 2).sum() / (n - 1)
        vt = max((msb - msw) / k, 0)
        return vt / (vt + msw)
    cells = d.groupby(["topo_id", "density"]).y; k = 5
    msw = (d.y - cells.transform("mean")).pow(2).sum() / (len(d) - len(cells))
    cm = cells.mean().reset_index(); cm["dm"] = cm.groupby("density").y.transform("mean")
    dfb = len(cm) - d.density.nunique()
    vt = max((k * ((cm.y - cm.dm) ** 2).sum() / dfb - msw) / k, 0)
    return vt / (vt + msw)


def add_logit(df, col):
    eps = 0.5 / df.n_nodes
    x = df[col].clip(eps, 1 - eps)
    df["y"] = np.log(x / (1 - x))
    return df


def run(floor=0.05, decay=0.5, transform="linear"):
    S.DECAY = decay
    tbl = build_table(CATALOG, EPSS, floor=floor, ceil=P_CEIL, transform=transform)[0]
    pk = sorted(tbl); ix = {s: i for i, s in enumerate(pk)}
    q = np.array([(tbl[s] - floor) / (P_CEIL - floor) for s in pk])
    rows = []
    for ti, t in enumerate(topology_names()):
        G, nodes, _ = load_topology(t); A, D, idx = prepare(G, nodes)
        for d in DENS:
            for s in range(1, 6):
                cn = json.load(open(os.path.join(CONFIGS, f"{t}_d{d}_s{s}.json")))["nodes"]
                sw = [cn[v]["software"] for v in nodes]
                p = np.array([tbl[x] for x in sw])
                e = idx[[v for v in nodes if cn[v]["is_entry_point"]][0]]
                f = simulate(A, D, p, e, TRIALS, mode="decay", seed=1000 * ti + 100 * int(d * 10) + s,
                             shared=(np.array([ix[x] for x in sw]), q, floor, P_CEIL))["frac"].mean()
                rows.append((t, len(nodes), d, f))
    df = add_logit(pd.DataFrame(rows, columns=["topo_id", "n_nodes", "density", "f"]), "f")
    out = {"topology_share": share(df), "mean_compromised": df.f.mean()}
    for d, g in df.groupby("density"):
        out[f"topology_share_d{d}"] = share(g, by_density=True)
    return out


if __name__ == "__main__":
    variants = [("baseline (floor 0.05, decay 0.5)", {}),
                ("floor 0.02", dict(floor=0.02)), ("floor 0.10", dict(floor=0.10)),
                ("decay 0.3", dict(decay=0.3)), ("decay 0.7", dict(decay=0.7)),
                ("EPSS sqrt mapping", dict(transform="sqrt"))]
    res = []
    for name, kw in variants:
        r = run(**kw); res.append({"variant": name, **r}); print(f"{name:34s} topology {r['topology_share']:.1%}  mean compromised {r['mean_compromised']:.1%}", flush=True)
    # bootstrap over networks, baseline sweep already saved
    sw = add_logit(pd.read_csv(SWEEP).rename(columns={"compromise_fraction": "f"}), "f")
    rng = np.random.default_rng(0); tops = sw.topo_id.unique(); bs = []
    groups = {t: sw[sw.topo_id == t] for t in tops}
    for _ in range(2000):
        parts = []
        for i, t in enumerate(rng.choice(tops, len(tops))):
            g = groups[t].copy(); g["topo_id"] = f"{t}_{i}"; parts.append(g)
        bs.append(share(pd.concat(parts)))
    lo, med, hi = np.percentile(bs, [2.5, 50, 97.5])
    print(f"\nBootstrap over networks, baseline topology share: median {med:.1%}, 95% interval {lo:.1%} to {hi:.1%}")
    os.makedirs(os.path.join(SENSITIVITY), exist_ok=True)
    pd.DataFrame(res).to_csv(os.path.join(SENSITIVITY, "parameter_sensitivity.csv"), index=False, float_format="%.4f")
    print("wrote results/sensitivity/parameter_sensitivity.csv")
