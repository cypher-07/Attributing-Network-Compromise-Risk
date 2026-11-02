"""How much do the headline findings depend on the two assumed settings?

    python src/uncertainty.py [n_draws] [n_trials]      (64 draws x 1,000 attacks: about 15 min)

The model has two settings nobody measured: the floor (chance of breaking into a
fully patched machine, 0.05 in the main run) and the decay (how much harder each
extra step away from the attacker is, 0.5 in the main run). The midterm grid tried
3 x 3 combinations. Here we draw n_draws pairs spread evenly over the whole box
(floor 0.02-0.10, decay 0.3-0.7, scrambled Sobol points), rerun the full
400-scenario sweep for each pair, and ask:

  * how wide is the range of the topology share?
  * in what fraction of the box does configuration still explain more than topology?
  * does the ranking of the networks change?
  * which of the two settings moves the answer?

Everything is "relative to this box": a fraction of draws is not a probability that
the real world behaves this way. The box is the same range as the midterm grid.

Writes results/uncertainty/draws.csv and summary.csv. Nothing existing is changed.
"""
import json, os, sys, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, qmc
import simulate as S
from simulate import prepare, simulate, percolation_threshold
from scoring import build_table, P_CEIL
from topology import load_topology, topology_names
from sensitivity import share, add_logit
from paths import CONFIGS, CATALOG, EPSS, RESULTS

OUT = os.path.join(RESULTS, "uncertainty")
FLOOR_RANGE, DECAY_RANGE = (0.02, 0.10), (0.3, 0.7)
DENS, SEEDS = (0.2, 0.4, 0.6, 0.8), (1, 2, 3, 4, 5)


def load_world(net_dir=None, cfg_dir=CONFIGS):
    """Everything that does not depend on floor/decay: graphs, paths, configs."""
    world = []
    kw = {} if net_dir is None else {"net_dir": net_dir}
    for ti, t in enumerate(topology_names(**kw)):
        G, nodes, _ = load_topology(t, **kw)
        A, D, idx = prepare(G, nodes)
        cfgs = {}
        for d in DENS:
            for s in SEEDS:
                cn = json.load(open(os.path.join(cfg_dir, f"{t}_d{d}_s{s}.json")))["nodes"]
                cfgs[(d, s)] = ([cn[v]["software"] for v in nodes],
                                idx[[v for v in nodes if cn[v]["is_entry_point"]][0]])
        world.append({"name": t, "ti": ti, "n": len(nodes), "A": A, "D": D,
                      "pc": percolation_threshold(G), "cfgs": cfgs})
    return world


def sweep(world, floor, decay, trials):
    """The 400-scenario assume-breach sweep at one (floor, decay)."""
    S.DECAY = decay
    tbl = build_table(CATALOG, EPSS, floor=floor, ceil=P_CEIL)[0]
    pk = sorted(tbl); ix = {s: i for i, s in enumerate(pk)}
    q = np.array([(tbl[s] - floor) / (P_CEIL - floor) for s in pk])
    rows = []
    for w in world:
        for (d, s), (sw, e) in w["cfgs"].items():
            p = np.array([tbl[x] for x in sw])
            f = simulate(w["A"], w["D"], p, e, trials, mode="decay",
                         seed=1000 * w["ti"] + 100 * int(d * 10) + s,
                         shared=(np.array([ix[x] for x in sw]), q, floor, P_CEIL))["frac"].mean()
            rows.append((w["name"], w["n"], w["pc"], d, s, float(p.mean()), f))
    df = pd.DataFrame(rows, columns=["topo_id", "n_nodes", "pc", "density", "config_seed",
                                     "mean_p", "f"])
    return add_logit(df, "f")


def summarise(df):
    out = {"topology_share": share(df), "mean_compromised": df.f.mean()}
    for d, g in df.groupby("density"):
        out[f"topology_share_d{d}"] = share(g, by_density=True)
    nets = df.groupby("topo_id").agg(r=("y", "mean"), pc=("pc", "first"))
    out["rho_percolation"] = spearmanr(nets.pc, nets.r)[0]
    # how well the software's average exploitability tracks risk, within density
    out["r_config"] = float(np.mean([np.corrcoef(g.mean_p, g.y)[0, 1] for _, g in df.groupby("density")]))
    return out


def main(n_draws=64, trials=1000):
    world = load_world()
    pts = qmc.Sobol(d=2, scramble=True, seed=0).random(n_draws)
    lo = np.array([FLOOR_RANGE[0], DECAY_RANGE[0]]); hi = np.array([FLOOR_RANGE[1], DECAY_RANGE[1]])
    pts = lo + pts * (hi - lo)
    rows, t0 = [], time.time()
    for k, (fl, dc) in enumerate(pts):
        r = summarise(sweep(world, float(fl), float(dc), trials))
        rows.append({"draw": k, "floor": fl, "decay": dc, **r})
        print(f"draw {k + 1:3d}/{n_draws}  floor {fl:.3f} decay {dc:.2f}  topology {r['topology_share']:.1%}  "
              f"mean {r['mean_compromised']:.1%}  [{time.time() - t0:5.0f}s]", flush=True)
    S.DECAY = 0.5
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "draws.csv"), index=False, float_format="%.5f")
    s = describe(df)
    s.to_csv(os.path.join(OUT, "summary.csv"), index=False, float_format="%.4f")
    with pd.option_context("display.width", 160, "display.max_colwidth", 80):
        print("\n", s.to_string(index=False))
    return df, s


def describe(df):
    ts = df.topology_share
    items = [
        ("draws", len(df)),
        ("topology share: minimum", ts.min()), ("topology share: 10th percentile", ts.quantile(.1)),
        ("topology share: median", ts.median()), ("topology share: 90th percentile", ts.quantile(.9)),
        ("topology share: maximum", ts.max()),
        ("fraction of draws where configuration explains more than topology", (ts < 0.5).mean()),
        ("fraction of draws where topology share is under one third", (ts < 1 / 3).mean()),
        ("rank correlation, topology share vs floor", spearmanr(df.floor, ts)[0]),
        ("rank correlation, topology share vs decay", spearmanr(df.decay, ts)[0]),
        ("rank correlation, mean compromised vs floor", spearmanr(df.floor, df.mean_compromised)[0]),
        ("rank correlation, mean compromised vs decay", spearmanr(df.decay, df.mean_compromised)[0]),
        ("percolation threshold vs network risk: weakest (rho)", df.rho_percolation.max()),
        ("percolation threshold vs network risk: strongest (rho)", df.rho_percolation.min()),
        ("software exploitability vs risk: weakest correlation", df.r_config.min()),
        ("software exploitability vs risk: strongest correlation", df.r_config.max()),
        ("mean compromised: minimum", df.mean_compromised.min()),
        ("mean compromised: maximum", df.mean_compromised.max()),
    ]
    return pd.DataFrame(items, columns=["quantity", "value"])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 64, int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
