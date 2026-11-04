"""Q2: does a risk predictor built on one family of networks work on another?

    python src/q2_predict.py [scenarios.csv] [features.csv] [out_dir]

Inputs (the same two tables the rest of the project uses)
  scenarios.csv   one row per (config, adversary): sweep_attackers.csv. Needs
                  topo_id, family, density, config_seed, adversary, n_nodes,
                  compromise_fraction, mean_p, vulnerable_machines.
  features.csv    one row per network: topology, pc + the 10 FEATURES from
                  topology.py (make_features.py builds it for any folder of
                  .graphml files, so synthetic and real use identical code).
`family` is real, scale_free, random, small_world, fat_tree, ... Whatever
families are in the file become the rows and columns of the transfer matrix.

Question and design
-------------------
Predict how much of a network gets compromised from (a) the network's
structure and (b) a summary of its software setup. The test of "does it
transfer" is strict: a model trained on every scenario of family A is scored
on family B, a family it never saw. Splitting is by network and family, never
by scenario, so no network's scenarios sit in both train and test.

  train family i, test family j  (i != j)   one cell of the transfer matrix
  i == j                                    leave-one-network-out inside the
                                            family: train on its other networks,
                                            test on the held-out one

Models
  mean            training-set mean of logit risk (the do-nothing floor)
  config_only     gradient boosting on configuration + attacker features only
  pc_linear       linear on percolation threshold + configuration + attacker
                  (one structural number, the simple baseline)
  structure_gb    gradient boosting on all structural + configuration features
                  (the model Q2 asks about)
"structure_gb vs config_only" is the value of knowing the network's structure.
"structure_gb vs pc_linear" is whether a richer model beats one number.

Target and scores
  Target: logit of mean share compromised (as in Q1). Scores are on the share
  scale: R^2 across the test scenarios, and Spearman rank correlation across
  test scenarios and across test networks (mean per network).

Features (all size-normalised, see topology.py): FEATURES + pc.
Configuration/attacker features: density, mean_p (mean susceptibility over
all nodes), and one-hot attacker (only if several attackers are present).
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from topology import FEATURES
from paths import RESULTS, TOPOLOGY_FEATURES

STRUCT = ["pc"] + FEATURES
MODELS = ["mean", "config_only", "pc_linear", "structure_gb"]


def logit(x, n):
    eps = 0.5 / n
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))


def expit(z):
    return 1 / (1 + np.exp(-z))


def build_table(scen_path, feat_path):
    s = pd.read_csv(scen_path)
    f = pd.read_csv(feat_path).rename(columns={"topology": "topo_id"})
    miss = set(s.topo_id) - set(f.topo_id)
    if miss:
        sys.exit(f"no structural features for: {sorted(miss)}")
    df = s.merge(f[["topo_id"] + STRUCT], on="topo_id", how="left")
    df["y"] = logit(df.compromise_fraction, df.n_nodes)
    advs = sorted(df.adversary.unique())
    cfg = ["density", "mean_p"]
    if len(advs) > 1:
        for a in advs:
            df[f"adv_{a}"] = (df.adversary == a).astype(float)
        cfg += [f"adv_{a}" for a in advs]
    return df, cfg


def feature_sets(cfg):
    return {"mean": [], "config_only": cfg, "pc_linear": ["pc"] + cfg,
            "structure_gb": STRUCT + cfg}


def fit_predict(model, cols, train, test):
    if model == "mean":
        return np.full(len(test), train.y.mean())
    if model == "pc_linear":
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    else:
        m = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, random_state=0)
    m.fit(train[cols], train.y)
    return m.predict(test[cols])


def score(test, pred_logit):
    pred = expit(pred_logit)
    true = test.compromise_fraction.to_numpy()
    r2 = r2_score(true, pred)
    rho = spearmanr(true, pred)[0] if np.ptp(pred) > 0 else np.nan
    g = pd.DataFrame({"t": test.topo_id.to_numpy(), "a": true, "p": pred}).groupby("t").mean()
    rho_net = spearmanr(g.a, g.p)[0] if len(g) > 2 and np.ptp(g.p) > 0 else np.nan
    return r2, rho, rho_net


def evaluate(df, cfg):
    """Long table: model, train family, test family, scores."""
    fsets = feature_sets(cfg)
    fams = sorted(df.family.unique())
    rows = []
    for mdl in MODELS:
        cols = fsets[mdl]
        for i in fams:
            for j in fams:
                if i != j:
                    tr, te = df[df.family == i], df[df.family == j]
                    pred = fit_predict(mdl, cols, tr, te)
                    r2, rho, rn = score(te, pred)
                else:                                    # leave one network out
                    fam = df[df.family == i]
                    nets = sorted(fam.topo_id.unique())
                    if len(nets) < 3:
                        rows.append({"model": mdl, "train": i, "test": j, "r2": np.nan,
                                     "spearman": np.nan, "spearman_networks": np.nan,
                                     "n_test": len(fam)})
                        continue
                    preds = pd.Series(index=fam.index, dtype=float)
                    for held in nets:
                        tr, te = fam[fam.topo_id != held], fam[fam.topo_id == held]
                        preds[te.index] = fit_predict(mdl, cols, tr, te)
                    r2, rho, rn = score(fam, preds.to_numpy())
                    te = fam
                rows.append({"model": mdl, "train": i, "test": j, "r2": r2, "spearman": rho,
                             "spearman_networks": rn, "n_test": len(te)})
    return pd.DataFrame(rows)


def matrix(res, model, col):
    m = res[res.model == model].pivot(index="train", columns="test", values=col)
    return m


def plot(res, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = ["config_only", "pc_linear", "structure_gb"]
    fig, ax = plt.subplots(1, len(models), figsize=(4.4 * len(models), 4.0), squeeze=False,
                           sharey=True, gridspec_kw={"wspace": 0.08})
    for k, (a, mdl) in enumerate(zip(ax[0], models)):
        m = matrix(res, mdl, "r2")
        im = a.imshow(m.values, vmin=-0.5, vmax=1, cmap="Blues")
        a.set_xticks(range(len(m.columns)), m.columns, rotation=40, ha="right")
        a.set_yticks(range(len(m.index)), m.index if k == 0 else [""] * len(m.index))
        a.set_title(mdl)
        a.set_xlabel("tested on")
        if k == 0:
            a.set_ylabel("trained on")
        for (r, c), v in np.ndenumerate(m.values):
            a.text(c, r, "" if np.isnan(v) else f"{v:.2f}", ha="center", va="center",
                   color="white" if v > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax[0].tolist(), label="R$^2$ (share compromised)")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(scen, feat, out_dir):
    df, cfg = build_table(scen, feat)
    res = evaluate(df, cfg)
    os.makedirs(out_dir, exist_ok=True)
    res.to_csv(os.path.join(out_dir, "q2_transfer.csv"), index=False, float_format="%.4f")
    plot(res, os.path.join(out_dir, "q2_transfer_r2.png"))
    with pd.option_context("display.width", 200, "display.float_format", "{:.2f}".format):
        for mdl in MODELS[1:]:
            print(f"\n== {mdl}: R^2 (rows trained on, columns tested on)")
            print(matrix(res, mdl, "r2").to_string())
            print(f"-- {mdl}: Spearman over networks")
            print(matrix(res, mdl, "spearman_networks").to_string())
    return res


if __name__ == "__main__":
    a = sys.argv
    main(a[1] if len(a) > 1 else os.path.join(RESULTS, "sweep_attackers.csv"),
         a[2] if len(a) > 2 else TOPOLOGY_FEATURES,
         a[3] if len(a) > 3 else os.path.join(RESULTS, "q2"))
