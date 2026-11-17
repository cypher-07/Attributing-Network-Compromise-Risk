"""Which structural features matter, and are they the same in every family?

    python src/q2_importance.py [sweep_attackers_all.csv] [features_all.csv]

The proposal promised this alongside the transfer matrix. For each family, train the
gradient-boosting model on that family alone and record (1) its built-in importances and
(2) permutation importance measured on the OTHER families (how much accuracy is lost
there if a feature is scrambled), which is the importance that counts for transfer.
Then compare the rankings across families with Spearman correlation.

Writes results/q2_all/feature_importance.csv and feature_importance_rank_agreement.csv.
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
import q2_predict as Q
from paths import RESULTS

OUT = os.path.join(RESULTS, "q2_all")


def run(scen, feat, repeats=5):
    df, cfg = Q.build_table(scen, feat)
    cols = Q.feature_sets(cfg)["structure_gb"]
    rows = []
    for fam in sorted(df.family.unique()):
        tr, te = df[df.family == fam], df[df.family != fam]
        m = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                      subsample=0.8, random_state=0).fit(tr[cols], tr.y)
        pi = permutation_importance(m, te[cols], te.y, n_repeats=repeats, random_state=0, scoring="r2")
        for c, a, b in zip(cols, m.feature_importances_, pi.importances_mean):
            rows.append({"trained_on": fam, "feature": c, "builtin": a, "permutation_on_other_families": b})
    return pd.DataFrame(rows)


def main(scen=os.path.join(RESULTS, "sweep_attackers_all.csv"),
         feat=os.path.join(RESULTS, "features_all.csv")):
    imp = run(scen, feat)
    os.makedirs(OUT, exist_ok=True)
    imp.to_csv(os.path.join(OUT, "feature_importance.csv"), index=False, float_format="%.4f")
    w = imp.pivot(index="feature", columns="trained_on", values="permutation_on_other_families")
    agree = w.corr(method="spearman")
    agree.to_csv(os.path.join(OUT, "feature_importance_rank_agreement.csv"), float_format="%.3f")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print("permutation importance on the other families (drop in R^2 when scrambled)")
        print(w.assign(mean=w.mean(axis=1)).sort_values("mean", ascending=False).to_string())
        print("\nrank agreement of feature importance between training families (Spearman)")
        print(agree.to_string())
        off = agree.values[np.triu_indices(len(agree), 1)]
        print(f"\naverage agreement {off.mean():.2f}, lowest {off.min():.2f}")
    return imp, agree


if __name__ == "__main__":
    main(*(sys.argv[1:3]))
