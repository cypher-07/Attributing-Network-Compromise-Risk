"""Put the real networks and the synthetic families in one table and run Q2.

    python src/combine_families.py

Needs (all produced by earlier steps):
  results/sweep_attackers.csv               real networks      (sweep_attackers.py)
  results/sweep_attackers_synthetic.csv     synthetic families (sweep_attackers.py ... auto)
  data/topology_features.csv                real features      (topology.py)
  data/synthetic_features.csv               synthetic features (synthetic.py or make_features.py)

Writes results/sweep_attackers_all.csv, results/features_all.csv and runs
q2_predict.py on them -> results/q2_all/ (transfer matrix over every family).
"""
import os
import pandas as pd
from paths import DATA, RESULTS, TOPOLOGY_FEATURES
import q2_predict


def combine(real_sweep, syn_sweep, real_feat, syn_feat):
    r, s = pd.read_csv(real_sweep), pd.read_csv(syn_sweep)
    if list(r.columns) != list(s.columns):
        raise SystemExit(f"sweep columns differ: {sorted(set(r.columns) ^ set(s.columns))}")
    clash = set(r.topo_id) & set(s.topo_id)
    if clash:
        raise SystemExit(f"network names in both real and synthetic: {sorted(clash)}")
    fr, fs = pd.read_csv(real_feat), pd.read_csv(syn_feat)
    cols = [c for c in fr.columns if c in fs.columns]
    return pd.concat([r, s], ignore_index=True), pd.concat([fr[cols], fs[cols]], ignore_index=True)


def main():
    sweep, feat = combine(os.path.join(RESULTS, "sweep_attackers.csv"),
                          os.path.join(RESULTS, "sweep_attackers_synthetic.csv"),
                          TOPOLOGY_FEATURES, os.path.join(DATA, "synthetic_features.csv"))
    sp, fp = os.path.join(RESULTS, "sweep_attackers_all.csv"), os.path.join(RESULTS, "features_all.csv")
    sweep.to_csv(sp, index=False); feat.to_csv(fp, index=False, float_format="%.6f")
    return q2_predict.main(sp, fp, os.path.join(RESULTS, "q2_all"))


if __name__ == "__main__":
    main()
