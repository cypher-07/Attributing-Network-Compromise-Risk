"""Structural features for any folder of .graphml networks (real or synthetic).

    python src/make_features.py <networks_dir> <out_csv>

Same code as topology.py (largest connected component, parallel links
collapsed, ten size-normalised features, percolation threshold pc), so a
synthetic family is described by exactly the features the real networks are.
Refuses to write a non-finite feature (e.g. a disconnected or one-node graph).
"""
import sys
import numpy as np
import pandas as pd
from topology import load_topology, features, topology_names
from simulate import percolation_threshold


def build(net_dir):
    rows = []
    for name in topology_names(net_dir):
        G, nodes, _ = load_topology(name, net_dir)
        f = features(G)
        bad = [k for k, v in f.items() if not np.isfinite(v)]
        if bad:
            sys.exit(f"{name}: non-finite features {bad}")
        rows.append({"topology": name, "n_nodes": G.number_of_nodes(),
                     "n_edges": G.number_of_edges(),
                     "pc": percolation_threshold(G), **f})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build(sys.argv[1])
    df.to_csv(sys.argv[2], index=False, float_format="%.6f")
    print(f"wrote {sys.argv[2]}: {len(df)} networks")
