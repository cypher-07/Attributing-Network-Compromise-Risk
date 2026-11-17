"""Checks for the per-family variance split and the Q2 sensitivity run.

    python -m pytest tests -q          (run from the project root)
"""
import os, sys
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import q1_families
import q2_uncertainty
from paths import RESULTS


def test_family_name_rule():
    assert q2_uncertainty.family_of("scale_free_7") == "scale_free"
    assert q2_uncertainty.family_of("fat_tree_10") == "fat_tree"
    assert q2_uncertainty.family_of("Aarnet") == "real"
    assert q2_uncertainty.family_of("HurricaneElectric") == "real"


def _toy(topology_sd, config_sd, nets=8, seed=0):
    """A sweep table where network effect and config effect have the given sd (logit)."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(nets):
        te = rng.normal(0, topology_sd)
        for d in (0.2, 0.4, 0.6, 0.8):
            for s in range(1, 6):
                y = -0.5 + te + rng.normal(0, config_sd) + 0.2 * d
                rows.append({"topo_id": f"n{t}", "family": "toy", "density": d, "config_seed": s,
                             "adversary": "assume_breach_random_entry", "n_nodes": 100,
                             "compromise_fraction": 1 / (1 + np.exp(-y))})
    return pd.DataFrame(rows)


def test_split_recovers_which_source_dominates():
    tp, cf = 1.0, 0.3
    c, lo, hi, n = q1_families.split(_toy(tp, cf, nets=40),
                                     ["assume_breach_random_entry"], B=50)
    assert n == 40 and c["topology"] > 0.8 and c["configuration"] < 0.2
    c2 = q1_families.split(_toy(0.2, 1.0, nets=40), ["assume_breach_random_entry"], B=50)[0]
    assert c2["configuration"] > 0.8 and c2["topology"] < 0.2
    assert sum(c.values()) == pytest.approx(1.0)


def test_identical_networks_give_zero_topology_share():
    c = q1_families.split(_toy(0.0, 0.5, nets=12), ["assume_breach_random_entry"], B=20)[0]
    assert c["topology"] < 0.05


def test_the_real_family_reproduces_the_midterm_split():
    p = os.path.join(RESULTS, "sweep_attackers_all.csv")
    if not os.path.exists(p):
        pytest.skip("combined sweep not present")
    df = pd.read_csv(p)
    c = q1_families.split(df[df.family == "real"], ["assume_breach_random_entry"], B=10)[0]
    assert c["topology"] == pytest.approx(0.258, abs=0.002)
