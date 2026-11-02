"""Checks for the uncertainty analysis, the reproduction number and the patching test.

    python -m pytest tests -q          (run from the Project folder)
"""
import os, sys
import numpy as np
import pandas as pd
import networkx as nx
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import uncertainty, reproduction, patching
import sensitivity
from paths import RESULTS
from sweep import PATCHED


# ---------- uncertainty ----------
def test_uncertainty_sweep_matches_the_midterm_sensitivity_code():
    """Same seeds, same model: our faster loop must give the numbers sensitivity.run gives."""
    sensitivity.TRIALS = 200
    ref = sensitivity.run(floor=0.05, decay=0.5)
    ours = uncertainty.summarise(uncertainty.sweep(uncertainty.load_world(), 0.05, 0.5, 200))
    sensitivity.S.DECAY = 0.5
    assert ours["topology_share"] == pytest.approx(ref["topology_share"], abs=1e-9)
    assert ours["mean_compromised"] == pytest.approx(ref["mean_compromised"], abs=1e-9)


def test_uncertainty_summary_counts_the_right_fractions():
    df = pd.DataFrame({"floor": [.02, .04, .06, .08], "decay": [.3, .4, .5, .6],
                       "topology_share": [.1, .3, .6, .9], "mean_compromised": [.2, .3, .4, .5],
                       "rho_percolation": [-.9, -.8, -.7, -.6], "r_config": [.5, .6, .7, .8]})
    s = uncertainty.describe(df).set_index("quantity").value
    assert s["fraction of draws where configuration explains more than topology"] == 0.5
    assert s["fraction of draws where topology share is under one third"] == 0.5
    assert s["rank correlation, topology share vs floor"] == pytest.approx(1.0)


# ---------- reproduction number ----------
def test_reproduction_number_on_a_path_by_hand():
    G = nx.path_graph(3)                            # 0 - 1 - 2
    A, D, idx = reproduction.prepare(G, [0, 1, 2])
    p = np.array([0.2, 0.4, 0.6])
    rm, re = reproduction.reproduction(D, p, [0], decay=0.5)
    # row sums: from 0: 0.4 + 0.5*0.6 = 0.7 ; from 1: 0.2 + 0.6 = 0.8 ; from 2: 0.5*0.2 + 0.4 = 0.5
    assert rm == pytest.approx((0.7 + 0.8 + 0.5) / 3)
    assert re == pytest.approx(0.7)


def test_reproduction_is_zero_without_vulnerability_and_ignores_unreachable():
    G = nx.Graph(); G.add_edges_from([(0, 1)]); G.add_node(2)
    A, D, idx = reproduction.prepare(G, [0, 1, 2])
    assert reproduction.reproduction(D, np.zeros(3), [0], decay=0.5) == (0.0, 0.0)
    rm, re = reproduction.reproduction(D, np.array([.5, .5, .9]), [0], decay=0.5)
    assert re == pytest.approx(0.5)                 # node 2 cannot be reached


def test_reproduction_rises_with_susceptibility_and_falls_with_decay():
    G = nx.cycle_graph(6)
    A, D, idx = reproduction.prepare(G, list(G))
    lo = reproduction.reproduction(D, np.full(6, .2), [0], decay=.5)[0]
    hi = reproduction.reproduction(D, np.full(6, .4), [0], decay=.5)[0]
    steep = reproduction.reproduction(D, np.full(6, .4), [0], decay=.2)[0]
    assert lo < hi and steep < hi


def test_the_reproduction_table_covers_every_scenario():
    R = pd.read_csv(os.path.join(RESULTS, "reproduction", "scenario_R.csv"))
    assert len(R) == 1200 and R.R_mean.gt(0).all() and R.R_entry.gt(0).all()


# ---------- patching ----------
def _cfg():
    sw = ["A", "B", "C", PATCHED, "A", "B"]
    p = np.array([.9, .5, .1, .05, .9, .5])
    deg = np.array([1., 5., 2., 9., 3., 1.])
    q = {"A": .9, "B": .5, "C": .1, PATCHED: 0.}
    return sw, p, deg, q


def test_budget_size():
    assert patching.budget_size(0) == 0
    assert patching.budget_size(1) == 1
    assert patching.budget_size(8) == 2 and patching.budget_size(9) == 3


def test_strategies_only_patch_vulnerable_machines_and_respect_the_budget():
    sw, p, deg, q = _cfg()
    for s in patching.STRATEGIES:
        pick = patching.choose(s, sw, p, deg, q, 3, np.random.default_rng(0))
        assert len(pick) == 3 and len(set(pick)) == 3 and 3 not in pick
        assert patching.choose(s, sw, p, deg, q, 99, np.random.default_rng(0)).size == 5


def test_each_strategy_picks_what_it_says():
    sw, p, deg, q = _cfg()
    rng = lambda: np.random.default_rng(0)
    assert set(patching.choose("exploit_first", sw, p, deg, q, 2, rng())) == {0, 4}   # the two p=0.9
    assert set(patching.choose("hub_first", sw, p, deg, q, 2, rng())) == {1, 4}       # degrees 5 and 3
    assert set(patching.choose("package_first", sw, p, deg, q, 2, rng())) == {0, 4}   # all of package A
    assert patching.choose("combined", sw, p, deg, q, 1, rng())[0] == 4               # .9*3 = 2.7 is the largest


def test_package_first_cuts_the_last_package_at_the_budget():
    sw, p, deg, q = _cfg()
    pick = patching.choose("package_first", sw, p, deg, q, 3, np.random.default_rng(0))
    assert {0, 4} <= set(pick) and len(pick) == 3 and sw[pick[2]] == "B"


def test_apply_patch_does_not_modify_the_original():
    sw, *_ = _cfg()
    out = patching.apply_patch(sw, [0, 1])
    assert out[:2] == [PATCHED, PATCHED] and sw[0] == "A"


def test_unpatched_rows_equal_the_attacker_sweep_and_patching_never_hurts_on_average():
    """'none' uses the same seeds as sweep_attackers.csv, so it must equal it exactly."""
    df = pd.read_csv(os.path.join(RESULTS, "patching", "patching.csv"))
    ref = pd.read_csv(os.path.join(RESULTS, "sweep_attackers.csv"))
    none = df[df.strategy == "none"].rename(columns={"risk": "risk_none"})
    m = none.merge(ref, left_on=["topo_id", "density", "config_seed", "adversary"],
                   right_on=["topo_id", "density", "config_seed", "adversary"])
    assert len(m) == 1200
    assert np.abs(m.risk_none - m.compromise_fraction).max() < 1e-5
    d = patching.analyse(df)
    assert (d.groupby("strategy").reduction.mean() > 0).all()
    assert d.groupby("strategy").size().nunique() == 1
