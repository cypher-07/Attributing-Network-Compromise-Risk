"""Checks for the attacker variants, the three-attacker variance split and Q2.

    pip install pytest scikit-learn scipy
    python -m pytest tests -q          (run from the Project folder)
"""
import os, shutil, sys, tempfile
import numpy as np
import pandas as pd
import networkx as nx
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import attackers
import q1_adversary
import q2_predict
import sweep_attackers
from paths import NETWORKS, SWEEP, RESULTS


# ---------- who starts where ----------
def test_border_nodes_are_the_dead_ends():
    G = nx.star_graph(4)                       # centre 0, four leaves
    assert sorted(attackers.border_nodes(G, list(G))) == [1, 2, 3, 4]


def test_border_falls_back_to_fewest_links_when_no_dead_end():
    G = nx.cycle_graph(6)                      # every node has 2 links
    assert sorted(attackers.border_nodes(G, list(G))) == list(range(6))
    H = nx.complete_graph(4)
    H.add_edge(3, 4); H.add_edge(4, 5); H.add_edge(5, 3)   # no leaf; nodes 4 and 5 have 2 links
    assert set(attackers.border_nodes(H, list(H))) == {4, 5}   # 2 is the minimum


def test_insider_starts_at_the_best_connected_machine_without_any_flag():
    G = nx.star_graph(5)                       # centre 0 has 5 links
    cfg = {v: {"is_entry_point": v == 1} for v in G}      # no is_privileged key at all
    assert attackers.entry_set("insider_privileged_entry", G, list(G), cfg) == [0]
    assert attackers.entry_set("assume_breach_random_entry", G, list(G), cfg) == [1]


def test_hub_ties_are_broken_by_name_and_one_percent_rule_applies():
    G = nx.cycle_graph(6)                      # all tied: the first name wins
    assert attackers.hub_nodes(G, list(G)) == [0]
    big = nx.barabasi_albert_graph(200, 2, seed=1)   # 1% of 200 = 2 hubs
    assert len(attackers.hub_nodes(big, list(big))) == 2


def test_hubs_equal_the_configs_privileged_flag_on_every_real_config():
    """The change from flag to graph rule must not move a single real scenario."""
    import json
    from paths import CONFIGS
    from topology import load_topology, topology_names
    for t in topology_names():
        G, nodes, _ = load_topology(t)
        hubs = set(attackers.hub_nodes(G, nodes))
        for f in os.listdir(CONFIGS):
            if f.startswith(t + "_d"):
                cn = json.load(open(os.path.join(CONFIGS, f)))["nodes"]
                assert {v for v in nodes if cn[v]["is_privileged"]} == hubs


def test_entries_stay_inside_the_set_and_are_reproducible():
    idx = {v: i for i, v in enumerate("abcdef")}
    e1 = attackers.entries_per_trial("x", ["a", "c", "e"], idx, 500, seed=5)
    e2 = attackers.entries_per_trial("x", ["a", "c", "e"], idx, 500, seed=5)
    assert set(e1) <= {0, 2, 4} and set(e1) == {0, 2, 4}
    assert (e1 == e2).all()
    assert (attackers.entries_per_trial("x", ["b"], idx, 10, seed=1) == 1).all()


# ---------- the model did not change ----------
def test_assume_breach_rows_reproduce_the_midterm_sweep():
    """Same seeds, same model: the assume-breach rows of the new sweep must equal
    results/sweep.csv for the same scenarios."""
    old = pd.read_csv(SWEEP)
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "n"))
        shutil.copy(os.path.join(NETWORKS, "Aarnet.graphml"), os.path.join(d, "n"))
        df = sweep_attackers.run(2000, os.path.join(d, "n"), densities=(0.2,), seeds=(1, 2),
                                 out=os.path.join(d, "o.csv"), verbose=False)
    new = df[df.adversary == "assume_breach_random_entry"]
    m = new.merge(old, on=["topo_id", "density", "config_seed"], suffixes=("", "_old"))
    assert len(m) == 2
    # the topology index differs when only one network is present, so the seed
    # differs: compare to Monte Carlo noise instead of exactly
    assert np.allclose(m.compromise_fraction, m.compromise_fraction_old, atol=0.01)


def test_each_adversary_present_and_balanced():
    p = os.path.join(RESULTS, "sweep_attackers.csv")
    df = pd.read_csv(p)
    assert set(df.adversary) == set(attackers.ADVERSARIES)
    assert df.groupby("adversary").size().nunique() == 1
    assert df.compromise_fraction.between(0, 1).all()


# ---------- three-attacker variance split ----------
def test_variance_code_reproduces_the_midterm_split():
    df = pd.read_csv(os.path.join(RESULTS, "sweep_attackers.csv"))
    Y, _, _ = q1_adversary.cube(df, [q1_adversary.BASE])
    c = q1_adversary.components(Y)
    assert c["topology"] == pytest.approx(0.258, abs=0.002)
    assert c["configuration"] == pytest.approx(0.742, abs=0.002)


def test_variance_shares_sum_to_one():
    df = pd.read_csv(os.path.join(RESULTS, "sweep_attackers.csv"))
    Y, _, _ = q1_adversary.cube(df, list(attackers.ADVERSARIES))
    c = q1_adversary.components(Y)
    assert sum(c.values()) == pytest.approx(1.0)
    assert all(v >= 0 for v in c.values())


def test_variance_code_recovers_planted_effects():
    """Planted: network effect sd 1, config sd 0.5, attacker sd 0.5, noise tiny."""
    rng = np.random.default_rng(0)
    D, t, k, a = 4, 400, 5, 3
    T = rng.normal(0, 1, (1, t, 1, 1))
    C = rng.normal(0, 0.5, (D, t, k, 1))
    A = rng.normal(0, 0.5, (1, 1, 1, a))
    Y = T + C + A + rng.normal(0, 0.05, (D, t, k, a))
    c = q1_adversary.components(Y)
    tot = 1 + 0.25 + 0.25 + 0.0025
    assert c["topology"] == pytest.approx(1 / tot, abs=0.05)
    assert c["configuration"] == pytest.approx(0.25 / tot, abs=0.05)
    assert c["attacker"] == pytest.approx(0.25 / tot, abs=0.1)   # only 3 attackers


# ---------- Q2: no leakage ----------
def _toy(n_fam=3, nets=4, rows=8, seed=0):
    rng = np.random.default_rng(seed)
    recs, feats = [], []
    for f in range(n_fam):
        for n in range(nets):
            name = f"fam{f}_net{n}"
            feats.append({"topology": name, "pc": rng.uniform(0.1, 0.5),
                          **{c: rng.normal() for c in q2_predict.FEATURES}})
            for r in range(rows):
                recs.append({"topo_id": name, "family": f"fam{f}", "density": 0.2 * (1 + r % 4),
                             "config_seed": r, "adversary": "assume_breach_random_entry",
                             "n_nodes": 40, "mean_p": rng.uniform(0.1, 0.6),
                             "compromise_fraction": rng.uniform(0.05, 0.9)})
    return pd.DataFrame(recs), pd.DataFrame(feats)


def test_q2_never_trains_on_the_network_it_is_tested_on(monkeypatch, tmp_path):
    s, f = _toy()
    s.to_csv(tmp_path / "s.csv", index=False); f.to_csv(tmp_path / "f.csv", index=False)
    df, cfg = q2_predict.build_table(tmp_path / "s.csv", tmp_path / "f.csv")
    seen = []
    orig = q2_predict.fit_predict
    def spy(model, cols, train, test):
        seen.append((set(train.topo_id), set(test.topo_id), set(train.family), set(test.family)))
        return orig(model, cols, train, test)
    monkeypatch.setattr(q2_predict, "fit_predict", spy)
    q2_predict.evaluate(df, cfg)
    assert seen
    for tr_nets, te_nets, tr_fam, te_fam in seen:
        assert not (tr_nets & te_nets)                       # no network in both
        if tr_fam == te_fam:                                 # diagonal: one held-out network
            assert len(te_nets) == 1
        else:                                                # off-diagonal: disjoint families
            assert not (tr_fam & te_fam)


def test_q2_matrix_has_every_family_pair(tmp_path):
    s, f = _toy(n_fam=4)
    s.to_csv(tmp_path / "s.csv", index=False); f.to_csv(tmp_path / "f.csv", index=False)
    df, cfg = q2_predict.build_table(tmp_path / "s.csv", tmp_path / "f.csv")
    res = q2_predict.evaluate(df, cfg)
    for mdl in q2_predict.MODELS:
        assert len(res[res.model == mdl]) == 16
    assert res.r2.notna().all()


def test_q2_a_predictor_with_signal_beats_the_mean(tmp_path):
    """Plant risk = f(pc, mean_p); structure_gb must beat the do-nothing model."""
    s, f = _toy(n_fam=3, nets=6, rows=40, seed=3)
    pc = f.set_index("topology").pc
    mp = s.mean_p.to_numpy()
    s["compromise_fraction"] = 1 / (1 + np.exp(-(-1 - 4 * pc[s.topo_id].to_numpy() + 3 * mp)))
    s.to_csv(tmp_path / "s.csv", index=False); f.to_csv(tmp_path / "f.csv", index=False)
    df, cfg = q2_predict.build_table(tmp_path / "s.csv", tmp_path / "f.csv")
    res = q2_predict.evaluate(df, cfg)
    off = res[res.train != res.test].groupby("model").r2.mean()
    assert off["structure_gb"] > off["mean"] + 0.3
