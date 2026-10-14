"""Automated checks for the simulation pipeline.

    pip install pytest statsmodels
    python -m pytest tests -q          (run from the Project folder)

Covers: EPSS scoring, the simulator on cases with a known answer, the 400
configuration files, the results table, and the variance-split function.
"""
import glob, json, os, sys
import numpy as np
import pandas as pd
import networkx as nx
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from scoring import build_table, foothold, attack_cves, susceptibility, P_FLOOR, P_CEIL
from simulate import prepare, simulate
from topology import load_topology, topology_names
from paths import CATALOG, EPSS, CONFIGS, SWEEP

PATCHED = "Fully patched (no known CVEs)"


@pytest.fixture(scope="module")
def table():
    return build_table(CATALOG, EPSS)


# ---------- scoring ----------
def test_patched_node_gets_floor(table):
    assert table[0][PATCHED] == P_FLOOR


def test_all_p_inside_floor_and_ceiling(table):
    assert all(P_FLOOR <= p <= P_CEIL for p in table[0].values())


def test_famous_version_scores_higher_than_its_patched_version(table):
    t = table[0]
    assert t["Apache HTTP Server 2.4.49"] > t["Apache HTTP Server 2.4.62"]
    assert t["Drupal 7.57"] > t["Drupal 10.3.6"]


def test_linear_map(table):
    _, epss, cat = table
    best = max(epss[v["cve"]] for v in attack_cves(cat["Apache HTTP Server 2.4.49"]))
    assert table[0]["Apache HTTP Server 2.4.49"] == pytest.approx(P_FLOOR + (P_CEIL - P_FLOOR) * best, abs=1e-3)


def test_heartbleed_is_not_a_foothold(table):
    cat = table[2]
    hb = [v for v in cat["OpenSSL 1.0.1"]["vulnerabilities"] if v["cve"] == "CVE-2014-0160"][0]
    assert not foothold(hb)          # confidentiality only


def test_dos_cve_is_not_a_foothold():
    v = {"cvss_version": "3.1", "conf": "NONE", "integ": "NONE", "avail": "HIGH"}
    assert not foothold(v)
    assert foothold({"cvss_version": "3.1", "conf": "HIGH", "integ": "HIGH", "avail": "HIGH"})


def test_missing_epss_score_raises_not_zero(table):
    _, _, cat = table
    with pytest.raises(KeyError):
        susceptibility(cat["Apache HTTP Server 2.4.49"], {})


# ---------- simulator, known answers ----------
@pytest.fixture(scope="module")
def graph():
    G, nodes, _ = load_topology("Aarnet")
    A, D, idx = prepare(G, nodes)
    return G, nodes, A, D


def test_p_zero_only_entry_node_falls(graph):
    _, nodes, A, D = graph
    f = simulate(A, D, np.zeros(len(nodes)), 0, 200, mode="hop", seed=1)["frac"]
    assert np.allclose(f, 1 / len(nodes))


def test_p_one_whole_connected_graph_falls(graph):
    _, nodes, A, D = graph
    f = simulate(A, D, np.ones(len(nodes)), 0, 50, mode="hop", seed=1)["frac"]
    assert np.allclose(f, 1.0)


def test_same_seed_same_result_and_higher_p_more_spread(graph):
    _, nodes, A, D = graph
    n = len(nodes)
    a = simulate(A, D, np.full(n, 0.2), 0, 500, seed=5)["frac"].mean()
    b = simulate(A, D, np.full(n, 0.2), 0, 500, seed=5)["frac"].mean()
    hi = simulate(A, D, np.full(n, 0.6), 0, 500, seed=5)["frac"].mean()
    assert a == b and hi > a


def test_inconsistent_shared_marginals_rejected(graph):
    _, nodes, A, D = graph
    n = len(nodes)
    with pytest.raises(ValueError):
        simulate(A, D, np.full(n, 0.5), 0, 10, shared=(np.zeros(n, int), np.array([0.9]), 0.05, 0.95))


# ---------- the 400 configuration files ----------
def test_400_configs_consistent_with_graphs_and_scoring(table):
    files = sorted(glob.glob(os.path.join(CONFIGS, "*.json")))
    assert len(files) == 400
    tbl = table[0]
    graphs = {t: set(load_topology(t)[1]) for t in topology_names()}
    for f in files:
        c = json.load(open(f)); n = c["nodes"]
        assert set(n) == graphs[c["topology"]], os.path.basename(f)
        assert sum(v["is_entry_point"] for v in n.values()) == 1
        assert sum(v["is_privileged"] for v in n.values()) >= 1
        for v in n.values():
            assert v["p"] == pytest.approx(tbl[v["software"]], abs=1e-3)


# ---------- results table ----------
def test_sweep_is_balanced_and_valid():
    df = pd.read_csv(SWEEP)
    assert len(df) == 400
    assert df.groupby(["topo_id", "density"]).size().eq(5).all()
    assert df.compromise_fraction.between(0, 1).all()
    assert df.adversary.nunique() == 1


# ---------- variance split ----------
def _synthetic(topology_effect):
    rng = np.random.default_rng(0)
    rows = []
    for t in range(20):
        eff = rng.normal(0, topology_effect)
        for d in (0.2, 0.4, 0.6, 0.8):
            for s in range(5):
                rows.append((f"T{t}", d, s, eff + rng.normal(0, 1)))
    return pd.DataFrame(rows, columns=["topo_id", "density", "config_seed", "y"])


def test_variance_split_recovers_known_shares():
    pytest.importorskip("statsmodels")
    import q1_anova as q
    none = q.components(q.anova(_synthetic(0.0), "y ~ C(topo_id) * C(density)"), 5)
    strong = q.components(q.anova(_synthetic(5.0), "y ~ C(topo_id) * C(density)"), 5)
    assert none["topology"] < 0.15
    assert strong["topology"] > 0.85
    assert none["topology"] + none["configuration"] == pytest.approx(1.0)
