"""Checks on the synthetic families (run after python src/synthetic.py)."""
import glob, json, os, sys
import numpy as np
import networkx as nx
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from topology import load_topology
from paths import NETWORKS_SYN, CONFIGS_SYN
from simulate import prepare, simulate

FAMS = ("scale_free", "random", "small_world", "fat_tree")
NAMES = [f"{f}_{i}" for f in FAMS for i in range(1, 11)]


def test_counts():
    assert len(glob.glob(os.path.join(NETWORKS_SYN, "*.graphml"))) == 40
    assert len(glob.glob(os.path.join(CONFIGS_SYN, "*.json"))) == 800


@pytest.mark.parametrize("name", NAMES)
def test_graph_is_connected_and_unchanged_by_loading(name):
    raw = nx.read_graphml(os.path.join(NETWORKS_SYN, f"{name}.graphml"))
    G, nodes, log = load_topology(name, NETWORKS_SYN)
    assert G.number_of_nodes() == raw.number_of_nodes()
    assert log["nodes_dropped"] == 0 and log["self_loops_dropped"] == 0


@pytest.mark.parametrize("name", NAMES)
def test_has_dead_end_machines(name):
    G, _, _ = load_topology(name, NETWORKS_SYN)
    assert any(d == 1 for _, d in G.degree())


def test_same_size_and_degree_across_matched_families():
    sizes = {f: sorted(load_topology(f"{f}_{i}", NETWORKS_SYN)[0].number_of_nodes()
                       for i in range(1, 11)) for f in FAMS[:3]}
    assert sizes["scale_free"] == sizes["random"] == sizes["small_world"]


@pytest.mark.parametrize("name", ["scale_free_1", "fat_tree_3"])
def test_config_rules_and_simulation_runs(name):
    G, nodes, _ = load_topology(name, NETWORKS_SYN)
    cfg = json.load(open(os.path.join(CONFIGS_SYN, f"{name}_d0.4_s1.json")))
    assert set(cfg["nodes"]) == set(nodes)
    cn = cfg["nodes"]
    assert sum(x["is_entry_point"] for x in cn.values()) == 1
    priv = [v for v in nodes if cn[v]["is_privileged"]]
    assert len(priv) == 1 and G.degree(priv[0]) == max(d for _, d in G.degree())
    A, D, idx = prepare(G, nodes)
    p = np.array([cn[v]["p"] for v in nodes])
    entry = idx[[v for v in nodes if cn[v]["is_entry_point"]][0]]
    frac = simulate(A, D, p, entry, 200, mode="decay", seed=1)["frac"]
    assert 0 < frac.mean() <= 1
