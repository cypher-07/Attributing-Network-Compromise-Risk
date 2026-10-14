"""
Version 3 of the configuration generator.
Same as v2, plus: each network gets privileged node(s), chosen as the
highest-degree machine(s). Needed for the insider attack type.
"""
import json
import random
import statistics
import networkx as nx

with open("vulnerability_catalog_live.json") as f:
    CATALOG = json.load(f)

PATCHED = "Fully patched (no known CVEs)"


def summarize(software, network_only=True):
    """Collapse a software's CVE list into a few numbers."""
    vulns = CATALOG[software]["vulnerabilities"]
    if network_only:
        vulns = [v for v in vulns if v["attack_vector"] == "NETWORK"]
    scores = [v["cvss"] for v in vulns]
    if not scores:
        return {"n_cves": 0, "max_cvss": 0.0, "mean_cvss": 0.0}
    return {
        "n_cves": len(scores),
        "max_cvss": max(scores),
        "mean_cvss": round(statistics.mean(scores), 2),
    }


def pick_privileged(G, fraction=0.01):
    """
    Privileged machines = the highest-degree machines (the hubs).
    About 1% of machines, minimum 1 per network.
    Deterministic: same network always gives the same privileged machine.
    """
    simple = nx.Graph(G)                       # collapse duplicate links
    simple.remove_edges_from(nx.selfloop_edges(simple))
    k = max(1, round(fraction * simple.number_of_nodes()))
    ranked = sorted(simple.nodes(), key=lambda n: (-simple.degree(n), str(n)))
    return {str(n) for n in ranked[:k]}


def assign_configuration(G, vulnerability_density=0.5, num_entry_points=1,
                         seed=0, network_only=True, privileged_fraction=0.01):
    rng = random.Random(seed)
    nodes = list(G.nodes())
    software_options = list(CATALOG.keys())

    entry_points = {str(n) for n in rng.sample(nodes, k=min(num_entry_points, len(nodes)))}
    privileged = pick_privileged(G, privileged_fraction)

    config = {}
    for node in nodes:
        if rng.random() < vulnerability_density:
            software = rng.choice(software_options)
            info = summarize(software, network_only)
        else:
            software = PATCHED
            info = {"n_cves": 0, "max_cvss": 0.0, "mean_cvss": 0.0}
        config[str(node)] = {
            "software": software,
            **info,
            "is_entry_point": str(node) in entry_points,
            "is_privileged": str(node) in privileged,
        }
    return config


if __name__ == "__main__":
    for name in ("Sprint", "Colt"):
        G = nx.read_graphml(f"real_networks/{name}.graphml")
        cfg = assign_configuration(G, vulnerability_density=0.5, num_entry_points=1, seed=1)
        priv = [n for n, v in cfg.items() if v["is_privileged"]]
        print(f"=== {name}: {G.number_of_nodes()} machines ===")
        print(f"  privileged machine(s): {priv}")
        for node in priv:
            print(f"  degree of {node}: {nx.Graph(G).degree(node)}")
        print()