"""Regenerate configs with the expanded (24-package) catalog, in place.

    python src/config_gen.py

Reads data/configs/ and rewrites it. Everything about a config except WHICH
software a vulnerable node runs is kept unchanged from the teammate's original
configs (configs_real, deleted 2026-10-13 because every field taken from them
is preserved here): the set of vulnerable nodes, the
single assume-breach entry point, and is_privileged. Only the software is
redrawn, uniformly from all 24 catalog packages (12 famous-vulnerable, 12
well-patched), so a config now varies in severity as well as placement.

Rule 5 (configuration never depends on topology): the draw is seeded on
(n, density, seed) only, n being the node count in the original config, and
ignores every structural property of the graph.

Each node also gets its resolved p (scoring.py) so the mapping is auditable
without reading simulator source. n_cves / max_cvss / mean_cvss are over
attack CVEs (NETWORK and foothold, scoring.py), so they can be 0 for a
package NVD lists no foothold CVE against.
"""
import json, os, sys
import numpy as np
import pandas as pd
from scoring import build_table, attack_cves
from paths import CATALOG, EPSS, CONFIGS, CONFIGS_MANIFEST, rel
PATCHED = "Fully patched (no known CVEs)"


def node_fields(sw, catalog, table):
    if sw == PATCHED:
        return {"software": sw, "n_cves": 0, "max_cvss": 0.0, "mean_cvss": 0.0,
                "p": table[sw]}
    net = attack_cves(catalog[sw])
    cv = [v["cvss"] for v in net]
    return {"software": sw, "n_cves": len(net),
            "max_cvss": max(cv) if cv else 0.0,
            "mean_cvss": round(float(np.mean(cv)), 4) if cv else 0.0,
            "p": table[sw]}


def regenerate(old, catalog, table, packages):
    n = len(old["nodes"])
    d = old["vulnerability_density"]
    rng = np.random.default_rng([n, int(round(d * 100)), old["seed"]])
    try:
        ids = sorted(old["nodes"], key=int)
    except ValueError:
        ids = sorted(old["nodes"])
    nodes = {}
    for v in ids:
        o = old["nodes"][v]
        sw = PATCHED if o["software"] == PATCHED else packages[rng.integers(len(packages))]
        nodes[v] = {**node_fields(sw, catalog, table),
                    "is_entry_point": o["is_entry_point"],
                    "is_privileged": o["is_privileged"]}
    return {**{k: old[k] for k in ("topology", "vulnerability_density",
                                   "num_entry_points", "seed")},
            "adversary": "assume_breach_random_entry",
            "catalog": rel(CATALOG), "epss": rel(EPSS),
            "nodes": nodes}


if __name__ == "__main__":
    src = dst = CONFIGS
    table, epss, catalog = build_table(CATALOG, EPSS)
    packages = sorted(catalog)
    os.makedirs(dst, exist_ok=True)
    rows = []
    for f in sorted(os.listdir(src)):
        if not f.endswith(".json"):
            continue
        old = json.load(open(os.path.join(src, f)))
        new = regenerate(old, catalog, table, packages)
        json.dump(new, open(os.path.join(dst, f), "w"), indent=1)
        ps = [x["p"] for x in new["nodes"].values() if x["software"] != PATCHED]
        rows.append({"file": f, "topology": new["topology"],
                     "density": new["vulnerability_density"], "seed": new["seed"],
                     "machines": len(new["nodes"]), "vulnerable_machines": len(ps),
                     "mean_p_vulnerable": round(float(np.mean(ps)), 4) if ps else None})
    man = pd.DataFrame(rows)
    man.to_csv(CONFIGS_MANIFEST, index=False)
    print(f"wrote {len(man)} configs to {rel(dst)}/ and {rel(CONFIGS_MANIFEST)}")
