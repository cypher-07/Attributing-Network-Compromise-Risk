"""Every file location in one place. Paths are absolute from the project root,
so scripts run from any directory: python src/sweep.py, cd src && python
sweep.py, ... all read and write the same files."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

NETWORKS = os.path.join(DATA, "networks")                  # Topology Zoo .graphml
CONFIGS = os.path.join(DATA, "configs")                    # one JSON per scenario
CONFIGS_MANIFEST = os.path.join(DATA, "configs_manifest.csv")
TOPOLOGY_FEATURES = os.path.join(DATA, "topology_features.csv")
INGESTION_LOG = os.path.join(DATA, "ingestion_log.csv")

CATALOG = os.path.join(DATA, "catalog", "vulnerability_catalog.json")
EPSS = os.path.join(DATA, "catalog", "epss_scores.json")

RESULTS = os.path.join(ROOT, "results")
SWEEP = os.path.join(RESULTS, "sweep.csv")
Q1 = os.path.join(RESULTS, "q1_variance_split.csv")
NETWORK_RISK = os.path.join(RESULTS, "network_risk.csv")
SUMMARY = os.path.join(RESULTS, "SUMMARY.md")
SENSITIVITY = os.path.join(RESULTS, "sensitivity")
FIGURES = os.path.join(ROOT, "figures")
VIZ = os.path.join(ROOT, "viz")


def rel(path):
    """Project-relative path with forward slashes, for printing and for
    references stored inside data files."""
    return os.path.relpath(path, ROOT).replace(os.sep, "/")
