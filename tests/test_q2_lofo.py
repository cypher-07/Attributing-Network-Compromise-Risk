"""Checks on the Q2 bootstrap intervals (results/q2_all/q2_transfer_ci.csv, q2_lofo.csv).

    python -m pytest tests -q          (run from the project root)
"""
import os
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "results", "q2_all")


def _load(name):
    p = os.path.join(D, name)
    if not os.path.exists(p):
        pytest.skip(name + " not present")
    return pd.read_csv(p)


@pytest.mark.parametrize("name", ["q2_transfer_ci.csv", "q2_lofo.csv"])
def test_interval_brackets_estimate(name):
    d = _load(name).dropna(subset=["r2", "rank"])
    assert (d.r2_lo <= d.r2 + 1e-9).all() and (d.r2 <= d.r2_hi + 1e-9).all()
    assert (d.rank_lo <= d["rank"] + 1e-9).all() and (d["rank"] <= d.rank_hi + 1e-9).all()


def test_lofo_covers_every_family_and_model():
    d = _load("q2_lofo.csv")
    assert set(d.model) == {"config_only", "pc_linear", "structure_gb"}
    assert d.groupby("model").held_out_family.nunique().eq(5).all()
    assert d.n_networks.min() >= 20


def test_structure_beats_software_only_in_lofo():
    d = _load("q2_lofo.csv").pivot(index="held_out_family", columns="model", values="r2")
    assert (d.structure_gb > d.config_only).all()
