"""Independent-cascade compromise simulator.

Spread rule
-----------
Each node j has an intrinsic susceptibility p_j from scoring.py. When a node is
newly compromised it makes ONE attempt against each uncompromised neighbour.
So a node with k newly-compromised neighbours in the same round falls with

    P(fall) = 1 - (1 - p_j) ** k

which is exactly the "more compromised neighbours -> higher risk" behaviour,
derived rather than bolted on. Each EDGE fires at most once, so risk does not
creep to 1.0 just because the simulation ran more rounds -- that one detail is
what separates independent cascade from a naive retry loop.

Correlated exploits (shared=..., PRIMARY since 2026-10-12)
---------------------------------------------------------
Real attackers either have a working exploit for a package or they do not,
and if they do it works on every node running that package -- the way worms
take out every unpatched box at once. Independent per-attempt draws miss this.
So at the start of each trial, for each package s, draw once

    armed_s ~ Bernoulli(q_s),   q_s = max EPSS over s's attack CVEs

and an attempt on a node running s succeeds at P_CEIL if armed_s, else at
P_FLOOR (credential reuse, still independent per attempt). The marginal
per-attempt success is q*P_CEIL + (1-q)*P_FLOOR = p_j exactly, so the two
modes differ ONLY in correlation: no new constant is introduced. This also
reads EPSS more naturally -- "is this exploited in the wild" becomes "does
this attacker have an exploit", not a per-attempt rate. shared=None is the
old model (one independent draw per attempt at p_j), kept for comparison.

Reachability modes
------------------
  decay  any routable node, success scaled by DECAY ** (hops - 1). Models a
         partly segmented network. PRIMARY MODEL (decided 2026-10-12; hop was
         dropped from the analysis).
  hop    neighbours only. Models a segmented network where every link is a
         trust boundary. Kept for check.py; not used in the analysis.
  flat   any routable node at full p. Models no segmentation at all.

In decay and flat, a newly compromised node makes one attempt on every
uncompromised reachable node, so each (attacker, target) pair fires at most
once -- the same rule as hop, over reachability instead of adjacency. These
modes are vectorised in log space: a node survives a round with probability
prod(1 - per_uv) over this round's attackers u, i.e. exp(newly @ log(1 - per)).
"""
import numpy as np
import networkx as nx

DECAY = 0.5
MAX_ROUNDS = 60


def prepare(G, nodelist):
    A = nx.to_numpy_array(G, nodelist=nodelist).astype(np.int8)
    n = len(nodelist)
    idx = {v: i for i, v in enumerate(nodelist)}
    D = np.full((n, n), -1, dtype=np.int16)
    for u, lengths in nx.all_pairs_shortest_path_length(G):
        for v, d in lengths.items():
            D[idx[u], idx[v]] = d
    return A, D, idx


def simulate(A, D, p, entry_idx, n_trials=500, mode="decay", seed=0,
             track_rounds=False, shared=None):
    """p: (n,) intrinsic susceptibility. entry_idx: int or (n_trials,) array.
    shared: None for independent attempts, or (pkg, q, lo, hi) for correlated
    exploits -- pkg (n,) package index per node (-1 = no attack CVE, always
    lo), q (n_pkg,) chance the attacker holds an exploit for each package, lo
    / hi the per-attempt success without / with one. p is then ignored except
    as a check that it equals the marginal lo + (hi - lo) * q[pkg].
    Returns dict with compromise fraction per trial."""
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    if shared is not None:
        pkg, q, lo, hi = shared
        pkg, q = np.asarray(pkg), np.asarray(q, dtype=float)
        marg = np.where(pkg >= 0, lo + (hi - lo) * q[np.maximum(pkg, 0)], lo)
        if not np.allclose(marg, p, atol=1e-3):
            raise ValueError("shared-exploit marginals disagree with p")
        has = rng.random((n_trials, len(q))) < q[None, :]          # per trial, per package
        armed = np.where(pkg[None, :] >= 0, has[:, np.maximum(pkg, 0)], False)
        p_lo, p_hi = np.full(n, lo), np.full(n, hi)
    else:
        armed, p_lo, p_hi = None, p, p
    owned = np.zeros((n_trials, n), dtype=bool)
    entry = np.full(n_trials, entry_idx) if np.isscalar(entry_idx) else np.asarray(entry_idx)
    owned[np.arange(n_trials), entry] = True
    newly = owned.copy()
    rounds = np.zeros(n_trials, dtype=np.int16)

    if mode in ("decay", "flat"):
        # per[u, v]: success of one attempt by u on v; log survival summed over
        # this round's attackers by a matmul. Correlated mode needs two tables
        # (armed / unarmed target), chosen per trial and node after the matmul.
        reach = D > 0
        scale = DECAY ** np.where(reach, D - 1, 0) if mode == "decay" else 1.0
        log_lo = np.log1p(-np.where(reach, p_lo[None, :] * scale, 0.0))
        log_hi = log_lo if armed is None else np.log1p(-np.where(reach, p_hi[None, :] * scale, 0.0))
    elif mode != "hop":
        raise ValueError(mode)

    for r in range(MAX_ROUNDS):
        if not newly.any():
            break
        if mode == "hop":
            # k[t, j] = number of newly-compromised neighbours of j in trial t
            k = newly @ A
            pfall = 1.0 - np.power(1.0 - p_lo[None, :], k)
            if armed is not None:
                pfall = np.where(armed, 1.0 - np.power(1.0 - p_hi[None, :], k), pfall)
        else:
            nf = newly.astype(np.float64)
            ls = nf @ log_lo
            if armed is not None:
                ls = np.where(armed, nf @ log_hi, ls)
            pfall = 1.0 - np.exp(ls)
        cand = ~owned
        hit = cand & (rng.random((n_trials, n)) < pfall)
        if not hit.any():
            break
        rounds[hit.any(axis=1)] = r + 1
        owned |= hit
        newly = hit

    res = {"frac": owned.sum(axis=1) / n}
    if track_rounds:
        res["rounds"] = rounds
    return res


def percolation_threshold(G):
    """Molloy-Reed site-percolation threshold from the degree distribution."""
    d = np.array([deg for _, deg in G.degree()], dtype=float)
    k1, k2 = d.mean(), (d ** 2).mean()
    return k1 / (k2 - k1) if k2 > k1 else float("inf")
