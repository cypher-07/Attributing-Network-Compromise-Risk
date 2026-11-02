"""Where each attacker starts.

Three adversaries, same simulator, same configs; only the entry rule differs.

  assume_breach_random_entry   the midterm attacker. One entry node per config
                               (the config's is_entry_point), chosen at random
                               independently of the graph. Unchanged.
  external_border_entry        an outsider who gets in at the edge of the
                               network. Each trial starts at a border node
                               picked at random.
  insider_privileged_entry     an insider holding a privileged account. Each
                               trial starts at a privileged node (the config's
                               is_privileged flag). Configs have one privileged
                               node, two on Colt, so this is almost always a
                               fixed start.

Border node. A node with a single link (a dead end: nothing connects through
it). If a network has none, the nodes with the fewest links. This is the plain
reading of "the edge of the network": machines you reach last, not first.
Border nodes are chosen from the graph alone and never look at the software,
so the configuration-independence rule holds.

Entry is drawn per trial, not once per config, so a scenario's risk averages
over every border node. That removes the luck of one entry draw. Compromise is
measured the same way as before: share of nodes compromised, entry included.
"""
import numpy as np

ADVERSARIES = ("assume_breach_random_entry", "external_border_entry",
               "insider_privileged_entry")


def border_nodes(G, nodes):
    """Node ids with degree 1; if none, those with the minimum degree."""
    deg = dict(G.degree())
    leaves = [v for v in nodes if deg[v] == 1]
    if leaves:
        return leaves
    m = min(deg[v] for v in nodes)
    return [v for v in nodes if deg[v] == m]


def entry_set(adversary, G, nodes, cfg_nodes):
    """List of node ids the attacker may start at, for one config."""
    if adversary == "assume_breach_random_entry":
        s = [v for v in nodes if cfg_nodes[v]["is_entry_point"]]
    elif adversary == "external_border_entry":
        s = border_nodes(G, nodes)
    elif adversary == "insider_privileged_entry":
        s = [v for v in nodes if cfg_nodes[v]["is_privileged"]]
    else:
        raise ValueError(adversary)
    if not s:
        raise ValueError(f"{adversary}: empty entry set")
    return s


def entries_per_trial(adversary, entry_ids, idx, n_trials, seed):
    """Entry index for every trial. One entry -> scalar semantics via an array;
    several -> uniform draw per trial, seeded so a row reproduces alone."""
    ents = np.array([idx[v] for v in entry_ids])
    if len(ents) == 1:
        return np.full(n_trials, ents[0])
    rng = np.random.default_rng(seed)
    return ents[rng.integers(len(ents), size=n_trials)]
