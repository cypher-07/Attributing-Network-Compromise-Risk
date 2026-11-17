# Does "configuration matters more than topology" hold in every family?

`python src/q1_families.py` (needs `results/sweep_attackers_all.csv`). Same variance split as
`q1_adversary.py`, run inside each family and on all 140 networks pooled. Share of the variation
in compromise (logit scale, within density), 95% interval over networks.

Midterm attacker only (the real row reproduces the midterm 25.8% / 74.2%):

| family (networks) | topology | configuration |
|---|---|---|
| real ISP backbones (20) | 25.8% (7-40) | 74.2% (60-93) |
| random (30) | 12.1% (4-19) | 87.9% (81-96) |
| small-world (30) | 4.7% (0-11) | 95.3% (89-100) |
| fat-tree (30) | 48.3% (32-58) | 51.7% (42-68) |
| scale-free (30) | 61.7% (48-70) | 38.3% (30-52) |
| all 140 pooled | 43.4% (34-51) | 56.6% (49-66) |

Three attackers, all 140 networks pooled: topology 42.1% (33-50), configuration 45.5% (39-53),
attacker 6.3% (5-8), network x attacker 0.7%, residual 5.4%.

## Reading
- The answer depends on the kind of network. Configuration dominates in real, random and
  small-world networks. Topology dominates in scale-free networks and is about even in fat-tree.
- This is not a contradiction. The split measures how much the networks in a sample differ
  from each other in structure, compared with how much different software setups differ.
  Small-world networks in this set are almost the same shape, so shape explains little.
  Scale-free networks differ a lot (hubs of very different size), so shape explains a lot.
- So "configuration usually matters more" is true for realistic backbones, but it is not a
  law. State it as: how much shape matters depends on how varied the shapes are.
- With 30 networks per family the intervals are about half as wide as with 10 (fat-tree 32-58
  instead of 20-69), and the ordering of the families is the same as before.
- Fat-tree networks 11-30 vary only in host counts, so its 48% comes mostly from the 10 distinct
  sizes.

## Does it survive the two assumed settings? (`src/q2_uncertainty.py`, 24 draws)
Topology share (midterm attacker, 1,000 attacks per scenario) rerun for each (floor, decay) draw:

| family | median | range over draws | draws where configuration beats topology |
|---|---|---|---|
| real | 39% | 7-80% | 83% |
| random | 18% | 3-90% | 71% |
| small-world | 23% | 0-86% | 83% |
| fat-tree | 53% | 12-94% | 50% |
| scale-free | 68% | 5-96% | 38% |
| all pooled | 54% | 25-92% | 50% |

- The ordering of families is fairly stable (scale-free and fat-tree most topology-driven), but the
  size of the topology share is not: it moves from near zero to over 80% in every family.
- A higher floor raises the topology share (rank correlation +0.70 pooled, +0.86 real): when
  patched machines are easy to break, differences in how well connected the network is matter more.
- Quote ranges, not single shares. See also `ROBUSTNESS_AND_DEFENCES.md` for the 128-draw range on
  the real networks.
