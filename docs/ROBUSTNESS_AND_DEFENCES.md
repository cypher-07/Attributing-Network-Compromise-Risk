# Three fixes for the weak points: how sure are we, what is the mechanism, does the advice work

Additive: no existing file changed. All numbers: 20 real networks, 400 configurations.
Code: `src/uncertainty.py`, `src/reproduction.py`, `src/patching.py`; tests in
`tests/test_robustness_defences.py` (12 tests, pass).

## 1. How much does the "75% configuration / 25% topology" split depend on our two assumptions?

The floor (chance of breaking into a fully patched machine, 0.05) and the decay (each
extra step away makes an attack half as likely, 0.5) were never measured. The midterm
tried 3 x 3 combinations. Here: 128 evenly spread combinations over floor 0.02-0.10 and
decay 0.3-0.7, the full 400-scenario sweep for each (1,000 attacks per scenario).
`python src/uncertainty.py 128 1000` (about 7 minutes).

| result | value |
|---|---|
| topology share of variation, across the box | 1.5% to 88%; median 38%; middle 80% of draws: 18% to 69% |
| draws where configuration explains more than topology | 73% |
| draws where topology share is under one third | 33% |
| rank correlation of topology share with the floor | +0.80 |
| rank correlation of topology share with the decay | +0.21 |
| median topology share by density (20/40/60/80% vulnerable) | 53% / 41% / 32% / 20% |
| percolation threshold vs network risk | rho between -0.31 and -0.94 (below -0.8 in 59% of draws) |
| software exploitability vs risk (correlation) | between 0.20 and 0.85 (above 0.5 in 77% of draws) |

Plain reading. The midterm's 25.8% is one point inside a very wide range: the split is
**not** a fixed property of these networks. What survives is a direction, not a number:
configuration explains more than topology in about three quarters of the plausible box,
and the topology share is driven mostly by the floor. The easier a patched machine is to
break, the more the layout matters (floor 0.02-0.047: median 24%; 0.073-0.10: median 52%),
because attacks then pass through patched machines and the wiring decides how far they go.
The number the report should quote is the range with its reason, not 25.8% alone. "A
fraction of draws" is relative to this box; it is not a probability that the real world is
like this.

## 2. The attack reproduction number: a mechanism, and an honest test

R = how many new machines one compromised machine takes over in its own first round of
attempts, built from the same two ingredients as the simulator and with no fitted
parameters (`src/reproduction.py`). Like the epidemic R number: above 1 an attack tends to
grow, below 1 it fizzles.

Share of machines compromised in the simulation, by the network's average R:

| R band | scenarios | mean share compromised |
|---|---|---|
| under 0.5 | 6 | 12% |
| 0.5 to 1 | 81 | 14% |
| 1 to 2 | 312 | 21% |
| 2 to 4 | 525 | 42% |
| over 4 | 276 | 58% |

Predicting a held-out network (leave-one-network-out, real networks):

| model | R^2 | Spearman over scenarios | Spearman across networks |
|---|---|---|---|
| software only (config_only) | 0.80 | 0.90 | -0.31 |
| R only | 0.74 | 0.91 | 0.75 |
| R + software | 0.90 | 0.96 | 0.81 |
| one number, percolation threshold + software | 0.91 | 0.96 | 0.90 |
| gradient boosting on 11 structure features | 0.92 | 0.96 | 0.89 |

**R explains but does not beat the percolation threshold.** It gives a plain mechanism
(R rises with how exploitable the software is and how connected the network is) and
orders scenarios as well as the other models (0.91). It ranks unseen *networks* worse
(0.75 vs 0.90) because R, summed over every reachable machine, grows with network size and
largely repeats the average exploitability. Say this in the report; do not claim R is the
better predictor.

## 3. Does the advice work? Patching, tested

Fix 25% of the vulnerable machines (rounded up) in each configuration, choosing them
five ways, then rerun the identical attacks (same seeds, paired) for all three attackers.
`python src/patching.py 2000` (about 3 minutes). Fall in the share of the network
compromised, in percentage points (95% interval over networks):

| strategy | fall | 95% interval | at 20% / 40% / 60% / 80% vulnerable |
|---|---|---|---|
| combined (easiest to break x best connected) | 16.9 | 16.0-17.7 | 6.7 / 14.7 / 21.4 / 24.8 |
| easiest to break first | 15.7 | 15.1-16.2 | 6.3 / 13.5 / 19.8 / 23.0 |
| whole software package first | 15.7 | 15.1-16.2 | 6.3 / 13.6 / 19.8 / 23.1 |
| best-connected first | 10.1 | 9.5-10.7 | 4.7 / 9.7 / 12.6 / 13.4 |
| random | 8.5 | 8.2-8.9 | 3.7 / 7.6 / 10.9 / 11.9 |

By attacker (percentage points): combined 17.5 (assume-breach) / 18.0 (external) / 15.2
(insider). Best-connected-first helps least against the insider (8.3), who starts at the
biggest hub: patching the hub does not stop someone who is already on it.

Plain reading. Targeted patching roughly doubles the benefit of patching at random, and
choosing by exploitability and connectivity together is best on 18 of 20 networks (the
other two: whole-package-first). Choosing by connectivity alone is **weaker** than choosing
by exploitability alone here. The gap between those two rules grows with the percolation
threshold (rho = +0.50 across networks), so structure changes how much the layout-based
rule is worth, but no network reverses the ranking of combined over random.

Limits to state: the software is assigned independent of position (a hub is no more likely
to be vulnerable than an edge machine), which favours exploit-first over hub-first; an
untested assumption, not a finding. Patching is instantaneous and perfect. The same floor
and decay assumptions apply (section 1). Fixing is a budget of machines, not of effort.

## 4. Checks

- The faster sweep in `uncertainty.py` reproduces `sensitivity.run` exactly (same seeds).
- Unpatched rows of the patching run equal `sweep_attackers.csv` for all 1,200 scenarios.
- R verified by hand on a three-machine path; zero without vulnerability; ignores
  unreachable machines.
- Each patching rule picks only vulnerable machines, stays within the budget and picks what
  its name says; patching lowers risk under every rule.
