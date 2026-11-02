# Attacker variants and the Q2 predictor

Everything here is additive: no existing file was changed. The midterm numbers
are reproduced exactly (see "Checks").

## 1. The two new attackers

Same simulator, same configurations, same 2,000 trials. Only the starting point changes.

| attacker | starts at | in plain words |
|---|---|---|
| `assume_breach_random_entry` | the config's `is_entry_point` (one node, random) | the midterm attacker |
| `external_border_entry` | a random **border node**, redrawn every trial | an outsider who gets in at the edge |
| `insider_privileged_entry` | a **privileged node** (`is_privileged`) | an insider with an admin account |

**Border node** = a node with a single link (a dead end). If a network has none,
the nodes with the fewest links. Chosen from the wiring alone, never from the
software. Code: `src/attackers.py`.

**Privileged node.** In every real config it is the network's biggest hub (the
node with the most links; Colt has two). So the insider attacker is "the attacker
starts at the biggest hub".

Entry is drawn per trial (not once per config), so a scenario's risk averages
over all its border nodes.

Run: `python src/sweep_attackers.py` (about 25 seconds for all 20 real networks:
2,000 trials x 400 configs x 3 attackers = 2.4 million simulated attacks)
-> `results/sweep_attackers.csv`.

## 2. Preliminary results, 20 real networks (2,000 trials)

Mean share of nodes compromised:

| attacker | all densities | 20% | 40% | 60% | 80% |
|---|---|---|---|---|---|
| assume-breach (midterm) | 37.9% | 17.0% | 32.5% | 47.5% | 54.9% |
| external (border) | 35.5% | 14.5% | 30.1% | 44.2% | 53.0% |
| insider (privileged hub) | 40.7% | 21.5% | 36.3% | 48.4% | 56.5% |

Plain reading: starting at the biggest hub does the most damage; starting at a
dead end at the edge does the least. The gap is widest when few machines are
vulnerable (20%: 14.5% vs 21.5%) and nearly vanishes when most are (80%).

**Who explains the differences** (`python src/q1_adversary.py`, logit scale, within
density, 95% intervals from resampling networks):

| source | share of variation | 95% interval |
|---|---|---|
| network wiring (topology) | 25.0% | 8.9-37.2% |
| software setup (configuration) | 56.4% | 45.4-71.0% |
| attacker | 9.4% | 7.2-13.0% |
| attacker depends on the network | 1.4% | 0.5-2.1% |
| attacker x configuration + noise | 7.7% | 6.4-9.5% |

By density the attacker's share is 18.2% at 20% vulnerable, 7.4% at 40%, 3.9% at
60%, 2.7% at 80%. Topology holds at about a quarter, as at midterm.

**Read this carefully.** Configuration fell from 74% to 56% mostly because the
entry point used to count as part of the configuration. With the entry point now
defined by the attacker, that variation moved to the attacker row. It is a change
in definition, not new evidence that configuration matters less. The report
should say so. The same two assumptions (floor 0.05, decay 0.5) apply; the
sensitivity grid has not been re-run for the new attackers.

## 3. Q2: does a predictor trained on one family work on another?

`src/q2_predict.py` builds the 4x4 (or NxN) transfer matrix.

- Train on every scenario of family i, test on family j. Never the same network
  on both sides. The diagonal is leave-one-network-out inside the family.
- Models: `mean` (do-nothing floor), `config_only` (software + attacker only),
  `pc_linear` (one structural number: percolation threshold, plus software),
  `structure_gb` (gradient boosting on all 11 structural features + software).
- Scores on the share-compromised scale: R^2 over test scenarios, Spearman over
  test scenarios and over test networks (per-network mean).
- Needs two tables: `sweep_attackers.csv` (run for each family) and a features
  table from `src/make_features.py <networks_dir> <out.csv>`.

**Real networks only, leave-one-network-out (20 networks, 3 attackers):**

| model | R^2 | Spearman across held-out networks |
|---|---|---|
| config_only | 0.80 | -0.31 |
| pc_linear | 0.91 | 0.90 |
| structure_gb | 0.92 | 0.89 |

Plain reading: the software setup alone gets the *levels* mostly right but cannot
rank networks it has not seen; adding structure fixes the ranking. One number
(percolation threshold) does almost as well as the full gradient-boosting model,
which is itself a finding for the report. These are within-family numbers; the
cross-family matrix needs the synthetic families.

**Dry run.** `src/dryrun_synthetic.py` builds four fake families (scale-free,
random, small-world, tree-like; 40 nodes; 5 networks each) in the real data
format so the whole matrix can be exercised. Its numbers mean nothing for the
project. They did show one thing to expect: R^2 can go very negative when a test
family is riskier than anything seen in training (the model extrapolates badly),
so report Spearman next to R^2.

## 4. What the synthetic families must look like (for Hiten and Faisal)

The sample synthetic files in the handoff do not match the pipeline's format, so
`sweep_attackers.py` cannot run them as they are. To run them, each synthetic
family needs:

1. `networks/<name>.graphml` (connected, simple graph) and, per network,
   configs named `<name>_d<density>_s<seed>.json`.
2. Densities 0.2, 0.4, 0.6, 0.8 and seeds 1-5 (the sample uses 0.3/0.5/0.7 and 2 seeds).
3. Per node: `software` drawn from the 24-package catalog (or
   `Fully patched (no known CVEs)`), `is_entry_point`, and `is_privileged`.
   The sample files have no `is_privileged` flag, so the insider attacker cannot
   start. Use the same rule as the real configs: privileged = highest-degree node.
4. Real configs list `p` per node; the sweep recomputes it from the catalog, so it
   is optional but must agree if present.
5. Network size: the samples have 15 nodes; the real networks span 11-153.
   `config.yaml` says 60. Pick one and state it; with 5 networks per family,
   more networks per family (10+) would make the cross-family matrix much less noisy.

Then: `python src/sweep_attackers.py 2000 <networks_dir> <configs_dir> <out.csv>`
for each family, set the `family` column, concatenate with the real results, and
run `q2_predict.py`.

## 5. Checks

- Assume-breach rows reproduce `results/sweep.csv` exactly (max difference 0.0
  over all 400 scenarios).
- The three-attacker variance code, run on assume-breach alone, returns the
  midterm split (25.8% / 74.2%).
- A planted-effect test recovers known variance shares.
- Q2 tests: no network is ever in both train and test; every family pair is
  present; a planted signal beats the do-nothing model.
- `tests/test_attackers_q2.py`: 12 tests, all pass.
