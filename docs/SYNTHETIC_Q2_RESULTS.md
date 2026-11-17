# Q2 across network families (30 networks per synthetic family)

Data: Hiten's `synthetic-30` set (4 families x 30 networks, 2,400 configs; scale-free, random,
small-world, fat-tree) plus the 20 real networks as a fifth family "real" (140 networks).
Same model, same three attackers, 2,000 attacks per scenario (3 attackers x 2,400 configs = 7,200
synthetic scenarios; 1,200 real). Run: `python src/sweep_attackers.py 2000
data/networks_synthetic data/configs_synthetic results/sweep_attackers_synthetic.csv auto`
then `python src/combine_families.py`, `python src/q2_importance.py`, `python src/q2_lofo.py`,
`python src/q1_families.py`, `python src/q2_uncertainty.py 24 1000`.

## Checks on the new data
- The first 10 networks per family are identical to the earlier 10-per-family set; results on them
  agree with the earlier run to within about 0.02 (different random seeds, so simulation noise).
- Fat-tree networks 11-30 differ only in the number of hosts per edge switch (+/-1), so fat-tree has
  the least real variety of the four families.
- Scale-free, random and small-world repeat the same 10 sizes with new seeds.

## The transfer matrix (train on the row family, test on the column family)
Averages over the 20 off-diagonal cells (a family never sees the family it is tested on):

| model | R^2 mean (worst) | rank of unseen networks, Spearman mean (worst) |
|---|---|---|
| mean only | -0.15 (-0.54) | - |
| software setup only | 0.68 (0.20) | 0.21 (-0.48) |
| percolation threshold + software | 0.73 (0.38) | 0.75 (0.50) |
| gradient boosting, 11 structure features + software | 0.88 (0.51) | 0.85 (0.37) |

- Structure beats software-only in all 20 train/test pairs, on both R^2 and ranking.
- Trained on real networks only, the structure model predicts the four synthetic families with
  R^2 0.87 and ranks their networks at 0.82 (software-only: 0.67 and -0.11).
- Trained on the synthetic families, it predicts the real networks with R^2 0.92, ranking 0.91
  (software-only: 0.70 and -0.15).
- Hardest target: scale-free (worst R^2 0.51 when trained on small-world; 0.68 from real).
  Scale-free has big hubs, and families without them do not show the model what a hub does.
- Software-only gets the level of risk mostly right but cannot rank networks.
- Full table with 95% intervals (bootstrap over test networks): `results/q2_all/q2_transfer_ci.csv`.

## Train on every other family, test on the one left out (`src/q2_lofo.py`)
This uses all four other families together and gives each target a confidence interval.

| held-out family | software only R^2 | structure R^2 | software only rank | structure rank |
|---|---|---|---|---|
| fat-tree (30) | 0.78 (0.72-0.82) | 0.96 (0.96-0.97) | 0.21 (-0.14-0.59) | 0.97 (0.91-0.99) |
| random (30) | 0.91 (0.87-0.94) | 0.97 (0.96-0.97) | 0.88 (0.74-0.94) | 0.97 (0.90-0.99) |
| real (20) | 0.77 (0.67-0.85) | 0.94 (0.92-0.96) | -0.23 (-0.65-0.29) | 0.96 (0.88-0.99) |
| scale-free (30) | 0.39 (0.13-0.60) | 0.87 (0.81-0.93) | 0.29 (-0.07-0.61) | 0.92 (0.74-0.98) |
| small-world (30) | 0.90 (0.88-0.93) | 0.97 (0.96-0.97) | 0.77 (0.56-0.87) | 0.93 (0.83-0.97) |

The structure model has the best R^2 and ranking in every family. The one-number predictor
(percolation threshold) sits in between (R^2 0.70-0.92) and is uneven: rank 0.96 on fat-tree but
0.50 on small-world.

## Does it survive the two assumed settings? (`src/q2_uncertainty.py`)
24 scrambled-Sobol (floor, decay) pairs over floor 0.02-0.10 and decay 0.3-0.7. For each pair the
full sweep of all 140 networks is rerun (midterm attacker, 1,000 attacks) and the transfer matrix
rebuilt, averaged over the 20 off-diagonal cells. (24 is not a power of 2, so the Sobol points are
slightly less balanced than ideal; the draws still cover the box.)

| predictor | R^2 median (worst draw) | rank of unseen networks, median (worst draw) |
|---|---|---|
| software only | 0.51 (0.04) | 0.20 (0.03) |
| percolation threshold + software | 0.55 (-0.68) | 0.67 (0.49) |
| gradient boosting, structure + software | 0.78 (0.40) | 0.80 (0.52) |

- Structure beat software-only in all 24 draws on R^2 and on ranking. In the worst draw it still won
  15 of 20 train/test pairs.
- What survives is the direction and the ranking: structure always helps and ranks unseen networks at
  0.52-0.86 where software-only ranks them at 0.03-0.39.
- The level of accuracy is not stable. R^2 falls as the floor rises (rank correlation -0.79):
  when patched machines are easy to break, shares compromised bunch up and every predictor does
  worse. Structure R^2 is above 0.7 in 79% of draws and above 0.5 in 96%.
- Quote "structure beats software-only throughout the plausible range; R^2 median 0.78, ranking
  about 0.8", not 0.88 alone.
- The one-number predictor is the least stable: negative R^2 in 4 draws.

## Which structural features matter? (`src/q2_importance.py`)
Permutation importance measured on the other families, for a model trained on each family:
- The software's average exploitability (`mean_p`) dominates (drop in R^2 about 1.5 when scrambled);
  structure features are small next to it for the level of risk.
- Among structure features, normalised average path length (about 0.09) is the most useful, then the
  percolation threshold (about 0.02); the rest are about 0.01 or less. They are correlated, so credit
  is shared.
- The feature ranking agrees only moderately between families (Spearman about 0.57 on average, see
  `feature_importance_rank_agreement.csv`).

## Limits
- All synthetic families are built by us (Hiten); the real family is only 20 ISP backbones.
- Synthetic families are denser (mean degree about 2.8-3.5 or more) than some real networks.
- Floor 0.05 and decay 0.5 are assumptions; the check above covers them, but not other model
  changes (for example software placed according to a machine's position).
- Intervals bootstrap over test networks, so they reflect network variety, not Monte Carlo noise in
  each scenario.
