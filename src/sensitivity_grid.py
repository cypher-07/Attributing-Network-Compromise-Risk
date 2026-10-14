"""Floor x decay grid: does the topology share depend on the two settings together?

    python src/sensitivity_grid.py        (about 2 minutes; needs matplotlib)

Runs the full 400-scenario sweep for every (floor, decay) pair, with 1,000
attacks per scenario, and writes
  results/sensitivity/floor_decay_grid.csv
  figures/figS1_floor_decay_grid.pdf   (heat map, no title: explain it in the text)
Nothing existing is changed.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import sensitivity as S
from paths import SENSITIVITY, FIGURES

FLOORS = (0.02, 0.05, 0.10)
DECAYS = (0.3, 0.5, 0.7)

if __name__ == "__main__":
    S.TRIALS = 1000
    rows = []
    for fl in FLOORS:
        for dc in DECAYS:
            r = S.run(floor=fl, decay=dc)
            rows.append({"floor": fl, "decay": dc, **r})
            print(f"floor {fl:.2f} decay {dc:.1f}: topology {r['topology_share']:.1%}  "
                  f"mean compromised {r['mean_compromised']:.1%}", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(SENSITIVITY, exist_ok=True)
    df.to_csv(os.path.join(SENSITIVITY, "floor_decay_grid.csv"), index=False, float_format="%.4f")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, col, cmap in ((axes[0], "topology_share", "Blues"), (axes[1], "mean_compromised", "Greys")):
        M = df.pivot(index="floor", columns="decay", values=col).loc[list(FLOORS), list(DECAYS)]
        im = ax.imshow(M.values, cmap=cmap, vmin=0, vmax=1, origin="lower", aspect="auto")
        ax.set_xticks(range(len(DECAYS)), [f"{d:g}" for d in DECAYS])
        ax.set_yticks(range(len(FLOORS)), [f"{f:g}" for f in FLOORS])
        ax.set_xlabel("decay per extra hop"); ax.set_ylabel("floor (patched machine)")
        for i in range(len(FLOORS)):
            for j in range(len(DECAYS)):
                v = M.values[i, j]
                ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                        color="white" if v > 0.55 else "#0b0b0b", fontsize=9)
        cb = fig.colorbar(im, ax=ax); cb.ax.yaxis.set_major_formatter(PercentFormatter(1, 0))
        ax.set_title({"topology_share": "topology share of variance",
                      "mean_compromised": "mean share compromised"}[col], fontsize=9)
    fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, "figS1_floor_decay_grid.pdf"))
    print("wrote results/sensitivity/floor_decay_grid.csv and figures/figS1_floor_decay_grid.pdf")
