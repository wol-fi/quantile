"""Annotated heatmap of results.json (Figure D.1 of the paper).

Run after quantile_accuracy_table.py. Requires numpy and matplotlib. Writes
quantile_accuracy_heatmap.{pdf,png} beside this script.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
LABELS = {  # row label in the figure, keyed by the label in results.json
    "Beta, shapes below one": "GB: beta, shapes below one",
    "Beta, shapes above one": "GB: beta, shapes above one",
    "Gamma, shape below one": "GG: gamma, shape below one",
    "Exponential (GEV)": "GEV: exponential",
    "Inverse Gaussian": "GIG: inverse Gaussian",
    "Hyperbolic": "GH: hyperbolic",
    "Normal inverse Gaussian": "GH: normal inverse Gaussian",
    "Normal": "Normal and log-normal",
    "a < 1": r"Student's t, $a<1$",
    "a = 5": r"Student's t, $a=5$",
    "Density unbounded at zero": "VG: density unbounded at 0",
    "Density bounded at zero": "VG: density bounded at 0",
}


def main():
    data = json.loads((OUT / "results.json").read_text())
    rows, ps = data["rows"], data["p_grid"]
    errors = np.array([[r["rel_err"][str(p)] for p in ps] for r in rows])
    logs = np.log10(errors)
    vmin, vmax = math.floor(logs.min()), math.ceil(logs.max())

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    im = ax.imshow(logs, aspect="auto", cmap="viridis_r", vmin=vmin, vmax=vmax)
    ax.set(xticks=range(len(ps)), xticklabels=[f"{p:g}" for p in ps],
           yticks=range(len(rows)), yticklabels=[LABELS[r["label"]] for r in rows],
           xlabel=r"Probability level $p$",
           title=f"Relative quantile error: {errors.size} contour integrals")
    ax.tick_params(length=0)
    for i in range(len(rows)):
        for j in range(len(ps)):
            ax.text(j, i, f"{errors[i, j]:.0e}", ha="center", va="center", fontsize=8.5,
                    color="black" if logs[i, j] < (vmin + vmax) / 2 else "white")
    for i in range(1, len(rows)):  # separate the distribution families
        if rows[i]["family"] != rows[i - 1]["family"]:
            ax.axhline(i - 0.5, color="white", lw=2.5)
    fig.colorbar(im, ax=ax, label="log10(relative error)", fraction=0.047, pad=0.03)
    fig.tight_layout()
    fig.savefig(OUT / "quantile_accuracy_heatmap.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "quantile_accuracy_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"color range 1e{vmin} .. 1e{vmax}")


if __name__ == "__main__":
    main()
