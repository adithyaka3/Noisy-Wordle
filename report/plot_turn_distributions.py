"""
plot_turn_distributions.py  —  Guess Distribution Comparison Across Noise
──────────────────────────────────────────────────────────────────────────
Generates: report/img/turn_dist_comparison.pdf

Violin + box plot of the turn distribution for each strategy,
overlaid across three noise configs (100/0/0, 80/10/10, 60/20/20).
English 5-letter dictionary.

Violin shows probability density; box inside shows quartiles.
This reveals *spread* and *tail risk*, not just the mean.
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from parse_outputs import parse_output_file_fast, STRATEGY_NAMES

IMG_DIR      = os.path.join(HERE, "img")
os.makedirs(IMG_DIR, exist_ok=True)
OUTPUTS_ROOT = os.path.join(HERE, "..", "outputs")

CONFIGS = [
    ("100_0_0",  "100/0/0\n(Clean)",  "#4CC9F0"),
    ("80_10_10", "80/10/10",           "#F72585"),
    ("60_20_20", "60/20/20\n(Noisy)", "#FFB347"),
]
STRAT_IDS = [1, 2, 3, 4]


def main():
    # load turns data for each config
    all_data = {}
    for nc, label, color in CONFIGS:
        fpath = os.path.join(OUTPUTS_ROOT, nc, "output_english5.txt")
        if not os.path.exists(fpath):
            print(f"  [skip] {fpath}")
            continue
        d = parse_output_file_fast(fpath)
        all_data[nc] = d

    fig, axes = plt.subplots(1, 4, figsize=(16, 6), facecolor="#0D1117",
                             sharey=False)
    fig.suptitle(
        "Turn Distribution per Strategy — Effect of Noise\n"
        "English 5-letter Dictionary · 100 games per condition",
        color="white", fontsize=13, fontweight="bold", y=1.02
    )

    n_cfg  = len(CONFIGS)
    positions_base = np.arange(1, n_cfg + 1) * 2   # spread violins out

    legend_patches = [
        mpatches.Patch(color=c, label=lbl.replace("\n", " "))
        for (_, lbl, c) in CONFIGS
    ]

    for ax_idx, sid in enumerate(STRAT_IDS):
        ax = axes[ax_idx]
        ax.set_facecolor("#161B22")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#444")
        ax.spines["bottom"].set_color("#444")
        ax.tick_params(colors="#CCCCCC", labelsize=9)

        vdata  = []
        colors = []
        pos    = []

        for gi, (nc, _, color) in enumerate(CONFIGS):
            if nc not in all_data:
                continue
            turns = all_data[nc][sid]["turns"]
            if not turns:
                continue
            vdata.append(turns)
            colors.append(color)
            pos.append(positions_base[gi])

        if vdata:
            parts = ax.violinplot(vdata, positions=pos, widths=1.4,
                                  showmeans=False, showmedians=False,
                                  showextrema=False)
            for body, c in zip(parts["bodies"], colors):
                body.set_facecolor(c)
                body.set_edgecolor("white")
                body.set_alpha(0.75)
                body.set_linewidth(0.5)

            # overlay box plots
            bp = ax.boxplot(vdata, positions=pos, widths=0.5,
                            patch_artist=True, notch=False,
                            manage_ticks=False,
                            medianprops=dict(color="white", lw=1.8),
                            whiskerprops=dict(color="#AAAAAA", lw=1.2),
                            capprops=dict(color="#AAAAAA", lw=1.2),
                            flierprops=dict(marker=".", color="#AAAAAA",
                                            markersize=3, alpha=0.5))
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.4)
                patch.set_edgecolor("white")

        ax.set_title(STRATEGY_NAMES[sid], color="#E6EDF3",
                     fontsize=9.5, fontweight="bold", pad=6, wrap=True)
        ax.set_ylabel("Turns to Solve", fontsize=9, color="#AAAAAA")
        ax.set_xticks(positions_base)
        ax.set_xticklabels([lbl.split("\n")[0] for (_, lbl, _) in CONFIGS],
                           fontsize=8, color="#CCCCCC")
        ax.grid(True, axis="y", color="#21262D", lw=0.6, ls="--")
        ax.set_ylim(bottom=0)

    # single shared legend
    fig.legend(handles=legend_patches, loc="lower center",
               bbox_to_anchor=(0.5, -0.07), ncol=3, fontsize=10,
               framealpha=0.25, labelcolor="white",
               facecolor="#21262D", edgecolor="#30363D")

    plt.tight_layout()
    out = os.path.join(IMG_DIR, "turn_dist_comparison.pdf")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out}")
    plt.close()


if __name__ == "__main__":
    main()
