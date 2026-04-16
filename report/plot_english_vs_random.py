"""
plot_english_vs_random.py  —  Linguistic Structure Advantage
──────────────────────────────────────────────────────────────
Generates: report/img/english_vs_random.pdf

Two panels:
  Left : Avg turns (English vs Random) for each word length,
         one grouped bar cluster per length, across all noise levels.
  Right: Same for avg time.

Shows whether linguistic patterns in English words help or hurt each
algorithm relative to purely random letter sequences.
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from parse_outputs import (
    parse_output_file_fast, STRATEGY_NAMES,
    NOISE_CONFIGS, NOISE_LABELS, WORD_LENGTHS
)

IMG_DIR      = os.path.join(HERE, "img")
os.makedirs(IMG_DIR, exist_ok=True)
OUTPUTS_ROOT = os.path.join(HERE, "..", "outputs")

STRAT_COLORS = {1: "#4CC9F0", 2: "#F72585", 3: "#7BF1A8", 4: "#FFB347"}
DICT_HATCH   = {"english": "", "random": "///"}
BAR_WIDTH    = 0.18

# representative noise: use 80/10/10
NOISE_CFG = "80_10_10"


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0D1117")
    fig.suptitle(
        "English vs. Random Dictionary — Performance Comparison\n"
        f"Noise Config: {NOISE_CFG.replace('_','/')}  ·  100 games each",
        color="white", fontsize=13, fontweight="bold", y=1.02
    )

    metrics = [("turns", "Avg Turns to Solve"),
               ("times", "Avg Computation Time (s)")]

    for ax_idx, (metric_key, ylabel) in enumerate(metrics):
        ax = axes[ax_idx]
        ax.set_facecolor("#161B22")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#444")
        ax.spines["bottom"].set_color("#444")
        ax.tick_params(colors="#CCCCCC", labelsize=10)

        x = np.arange(len(WORD_LENGTHS))   # one group per word length

        for sid in range(1, 5):
            offset_e = (sid - 2.5) * BAR_WIDTH * 2
            offset_r = offset_e + BAR_WIDTH

            vals_e, vals_r = [], []
            for wl in WORD_LENGTHS:
                def _get(dt):
                    fp = os.path.join(OUTPUTS_ROOT, NOISE_CFG,
                                      f"output_{dt}{wl}.txt")
                    if not os.path.exists(fp):
                        return np.nan
                    d = parse_output_file_fast(fp)
                    v = d[sid][metric_key]
                    return np.mean(v) if v else np.nan

                vals_e.append(_get("english"))
                vals_r.append(_get("random"))

            ax.bar(x + offset_e, vals_e, BAR_WIDTH,
                   color=STRAT_COLORS[sid], alpha=0.88,
                   hatch=DICT_HATCH["english"], edgecolor="black",
                   linewidth=0.3, label=f"S{sid} English" if ax_idx == 0 else None)
            ax.bar(x + offset_r, vals_r, BAR_WIDTH,
                   color=STRAT_COLORS[sid], alpha=0.55,
                   hatch=DICT_HATCH["random"], edgecolor="black",
                   linewidth=0.3, label=f"S{sid} Random" if ax_idx == 0 else None)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{w}-letter" for w in WORD_LENGTHS],
                           color="#CCCCCC", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11, color="#AAAAAA")
        ax.set_title(ylabel, fontsize=11, color="#E6EDF3", pad=6)
        ax.grid(True, axis="y", color="#21262D", lw=0.6, ls="--")

    # legend — combine strategy colours + hatch for dict type
    legend_handles = []
    for sid in range(1, 5):
        legend_handles.append(
            mpatches.Patch(color=STRAT_COLORS[sid],
                           label=f"S{sid}: {STRATEGY_NAMES[sid]}")
        )
    legend_handles += [
        mpatches.Patch(facecolor="#888", hatch="", edgecolor="black",
                       label="English dict"),
        mpatches.Patch(facecolor="#888", hatch="///", edgecolor="black",
                       label="Random dict"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.10), ncol=3, fontsize=9,
               framealpha=0.25, labelcolor="white",
               facecolor="#21262D", edgecolor="#30363D")

    plt.tight_layout()
    out = os.path.join(IMG_DIR, "english_vs_random.pdf")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out}")
    plt.close()


if __name__ == "__main__":
    main()
