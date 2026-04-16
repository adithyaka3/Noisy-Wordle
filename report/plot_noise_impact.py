"""
plot_noise_impact.py  —  Effect of Noise on Algorithm Performance
─────────────────────────────────────────────────────────────────
Generates:
  report/img/noise_impact.pdf

Two-panel figure:
  Left  : Average turns to solve  vs. noise level (p_correct ↓)
  Right : Average wall-clock time vs. noise level

English 5-letter dictionary, all 4 strategies.
Noise axis: 100% → 90% → 80% → 70% → 60% correct.
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from parse_outputs import (
    parse_output_file_fast, STRATEGY_NAMES,
    NOISE_CONFIGS, NOISE_LABELS
)

IMG_DIR      = os.path.join(HERE, "img")
os.makedirs(IMG_DIR, exist_ok=True)
OUTPUTS_ROOT = os.path.join(HERE, "..", "outputs")

STRAT_COLORS = {1: "#4CC9F0", 2: "#F72585", 3: "#7BF1A8", 4: "#FFB347"}
LINESTYLES   = {1: "-",       2: "--",      3: "-.",      4: ":"}
MARKERS      = {1: "o",       2: "s",       3: "^",       4: "D"}
# p_correct values (x-axis)
P_CORRECT    = [100, 90, 80, 70, 60]


def main():
    # collect avg_turns and avg_time per (strategy, noise_config)
    avg_turns = {sid: [] for sid in range(1, 5)}
    avg_times = {sid: [] for sid in range(1, 5)}

    for nc in NOISE_CONFIGS:
        fpath = os.path.join(OUTPUTS_ROOT, nc, "output_english5.txt")
        if not os.path.exists(fpath):
            for sid in range(1, 5):
                avg_turns[sid].append(np.nan)
                avg_times[sid].append(np.nan)
            continue
        data = parse_output_file_fast(fpath)
        for sid in range(1, 5):
            t = data[sid]["turns"]
            tm = data[sid]["times"]
            avg_turns[sid].append(np.mean(t) if t else np.nan)
            avg_times[sid].append(np.mean(tm) if tm else np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="#0D1117")
    fig.suptitle(
        "Impact of Noise on Algorithm Performance\n"
        "English 5-letter Dictionary · 100 games per noise level",
        color="white", fontsize=13, fontweight="bold", y=1.02
    )

    ylabels = ["Average Turns to Solve", "Average Computation Time (s)"]
    ydata   = [avg_turns, avg_times]
    titles  = ["Sample Efficiency — Avg Turns vs Noise",
               "Computational Cost — Avg Time vs Noise"]

    for ax_idx in range(2):
        ax = axes[ax_idx]
        ax.set_facecolor("#161B22")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#444")
        ax.spines["bottom"].set_color("#444")
        ax.tick_params(colors="#CCCCCC", labelsize=10)

        for sid in range(1, 5):
            ys = ydata[ax_idx][sid]
            ax.plot(P_CORRECT, ys,
                    color=STRAT_COLORS[sid], lw=2,
                    ls=LINESTYLES[sid], marker=MARKERS[sid],
                    markersize=7, label=STRATEGY_NAMES[sid])
            # annotate last value
            y_last = ys[-1]
            if not np.isnan(y_last):
                ax.annotate(f"{y_last:.1f}",
                            xy=(P_CORRECT[-1], y_last),
                            xytext=(4, 0), textcoords="offset points",
                            color=STRAT_COLORS[sid], fontsize=7.5,
                            va="center")

        ax.set_xlabel("P(Correct) %  →  more noise", fontsize=11, color="#AAAAAA")
        ax.set_ylabel(ylabels[ax_idx], fontsize=11, color="#AAAAAA")
        ax.set_title(titles[ax_idx], fontsize=11, color="#E6EDF3", pad=8)
        ax.set_xticks(P_CORRECT)
        ax.set_xticklabels([f"{p}%" for p in P_CORRECT], color="#CCCCCC")
        ax.invert_xaxis()  # left = clean (100%), right = most noisy (60%)
        ax.grid(True, color="#21262D", lw=0.7, ls="--")
        ax.legend(fontsize=9, loc="upper left" if ax_idx == 0 else "upper right",
                  framealpha=0.25, labelcolor="white",
                  facecolor="#21262D", edgecolor="#30363D")

    plt.tight_layout()
    out = os.path.join(IMG_DIR, "noise_impact.pdf")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out}")
    plt.close()


if __name__ == "__main__":
    main()
