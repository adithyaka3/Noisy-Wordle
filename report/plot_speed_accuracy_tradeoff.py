"""
plot_speed_accuracy_tradeoff.py  —  Speed-Accuracy Frontier
─────────────────────────────────────────────────────────────
Generates: report/img/speed_accuracy_tradeoff.pdf

Scatter plot: avg computation time (x, log scale) vs avg turns (y).
Each strategy traces a curve through the 5 noise levels.
Point size encodes noise severity (smaller p_correct = larger point).

Shows the Pareto-like tradeoff: low-time strategies need more turns;
high-accuracy strategies cost more compute.
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
MARKER_S     = [80, 120, 160, 200, 240]   # size grows with noise


def main():
    avg_turns = {sid: [] for sid in range(1, 5)}
    avg_times = {sid: [] for sid in range(1, 5)}

    for nc in NOISE_CONFIGS:
        fpath = os.path.join(OUTPUTS_ROOT, nc, "output_english5.txt")
        if not os.path.exists(fpath):
            for sid in range(1, 5):
                avg_turns[sid].append(np.nan)
                avg_times[sid].append(np.nan)
            continue
        d = parse_output_file_fast(fpath)
        for sid in range(1, 5):
            avg_turns[sid].append(np.mean(d[sid]["turns"]) if d[sid]["turns"] else np.nan)
            avg_times[sid].append(np.mean(d[sid]["times"]) if d[sid]["times"] else np.nan)

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="#0D1117")
    ax.set_facecolor("#161B22")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444")
    ax.spines["bottom"].set_color("#444")
    ax.tick_params(colors="#CCCCCC", labelsize=10)

    for sid in range(1, 5):
        xs = avg_times[sid]
        ys = avg_turns[sid]
        color = STRAT_COLORS[sid]
        # connecting line
        ax.plot(xs, ys, color=color, lw=1.5, ls=LINESTYLES[sid], alpha=0.6)
        # one scatter point per noise level
        for i, (x, y, s, nl) in enumerate(zip(xs, ys, MARKER_S, NOISE_LABELS)):
            if np.isnan(x) or np.isnan(y):
                continue
            sc = ax.scatter(x, y, s=s, color=color, edgecolors="white",
                            linewidths=0.6, zorder=5)
            if sid == 1:   # label noise levels only once
                ax.annotate(nl, xy=(x, y), xytext=(5, 5), fontsize=7.5,
                            textcoords="offset points", color="#AAAAAA")

        # strategy label at 80/10/10 point (index 2)
        lx, ly = xs[2], ys[2]
        if not (np.isnan(lx) or np.isnan(ly)):
            ax.annotate(STRATEGY_NAMES[sid], xy=(lx, ly),
                        xytext=(6, -10), textcoords="offset points",
                        color=color, fontsize=8.5, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("Avg Computation Time (s, log scale)", fontsize=11, color="#AAAAAA")
    ax.set_ylabel("Avg Turns to Solve", fontsize=11, color="#AAAAAA")
    ax.set_title(
        "Speed vs. Accuracy Tradeoff\n"
        "Each curve = one strategy across noise levels  ·  "
        "Point size ∝ noise severity",
        color="white", fontsize=12, fontweight="bold", pad=10
    )
    ax.grid(True, color="#21262D", lw=0.7, ls="--", which="both")

    # custom legend for strategies only
    handles = [
        plt.Line2D([0], [0], color=STRAT_COLORS[sid], lw=2,
                   ls=LINESTYLES[sid], label=STRATEGY_NAMES[sid])
        for sid in range(1, 5)
    ]
    ax.legend(handles=handles, fontsize=9, loc="upper left",
              framealpha=0.25, labelcolor="white",
              facecolor="#21262D", edgecolor="#30363D")

    plt.tight_layout()
    out = os.path.join(IMG_DIR, "speed_accuracy_tradeoff.pdf")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out}")
    plt.close()


if __name__ == "__main__":
    main()
