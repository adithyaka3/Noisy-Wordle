"""
plot_convergence.py  —  Convergence Analysis
─────────────────────────────────────────────
Generates: report/img/convergence.pdf

Plots: cumulative fraction of 100 games that have converged by turn N.
  - Strategies 1 & 2: converged when Leader Gap >= 4.60 (99% confidence)
  - Strategies 3 & 4: converged when game resolved (turns recorded)

Two panels: 100/0/0 (clean) vs 70/15/15 (moderate noise). English 5-letter.
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from parse_outputs import parse_output_file, STRATEGY_NAMES

IMG_DIR      = os.path.join(HERE, "img")
os.makedirs(IMG_DIR, exist_ok=True)
OUTPUTS_ROOT = os.path.join(HERE, "..", "outputs")

THRESHOLD    = 4.60
STRAT_COLORS = {1: "#4CC9F0", 2: "#F72585", 3: "#7BF1A8", 4: "#FFB347"}
LINESTYLES   = {1: "-",       2: "--",      3: "-.",      4: ":"}
MARKERS      = {1: "o",       2: "s",       3: "^",       4: "D"}


def convergence_turns(data: dict):
    """
    For each strategy return a list of the turn at which each game converged.
    Strategies 1&2: first turn where Leader Gap >= THRESHOLD.
    Strategies 3&4: just the recorded turn count.
    """
    conv = {}
    for sid in range(1, 5):
        d = data[sid]
        if sid in (1, 2):
            t_list = []
            for gap_seq in d["gaps"]:
                hit = next((i + 1 for i, g in enumerate(gap_seq) if g >= THRESHOLD), None)
                t_list.append(hit if hit is not None else len(gap_seq) + 1)
        else:
            t_list = list(d["turns"])
        conv[sid] = t_list
    return conv


def cdf_curve(t_list, max_t):
    x = np.arange(1, max_t + 2)
    arr = np.array(t_list)
    y = np.array([np.mean(arr <= t) * 100 for t in x])
    return x, y


def main():
    configs = [
        ("100_0_0",  "100/0/0  (No Noise)"),
        ("70_15_15", "70/15/15  (Moderate Noise)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="#0D1117")
    fig.suptitle(
        "Convergence Speed — Cumulative % of Games Solved by Turn N\n"
        "English 5-letter Dictionary · 100 games per condition",
        color="white", fontsize=13, fontweight="bold", y=1.02
    )

    for ax_idx, (nc, label) in enumerate(configs):
        fpath = os.path.join(OUTPUTS_ROOT, nc, "output_english5.txt")
        if not os.path.exists(fpath):
            print(f"  [skip] {fpath}")
            continue

        data = parse_output_file(fpath)
        conv = convergence_turns(data)
        max_t = max(max(v) for v in conv.values() if v)

        ax = axes[ax_idx]
        ax.set_facecolor("#161B22")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#444")
        ax.spines["bottom"].set_color("#444")
        ax.tick_params(colors="#CCCCCC", labelsize=10)

        for sid in range(1, 5):
            x, y = cdf_curve(conv[sid], max_t)
            ax.step(x, y, where="post",
                    color=STRAT_COLORS[sid], lw=2, ls=LINESTYLES[sid],
                    label=STRATEGY_NAMES[sid])
            # markers every 5 turns for legibility
            mask = (x % 5 == 0)
            ax.plot(x[mask], y[mask],
                    marker=MARKERS[sid], color=STRAT_COLORS[sid],
                    markersize=5, ls="none")

        ax.axhline(100, color="#555", lw=0.8, ls="--")
        ax.set_xlabel("Turn Number", fontsize=11, color="#AAAAAA")
        ax.set_ylabel("Games Converged (%)", fontsize=11, color="#AAAAAA")
        ax.set_title(label, fontsize=12, fontweight="bold", color="#E6EDF3", pad=8)
        ax.set_ylim(0, 107)
        ax.set_xlim(left=1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
        ax.grid(True, axis="y", color="#21262D", lw=0.7, ls="--")
        ax.legend(fontsize=9, loc="lower right",
                  framealpha=0.25, labelcolor="white",
                  facecolor="#21262D", edgecolor="#30363D")

    plt.tight_layout()
    out = os.path.join(IMG_DIR, "convergence.pdf")
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out}")
    plt.close()


if __name__ == "__main__":
    main()
