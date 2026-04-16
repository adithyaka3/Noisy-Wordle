"""
plot_word_length_scaling.py  (Figure: Scaling Across Word Lengths)
───────────────────────────────────────────────────────────────────
Generates:
  report/img/scaling_turns.pdf    — avg turns  vs word length (5-8)
  report/img/scaling_time.pdf     — avg time   vs word length (5-8)

Each figure shows one line per strategy, for both English and Random dicts,
using the representative 80/10/10 noise configuration.

Run from inside the report/ folder:
    python plot_word_length_scaling.py
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from parse_outputs import (
    parse_output_file, STRATEGY_NAMES, NOISE_CONFIGS, WORD_LENGTHS, DICT_TYPES
)

IMG_DIR      = os.path.join(HERE, "img")
os.makedirs(IMG_DIR, exist_ok=True)
OUTPUTS_ROOT = os.path.join(HERE, "..", "outputs")

# primary noise config used for scaling comparison
NOISE_CFG   = "80_10_10"
NOISE_LABEL = "80/10/10"

STRAT_COLORS = ["#4CC9F0", "#F72585", "#7BF1A8", "#FFB347"]
DICT_LINESTYLE = {"english": "-", "random": "--"}
DICT_MARKER    = {"english": "o", "random": "s"}


def collect_scaling_data():
    """
    Returns nested dict:
        data[dict_type][word_length][strategy_id] = {avg_turns, avg_time}
    """
    result = {}
    for dt in DICT_TYPES:
        result[dt] = {}
        for wl in WORD_LENGTHS:
            fpath = os.path.join(OUTPUTS_ROOT, NOISE_CFG, f"output_{dt}{wl}.txt")
            if not os.path.exists(fpath):
                print(f"  [skip] {fpath}")
                continue
            parsed = parse_output_file(fpath)
            result[dt][wl] = {}
            for sid in range(1, 5):
                turns = parsed[sid]["turns"]
                times = parsed[sid]["times"]
                result[dt][wl][sid] = {
                    "avg_turns": np.mean(turns) if turns else np.nan,
                    "avg_time" : np.mean(times) if times else np.nan,
                }
    return result


def make_plot(data, metric_key, ylabel, title_suffix, filename):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0D1117")
    ax.set_facecolor("#161B22")
    ax.spines[:].set_color("#30363D")
    ax.tick_params(colors="#8B949E", labelsize=10)

    for sid in range(1, 5):
        color = STRAT_COLORS[sid - 1]
        for dt in DICT_TYPES:
            ys = [data[dt].get(wl, {}).get(sid, {}).get(metric_key, np.nan)
                  for wl in WORD_LENGTHS]
            ls = DICT_LINESTYLE[dt]
            mk = DICT_MARKER[dt]
            label = f"{STRATEGY_NAMES[sid]}  ({dt})"
            ax.plot(WORD_LENGTHS, ys, color=color, lw=2, ls=ls,
                    marker=mk, markersize=7, label=label)

    ax.set_xticks(WORD_LENGTHS)
    ax.set_xticklabels([f"{w}-letter" for w in WORD_LENGTHS])
    ax.set_xlabel("Word Length", color="#8B949E", fontsize=11)
    ax.set_ylabel(ylabel, color="#8B949E", fontsize=11)
    ax.set_title(
        f"Scaling Across Word Lengths — {title_suffix}\n"
        f"(noise {NOISE_LABEL}, 100 games each)",
        color="white", fontsize=13, fontweight="bold", pad=10
    )
    # legend: 2-column outside
    leg = ax.legend(ncol=2, loc="upper left",
                    framealpha=0.2, labelcolor="white", fontsize=8,
                    facecolor="#21262D", edgecolor="#30363D")
    ax.grid(True, color="#21262D", lw=0.8, ls="--")

    out_path = os.path.join(IMG_DIR, filename)
    plt.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved → {out_path}")
    plt.close()


def main():
    print(f"Collecting scaling data from noise config '{NOISE_CFG}' …")
    data = collect_scaling_data()

    make_plot(data,
              metric_key="avg_turns",
              ylabel="Average Turns to Solve",
              title_suffix="Avg Turns",
              filename="scaling_turns.pdf")

    make_plot(data,
              metric_key="avg_time",
              ylabel="Average Computation Time (s)",
              title_suffix="Avg Time (s)",
              filename="scaling_time.pdf")


if __name__ == "__main__":
    main()
