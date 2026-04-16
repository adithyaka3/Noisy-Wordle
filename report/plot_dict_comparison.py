"""
plot_dict_comparison.py  (Figure: Dict-Type Comparison)
────────────────────────────────────────────────────────
Generates:
  report/img/dict_compare_5.pdf   — English vs Random for 5-letter words
  report/img/dict_compare_6.pdf   — English vs Random for 6-letter words
  report/img/dict_compare_7.pdf   — English vs Random for 7-letter words
  report/img/dict_compare_8.pdf   — English vs Random for 8-letter words

Each figure shows grouped bar charts of avg turns and avg time separated
by dictionary type, for all four strategies, for a given word length.
All noise configs are overlaid (one grouped cluster per noise level).

The 5 & 6-letter figures cover all five noise configs (for section 3.4.a).
The 7 & 8-letter figures the same.

Run from inside the report/ folder:
    python plot_dict_comparison.py
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
    parse_output_file, STRATEGY_NAMES, NOISE_CONFIGS, NOISE_LABELS,
    WORD_LENGTHS, DICT_TYPES
)

IMG_DIR      = os.path.join(HERE, "img")
os.makedirs(IMG_DIR, exist_ok=True)
OUTPUTS_ROOT = os.path.join(HERE, "..", "outputs")

NOISE_COLORS = ["#E0AFA0", "#F4A261", "#F72585", "#7209B7", "#4361EE"]
STRAT_LABELS = [STRATEGY_NAMES[i] for i in range(1, 5)]
DICT_HATCH   = {"english": "", "random": "//"}


def collect_data(word_length: int):
    """
    Returns:
        {noise_cfg: {dict_type: {strategy_id: {avg_turns, avg_time}}}}
    """
    result = {}
    for nc in NOISE_CONFIGS:
        result[nc] = {}
        for dt in DICT_TYPES:
            fpath = os.path.join(OUTPUTS_ROOT, nc, f"output_{dt}{word_length}.txt")
            if not os.path.exists(fpath):
                continue
            parsed = parse_output_file(fpath)
            result[nc][dt] = {}
            for sid in range(1, 5):
                turns = parsed[sid]["turns"]
                times = parsed[sid]["times"]
                result[nc][dt][sid] = {
                    "avg_turns": np.mean(turns) if turns else np.nan,
                    "avg_time" : np.mean(times) if times else np.nan,
                }
    return result


def make_comparison_plot(word_length: int, data: dict):
    """
    Grouped bar chart:
      x-axis  : strategies (4 groups)
      clusters : noise configs × dict types (up to 10 bars per strategy group)
    One subplot for turns, one for time.
    """
    strat_ids = list(range(1, 5))
    n_strat   = len(strat_ids)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0D1117")
    fig.suptitle(
        f"Dictionary-Type Comparison — {word_length}-letter words\n"
        "(English vs Random, all noise configs, avg over 100 games)",
        color="white", fontsize=13, fontweight="bold", y=1.01
    )

    for ax_idx, (metric, ylabel) in enumerate([
        ("avg_turns", "Average Turns to Solve"),
        ("avg_time",  "Average Computation Time (s)"),
    ]):
        ax = axes[ax_idx]
        ax.set_facecolor("#161B22")
        ax.spines[:].set_color("#30363D")
        ax.tick_params(colors="#8B949E", labelsize=9)

        n_per_group  = len(NOISE_CONFIGS) * len(DICT_TYPES)  # 10
        group_width  = 0.8
        bar_width    = group_width / n_per_group
        x_centers    = np.arange(n_strat)

        legend_patches = []
        for nc_idx, (nc, nl) in enumerate(zip(NOISE_CONFIGS, NOISE_LABELS)):
            color = NOISE_COLORS[nc_idx]
            for dt_idx, dt in enumerate(DICT_TYPES):
                bar_offset = (nc_idx * len(DICT_TYPES) + dt_idx) * bar_width
                xs = x_centers - group_width / 2 + bar_offset + bar_width / 2
                ys = [data.get(nc, {}).get(dt, {}).get(sid, {}).get(metric, np.nan)
                      for sid in strat_ids]
                ax.bar(xs, ys, width=bar_width * 0.88, color=color,
                       hatch=DICT_HATCH[dt], alpha=0.85, edgecolor="black",
                       linewidth=0.3)

        # legend items
        for nc_idx, nl in enumerate(NOISE_LABELS):
            legend_patches.append(
                mpatches.Patch(color=NOISE_COLORS[nc_idx], label=nl)
            )
        for dt, hatch in DICT_HATCH.items():
            legend_patches.append(
                mpatches.Patch(facecolor="white", hatch=hatch,
                               edgecolor="black", label=dt.capitalize())
            )

        ax.set_xticks(x_centers)
        ax.set_xticklabels(
            [f"S{i}\n{STRATEGY_NAMES[i][:14]}" for i in strat_ids],
            color="#C9D1D9", fontsize=8
        )
        ax.set_ylabel(ylabel, color="#8B949E", fontsize=10)
        ax.set_title(ylabel, color="#C9D1D9", fontsize=11, pad=6)
        ax.legend(handles=legend_patches, fontsize=7, ncol=2,
                  framealpha=0.2, labelcolor="white",
                  facecolor="#21262D", edgecolor="#30363D")
        ax.grid(axis="y", color="#21262D", lw=0.6, ls="--")

    plt.tight_layout()
    out_path = os.path.join(IMG_DIR, f"dict_compare_{word_length}.pdf")
    plt.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved → {out_path}")
    plt.close()


def main():
    for wl in WORD_LENGTHS:
        print(f"Building dict-comparison plot for {wl}-letter words …")
        data = collect_data(wl)
        make_comparison_plot(wl, data)


if __name__ == "__main__":
    main()
