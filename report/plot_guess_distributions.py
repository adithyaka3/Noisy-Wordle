"""
plot_guess_distributions.py  (Figure: Impact of Noise)
────────────────────────────────────────────────────────
Generates: report/img/guess_distributions_{noise_config}.pdf
  → one figure per noise config, each a 2×2 grid of histograms
    (one subplot per algorithm)

Run from inside the  report/  folder:
    python plot_guess_distributions.py
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from parse_outputs import (
    parse_output_file, STRATEGY_NAMES, NOISE_CONFIGS, NOISE_LABELS
)

IMG_DIR = os.path.join(HERE, "img")
os.makedirs(IMG_DIR, exist_ok=True)
OUTPUTS_ROOT = os.path.join(HERE, "..", "outputs")

PALETTE = ["#4CC9F0", "#F72585", "#7BF1A8", "#FFB347"]
STRAT_IDS = [1, 2, 3, 4]


def plot_for_noise(noise_cfg: str, noise_label: str, dict_type="english", wl=5):
    fpath = os.path.join(OUTPUTS_ROOT, noise_cfg, f"output_{dict_type}{wl}.txt")
    if not os.path.exists(fpath):
        print(f"  [skip] {fpath} not found")
        return

    data = parse_output_file(fpath)

    fig = plt.figure(figsize=(12, 9), facecolor="#0D1117")
    fig.suptitle(
        f"Guess-Count Distribution — {dict_type.capitalize()} {wl}-letter words\n"
        f"Noise Config: {noise_label}  (100 games)",
        color="white", fontsize=13, fontweight="bold", y=1.01
    )

    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    for idx, sid in enumerate(STRAT_IDS):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.set_facecolor("#161B22")
        ax.spines[:].set_color("#30363D")
        ax.tick_params(colors="#8B949E", labelsize=9)

        turns = data[sid]["turns"]
        if not turns:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="#8B949E")
            continue

        color = PALETTE[idx]
        bins  = range(1, max(turns) + 2)
        ax.hist(turns, bins=bins, color=color, alpha=0.85, edgecolor="black",
                linewidth=0.4, rwidth=0.8)

        mean_t = np.mean(turns)
        ax.axvline(mean_t, color="white", lw=1.5, ls="--", alpha=0.8,
                   label=f"Mean = {mean_t:.1f}")

        ax.set_title(STRATEGY_NAMES[sid], color="#C9D1D9", fontsize=10, pad=6)
        ax.set_xlabel("Turns to Solve", color="#8B949E", fontsize=9)
        ax.set_ylabel("Number of Games", color="#8B949E", fontsize=9)
        ax.legend(framealpha=0.2, labelcolor="white", fontsize=8,
                  facecolor="#21262D", edgecolor="#30363D")

    out_name = f"guess_dist_{noise_cfg}_{dict_type}{wl}.pdf"
    out_path = os.path.join(IMG_DIR, out_name)
    plt.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Saved → {out_path}")
    plt.close()


def main():
    # Primary figures the report uses: english-5 across all noise configs
    for nc, nl in zip(NOISE_CONFIGS, NOISE_LABELS):
        print(f"Plotting guess distributions for noise={nc} ...")
        plot_for_noise(nc, nl, dict_type="english", wl=5)


if __name__ == "__main__":
    main()
