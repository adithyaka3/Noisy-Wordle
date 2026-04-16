"""
plot_heatmap.py  —  Performance Heatmap
─────────────────────────────────────────
Generates:
  report/img/heatmap_avg_turns.pdf
  report/img/heatmap_avg_time.pdf

2-D heatmaps: rows = strategies, columns = noise configs.
Cell colour encodes avg turns (or avg time).
English 5-letter dictionary.

Also a second figure: rows = word lengths, columns = noise configs,
one subplot per strategy — shows how complexity grows.
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from parse_outputs import (
    parse_output_file_fast, STRATEGY_NAMES,
    NOISE_CONFIGS, NOISE_LABELS, WORD_LENGTHS
)

IMG_DIR      = os.path.join(HERE, "img")
os.makedirs(IMG_DIR, exist_ok=True)
OUTPUTS_ROOT = os.path.join(HERE, "..", "outputs")

SHORT_NAMES = {sid: n.replace(" ", "\n") for sid, n in STRATEGY_NAMES.items()}
SHORT_NOISE = ["100%", "90%", "80%", "70%", "60%"]      # p_correct shorthand


# ── Figure 1: strategy × noise heatmap ──────────────────────────────────────
def heatmap_strategy_noise(metric: str, cmap: str, title: str, fname: str):
    """metric: 'avg_turns' or 'avg_time'"""
    n_strat = 4
    n_noise = len(NOISE_CONFIGS)
    matrix  = np.full((n_strat, n_noise), np.nan)

    for nc_i, nc in enumerate(NOISE_CONFIGS):
        fpath = os.path.join(OUTPUTS_ROOT, nc, "output_english5.txt")
        if not os.path.exists(fpath):
            continue
        d = parse_output_file_fast(fpath)
        for sid in range(1, 5):
            vals = d[sid]["turns"] if metric == "avg_turns" else d[sid]["times"]
            if vals:
                matrix[sid - 1, nc_i] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#0D1117")
    ax.set_facecolor("#161B22")

    im = ax.imshow(matrix, cmap=cmap, aspect="auto")

    # annotate cells
    for r in range(n_strat):
        for c in range(n_noise):
            val = matrix[r, c]
            if not np.isnan(val):
                txt = f"{val:.1f}" if metric == "avg_turns" else f"{val:.2f}s"
                ax.text(c, r, txt, ha="center", va="center",
                        fontsize=10, color="white", fontweight="bold")

    ax.set_xticks(range(n_noise))
    ax.set_xticklabels(SHORT_NOISE, color="#CCCCCC", fontsize=10)
    ax.set_yticks(range(n_strat))
    ax.set_yticklabels([STRATEGY_NAMES[s] for s in range(1, 5)],
                       color="#CCCCCC", fontsize=9)
    ax.set_xlabel("P(Correct)  →  more noise to right", color="#AAAAAA", fontsize=10)
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)

    cb = plt.colorbar(im, ax=ax, pad=0.02)
    cb.ax.yaxis.set_tick_params(color="#CCCCCC")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#CCCCCC")
    cb.set_label("Avg Turns" if metric == "avg_turns" else "Avg Time (s)",
                 color="#AAAAAA")

    plt.tight_layout()
    out = os.path.join(IMG_DIR, fname)
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out}")
    plt.close()


# ── Figure 2: word-length × noise heatmap (one subplot per strategy) ────────
def heatmap_length_noise(metric: str, cmap: str, title: str, fname: str,
                          dict_type: str = "english"):
    n_len   = len(WORD_LENGTHS)
    n_noise = len(NOISE_CONFIGS)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), facecolor="#0D1117")
    fig.suptitle(title, color="white", fontsize=13, fontweight="bold", y=1.02)

    for ax_idx, sid in enumerate(range(1, 5)):
        matrix = np.full((n_len, n_noise), np.nan)
        for nc_i, nc in enumerate(NOISE_CONFIGS):
            for wl_i, wl in enumerate(WORD_LENGTHS):
                fpath = os.path.join(OUTPUTS_ROOT, nc,
                                     f"output_{dict_type}{wl}.txt")
                if not os.path.exists(fpath):
                    continue
                d = parse_output_file_fast(fpath)
                vals = d[sid]["turns"] if metric == "avg_turns" else d[sid]["times"]
                if vals:
                    matrix[wl_i, nc_i] = np.mean(vals)

        ax = axes[ax_idx]
        ax.set_facecolor("#161B22")
        im = ax.imshow(matrix, cmap=cmap, aspect="auto")

        for r in range(n_len):
            for c in range(n_noise):
                val = matrix[r, c]
                if not np.isnan(val):
                    txt = f"{val:.1f}" if metric == "avg_turns" else f"{val:.1f}s"
                    ax.text(c, r, txt, ha="center", va="center",
                            fontsize=8.5, color="white", fontweight="bold")

        ax.set_xticks(range(n_noise))
        ax.set_xticklabels(SHORT_NOISE, color="#CCCCCC", fontsize=8, rotation=30)
        ax.set_yticks(range(n_len))
        ax.set_yticklabels([f"{w}-letter" for w in WORD_LENGTHS],
                           color="#CCCCCC", fontsize=8)
        ax.set_title(STRATEGY_NAMES[sid], color="#E6EDF3",
                     fontsize=9, fontweight="bold", pad=6)
        ax.set_xlabel("P(Correct)", color="#AAAAAA", fontsize=8)
        plt.colorbar(im, ax=ax, pad=0.02).ax.yaxis.set_tick_params(
            color="#CCCCCC", labelsize=7)

    plt.tight_layout()
    out = os.path.join(IMG_DIR, fname)
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out}")
    plt.close()


def main():
    heatmap_strategy_noise(
        "avg_turns", "YlOrRd",
        "Avg Turns Heatmap — Strategy × Noise Level\n(English 5-letter, 100 games)",
        "heatmap_avg_turns.pdf"
    )
    heatmap_strategy_noise(
        "avg_time", "PuBuGn",
        "Avg Computation Time Heatmap — Strategy × Noise Level\n(English 5-letter, 100 games)",
        "heatmap_avg_time.pdf"
    )
    heatmap_length_noise(
        "avg_turns", "YlOrRd",
        "Avg Turns: Word Length × Noise Level  (English dictionary)",
        "heatmap_length_noise_turns.pdf",
        dict_type="english"
    )


if __name__ == "__main__":
    main()
