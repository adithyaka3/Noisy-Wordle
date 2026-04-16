"""
generate_all_plots.py
─────────────────────
Master script — run this once to produce every image for the report.

Usage (from the report/ directory, or anywhere):
    source ~/myenv/bin/activate
    python generate_all_plots.py

All images are written to  report/img/
"""

import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ordered: fast-parsing scripts first, then scripts that need gap data last
scripts = [
    # Original plots
    ("plot_guess_distributions.py",    "Guess distributions (2×2 grid per noise config)"),
    ("plot_word_length_scaling.py",    "Scaling: turns + time vs word length"),
    ("plot_dict_comparison.py",        "Dict-type comparison grouped bars"),
    # New insightful plots (all fast-parse)
    ("plot_noise_impact.py",           "Noise impact: avg turns + time vs p_correct"),
    ("plot_speed_accuracy_tradeoff.py","Speed-accuracy scatter tradeoff"),
    ("plot_turn_distributions.py",     "Turn distribution violins across noise"),
    ("plot_heatmap.py",                "Performance heatmaps (strategy×noise, length×noise)"),
    ("plot_english_vs_random.py",      "English vs Random dict grouped bars"),
    # Convergence last (needs full parse for Leader Gap)
    ("plot_convergence.py",            "Convergence CDF (fixed)"),
]

print("=" * 65)
print("  Noisy-Wordle — generating all report figures")
print("=" * 65)

for script, description in scripts:
    path = os.path.join(HERE, script)
    print(f"\n▶  {script}")
    print(f"   {description}")
    result = subprocess.run([sys.executable, path], capture_output=False)
    if result.returncode != 0:
        print(f"   ✗  exited with code {result.returncode}")
    else:
        print(f"   ✓  done")

print("\n" + "=" * 65)
print("  All figures saved to report/img/")
print("=" * 65)
