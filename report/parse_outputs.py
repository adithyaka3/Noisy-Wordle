"""
parse_outputs.py
────────────────
Shared parsing utilities for Noisy-Wordle output files.

Every output file has the same structure:
  • One header block (dataset / noise model info)
  • 100 ITERATION blocks, each containing 4 STRATEGY blocks
  • Each STRATEGY block has TURN sub-blocks and ends with RESOLUTION ACHIEVED

Parsed data returned:
    {
        'turns'  : [int, ...]   # turns taken per game (100 entries)
        'times'  : [float, ...] # total wall-clock time per game (100 entries)
        'gaps'   : [[float,...], ...] # leader-gap per turn, per game
    }
"""

import re
import os
from collections import defaultdict

# ── label maps ──────────────────────────────────────────────────────────────
STRATEGY_NAMES = {
    1: "Greedy Parallel Trie",
    2: "Thompson Sampling",
    3: "GPU Accelerated Unique Bins",
    4: "POMCP Deep Search",
}

STRATEGY_SHORT = {
    1: "Greedy\nParallel Trie",
    2: "Thompson\nSampling",
    3: "GPU Accel.\nUnique Bins",
    4: "POMCP\nDeep Search",
}

NOISE_CONFIGS = ["100_0_0", "90_5_5", "80_10_10", "70_15_15", "60_20_20"]
NOISE_LABELS  = ["100/0/0 (Clean)", "90/5/5", "80/10/10", "70/15/15", "60/20/20 (Noisy)"]

DICT_TYPES = ["english", "random"]
WORD_LENGTHS = [5, 6, 7, 8]

# ── core parser ──────────────────────────────────────────────────────────────
def parse_output_file(filepath: str) -> dict:
    """
    Parse one output file and return per-strategy data.

    Returns:
        {
            strategy_id (int 1-4): {
                'turns': [int, ...],
                'times': [float, ...],
                'gaps' : [ [float, ...], ... ]   # list of per-game gap progressions
            }
        }
    """
    data = {sid: {"turns": [], "times": [], "gaps": []} for sid in range(1, 5)}

    current_strategy = None
    current_turns    = 0
    current_gaps     = []

    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            # ── strategy block start ─────────────────────────────────────
            m = re.search(r"RUNNING STRATEGY:\s*(\d+)\.", line)
            if m:
                current_strategy = int(m.group(1))
                current_turns    = 0
                current_gaps     = []
                continue

            if current_strategy is None:
                continue

            # ── turn counter ─────────────────────────────────────────────
            m = re.search(r"--- TURN (\d+) ---", line)
            if m:
                current_turns = int(m.group(1))
                continue

            # ── leader-gap metric (strategies 1 & 2) ────────────────────
            m = re.search(r"Leader Gap is ([\d.]+)", line)
            if m:
                current_gaps.append(float(m.group(1)))
                continue

            # ── resolution / end-of-strategy block ──────────────────────
            m = re.search(r"Total Computation Time:\s*([\d.]+)", line)
            if m:
                time_val = float(m.group(1))
                data[current_strategy]["turns"].append(current_turns)
                data[current_strategy]["times"].append(time_val)
                if current_strategy in (1, 2):
                    data[current_strategy]["gaps"].append(current_gaps[:])
                current_strategy = None   # wait for next RUNNING STRATEGY line
                continue

    return data


# ── fast parser (turns + times only, skips Leader Gap) ──────────────────────
def parse_output_file_fast(filepath: str) -> dict:
    """
    Lightweight version of parse_output_file.
    Only extracts turns and times; skips Leader Gap collection.
    ~4-6x faster on large files.

    Returns:
        {strategy_id: {'turns': [...], 'times': [...]}}
    """
    data = {sid: {"turns": [], "times": []} for sid in range(1, 5)}
    strat_re  = re.compile(r"RUNNING STRATEGY:\s*(\d+)\.")
    turn_re   = re.compile(r"--- TURN (\d+) ---")
    time_re   = re.compile(r"Total Computation Time:\s*([\d.]+)")

    current_strategy = None
    current_turns    = 0

    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = strat_re.search(line)
            if m:
                current_strategy = int(m.group(1))
                current_turns    = 0
                continue
            if current_strategy is None:
                continue
            m = turn_re.search(line)
            if m:
                current_turns = int(m.group(1))
                continue
            m = time_re.search(line)
            if m:
                data[current_strategy]["turns"].append(current_turns)
                data[current_strategy]["times"].append(float(m.group(1)))
                current_strategy = None
                continue
    return data


def load_all(outputs_root: str = None, fast: bool = False) -> dict:
    """
    Load every (noise_config, dict_type, word_length) combination.

    Args:
        fast: if True uses the faster parser (no gap data).

    Returns:
        {(noise_config, dict_type, word_length): parsed_data_dict}
    """
    if outputs_root is None:
        here = os.path.dirname(os.path.abspath(__file__))
        outputs_root = os.path.join(here, "..", "outputs")

    parser = parse_output_file_fast if fast else parse_output_file
    results = {}
    for nc in NOISE_CONFIGS:
        folder = os.path.join(outputs_root, nc)
        if not os.path.isdir(folder):
            continue
        for dt in DICT_TYPES:
            for wl in WORD_LENGTHS:
                fname = f"output_{dt}{wl}.txt"
                fpath = os.path.join(folder, fname)
                if not os.path.isfile(fpath):
                    continue
                results[(nc, dt, wl)] = parser(fpath)
    return results
