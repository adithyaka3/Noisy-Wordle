import sys
import json
import random
import os

class SuppressOutput:
    def __init__(self, suppress=True):
        self.suppress = suppress
    def __enter__(self):
        if self.suppress:
            self.old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.suppress:
            sys.stdout.close()
            sys.stdout = self.old_stdout

from strategies.sprt_greedyLL_parallel_trie import play_game as play_game_trie, build_trie
from strategies.sprt_thompson import play_game as play_game_thompson
from strategies.sprt_unique_gpu import play_msprt_game as play_game_gpu
from strategies.pomcp import play_msprt_game as play_game_pomcp

DATASET_FILE = "datasets/english5.json"

def main():
    dataset_file = sys.argv[1] if len(sys.argv) > 1 else DATASET_FILE
    with open(dataset_file, "r") as f:
        DICTIONARY = json.load(f)

    ITERATIONS = 100 # Set the number of test iterations here
    MAX_TURNS = 100 # Maximum allowed turns before forced timeout
    
    print("============================================================")
    print("                 NOISY WORDLE TEST SUITE                      ")
    print("============================================================")
    print(f"System: Using dataset '{dataset_file}' with {len(DICTIONARY)} words.")
    print("System: Building Shared Lexical Trie...")
    TRIE_ROOT = build_trie(DICTIONARY)

    results = {
        "1. Greedy Parallel Trie": {"turns": [], "time": [], "success": 0},
        "2. Thompson Sampling":    {"turns": [], "time": [], "success": 0},
        "3. GPU Accelerated Unique Bins": {"turns": [], "time": [], "success": 0},
        "4. POMCP Deep Search":    {"turns": [], "time": [], "success": 0},
    }

    strategies = [
        ("1. Greedy Parallel Trie", lambda word: play_game_trie(word, DICTIONARY, TRIE_ROOT, max_turns=MAX_TURNS)),
        ("2. Thompson Sampling", lambda word: play_game_thompson(word, DICTIONARY, TRIE_ROOT, max_turns=MAX_TURNS)),
        ("3. GPU Accelerated Unique Bins", lambda word: play_game_gpu(word, DICTIONARY, max_turns=MAX_TURNS)),
        ("4. POMCP Deep Search", lambda word: play_game_pomcp(word, DICTIONARY, max_turns=MAX_TURNS))
    ]

    if ITERATIONS > 1:
        print(f"Benchmarking silently over {ITERATIONS} scale factors. Please hold...")

    for i in range(1, ITERATIONS + 1):
        target_word = random.choice(DICTIONARY)
        
        if ITERATIONS == 1:
            print("\n\n" + "#" * 60)
            print(f"ITERATION {i}/{ITERATIONS} - TARGET WORD: {target_word.upper()}")
            print("#" * 60 + "\n")

        
        for name, run_func in strategies:
            if ITERATIONS == 1:
                print(f"\n>>> RUNNING STRATEGY: {name} <<<")
                
            with SuppressOutput(suppress=(ITERATIONS > 1)):
                res = run_func(target_word)
            
            results[name]["turns"].append(res["turns"])
            results[name]["time"].append(res["time"])
            if res["success"]:
                results[name]["success"] += 1
                
    print("\n\n" + "=" * 80)
    print("                   FINAL BENCHMARK AVERAGES                 ")
    print(f"                             (Over {ITERATIONS} Iterations)            ")
    print("=" * 80)
    print(f"{'Strategy':<35} | {'Success Rate':<12} | {'Avg Turns':<10} | {'Avg Time (s)':<12}")
    print("-" * 80)
    
    for name, _ in strategies:
        data = results[name]
        succ_rate = (data["success"] / ITERATIONS) * 100
        avg_turns = sum(data["turns"]) / len(data["turns"]) if data["turns"] else 0
        avg_time = sum(data["time"]) / len(data["time"]) if data["time"] else 0
        
        print(f"{name:<35} | {succ_rate:>8.1f}%   | {avg_turns:<10.2f} | {avg_time:<12.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
