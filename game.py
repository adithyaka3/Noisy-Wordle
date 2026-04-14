import sys
import json
import random
from strategies.sprt_greedyLL_parallel_trie import play_game as play_game_trie, build_trie
from strategies.sprt_thompson import play_game as play_game_thompson
from strategies.sprt_unique_gpu import play_msprt_game as play_game_gpu
from strategies.pomcp import play_msprt_game as play_game_pomcp

DATASET_FILE = "datasets/english5.json"

def main():
    p_correct, p_wrong1, p_wrong2 = 60, 20, 20
    dataset_file = DATASET_FILE
    
    args = sys.argv[1:]
    if len(args) == 3 and all(float(a) >= 0 for a in args):
        p_correct, p_wrong1, p_wrong2 = map(float, args)
    elif len(args) == 4 and all(float(a) >= 0 for a in args[1:]):
        dataset_file = args[0]
        p_correct, p_wrong1, p_wrong2 = map(float, args[1:])
    elif len(args) == 1:
        dataset_file = args[0]

    p_c = p_correct / 100.0
    p_w = p_wrong1 / 100.0

    if abs(p_correct + p_wrong1 + p_wrong2 - 100) > 0.01:
        print("Warning: Probability values do not sum to 100%.")

    with open(dataset_file, "r") as f:
        DICTIONARY = json.load(f)

    ITERATIONS = 100 # Set the number of test iterations here
    MAX_TURNS = 100 # Maximum allowed turns before forced timeout
    
    print("============================================================")
    print("                 NOISY WORDLE TEST SUITE                      ")
    print("============================================================")
    print(f"System: Using dataset '{dataset_file}' with {len(DICTIONARY)} words.")
    print(f"System: Noise Model - P(Correct): {p_correct}%, P(Wrong 1): {p_wrong1}%, P(Wrong 2): {p_wrong2}%")
    print("System: Building Shared Lexical Trie...")
    TRIE_ROOT = build_trie(DICTIONARY)

    results = {
        "1. Greedy Parallel Trie": {"turns": [], "time": [], "success": 0},
        "2. Thompson Sampling":    {"turns": [], "time": [], "success": 0},
        "3. GPU Accelerated Unique Bins": {"turns": [], "time": [], "success": 0},
        "4. POMCP Deep Search":    {"turns": [], "time": [], "success": 0},
    }

    strategies = [
        ("1. Greedy Parallel Trie", lambda word: play_game_trie(word, DICTIONARY, TRIE_ROOT, p_correct=p_c, p_wrong=p_w, max_turns=MAX_TURNS)),
        ("2. Thompson Sampling", lambda word: play_game_thompson(word, DICTIONARY, TRIE_ROOT, p_correct=p_c, p_wrong=p_w, max_turns=MAX_TURNS)),
        ("3. GPU Accelerated Unique Bins", lambda word: play_game_gpu(word, DICTIONARY, p_correct=p_c, p_wrong=p_w, max_turns=MAX_TURNS)),
        ("4. POMCP Deep Search", lambda word: play_game_pomcp(word, DICTIONARY, p_correct=p_c, p_wrong=p_w, max_turns=MAX_TURNS))
    ]

    for i in range(1, ITERATIONS + 1):
        target_word = random.choice(DICTIONARY)
        print("\n\n" + "#" * 60)
        print(f"ITERATION {i}/{ITERATIONS} - TARGET WORD: {target_word.upper()}")
        print("#" * 60 + "\n")
        
        for name, run_func in strategies:
            print(f"\n>>> RUNNING STRATEGY: {name} <<<")
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
