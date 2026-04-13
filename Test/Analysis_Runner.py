import csv
import time
import random
from noisy_wordle import load_words
from Entropy_Strat_New import run_single_game as run_entropy
from SPRT_Strat import run_single_game_sprt as run_sprt

def run_benchmark(num_games=30, p_corr=0.8, p_wr=0.1):
    targets = load_words("possible_words.txt")
    test_batch = random.choices(targets, k=num_games)
    
    results = []
    
    for i, target in enumerate(test_batch):
        print(f"--- Game {i+1}/{num_games}: {target.upper()} ---")
        
        # 1. Test Entropy
        start_e = time.time()
        turns_e = run_entropy(target, p_corr, p_wr)
        time_e = time.time() - start_e
        
        # 2. Test SPRT
        start_s = time.time()
        turns_s = run_sprt(target, p_corr, p_wr)
        time_s = time.time() - start_s
        
        results.append({
            "Target": target,
            "Entropy_Turns": turns_e,
            "Entropy_Time": round(time_e, 3),
            "SPRT_Turns": turns_s,
            "SPRT_Time": round(time_s, 3)
        })
        print(f"   Entropy: {turns_e} turns ({time_e:.2f}s) | SPRT: {turns_s} turns ({time_s:.2f}s)")

    # Save to CSV for plotting
    with open("strategy_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    # Settings for your IISc Project Analysis
    # Start with a small batch (10) to verify, then 100+ for final report
    run_benchmark(num_games=10, p_corr=0.8, p_wr=0.1)