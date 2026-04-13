import csv
import time
import math
import multiprocessing
from noisy_wordle import load_words
from Entropy_Strat_New import calculate_expected_entropy

# --- CONFIGURATION ---
ALLOWED_WORDS_FILE = "allowed_words.txt"
POSSIBLE_WORDS_FILE = "possible_words.txt"
OUTPUT_FILE = "best_openers_baseline.csv"

# Baseline: 0% Noise to replicate the video results
P_CORRECT = 0.8
P_WRONG = 0.1

def compute_info_gain_worker(args):
    """Worker function to calculate Information Gain for a single word."""
    guess, h_prior, beliefs, possible_words = args
    
    # Calculate expected remaining entropy (H_post)
    h_post = calculate_expected_entropy(guess, beliefs, possible_words, P_CORRECT, P_WRONG)
    
    # Information Gain = How many bits of uncertainty were removed
    info_gain = h_prior - h_post
    return (guess, info_gain)

def run_opener_analysis():
    # 1. Load Dictionaries
    allowed_guesses = load_words(ALLOWED_WORDS_FILE)
    possible_answers = load_words(POSSIBLE_WORDS_FILE)
    
    if not allowed_guesses or not possible_answers:
        print("Error: Word lists are missing or empty.")
        return

    # 2. Setup Initial State (Turn 1)
    num_targets = len(possible_answers)
    initial_prob = 1.0 / num_targets
    beliefs = {w: initial_prob for w in possible_answers}
    
    # Calculate Initial Entropy: H = log2(N) for uniform distribution
    h_prior = math.log2(num_targets)

    print(f"📊 Dictionary: {len(allowed_guesses)} guesses | {num_targets} targets")
    print(f"📉 Starting Uncertainty: {h_prior:.4f} bits")
    print(f"🧬 Calculating Information Gain (Multiprocessing)...")

    # 3. Parallel Execution
    # Pass h_prior to the worker so it can perform the subtraction
    tasks = [(word, h_prior, beliefs, possible_answers) for word in allowed_guesses]
    
    start_time = time.time()
    with multiprocessing.Pool() as pool:
        results = pool.map(compute_info_gain_worker, tasks)
    
    # 4. Sorting
    # We want the HIGHEST Information Gain first
    results.sort(key=lambda x: x[1], reverse=True)
    
    duration = time.time() - start_time
    print(f"✅ Finished in {duration:.2f} seconds.")

    # 5. Output to Console & CSV
    print("\n🏆 Top 10 Openers by Information Gain:")
    for i, (word, gain) in enumerate(results[:10]):
        print(f"{i+1}. {word.upper():<6} | {gain:.4f} bits")

    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "Word", "Information_Gain_Bits"])
        for i, (word, gain) in enumerate(results):
            writer.writerow([i+1, word, round(gain, 6)])

    print(f"\n📁 Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_opener_analysis()