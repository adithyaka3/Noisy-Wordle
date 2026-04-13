import json
import math
import random
import os
import string
import time

# --- 1. CONFIGURATION ---
DATASET_FILE = "wordle_dataset_5000.json"

# Clean text representation instead of emojis
FEEDBACK_STR = {0: "[GRAY]", 1: "[YELLOW]", 2: "[GREEN]"}

P_CORRECT = 0.6
P_WRONG = 0.2
MAX_ROUND = 100

# SPRT Thresholds
ALPHA = 0.01  # 99% Confidence target
THRESHOLD = math.log((1 - ALPHA) / ALPHA) # ~4.59

# --- 2. DATASET HANDLING ---
def ensure_dataset(num_words=10000, length=7):
    """Loads the dataset, or generates it if it doesn't exist."""
    if os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, 'r') as f:
            return json.load(f)
            
    print(f"System: Dataset not found. Generating {num_words} random {length}-letter words...")
    dataset = set()
    while len(dataset) < num_words:
        word = ''.join(random.choices(string.ascii_lowercase, k=length))
        dataset.add(word)
        
    word_list = list(dataset)
    with open(DATASET_FILE, 'w') as f:
        json.dump(word_list, f, indent=4)
        
    print(f"System: Saved {len(word_list)} words to {DATASET_FILE}")
    return word_list

# --- 3. GAME MECHANICS ---
def calculate_true_feedback(guess, target):
    """Calculates standard Wordle feedback: 0=Gray, 1=Yellow, 2=Green"""
    feedback = [0] * 5
    target_counts = {c: target.count(c) for c in set(target)}
    
    for i in range(5):
        if guess[i] == target[i]:
            feedback[i] = 2
            target_counts[guess[i]] -= 1
            
    for i in range(5):
        if feedback[i] == 0 and guess[i] in target_counts and target_counts[guess[i]] > 0:
            feedback[i] = 1
            target_counts[guess[i]] -= 1
            
    return tuple(feedback)

def apply_noise(true_feedback):
    """Applies the symmetric noise."""
    obs_feedback = list(true_feedback)
    for i in range(5):
        r = random.random()
        if r > P_CORRECT:
            # Pick one of the other two colors randomly
            other_colors = [c for c in [0, 1, 2] if c != true_feedback[i]]
            obs_feedback[i] = other_colors[0] if r < P_CORRECT + P_WRONG else other_colors[1]
    return tuple(obs_feedback)

# --- 4. MULTIPLE HYPOTHESIS TESTING ---
def get_best_discriminator_guess(ll_dict, dictionary):
    """Finds the guess that best splits the leading hypotheses."""
    sorted_words = sorted(ll_dict.keys(), key=lambda w: ll_dict[w], reverse=True)
    best_ll = ll_dict[sorted_words[0]]
    contenders = [w for w in sorted_words if best_ll - ll_dict[w] < 5.0]
    
    if len(contenders) == 1:
        return contenders[0]
        
    best_guess = None
    max_unique_splits = -1
    
    for guess in dictionary:
        feedbacks = set(calculate_true_feedback(guess, w) for w in contenders)
        if len(feedbacks) > max_unique_splits:
            max_unique_splits = len(feedbacks)
            best_guess = guess
            
    return best_guess

def update_log_likelihoods(ll_dict, guess, obs_fb, dictionary):
    """Adds the log-likelihood of the observation to each word's running total."""
    for word in dictionary:
        true_fb = calculate_true_feedback(guess, word)
        
        for o, t in zip(obs_fb, true_fb):
            if o == t:
                ll_dict[word] += math.log(P_CORRECT)
            else:
                ll_dict[word] += math.log(P_WRONG)

# --- 5. GAME SIMULATION ---
def play_msprt_game(target_word, dictionary):
    print("=" * 60)
    print(f"GAME INITIALIZATION (LINEAR SCAN ALGORITHM)")
    print(f"Dictionary Size: {len(dictionary)} words")
    print(f"Target Word:     {target_word.upper()}")
    print("=" * 60)
    
    # Initialize Log-Likelihoods to 0.0 for all words
    ll_dict = {w: 0.0 for w in dictionary}
    total_game_time = 0.0
    
    turn = 0
    while True:
        turn += 1
        print(f"\n--- TURN {turn:02d} ---")
        
        # Timing the Guessing phase
        t0 = time.time()
        guess = get_best_discriminator_guess(ll_dict, dictionary)
        t_guess = time.time() - t0
        
        true_fb = calculate_true_feedback(guess, target_word)
        obs_fb = apply_noise(true_fb)
        
        print(f"Action:       AI Guesses '{guess.upper()}'")
        print(f"True Signal:  {' '.join([FEEDBACK_STR[c] for c in true_fb])}")
        print(f"Noisy Sensor: {' '.join([FEEDBACK_STR[c] for c in obs_fb])}")
        
        # Timing the Update phase
        t1 = time.time()
        update_log_likelihoods(ll_dict, guess, obs_fb, dictionary)
        t_update = time.time() - t1
        
        total_game_time += (t_guess + t_update)
        
        ranked = sorted(ll_dict.items(), key=lambda x: x[1], reverse=True)
        top1, ll1 = ranked[0]
        top2, ll2 = ranked[1]
        gap = ll1 - ll2
        
        print("\nStatus:       Top 3 Hypotheses")
        for i in range(3):
            print(f"              {i+1}. {ranked[i][0].upper():<6} | LL: {ranked[i][1]:>6.2f}")
            
        print(f"\nMetrics:      Current LLR Gap is {gap:.2f} / {THRESHOLD:.2f}")
        print(f"Performance:  Guess calculated in {t_guess:.4f}s | Update computed in {t_update:.4f}s")
        
        if gap >= THRESHOLD:
            print("\n" + "=" * 60)
            print("RESOLUTION ACHIEVED")
            print(f"Target identified as {top1.upper()} on Turn {turn}.")
            print(f"Total Computation Time: {total_game_time:.4f} seconds.")
            print("=" * 60)
            break
            
        if turn >= MAX_ROUND:
            print("\nSystem: Reached safety limit. The noise was too persistent.")
            break

if __name__ == "__main__":
    DICTIONARY = ensure_dataset()
    target = random.choice(DICTIONARY)
    play_msprt_game(target, DICTIONARY)
