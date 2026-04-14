import json
import math
import random
import os
import string
import time
import numpy as np
from numba import njit, prange

# --- 1. CONFIGURATION ---
DATASET_FILE = "english5.json"

FEEDBACK_STR = {0: "[GRAY]", 1: "[YELLOW]", 2: "[GREEN]"}

P_CORRECT = 0.6
P_WRONG = 0.2
MAX_ROUND = 100

ALPHA = 0.01  
THRESHOLD = math.log((1 - ALPHA) / ALPHA)

# --- 2. DATASET HANDLING & ENCODING ---
def ensure_dataset(num_words=10000, length=5):
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

def encode_dictionary(dictionary):
    """Converts a list of strings into a 2D numpy array of int8."""
    if not dictionary: return np.zeros((0, 0), dtype=np.int8)
    word_len = len(dictionary[0])
    N = len(dictionary)
    matrix = np.zeros((N, word_len), dtype=np.int8)
    for i, word in enumerate(dictionary):
        for j, char in enumerate(word):
            matrix[i, j] = ord(char) - ord('a')
    return matrix

# --- 3. NUMBA JIT PARALLEL KERNELS ---
@njit
def get_feedback_int(guess, target):
    """Calculates feedback and packs the 7-tile sequence into a single Base-3 integer."""
    word_len = len(guess)
    fb = np.zeros(word_len, dtype=np.int8)
    counts = np.zeros(26, dtype=np.int8)
    
    # Pass 1: Greens
    for i in range(word_len):
        if guess[i] == target[i]:
            fb[i] = 2
        else:
            counts[target[i]] += 1
            
    # Pass 2: Yellows
    for i in range(word_len):
        if fb[i] == 0 and counts[guess[i]] > 0:
            fb[i] = 1
            counts[guess[i]] -= 1
            
    # Pack array into a single int (Base 3)
    packed = 0
    mult = 1
    for i in range(word_len):
        packed += fb[i] * mult
        mult *= 3
    return packed

@njit(parallel=True)
def find_best_discriminator_numba(dict_matrix, contender_indices):
    """Parallelized loop to find the guess that produces the most unique feedbacks."""
    N = dict_matrix.shape[0]
    C = contender_indices.shape[0]
    num_patterns = 3 ** dict_matrix.shape[1]
    
    split_counts = np.zeros(N, dtype=np.int32)
    
    # prange distributes this loop across all CPU cores automatically
    for i in prange(N):
        guess = dict_matrix[i]
        # Track seen patterns dynamically based on word length.
        seen = np.zeros(num_patterns, dtype=np.bool_) 
        unique_count = 0
        
        for j in range(C):
            target = dict_matrix[contender_indices[j]]
            fb_int = get_feedback_int(guess, target)
            if not seen[fb_int]:
                seen[fb_int] = True
                unique_count += 1
                
        split_counts[i] = unique_count
        
    best_idx = 0
    max_splits = -1
    for i in range(N):
        if split_counts[i] > max_splits:
            max_splits = split_counts[i]
            best_idx = i
            
    return best_idx

@njit(parallel=True)
def update_log_likelihoods_numba(dict_matrix, guess_idx, obs_fb_array, ll_array, p_corr, p_wrong):
    """Parallelized log-likelihood update."""
    N = dict_matrix.shape[0]
    word_len = dict_matrix.shape[1]
    guess = dict_matrix[guess_idx]
    log_p_corr = math.log(p_corr)
    log_p_wrong = math.log(p_wrong)
    
    for i in prange(N):
        target = dict_matrix[i]
        
        # Inline feedback calculation for speed
        fb = np.zeros(word_len, dtype=np.int8)
        counts = np.zeros(26, dtype=np.int8)
        for j in range(word_len):
            if guess[j] == target[j]:
                fb[j] = 2
            else:
                counts[target[j]] += 1
        for j in range(word_len):
            if fb[j] == 0 and counts[guess[j]] > 0:
                fb[j] = 1
                counts[guess[j]] -= 1
                
        # Apply the log jump based on the noisy observation
        for j in range(word_len):
            if obs_fb_array[j] == fb[j]:
                ll_array[i] += log_p_corr
            else:
                ll_array[i] += log_p_wrong

# --- 4. GAME PHYSICS (PYTHON BOUNDARY) ---
def apply_noise(true_feedback):
    obs_feedback = list(true_feedback)
    word_len = len(true_feedback)
    for i in range(word_len):
        if random.random() > P_CORRECT:
            other_colors = [c for c in [0, 1, 2] if c != true_feedback[i]]
            obs_feedback[i] = other_colors[0] if random.random() < 0.5 else other_colors[1]
    return tuple(obs_feedback)

def calculate_true_feedback_py(guess_str, target_str):
    word_len = len(target_str)
    feedback = [0] * word_len
    target_counts = {c: target_str.count(c) for c in set(target_str)}
    
    for i in range(word_len):
        if guess_str[i] == target_str[i]:
            feedback[i] = 2
            target_counts[guess_str[i]] -= 1
            
    for i in range(word_len):
        if feedback[i] == 0 and target_counts.get(guess_str[i], 0) > 0:
            feedback[i] = 1
            target_counts[guess_str[i]] -= 1
            
    return tuple(feedback)

# --- 5. GAME SIMULATION ---
def play_msprt_game(target_word, dictionary, max_turns=100):
    print("=" * 60)
    print(f"GAME INITIALIZATION (NUMBA JIT PARALLEL)")
    print(f"Dictionary Size: {len(dictionary)} words")
    print(f"Target Word:     {target_word.upper()}")
    print("=" * 60)
    
    # 1. Compile Data for Numba
    print("System: Encoding dictionary and compiling LLVM Kernels...")
    t_compile_start = time.time()
    word_len = len(dictionary[0]) if dictionary else 5
    dict_matrix = encode_dictionary(dictionary)
    ll_array = np.zeros(len(dictionary), dtype=np.float64)
    
    # Trigger a dummy run to force Numba to compile the C-code before the clock starts
    if len(dictionary) > 0:
        find_best_discriminator_numba(dict_matrix[:10], np.array([0], dtype=np.int32))
        update_log_likelihoods_numba(dict_matrix[:10], 0, np.zeros(word_len, dtype=np.int8), np.zeros(10, dtype=np.float64), P_CORRECT, P_WRONG)
    print(f"System: Compilation finished in {time.time() - t_compile_start:.2f}s\n")
    
    total_game_time = 0.0
    turn = 0
    
    while True:
        turn += 1
        print(f"--- TURN {turn:02d} ---")
        
        # --- GUESS PHASE ---
        t0 = time.time()
        
        # Find contenders (Python -> Numba bridge)
        best_ll = np.max(ll_array)
        contender_indices = np.where(best_ll - ll_array < 5.0)[0].astype(np.int32)
        
        if len(contender_indices) == 1:
            best_idx = contender_indices[0]
        else:
            # Execute C-level parallel search
            best_idx = find_best_discriminator_numba(dict_matrix, contender_indices)
            
        guess = dictionary[best_idx]
        t_guess = time.time() - t0
        
        # --- PHYSICS PHASE ---
        true_fb = calculate_true_feedback_py(guess, target_word)
        obs_fb = apply_noise(true_fb)
        
        print(f"Action:       AI Guesses '{guess.upper()}'")
        print(f"True Signal:  {' '.join([FEEDBACK_STR[c] for c in true_fb])}")
        print(f"Noisy Sensor: {' '.join([FEEDBACK_STR[c] for c in obs_fb])}")
        
        # --- UPDATE PHASE ---
        t1 = time.time()
        obs_fb_array = np.array(obs_fb, dtype=np.int8)
        update_log_likelihoods_numba(dict_matrix, best_idx, obs_fb_array, ll_array, P_CORRECT, P_WRONG)
        t_update = time.time() - t1
        
        total_game_time += (t_guess + t_update)
        
        # --- LEADERBOARD ---
        top_indices = np.argsort(ll_array)[::-1][:3]
        top1_idx, top2_idx = top_indices[0], top_indices[1]
        gap = ll_array[top1_idx] - ll_array[top2_idx]
        
        print("Status:       Top 3 Hypotheses")
        for rank, idx in enumerate(top_indices):
            print(f"              {rank+1}. {dictionary[idx].upper():<7} | LL: {ll_array[idx]:>6.2f}")
            
        print(f"Metrics:      Current LLR Gap is {gap:.2f} / {THRESHOLD:.2f}")
        print(f"Performance:  Guess calculated in {t_guess:.4f}s | Update computed in {t_update:.4f}s\n")
        
        if gap >= THRESHOLD:
            print("=" * 60)
            print("RESOLUTION ACHIEVED")
            print(f"Total Computation Time: {total_game_time:.4f} seconds.")
            top1 = dictionary[top1_idx]
            if top1 == target_word:
                print(f"The secret word was {target_word.upper()} and the model has guessed it correctly!")
            else:
                print(f"The secret word was {target_word.upper()} but the model guessed it incorrectly.")
            print("=" * 60)
            return {"success": top1 == target_word, "turns": turn, "time": total_game_time, "guess": top1}
            
        if turn >= max_turns:
            print("System: Reached safety limit. The noise was too persistent.")
            top1 = dictionary[top1_idx]
            return {"success": False, "turns": turn, "time": total_game_time, "guess": top1}

if __name__ == "__main__":
    DICTIONARY = ensure_dataset()
    target = random.choice(DICTIONARY)
    play_msprt_game(target, DICTIONARY)