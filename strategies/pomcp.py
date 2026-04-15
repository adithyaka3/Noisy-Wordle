import json
import math
import random
import os
import string
import time
import numpy as np
from numba import njit, prange

# --- 1. CONFIGURATION ---
DATASET_FILE = "datasets/english5.json"
FEEDBACK_STR = {0: "⬛", 1: "🟨", 2: "🟩"}

P_CORRECT_DEFAULT = 0.6
P_WRONG_DEFAULT = 0.2
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
    """Converts the dictionary into a 2D numpy int8 array for Numba."""
    if not dictionary: return np.zeros((0,0), dtype=np.int8)
    word_len = len(dictionary[0])
    N = len(dictionary)
    matrix = np.zeros((N, word_len), dtype=np.int8)
    for i, word in enumerate(dictionary):
        for j, char in enumerate(word):
            matrix[i, j] = ord(char) - ord('a')
    return matrix

# --- 3. NUMBA JIT PHYSICS ENGINE (THE BLACK BOX) ---
@njit
def generate_noisy_obs_numba(state_arr, action_arr, p_corr, p_wrong):
    """Simulates the physical game environment and noise for POMCP futures."""
    word_len = len(state_arr)
    fb = np.zeros(word_len, dtype=np.int8)
    counts = np.zeros(26, dtype=np.int8)
    
    # Calculate True Greens
    for i in range(word_len):
        if action_arr[i] == state_arr[i]:
            fb[i] = 2
        else:
            counts[state_arr[i]] += 1
            
    # Calculate True Yellows
    for i in range(word_len):
        if fb[i] == 0 and counts[action_arr[i]] > 0:
            fb[i] = 1
            counts[action_arr[i]] -= 1

    # Apply 60/20/20 Noise and Pack into Base-3 Integer
    packed = 0
    mult = 1
    for i in range(word_len):
        if np.random.random() > p_corr:
            r = np.random.random()
            if fb[i] == 0:
                fb[i] = 1 if r < 0.5 else 2
            elif fb[i] == 1:
                fb[i] = 0 if r < 0.5 else 2
            else:
                fb[i] = 0 if r < 0.5 else 1
                
        packed += fb[i] * mult
        mult *= 3
        
    return packed

@njit(parallel=True)
def update_ll_numba(dict_matrix, action_idx, obs_int, ll_array, p_corr, p_wrong):
    """Updates the global exact Log-Likelihood array in parallel."""
    N = dict_matrix.shape[0]
    word_len = dict_matrix.shape[1]
    action_arr = dict_matrix[action_idx]
    log_p_corr = math.log(p_corr)
    log_p_wrong = math.log(p_wrong)

    # Unpack the Base-3 integer back into an array
    obs_arr = np.zeros(word_len, dtype=np.int8)
    temp = obs_int
    for i in range(word_len):
        obs_arr[i] = temp % 3
        temp //= 3

    for i in prange(N):
        state_arr = dict_matrix[i]
        fb = np.zeros(word_len, dtype=np.int8)
        counts = np.zeros(26, dtype=np.int8)
        
        for j in range(word_len):
            if action_arr[j] == state_arr[j]:
                fb[j] = 2
            else:
                counts[state_arr[j]] += 1
                
        for j in range(word_len):
            if fb[j] == 0 and counts[action_arr[j]] > 0:
                fb[j] = 1
                counts[action_arr[j]] -= 1

        for j in range(word_len):
            if obs_arr[j] == fb[j]:
                ll_array[i] += log_p_corr
            else:
                ll_array[i] += log_p_wrong

# --- 4. POMCP TREE STRUCTURE (PYTHON) ---
class HistoryNode:
    def __init__(self):
        self.visits = 0
        self.action_nodes = {}

class ActionNode:
    def __init__(self):
        self.visits = 0
        self.value = 0.0
        self.history_nodes = {}

def simulate(state_idx, h_node, depth, legal_actions, dict_matrix, p_corr, p_wrong):
    """Recursive Monte Carlo Tree Search execution."""
    if depth > 3:
        # Heuristic Rollout: Truncate deep searches and assume it takes 2 more turns
        return -2.0 

    if len(h_node.action_nodes) == 0:
        # Expand the node with the top filtered actions
        for a in legal_actions:
            h_node.action_nodes[a] = ActionNode()
        return -2.0 

    # UCB Selection (Exploration vs Exploitation)
    best_a = -1
    max_ucb = -float('inf')
    C = 2.0 # Exploration constant
    log_h_visits = math.log(h_node.visits + 1)

    for a in legal_actions:
        a_node = h_node.action_nodes[a]
        if a_node.visits == 0:
            ucb = float('inf') + random.random() # Break ties dynamically
        else:
            explore = math.sqrt(log_h_visits / a_node.visits)
            ucb = a_node.value + C * explore

        if ucb > max_ucb:
            max_ucb = ucb
            best_a = a

    a_node = h_node.action_nodes[best_a]

    # Advance the Simulation Physics
    obs_int = generate_noisy_obs_numba(dict_matrix[state_idx], dict_matrix[best_a], p_corr, p_wrong)

    # Reward Calculation (Minimizing turns)
    reward = -1.0
    if state_idx == best_a:
        reward = 0.0 # Future simulation reached a win
    else:
        if obs_int not in a_node.history_nodes:
            a_node.history_nodes[obs_int] = HistoryNode()
        reward += simulate(state_idx, a_node.history_nodes[obs_int], depth + 1, legal_actions, dict_matrix, p_corr, p_wrong)

    # Backpropagate Values
    h_node.visits += 1
    a_node.visits += 1
    a_node.value += (reward - a_node.value) / a_node.visits

    return reward

# --- 5. GAME PHYSICS (REALITY BOUNDARY) ---
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

def apply_noise(true_feedback, p_correct, p_wrong):
    obs_feedback = list(true_feedback)
    word_len = len(true_feedback)
    for i in range(word_len):
        if random.random() > p_correct:
            other_colors = [c for c in [0, 1, 2] if c != true_feedback[i]]
            obs_feedback[i] = other_colors[0] if random.random() < 0.5 else other_colors[1]
    return tuple(obs_feedback)

def pack_tuple_to_int(fb_tuple):
    packed = 0
    mult = 1
    word_len = len(fb_tuple)
    for i in range(word_len):
        packed += fb_tuple[i] * mult
        mult *= 3
    return packed

def get_particles(ll_array, num_particles):
    """Converts exact LLs into Softmax probability and samples a particle bucket."""
    max_ll = np.max(ll_array)
    ll_shifted = ll_array - max_ll
    # Prevent extreme floating point overflow/underflow
    probs = np.where(ll_shifted > -20, np.exp(ll_shifted), 0.0)
    probs /= np.sum(probs)
    
    particles = np.random.choice(len(ll_array), size=num_particles, p=probs)
    return particles

# --- 6. AUTONOMOUS GAME LOOP ---
def play_msprt_game(target_word, dictionary, p_correct=P_CORRECT_DEFAULT, p_wrong=P_WRONG_DEFAULT, max_turns=100):
    print("=" * 60)
    print(f"GAME INITIALIZATION (POMCP DEEP SEARCH)")
    print(f"Dictionary Size: {len(dictionary)} words")
    print(f"Target Word:     {target_word.upper()}")
    print("=" * 60)
    
    # Compile Data for Numba
    print("System: Encoding dictionary and compiling LLVM Kernels...")
    t_compile_start = time.time()
    dict_matrix = encode_dictionary(dictionary)
    ll_array = np.zeros(len(dictionary), dtype=np.float64)
    
    # Dummy run to force JIT compilation
    if len(dictionary) > 1:
        generate_noisy_obs_numba(dict_matrix[0], dict_matrix[1], p_correct, p_wrong)
        update_ll_numba(dict_matrix, 0, 0, ll_array, p_correct, p_wrong)
    print(f"System: Compilation finished in {time.time() - t_compile_start:.2f}s\n")
    
    total_game_time = 0.0
    turn = 0
    
    while True:
        turn += 1
        print(f"--- TURN {turn:02d} ---")
        
        # --- POMCP SEARCH PHASE ---
        t0 = time.time()
        
        # Funnel the Action Space: Only let POMCP branch on the top 50 viable candidates
        M_contenders = min(50, len(dictionary))
        top_indices = np.argsort(ll_array)[::-1][:M_contenders]
        legal_actions = list(top_indices)
        
        # Generate 15,000 Particle Futures based on current belief state
        particles = get_particles(ll_array, num_particles=15000)
        
        # Run Monte Carlo Tree Search
        root = HistoryNode()
        for p in particles:
            simulate(p, root, 0, legal_actions, dict_matrix, p_correct, p_wrong)
            
        # Extract Best Action (Robust selection based on most tree visits)
        best_a = -1
        max_visits = -1
        for a, a_node in root.action_nodes.items():
            if a_node.visits > max_visits:
                max_visits = a_node.visits
                best_a = a
                
        guess_idx = best_a
        guess = dictionary[guess_idx]
        t_guess = time.time() - t0
        
        # --- PHYSICS PHASE ---
        true_fb = calculate_true_feedback_py(guess, target_word)
        obs_fb = apply_noise(true_fb, p_correct, p_wrong)
        
        print(f"Action:       AI Guesses '{guess.upper()}' (Deep Search Visits: {max_visits})")
        print(f"True Signal:  {' '.join([FEEDBACK_STR[c] for c in true_fb])}")
        print(f"Noisy Sensor: {' '.join([FEEDBACK_STR[c] for c in obs_fb])}")
        
        # --- UPDATE PHASE ---
        t1 = time.time()
        obs_int = pack_tuple_to_int(obs_fb)
        update_ll_numba(dict_matrix, guess_idx, obs_int, ll_array, p_correct, p_wrong)
        t_update = time.time() - t1
        
        total_game_time += (t_guess + t_update)
        
        # --- LEADERBOARD ---
        top_indices = np.argsort(ll_array)[::-1][:3]
        top1_idx, top2_idx = top_indices[0], top_indices[1]
        gap = ll_array[top1_idx] - ll_array[top2_idx]
        
        print("Status:       Top 3 Hypotheses by Log-Likelihood")
        for rank, idx in enumerate(top_indices):
            print(f"              {rank+1}. {dictionary[idx].upper():<7} | LL: {ll_array[idx]:>6.2f}")
            
        print(f"Metrics:      Current LLR Gap is {gap:.2f} / {THRESHOLD:.2f}")
        print(f"Performance:  POMCP sim computed in {t_guess:.4f}s | Update computed in {t_update:.4f}s\n")
        
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