import json
import math
import random
import os
import string
import time
from multiprocessing import Pool, cpu_count

# --- 1. CONFIGURATION ---
DATASET_FILE = "wordle_dataset.json"

# Colorful text representation instead of text
FEEDBACK_STR = {0: "⬛", 1: "🟨", 2: "🟩"}

# Noise Model: 80% True, 10% Wrong Option A, 10% Wrong Option B
P_CORRECT_DEFAULT = 0.6
P_WRONG_DEFAULT = 0.2
MAX_ROUND = 100

# SPRT Thresholds
ALPHA = 0.01  # 99% Confidence required to declare victory
THRESHOLD = math.log((1 - ALPHA) / ALPHA)

# --- 2. DATASET GENERATION ---
def ensure_dataset(num_words=10000, length=5):
    """Generates the dataset on the fly if it does not exist."""
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
    """Calculates standard deterministic Wordle feedback."""
    word_len = len(target)
    feedback = [0] * word_len
    target_counts = {c: target.count(c) for c in set(target)}
    
    for i in range(word_len):
        if guess[i] == target[i]:
            feedback[i] = 2
            target_counts[guess[i]] -= 1
            
    for i in range(word_len):
        if feedback[i] == 0 and guess[i] in target_counts and target_counts[guess[i]] > 0:
            feedback[i] = 1
            target_counts[guess[i]] -= 1
            
    return tuple(feedback)

def apply_noise(true_feedback, p_correct, p_wrong):
    """Applies the symmetric probability noise model to the feedback."""
    obs_feedback = list(true_feedback)
    word_len = len(true_feedback)
    for i in range(word_len):
        r = random.random()
        if r > p_correct:
            other_colors = [c for c in [0, 1, 2] if c != true_feedback[i]]
            obs_feedback[i] = other_colors[0] if r < p_correct + p_wrong else other_colors[1]
    return tuple(obs_feedback)

# --- 4. TRIE STRUCTURE ---
class TrieNode:
    def __init__(self, prefix=""):
        self.prefix = prefix
        self.children = {}
        self.words = [] # Stores valid dictionary words in this subtree
        self.ll = 0.0   # Upper bound Log-Likelihood

def build_trie(dictionary):
    """Constructs the lexical Trie for Branch-and-Bound."""
    root = TrieNode()
    for word in dictionary:
        current = root
        current.words.append(word)
        prefix = ""
        for char in word:
            prefix += char
            if char not in current.children:
                current.children[char] = TrieNode(prefix)
            current = current.children[char]
            current.words.append(word)
    return root

# --- 5. PARALLEL BFS LOGIC ---
def bfs_worker(args):
    """
    Executed by individual CPU cores.
    Performs a bound update on a specific top-level branch of the Trie.
    """
    node_dict, guess, obs_fb, p_correct, p_wrong = args
    updates = {}
    
    prefix = node_dict['prefix']
    words_in_subtree = node_dict['words']
    children_prefixes = node_dict['children']
    current_ll = node_dict['ll']
    
    # Calculate exact LLR jump for all leaf words in this branch
    word_jumps = {}
    for w in words_in_subtree:
        true_fb = calculate_true_feedback(guess, w)
        jump = 0.0
        for o, t in zip(obs_fb, true_fb):
            if o == t:
                jump += math.log(p_correct)
            else:
                jump += math.log(p_wrong)
        word_jumps[w] = jump

    # Emulate the BFS upper bound update
    max_jump = max(word_jumps.values()) if word_jumps else 0.0
    updates[prefix] = current_ll + max_jump
    
    queue = children_prefixes
    while queue:
        child_prefix = queue.pop(0)
        child_words = [w for w in words_in_subtree if w.startswith(child_prefix)]
        if not child_words: continue
        
        c_max_jump = max(word_jumps[w] for w in child_words)
        updates[child_prefix] = current_ll + c_max_jump 
        
    return updates, word_jumps

def parallel_trie_update(root, guess, obs_fb, pool, p_correct, p_wrong):
    """Distributes the Trie subtrees to the multiprocessing pool."""
    tasks = []
    for char, child_node in root.children.items():
        node_data = {
            'prefix': child_node.prefix,
            'words': child_node.words,
            'children': [c.prefix for c in child_node.children.values()],
            'll': child_node.ll
        }
        tasks.append((node_data, guess, obs_fb, p_correct, p_wrong))
    
    results = pool.map(bfs_worker, tasks)
    
    exact_word_jumps = {}
    for prefix_updates, exact_jumps in results:
        exact_word_jumps.update(exact_jumps)
        
        # Apply updates to the main Trie memory structure
        queue = [root.children[k] for k in root.children if root.children[k].prefix in prefix_updates]
        while queue:
            node = queue.pop(0)
            if node.prefix in prefix_updates:
                node.ll = prefix_updates[node.prefix]
            queue.extend(node.children.values())
            
    return exact_word_jumps

# --- 6. AUTONOMOUS GAME LOOP ---
def play_game(target_word, dictionary, trie_root, p_correct=P_CORRECT_DEFAULT, p_wrong=P_WRONG_DEFAULT, max_turns=100):
    print("=" * 60)
    print(f"GAME INITIALIZATION (PARALLEL TRIE ALGORITHM)")
    print(f"Dictionary Size: {len(dictionary)} words")
    print(f"Target Word:     {target_word.upper()}")
    print(f"Target Gap:      {THRESHOLD:.2f} (99% Confidence)")
    print("=" * 60)
    
    # Track the exact Log-Likelihood for guessing purposes
    word_lls = {w: 0.0 for w in dictionary}
    total_game_time = 0.0
    
    cores = cpu_count()
    print(f"System: Booting parallel pool with {cores} CPU cores...")
    
    with Pool(processes=cores) as pool:
        for turn in range(1, max_turns + 1):
            print(f"\n--- TURN {turn:02d} ---")
            
            # 1. Timing the Guess Phase
            t0 = time.time()
            # The AI takes the greedy approach: guess the current most likely word
            # On Turn 1, this will naturally be a random choice as all LLs are 0.0
            guess = max(word_lls, key=word_lls.get)
            t_guess = time.time() - t0
            
            # Simulate environment physics
            true_fb = calculate_true_feedback(guess, target_word)
            obs_fb = apply_noise(true_fb, p_correct, p_wrong)
            
            print(f"Action:       AI Guesses '{guess.upper()}'")
            print(f"True Signal:  {' '.join([FEEDBACK_STR[c] for c in true_fb])}")
            print(f"Noisy Sensor: {' '.join([FEEDBACK_STR[c] for c in obs_fb])}")
            
            # 2. Timing the Update Phase
            print("Processing:   Running parallel bounds update...")
            t1 = time.time()
            exact_jumps = parallel_trie_update(trie_root, guess, obs_fb, pool, p_correct, p_wrong)
            
            # Update absolute word scores
            for w, jump in exact_jumps.items():
                word_lls[w] += jump
            t_update = time.time() - t1
            
            total_game_time += (t_guess + t_update)
            
            # Evaluate Leaderboard
            ranked = sorted(word_lls.items(), key=lambda x: x[1], reverse=True)
            top1, ll1 = ranked[0]
            top2, ll2 = ranked[1]
            current_gap = ll1 - ll2
            
            print("Status:       Top 3 Hypotheses")
            for i in range(3):
                print(f"              {i+1}. {ranked[i][0].upper()} (LL: {ranked[i][1]:>6.2f})")
                
            print(f"Metric:       Current Leader Gap is {current_gap:.2f} / {THRESHOLD:.2f}")
            print(f"Performance:  Guess calculated in {t_guess:.4f}s | Update computed in {t_update:.4f}s")
            
            if current_gap >= THRESHOLD:
                print("\n" + "=" * 60)
                print("RESOLUTION ACHIEVED")
                print(f"Result: AI has established mathematical dominance.")
                print(f"Target identified as {top1.upper()} on Turn {turn}.")
                print(f"Total Computation Time: {total_game_time:.4f} seconds.")
                if top1 == target_word:
                    print(f"The secret word was {target_word.upper()} and the model has guessed it correctly!")
                else:
                    print(f"The secret word was {target_word.upper()} but the model guessed it incorrectly.")
                print("=" * 60)
                return {"success": top1 == target_word, "turns": turn, "time": total_game_time, "guess": top1}
                
        return {"success": False, "turns": turn, "time": total_game_time, "guess": top1}

if __name__ == '__main__':
    # Initialize the data and data structures
    DICTIONARY = ensure_dataset()
    print("System: Building Lexical Trie...")
    TRIE_ROOT = build_trie(DICTIONARY)
    
    # Run the simulation
    target = random.choice(DICTIONARY)
    play_game(target, DICTIONARY, TRIE_ROOT)
