import json
import math
import random
import os
import string
import time
from multiprocessing import Pool, cpu_count

DATASET_FILE = "wordle_dataset.json"
FEEDBACK_STR = {0: "⬛", 1: "🟨", 2: "🟩"}
P_CORRECT = 0.6
P_WRONG = 0.2
MAX_ROUND = 100
ALPHA = 0.01
THRESHOLD = math.log((1 - ALPHA) / ALPHA)

def calculate_true_feedback(guess, target):
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
    obs_feedback = list(true_feedback)
    for i in range(5):
        r = random.random()
        if r > P_CORRECT:
            other_colors = [c for c in [0, 1, 2] if c != true_feedback[i]]
            obs_feedback[i] = other_colors[0] if r < P_CORRECT + P_WRONG else other_colors[1]
    return tuple(obs_feedback)

# ---------------------------------------------------------
# TRIE AND LLR UPDATE LOGIC
# ---------------------------------------------------------
class TrieNode:
    def __init__(self, prefix=""):
        self.prefix = prefix
        self.children = {}
        self.words = [] 
        self.ll = 0.0   

def build_trie(dictionary):
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

def bfs_worker(args):
    node_dict, guess, obs_fb = args
    updates = {}
    
    prefix = node_dict['prefix']
    words_in_subtree = node_dict['words']
    children_prefixes = node_dict['children']
    current_ll = node_dict['ll']
    
    word_jumps = {}
    for w in words_in_subtree:
        true_fb = calculate_true_feedback(guess, w)
        jump = 0.0
        for o, t in zip(obs_fb, true_fb):
            if o == t:
                jump += math.log(P_CORRECT)
            else:
                jump += math.log(P_WRONG)
        word_jumps[w] = jump

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

def parallel_trie_update(root, guess, obs_fb, pool):
    tasks = []
    for char, child_node in root.children.items():
        node_data = {
            'prefix': child_node.prefix,
            'words': child_node.words,
            'children': [c.prefix for c in child_node.children.values()],
            'll': child_node.ll
        }
        tasks.append((node_data, guess, obs_fb))
    
    results = pool.map(bfs_worker, tasks)
    
    exact_word_jumps = {}
    for prefix_updates, exact_jumps in results:
        exact_word_jumps.update(exact_jumps)
        
        queue = [root.children[k] for k in root.children if root.children[k].prefix in prefix_updates]
        while queue:
            node = queue.pop(0)
            if node.prefix in prefix_updates:
                node.ll = prefix_updates[node.prefix]
            queue.extend(node.children.values())
            
    return exact_word_jumps


# ---------------------------------------------------------
# PARALLEL DISCRIMINATOR GUESS
# ---------------------------------------------------------
def discriminator_worker(args):
    guesses_chunk, contenders = args
    best_guess = None
    max_unique_splits = -1
    for guess in guesses_chunk:
        feedbacks = set(calculate_true_feedback(guess, w) for w in contenders)
        if len(feedbacks) > max_unique_splits:
            max_unique_splits = len(feedbacks)
            best_guess = guess
    return best_guess, max_unique_splits

def parallel_discriminator_guess(ll_dict, dictionary, pool):
    sorted_words = sorted(ll_dict.keys(), key=lambda w: ll_dict[w], reverse=True)
    best_ll = ll_dict[sorted_words[0]]
    contenders = [w for w in sorted_words if best_ll - ll_dict[w] < 5.0]
    
    if len(contenders) == 1:
        return contenders[0]

    # Split dictionary into chunks for the pool
    cores = pool._processes
    chunk_size = math.ceil(len(dictionary) / cores)
    if chunk_size == 0:
        chunk_size = 1
        
    chunks = [dictionary[i:i + chunk_size] for i in range(0, len(dictionary), chunk_size)]
    tasks = [(chunk, contenders) for chunk in chunks]
    
    results = pool.map(discriminator_worker, tasks)
    
    best_guess = None
    global_max = -1
    for local_guess, local_max in results:
        if local_max > global_max:
            global_max = local_max
            best_guess = local_guess
            
    return best_guess


# ---------------------------------------------------------
# AUTONOMOUS GAME LOOP
# ---------------------------------------------------------
def play_game(target_word, dictionary, trie_root):
    print("=" * 60)
    print(f"GAME INITIALIZATION (PARALLEL DISCRIMINATOR TRIE ALGORITHM)")
    print(f"Dictionary Size: {len(dictionary)} words")
    print(f"Target Word:     {target_word.upper()}")
    print(f"Target Gap:      {THRESHOLD:.2f} (99% Confidence)")
    print("=" * 60)
    
    word_lls = {w: 0.0 for w in dictionary}
    total_game_time = 0.0
    
    cores = cpu_count()
    print(f"System: Booting parallel pool with {cores} CPU cores...")
    
    with Pool(processes=cores) as pool:
        for turn in range(1, MAX_ROUND):
            print(f"\n--- TURN {turn:02d} ---")
            
            t0 = time.time()
            guess = parallel_discriminator_guess(word_lls, list(dictionary), pool)
            t_guess = time.time() - t0
            
            true_fb = calculate_true_feedback(guess, target_word)
            obs_fb = apply_noise(true_fb)
            
            print(f"Action:       AI Guesses '{guess.upper()}'")
            print(f"True Signal:  {' '.join([FEEDBACK_STR[c] for c in true_fb])}")
            print(f"Noisy Sensor: {' '.join([FEEDBACK_STR[c] for c in obs_fb])}")
            
            print("Processing:   Running parallel bounds update...")
            t1 = time.time()
            exact_jumps = parallel_trie_update(trie_root, guess, obs_fb, pool)
            
            for w, jump in exact_jumps.items():
                word_lls[w] += jump
            t_update = time.time() - t1
            
            total_game_time += (t_guess + t_update)
            
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
