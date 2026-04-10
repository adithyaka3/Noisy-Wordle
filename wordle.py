import os
import math
import random
import urllib.request
from collections import defaultdict
from typing import List, Tuple, Callable

# 0: Gray, 1: Yellow, 2: Green
Pattern = Tuple[int, int, int, int, int]

def load_wordle_lists() -> Tuple[List[str], List[str]]:
    """Fetches and locally caches the answers list and the allowed guesses list."""
    answers_url = "https://gist.githubusercontent.com/cfreshman/a03ef2cba789d8cf00c08f767e0fad7b/raw/5d752e5f0702da315298a6bb5a771586d6ff445c/wordle-answers-alphabetical.txt"
    guesses_url = "https://gist.githubusercontent.com/cfreshman/cdcdf777450c5b5301e439061d29694c/raw/de1df631b45492e0974f7affe266ce36fb986676/wordle-allowed-guesses.txt"
    
    def get_words(url: str, filename: str) -> List[str]:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                return [w.strip().lower() for w in f.readlines() if len(w.strip()) == 5]
        
        print(f"Downloading {filename}...")
        try:
            response = urllib.request.urlopen(url)
            words = [w.strip().lower() for w in response.read().decode('utf-8').splitlines() if len(w.strip()) == 5]
            with open(filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(words))
            return words
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            return []

    answers = get_words(answers_url, "wordle_answers.txt")
    allowed_guesses = get_words(guesses_url, "wordle_allowed_guesses.txt")
    
    if not answers:
        answers = ["apple", "crane", "roast", "slate", "trace", "batch", "hatch"]
        
    all_valid_guesses = list(set(answers + allowed_guesses))
    print(f"Loaded {len(answers)} answers and {len(all_valid_guesses)} total allowed guesses.\n")
    return answers, all_valid_guesses

def get_pattern(guess: str, secret: str) -> Pattern:
    """Calculates the Wordle feedback pattern for a guess against a secret word."""
    pattern = [0] * 5
    secret_counts = {}
    for char in secret:
        secret_counts[char] = secret_counts.get(char, 0) + 1

    for i in range(5):
        if guess[i] == secret[i]:
            pattern[i] = 2
            secret_counts[guess[i]] -= 1

    for i in range(5):
        if pattern[i] == 0 and guess[i] in secret_counts and secret_counts[guess[i]] > 0:
            pattern[i] = 1
            secret_counts[guess[i]] -= 1

    return tuple(pattern)

# ==========================================
# HEURISTICS
# ==========================================

def calculate_entropy(guess: str, possible_secrets: List[str]) -> float:
    """Calculates Information Gain (Shannon Entropy). Higher is better."""
    pattern_counts = defaultdict(int)
    for secret in possible_secrets:
        pattern_counts[get_pattern(guess, secret)] += 1

    entropy = 0.0
    total_words = len(possible_secrets)
    for count in pattern_counts.values():
        p = count / total_words
        entropy -= p * math.log2(p)
    return entropy

def calculate_expected_size(guess: str, possible_secrets: List[str]) -> float:
    """Calculates Expected Remaining Pool Size. Lower is better."""
    pattern_counts = defaultdict(int)
    for secret in possible_secrets:
        pattern_counts[get_pattern(guess, secret)] += 1

    expected_size = 0.0
    total_words = len(possible_secrets)
    for count in pattern_counts.values():
        # E[Size] = sum( (count/total) * count )
        expected_size += (count * count) / total_words
    return expected_size

# ==========================================
# STRATEGIES
# ==========================================

def entropy_strategy(all_words: List[str], possible_secrets: List[str], total_initial_answers: int) -> str:
    if len(possible_secrets) == total_initial_answers: return "roast"
    if len(possible_secrets) == 1: return possible_secrets[0]

    best_guess = ""
    max_entropy = -1.0
    for guess in all_words:
        entropy = calculate_entropy(guess, possible_secrets)
        if entropy > max_entropy:
            max_entropy = entropy
            best_guess = guess
    return best_guess

def expected_size_strategy(all_words: List[str], possible_secrets: List[str], total_initial_answers: int) -> str:
    if len(possible_secrets) == total_initial_answers: return "roast"
    if len(possible_secrets) == 1: return possible_secrets[0]

    best_guess = ""
    min_exp_size = float('inf')
    for guess in all_words:
        exp_size = calculate_expected_size(guess, possible_secrets)
        # Tie-breaker: If they yield the same size, prefer words that could actually be the answer
        if exp_size < min_exp_size or (exp_size == min_exp_size and guess in possible_secrets):
            min_exp_size = exp_size
            best_guess = guess
    return best_guess

def aco_expected_size_strategy(all_words: List[str], possible_secrets: List[str], total_initial_answers: int) -> str:
    """Stochastic search using Ant Colony Optimization concepts based on Expected Size."""
    if len(possible_secrets) == total_initial_answers: return "roast"
    if len(possible_secrets) == 1: return possible_secrets[0]

    # ACO Parameters
    num_ants = 15
    iterations = 3
    evaporation_rate = 0.4
    
    # Prune search space: Ants will explore all remaining valid answers + a random sample of 300 other guesses
    sample_size = min(300, len(all_words))
    candidates = list(set(possible_secrets + random.sample(all_words, sample_size)))
    
    # Initialize pheromones
    pheromones = {word: 1.0 for word in candidates}
    
    best_overall_guess = ""
    min_overall_exp_size = float('inf')

    for _ in range(iterations):
        ant_results = []
        
        # 1. Ants construct solutions probabilistically
        total_pheromone = sum(pheromones.values())
        probs = [pheromones[w] / total_pheromone for w in candidates]
        
        for ant in range(num_ants):
            chosen_word = random.choices(candidates, weights=probs, k=1)[0]
            exp_size = calculate_expected_size(chosen_word, possible_secrets)
            ant_results.append((chosen_word, exp_size))
            
            if exp_size < min_overall_exp_size:
                min_overall_exp_size = exp_size
                best_overall_guess = chosen_word
                
        # 2. Evaporate old pheromones
        for w in candidates:
            pheromones[w] *= (1.0 - evaporation_rate)
            
        # 3. Ants deposit new pheromones (inversely proportional to expected size)
        for word, exp_size in ant_results:
            # Add small epsilon to prevent division by zero
            deposit = 10.0 / (exp_size + 0.01) 
            pheromones[word] += deposit

    return best_overall_guess

# Global cache for AO* Strategy
ao_memo = {}

def ao_star_strategy(all_words: List[str], possible_secrets: List[str], total_initial_answers: int) -> str:
    global ao_memo
    if len(possible_secrets) == total_initial_answers: return "roast"
    if len(possible_secrets) == 1: return possible_secrets[0]

    def ao_cost(secrets_pool: Tuple[str], depth: int) -> float:
        if len(secrets_pool) == 1: return 1.0  
        if len(secrets_pool) == 2: return 1.5 
        if secrets_pool in ao_memo: return ao_memo[secrets_pool]
        if depth == 0: return math.log2(len(secrets_pool))

        best_cost = float('inf')
        for guess in secrets_pool:
            partitions = defaultdict(list)
            for secret in secrets_pool:
                partitions[get_pattern(guess, secret)].append(secret)
                
            expected_cost = 1.0 
            for pattern, partition in partitions.items():
                prob = len(partition) / len(secrets_pool)
                if pattern != (2, 2, 2, 2, 2):
                    expected_cost += prob * ao_cost(tuple(sorted(partition)), depth - 1)
                    
            if expected_cost < best_cost:
                best_cost = expected_cost

        ao_memo[secrets_pool] = best_cost
        return best_cost

    best_guess = ""
    best_expected_cost = float('inf')
    candidate_guesses = all_words if len(possible_secrets) <= 15 else possible_secrets

    for guess in candidate_guesses:
        partitions = defaultdict(list)
        for secret in possible_secrets:
            partitions[get_pattern(guess, secret)].append(secret)
            
        expected_cost = 1.0
        for pattern, partition in partitions.items():
            prob = len(partition) / len(possible_secrets)
            if pattern != (2, 2, 2, 2, 2):
                expected_cost += prob * ao_cost(tuple(sorted(partition)), depth=1)
                
        if expected_cost < best_expected_cost:
            best_expected_cost = expected_cost
            best_guess = guess

    return best_guess

# ==========================================
# UI & GAME ENGINE
# ==========================================

def colorize(word: str, pattern: Pattern) -> str:
    colors = {
        0: "\033[100m\033[97m",  
        1: "\033[43m\033[30m",   
        2: "\033[42m\033[30m"    
    }
    reset = "\033[0m"
    colored_word = ""
    for char, p in zip(word, pattern):
        colored_word += f"{colors[p]} {char.upper()} {reset}"
    return colored_word

def play_wordle(secret: str, all_words: List[str], initial_answers: List[str], strategy: Callable, verbose: bool = True) -> int:
    possible_secrets = initial_answers.copy()
    total_initial_answers = len(initial_answers)
    attempts = 0
    
    if verbose:
        print(f"--- Starting Wordle ---")
        print(f"Secret Word is hidden. Let's see how the solver does!")
        print("-" * 25)
    
    while True:
        attempts += 1
        guess = strategy(all_words, possible_secrets, total_initial_answers)
        pattern = get_pattern(guess, secret)
        
        if verbose:
            colored_output = colorize(guess, pattern)
            print(f"Attempt {attempts}: {colored_output}  |  Possible answers remaining: {len(possible_secrets)}")
        
        if pattern == (2, 2, 2, 2, 2):
            if verbose:
                print("-" * 25)
                print(f"Solved in {attempts} attempts! The word was {secret.upper()}.\n")
            return attempts
            
        possible_secrets = [w for w in possible_secrets if get_pattern(guess, w) == pattern]

def run_experiment(answers_list: List[str], all_guesses_list: List[str], trials: int = 100):
    print(f"\n==========================================")
    print(f"RUNNING EXPERIMENT: {trials} TRIALS")
    print(f"==========================================")
    print("This will take a few minutes. Running silently...")
    
    strategies = {
        "Entropy Strategy": entropy_strategy,
        "Expected Size Strategy": expected_size_strategy,
        # "ACO Expected Size": aco_expected_size_strategy,
        "AO* Search (Depth-Bounded)": ao_star_strategy
    }
    
    results = {name: [] for name in strategies}
    
    # Pick the exact same batch of random secrets for every strategy to ensure a fair fight
    test_secrets = random.choices(answers_list, k=trials)
    
    for i, secret in enumerate(test_secrets):
        if (i + 1) % 5 == 0:
            print(f"Completed {i + 1} / {trials} games...")
            
        for name, strategy in strategies.items():
            attempts = play_wordle(secret, all_guesses_list, answers_list, strategy, verbose=False)
            results[name].append(attempts)

    print("\n==========================================")
    print("FINAL RESULTS (Average Guesses):")
    print("==========================================")
    for name, attempts_list in results.items():
        avg = sum(attempts_list) / len(attempts_list)
        print(f"{name:<30}: {avg:.3f} average guesses")

if __name__ == "__main__":
    answers_list, all_guesses_list = load_wordle_lists()
    
    # Run the experiment over 100 games
    run_experiment(answers_list, all_guesses_list, trials=1000)
