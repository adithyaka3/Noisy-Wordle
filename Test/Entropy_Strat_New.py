import math
import time
import csv
import itertools
from noisy_wordle import (
    load_words, calculate_true_feedback, apply_noise, update_beliefs
)

# Load dictionaries
ALLOWED_GUESSES = load_words("allowed_words.txt")
POSSIBLE_ANSWERS = load_words("possible_words.txt")


def calculate_expected_entropy(guess, beliefs, possible_words, p_correct, p_wrong):
    """
    Calculates Shannon Entropy for a guess, accounting for the noisy channel.
    H(W|G) = sum_{O} P(O|G) * H(W|O,G)
    """
    # 1. Pre-calculate true feedbacks for all words we still care about
    active_words = {w: calculate_true_feedback(guess, w) for w in possible_words if beliefs[w] > 1e-6}
    if not active_words:
        return 0

    # 2. Determine which patterns to iterate over
    # Optimization: At 0% noise, only "true" patterns are possible.
    # At >0% noise, all 243 patterns are theoretically possible.
    if p_correct == 1.0:
        patterns_to_check = set(active_words.values())
    else:
        patterns_to_check = list(itertools.product([0, 1, 2], repeat=5))

    expected_h = 0
    
    # 3. Sum over all possible observed patterns (O)
    for obs_fb in patterns_to_check:
        prob_obs = 0
        pattern_posteriors = {}

        # Calculate P(O | G) = sum_w P(O | w, G) * P(w)
        for w, true_fb in active_words.items():
            # Get P(O | True_FB) based on our noise model
            lik = 1.0
            for o, t in zip(obs_fb, true_fb):
                lik *= p_correct if o == t else p_wrong
            
            joint_prob = lik * beliefs[w]
            if joint_prob > 0:
                prob_obs += joint_prob
                pattern_posteriors[w] = joint_prob

        # 4. If this pattern is possible, calculate its entropy contribution
        if prob_obs > 1e-12:
            h_obs = 0
            for w, joint_prob in pattern_posteriors.items():
                p_w_given_obs = joint_prob / prob_obs
                h_obs -= p_w_given_obs * math.log2(p_w_given_obs)
            
            # Weighted average of entropy
            expected_h += prob_obs * h_obs

    return expected_h

def get_best_guess(beliefs, turn, p_correct, p_wrong):
    """Selects the guess that maximizes information gain."""
    if turn == 1: return "roast" # Speed optimization for opening
    
    candidates = [w for w, p in beliefs.items() if p > 1e-6]
    if len(candidates) == 1: return candidates[0]
    
    # To run faster on the full dictionary, we only evaluate entropy for remaining candidates
    # In a full 3b1b solver, you'd check all ALLOWED_GUESSES.
    best_guess = ""
    min_entropy = float('inf')
    
    for guess in candidates:
        h = calculate_expected_entropy(guess, beliefs, POSSIBLE_ANSWERS, p_correct, p_wrong)
        if h < min_entropy:
            min_entropy = h
            best_guess = guess
    return best_guess

def run_single_game(target, p_correct, p_wrong):
    """Plays one game and returns stats."""
    beliefs = {w: 1.0 / len(POSSIBLE_ANSWERS) for w in POSSIBLE_ANSWERS}
    
    for turn in range(1, 11):
        guess = get_best_guess(beliefs, turn, p_correct, p_wrong)
        true_fb = calculate_true_feedback(guess, target)
        obs_fb = apply_noise(true_fb, p_correct, p_wrong)
        
        beliefs = update_beliefs(beliefs, guess, obs_fb, p_correct, p_wrong, POSSIBLE_ANSWERS)
        
        # Check for 99% certainty
        top_word = max(beliefs, key=beliefs.get)
        if beliefs[top_word] > 0.99 and top_word == target:
            return turn if guess == target else turn + 1
    return "DNF"