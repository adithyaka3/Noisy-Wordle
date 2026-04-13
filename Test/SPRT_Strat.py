import math
from noisy_wordle import calculate_true_feedback, load_words

# SPRT Constants
ALPHA = 0.01  # 99% Confidence target
THRESHOLD = math.log((1 - ALPHA) / ALPHA)

POSSIBLE_WORDS = load_words("possible_words.txt")

def get_discriminator_guess(ll_dict):
    """Finds the guess that best splits the top hypotheses."""
    ranked = sorted(ll_dict.keys(), key=lambda w: ll_dict[w], reverse=True)
    # Focus on the top contenders (those within a small LL gap of the leader)
    contenders = [w for w in ranked if ll_dict[ranked[0]] - ll_dict[w] < 5.0]
    
    if len(contenders) <= 1:
        return ranked[0]
        
    best_guess = ""
    max_splits = -1
    
    # Heuristic: Find a word that generates the most unique patterns among contenders
    for guess in contenders:
        patterns = set(calculate_true_feedback(guess, w) for w in contenders)
        if len(patterns) > max_splits:
            max_splits = len(patterns)
            best_guess = guess
    return best_guess

def run_single_game_sprt(target, p_correct, p_wrong):
    """Plays one game using MSPRT logic."""
    ll_dict = {w: 0.0 for w in POSSIBLE_WORDS}
    
    for turn in range(1, 15):
        guess = get_discriminator_guess(ll_dict)
        true_fb = calculate_true_feedback(guess, target)
        
        # Simulating the noisy observation
        import random
        obs_fb = list(true_fb)
        for i in range(5):
            r = random.random()
            if r > p_correct:
                others = [c for c in [0, 1, 2] if c != true_fb[i]]
                obs_fb[i] = others[0] if r < p_correct + p_wrong else others[1]
        obs_fb = tuple(obs_fb)

        # Update Log-Likelihoods: LL = sum(log(P(O|T)))
        for word in POSSIBLE_WORDS:
            word_fb = calculate_true_feedback(guess, word)
            for o, t in zip(obs_fb, word_fb):
                prob = p_correct if o == t else p_wrong
                ll_dict[word] += math.log(prob)

        # Evaluate the Gap
        ranked = sorted(ll_dict.items(), key=lambda x: x[1], reverse=True)
        ll1 = ranked[0][1]
        ll2 = ranked[1][1]
        
        if (ll1 - ll2) >= THRESHOLD:
            return turn if guess == target else turn + 1
            
    return "DNF"