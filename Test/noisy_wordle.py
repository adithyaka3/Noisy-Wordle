import random

def load_words(filepath):
    """Loads words from a text file, one per line."""
    try:
        with open(filepath, 'r') as f:
            return [line.strip().lower() for line in f if len(line.strip()) == 5]
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return []

def calculate_true_feedback(guess, target):
    """Standard Wordle logic: 0=Gray, 1=Yellow, 2=Green."""
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

def apply_noise(true_fb, p_correct, p_wrong):
    """Applies noise to the feedback based on parameters."""
    if p_correct == 1.0: return true_fb
    
    obs_fb = list(true_fb)
    for i in range(5):
        r = random.random()
        if r > p_correct:
            others = [c for c in [0, 1, 2] if c != true_fb[i]]
            # 50/50 split between the two wrong colors if p_wrong is generic
            obs_fb[i] = others[0] if r < p_correct + p_wrong else others[1]
    return tuple(obs_fb)

def get_likelihood(obs_fb, true_fb, p_correct, p_wrong):
    """P(Observed | True) math for Bayesian updates."""
    prob = 1.0
    for o, t in zip(obs_fb, true_fb):
        prob *= p_correct if o == t else p_wrong
    return prob

def update_beliefs(beliefs, guess, obs_fb, p_correct, p_wrong, possible_words):
    """Updates the probability distribution (priors -> posteriors)."""
    new_beliefs = {}
    for word in possible_words:
        if beliefs.get(word, 0) == 0:
            new_beliefs[word] = 0
            continue
        true_fb = calculate_true_feedback(guess, word)
        lik = get_likelihood(obs_fb, true_fb, p_correct, p_wrong)
        new_beliefs[word] = beliefs[word] * lik
        
    total = sum(new_beliefs.values())
    if total == 0: return beliefs 
    return {w: val / total for w, val in new_beliefs.items()}