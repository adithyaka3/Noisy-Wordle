import random
import math
import itertools

# --- 1. GAME SETUP & DICTIONARY ---
DICTIONARY = [
    "apple", "amber", "alert", "brave", "brick", "crane", "cramp", 
    "crisp", "drain", "dream", "flame", "flock", "grape", "ghost", 
    "heart", "hover", "light", "lemon", "mango", "mouse", "night", 
    "pride", "round", "smart", "train"
]

COLORS = {0: "⬛", 1: "🟨", 2: "🟩"}
COLOR_NAMES = {0: "Gray", 1: "Yellow", 2: "Green"}

# The new universal symmetric noise model
P_CORRECT = 0.8
P_WRONG = 0.1 # Applies to the two other incorrect colors

# --- 2. GAME MECHANICS ---
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
    """Applies the 60/20/20 probabilistic noise to ALL tiles."""
    obs_feedback = list(true_feedback)
    noise_log = []
    
    for i in range(5):
        r = random.random()
        true_c = true_feedback[i]
        
        # Figure out what the *other* two colors are for this specific tile
        other_colors = [c for c in [0, 1, 2] if c != true_c]
        
        if r < P_CORRECT: # 60% chance
            noise_log.append(f"✅ Stable: {COLOR_NAMES[true_c]} stayed {COLOR_NAMES[true_c]}")
        elif r < P_CORRECT + P_WRONG: # 20% chance
            obs_feedback[i] = other_colors[0]
            noise_log.append(f"⚠️ NOISE: {COLOR_NAMES[true_c]} became {COLOR_NAMES[other_colors[0]]}!")
        else: # Remaining 20% chance
            obs_feedback[i] = other_colors[1]
            noise_log.append(f"⚠️ NOISE: {COLOR_NAMES[true_c]} became {COLOR_NAMES[other_colors[1]]}!")
            
    return tuple(obs_feedback), noise_log

# --- 3. BAYESIAN MATH ---
def get_likelihood(obs_feedback, true_feedback):
    """
    Calculates P(Observed | True).
    Because noise is symmetric across all tiles, this is beautifully simple.
    """
    prob = 1.0
    for o, t in zip(obs_feedback, true_feedback):
        if o == t:
            prob *= P_CORRECT # 0.6
        else:
            prob *= P_WRONG   # 0.2
    return prob

def update_beliefs(beliefs, guess, obs_feedback):
    """Updates the probability distribution using Bayes' Theorem."""
    new_beliefs = {}
    for word in DICTIONARY:
        true_fb = calculate_true_feedback(guess, word)
        lik = get_likelihood(obs_feedback, true_fb)
        new_beliefs[word] = beliefs[word] * lik
        
    total = sum(new_beliefs.values())
    if total == 0: return beliefs 
    return {w: val / total for w, val in new_beliefs.items()}

def calculate_expected_entropy(guess, beliefs):
    """Calculates the expected Shannon entropy of making a specific guess."""
    expected_h = 0
    true_fbs = {w: calculate_true_feedback(guess, w) for w in DICTIONARY}
    
    for obs_fb in itertools.product([0, 1, 2], repeat=5):
        prob_obs = 0
        posteriors = {}
        
        for w in DICTIONARY:
            prior = beliefs[w]
            if prior < 1e-6: continue
            
            lik = get_likelihood(obs_fb, true_fbs[w])
            joint_prob = prior * lik
            prob_obs += joint_prob
            posteriors[w] = joint_prob
            
        if prob_obs > 0:
            h = 0
            for w, joint_prob in posteriors.items():
                if joint_prob > 0:
                    p_w_given_obs = joint_prob / prob_obs
                    h -= p_w_given_obs * math.log2(p_w_given_obs)
            expected_h += prob_obs * h
            
    return expected_h

def get_best_guess(beliefs):
    """Finds the guess that minimizes expected future uncertainty."""
    best_guess = None
    min_entropy = float('inf')
    
    for guess in DICTIONARY:
        entropy = calculate_expected_entropy(guess, beliefs)
        if entropy < min_entropy:
            min_entropy = entropy
            best_guess = guess
            
    return best_guess

# --- 4. GAME SIMULATION ---
def play_game(target_word):
    print(f"\n🎯 TARGET WORD: {target_word.upper()}")
    print("🎲 Universal Noise: 60% Correct, 20% Wrong Option A, 20% Wrong Option B.\n")
    
    beliefs = {w: 1.0 / len(DICTIONARY) for w in DICTIONARY}
    
    for turn in range(1, 10):
        print("=" * 45)
        print(f"🤖 TURN {turn} - Thinking...")
        
        guess = get_best_guess(beliefs)
        print(f"🗣️ AI Guesses: {guess.upper()}")
        
        true_fb = calculate_true_feedback(guess, target_word)
        obs_fb, noise_log = apply_noise(true_fb)
        
        true_emoji = "".join([COLORS[c] for c in true_fb])
        obs_emoji = "".join([COLORS[c] for c in obs_fb])
        
        print("\n--- WHAT HAPPENED UNDER THE HOOD ---")
        print(f"Actual Truth : {true_emoji}")
        for i, log in enumerate(noise_log):
            print(f"  Tile {i+1}: {log}")
        print(f"What AI sees : {obs_emoji}")
        print("------------------------------------")
        
        beliefs = update_beliefs(beliefs, guess, obs_fb)
        
        sorted_beliefs = sorted(beliefs.items(), key=lambda x: x[1], reverse=True)
        print("\n📊 AI Top Beliefs:")
        for w, prob in sorted_beliefs[:5]:
            if prob > 0.001:
                print(f"   {w.upper()}: {prob*100:.1f}%")
                
        # The AI needs to be extremely confident to declare victory
        if sorted_beliefs[0][1] > 0.99 and sorted_beliefs[0][0] == target_word:
            print(f"\n🎉 AI is 99%+ mathematically certain the word is {target_word.upper()} on turn {turn}!")
            print("It has successfully cut through the noise.")
            break
            
        if turn == 9:
            print("\n❌ Reached turn limit. The noise was too strong!")

if __name__ == "__main__":
    target = random.choice(DICTIONARY)
    play_game(target)
