import math
import random

# --- 1. GAME SETUP ---
DICTIONARY = [
    "apple", "amber", "alert", "brave", "brick", "crane", "cramp", 
    "crisp", "drain", "dream", "flame", "flock", "grape", "ghost", 
    "heart", "hover", "light", "lemon", "mango", "mouse", "night", 
    "pride", "round", "smart", "train"
]

COLORS = {0: "⬛", 1: "🟨", 2: "🟩"}
P_CORRECT = 0.8
P_WRONG = 0.1

# SPRT Thresholds
ALPHA = 0.01  # 99% Confidence target
THRESHOLD = math.log((1 - ALPHA) / ALPHA) # ~4.59

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
    """Applies the 60/20/20 symmetric noise."""
    obs_feedback = list(true_feedback)
    for i in range(5):
        r = random.random()
        if r > P_CORRECT:
            # Pick one of the other two colors randomly
            other_colors = [c for c in [0, 1, 2] if c != true_feedback[i]]
            obs_feedback[i] = other_colors[0] if r < P_CORRECT + P_WRONG else other_colors[1]
    return tuple(obs_feedback)

# --- 3. MULTIPLE HYPOTHESIS TESTING ---
def get_best_discriminator_guess(ll_dict):
    """Finds the guess that best splits the leading hypotheses."""
    # Find the top candidates (those within 5.0 LL of the leader)
    sorted_words = sorted(ll_dict.keys(), key=lambda w: ll_dict[w], reverse=True)
    best_ll = ll_dict[sorted_words[0]]
    contenders = [w for w in sorted_words if best_ll - ll_dict[w] < 5.0]
    
    # If only one contender is left, just guess it to finish the math
    if len(contenders) == 1:
        return contenders[0]
        
    best_guess = None
    max_unique_splits = -1
    
    # We want a guess that produces the most diverse feedbacks among contenders
    for guess in DICTIONARY:
        feedbacks = set(calculate_true_feedback(guess, w) for w in contenders)
        if len(feedbacks) > max_unique_splits:
            max_unique_splits = len(feedbacks)
            best_guess = guess
            
    return best_guess

def update_log_likelihoods(ll_dict, guess, obs_fb):
    """Adds the log-likelihood of the observation to each word's running total."""
    for word in DICTIONARY:
        true_fb = calculate_true_feedback(guess, word)
        
        for o, t in zip(obs_fb, true_fb):
            if o == t:
                ll_dict[word] += math.log(P_CORRECT) # + log(0.6)
            else:
                ll_dict[word] += math.log(P_WRONG)   # + log(0.2)

# --- 4. GAME SIMULATION ---
def play_msprt_game(target_word):
    print(f"\n🎯 TARGET WORD: {target_word.upper()}")
    
    # Initialize Log-Likelihoods to 0.0 for all words
    ll_dict = {w: 0.0 for w in DICTIONARY}
    
    turn = 0
    while True:
        turn += 1
        print("=" * 50)
        
        # 1. Pick guess and simulate game
        guess = get_best_discriminator_guess(ll_dict)
        true_fb = calculate_true_feedback(guess, target_word)
        obs_fb = apply_noise(true_fb)
        
        print(f"🤖 Turn {turn:02d} | AI Guesses: {guess.upper()}")
        print(f"   Truth: {''.join([COLORS[c] for c in true_fb])}")
        print(f"   Noise: {''.join([COLORS[c] for c in obs_fb])}")
        
        # 2. Update hypothesis scores
        update_log_likelihoods(ll_dict, guess, obs_fb)
        
        # 3. Rank and evaluate
        ranked = sorted(ll_dict.items(), key=lambda x: x[1], reverse=True)
        top1, ll1 = ranked[0]
        top2, ll2 = ranked[1]
        gap = ll1 - ll2
        
        print(f"\n📊 Hypothesis Leaderboard:")
        for i in range(3):
            print(f"   {i+1}. {ranked[i][0].upper():<6} | LL: {ranked[i][1]:>6.2f}")
            
        print(f"\n📈 Current LLR Gap (Top 1 vs Top 2): {gap:.2f} / {THRESHOLD:.2f}")
        
        # 4. Stop Condition
        if gap >= THRESHOLD:
            print(f"\n✅ SPRT THRESHOLD CROSSED!")
            print(f"The algorithm has established 99% mathematical dominance.")
            print(f"The word is {top1.upper()}.")
            break
            
        if turn >= 20:
            print("\n❌ Reached safety limit. The noise was too persistent.")
            break

if __name__ == "__main__":
    target = random.choice(DICTIONARY)
    play_msprt_game(target)
