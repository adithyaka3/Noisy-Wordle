import json
import urllib.request

def create_frequent_small_datasets():
    # A widely used public list of the 10,000 most common English words (derived from Google's Trillion Word Corpus)
    url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-no-swears.txt"
    print(f"Fetching common English word frequencies from: {url}")
    
    try:
        req = urllib.request.urlopen(url)
        common_words = req.read().decode('utf-8').splitlines()
    except Exception as e:
        print(f"Failed to download frequency list: {e}")
        return

    # Load original datasets as sets so we only pick strictly valid Wordle-dictionary accepted words
    try:
        with open(f"datasets/english5.json", "r") as f:
            valid_5 = set(json.load(f))
        with open(f"datasets/english6.json", "r") as f:
            valid_6 = set(json.load(f))
    except Exception as e:
        print(f"Failed to load original dictionaries: {e}")
        return

    small_5 = []
    small_6 = []
    
    # Iterate through the common words list (which is strictly sorted by frequency from most to least)
    for word in common_words:
        word = word.lower()
        if len(word) == 5 and word in valid_5:
            small_5.append(word)
        elif len(word) == 6 and word in valid_6:
            small_6.append(word)
            
        # We only need the top 1000 most common valid words for each
        if len(small_5) >= 1000 and len(small_6) >= 1000:
            break

    # Enforce strictly 1000 items subset just in case
    small_5 = small_5[:1000]
    small_6 = small_6[:1000]

    with open("datasets/english_small_5.json", "w") as f:
        json.dump(small_5, f, indent=4)
        
    with open("datasets/english_small_6.json", "w") as f:
        json.dump(small_6, f, indent=4)

    print(f"Successfully created 'english_small_5.json' with {len(small_5)} hyper-common 5-letter words.")
    print(f"Successfully created 'english_small_6.json' with {len(small_6)} hyper-common 6-letter words.")

if __name__ == "__main__":
    print("Generating Frequency-Based Mapped subset dictionaries...")
    create_frequent_small_datasets()
    print("Execution complete!")
