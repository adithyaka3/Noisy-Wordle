import json
import random
import string

# Load all words
with open("all_words_dictionary.json", "r") as f:
    all_words_dict = json.load(f)

# Extract 5 and 6 letter words
english5 = [word for word in all_words_dict.keys() if len(word) == 5 and word.isalpha()]
english6 = [word for word in all_words_dict.keys() if len(word) == 6 and word.isalpha()]

# Generate 10000 random 5-letter words
dataset = set()
while len(dataset) < 10000:
    word = ''.join(random.choices(string.ascii_lowercase, k=5))
    dataset.add(word)
random5 = list(dataset)

# Save datasets
with open("english5.json", "w") as f:
    json.dump(english5, f, indent=4)

with open("english6.json", "w") as f:
    json.dump(english6, f, indent=4)

with open("random5.json", "w") as f:
    json.dump(random5, f, indent=4)

print(f"Created english5.json with {len(english5)} words.")
print(f"Created english6.json with {len(english6)} words.")
print(f"Created random5.json with {len(random5)} words.")
