import json
import random
import string

# Load all words
with open("all_words_dictionary.json", "r") as f:
    all_words_dict = json.load(f)

# Extract 7 and 8 letter words
english7 = [word for word in all_words_dict.keys() if len(word) == 7 and word.isalpha()]
english8 = [word for word in all_words_dict.keys() if len(word) == 8 and word.isalpha()]

def generate_random_words(length, count=10000):
    dataset = set()
    while len(dataset) < count:
        word = ''.join(random.choices(string.ascii_lowercase, k=length))
        dataset.add(word)
    return list(dataset)

random6 = generate_random_words(6)
random7 = generate_random_words(7)
random8 = generate_random_words(8)

# Save datasets
with open("english7.json", "w") as f:
    json.dump(english7, f, indent=4)

with open("english8.json", "w") as f:
    json.dump(english8, f, indent=4)

with open("random6.json", "w") as f:
    json.dump(random6, f, indent=4)

with open("random7.json", "w") as f:
    json.dump(random7, f, indent=4)

with open("random8.json", "w") as f:
    json.dump(random8, f, indent=4)

print(f"Created english7.json with {len(english7)} words.")
print(f"Created english8.json with {len(english8)} words.")
print(f"Created random6.json with {len(random6)} words.")
print(f"Created random7.json with {len(random7)} words.")
print(f"Created random8.json with {len(random8)} words.")
