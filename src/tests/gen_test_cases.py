import json
import random
from configs.intents import INTENTS

# -------------------------------
# Configuration
# -------------------------------

POLITE_PREFIXES = [
    "please",
    "can you",
    "could you",
    "would you",
    "hey",
    "hey please"
]

POLITE_SUFFIXES = [
    "for me",
    "please",
    "right now",
    "quickly"
]

VARIATION_TEMPLATES = [
    "{cmd}",
    "{cmd} please",
    "please {cmd}",
    "can you {cmd}",
    "could you {cmd} for me",
    "hey {cmd}",
    "{cmd} right now"
]

CONFIDENCE_BY_LENGTH = {
    "short": 0.85,
    "medium": 0.7,
    "long": 0.6
}

MAX_PHRASES_PER_INTENT = 12


# -------------------------------
# Helper functions
# -------------------------------

def estimate_confidence(text: str) -> float:
    length = len(text.split())
    if length <= 2:
        return CONFIDENCE_BY_LENGTH["short"]
    elif length <= 5:
        return CONFIDENCE_BY_LENGTH["medium"]
    return CONFIDENCE_BY_LENGTH["long"]


def generate_variations(phrase: str, limit: int):
    variations = set()

    for tmpl in VARIATION_TEMPLATES:
        variations.add(tmpl.format(cmd=phrase))

    # Politeness noise
    variations.add(
        f"{random.choice(POLITE_PREFIXES)} {phrase}"
    )
    variations.add(
        f"{phrase} {random.choice(POLITE_SUFFIXES)}"
    )

    return list(variations)[:limit]


# -------------------------------
# Main generator
# -------------------------------

def generate_tests_from_intents(INTENTS):
    dataset = {}

    for intent, phrases in INTENTS.items():
        cases = []
        sampled_phrases = phrases[:MAX_PHRASES_PER_INTENT]

        for phrase in sampled_phrases:
            variations = generate_variations(phrase, limit=3)

            for text in variations:
                cases.append({
                    "input": text,
                    "expected": intent,
                    "min_confidence": estimate_confidence(text)
                })

        dataset[intent] = cases

    return dataset

CODE_EXPANSIONS = [
    "write me an article on {}",
    "build a program that {}",
    "create a function to {}",
    "generate code for {}",
    "write a script that {}"
]

CODE_TOPICS = [
    "web development",
    "sorting an array",
    "user authentication",
    "file handling",
    "data validation"
]

def enrich_code_intent(dataset):
    for topic in CODE_TOPICS:
        for tmpl in CODE_EXPANSIONS:
            text = tmpl.format(topic)
            dataset["code"].append({
                "input": text,
                "expected": "code",
                "min_confidence": 0.8
            })

def add_ambiguous_cases(dataset):
    dataset["ambiguous"] = []

    for intent in list(dataset.keys())[:10]:
        dataset["ambiguous"].append({
            "input": intent,
            "expected": intent,
            "should_be_ambiguous": True
        })




# -------------------------------
# Main
# -------------------------------

def gen():

    dataset = generate_tests_from_intents(INTENTS)

    enrich_code_intent(dataset)
    add_ambiguous_cases(dataset)

    with open("tests/data/test_cases.json", "w") as f:
        json.dump(dataset, f, indent=2)

