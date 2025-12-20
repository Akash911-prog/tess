from pathlib import Path

# Placeholder for your file path
APP_NAMES_FILE = Path("./app_names.txt") 

def load_app_patterns(file_path):
    """Generates phrase patterns from a text file."""
    if not file_path.exists():
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # Create a phrase pattern {"label": "app", "pattern": "App Name"} for each line
        patterns = [{"label": "app", "pattern": line.strip()} for line in f if line.strip()]
        return patterns

# 1. Load the manual phrase patterns from the file
manual_app_patterns = load_app_patterns(APP_NAMES_FILE)

keyboard_pattern = [
    {"LOWER": {"REGEX": "^(control|shift|alt|win|super|meta|enter|return|space|tab|caps|escape|esc|backspace|delete|del|home|end|pageup|pagedown|pgup|pgdn|insert|ins|up|down|left|right|printscreen|prtsc|scrolllock|pause|break|menu)$"}, "OP": "*"},
    {"LOWER": {"REGEX": "^f([1-9]|1[0-9]|2[0-4])$"}, "OP": "*"},  # F1–F24
    {"TEXT": {"REGEX": "^[a-zA-Z0-9]$"}, "OP": "*"},  # single letters/digits
    # {"TEXT": {"REGEX": r"^[`~!@#$%^&*()_+\-=\[\]{};':\",.<>/?\\|]$"}}  # symbol keys
]

params_pattern = manual_app_patterns + [

    # 1a. PERCENTAGE: Matches a token that looks like a number, followed by the literal "%" character.
    {"label": "percentage", "pattern": [{"LIKE_NUM": True}, {"ORTH": "%"}]},

    # 1b. PERCENTAGE: Matches a token that looks like a number, followed by "percentage"
    {"label": "percentage", "pattern": [{"LIKE_NUM": True}, {"ORTH": "percentage"}]},

    {"label": "direction", "pattern": [{"ORTH": {"IN": ["up", "down", "left", "right"]}}]},
    
    {"label": "btn", "pattern": keyboard_pattern },

    # 2. DIRECTION: Matches a single token that is exactly one of the listed words.

    # 3a. VALUE (Qualifiers): Matches a single token that is one of the qualifier words.
    {"label": "value", "pattern": [{"ORTH": {"IN": ["bit", "little", "notch"]}}]},
    
    # 3b. VALUE (Numbers): Matches a single token that is a number.
    # Note: This overlaps with the PERCENTAGE rule, which will take precedence if added first.
    {"label": "value", "pattern": [{"LIKE_NUM": True}]},

    {"label": "action", "pattern": [{"ORTH": {"IN": ["off", "on", "toggle"]}}]},

    # 4. APP: Matches one optional NOUN or PROPN. (Matches a single word that is a noun/proper noun)
    {"label": "app", "pattern": [{"POS": {"IN": ["NOUN", "PROPN"]}}]},

]