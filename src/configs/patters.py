from pathlib import Path

# Placeholder for your file path
APP_NAMES_FILE = Path("./app_names.txt") 

def load_app_patterns(file_path):
    """Generates phrase patterns from a text file."""
    if not file_path.exists():
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # Create a phrase pattern {"label": "app", "pattern": "App Name"} for each line
        patterns = [{"label": "target", "pattern": line.strip()} for line in f if line.strip()]
        return patterns

# 1. Load the manual phrase patterns from the file
manual_app_patterns = load_app_patterns(APP_NAMES_FILE)

keyboard_pattern = [
    {"LOWER": {"REGEX": "^(control|ctrl|shift|alt|win|super|meta|enter|return|space|tab|caps|escape|esc|backspace|delete|del|home|end|pageup|pagedown|pgup|pgdn|insert|ins|up|down|left|right|printscreen|prtsc|scrolllock|pause|break|menu)$"}, "OP": "+"},
    {"LOWER": {"REGEX": "^f([1-9]|1[0-9]|2[0-4])$"}, "OP": "?"},
    {"TEXT": {"REGEX": "^[a-zA-Z0-9]$"}, "OP": "?"}
]


params_pattern = manual_app_patterns + [

    # Percentage (highest priority)
    {"label": "percentage", "pattern": [{"LIKE_NUM": True}, {"ORTH": "%"}]},
    {"label": "percentage", "pattern": [{"LIKE_NUM": True}, {"LOWER": "percentage"}]},

    # Direction
    {"label": "direction", "pattern": [{"LOWER": {"IN": ["up", "down", "left", "right"]}}]},

    # Keyboard / buttons (fixed)
    {"label": "btn", "pattern": keyboard_pattern},

    # Qualitative values
    {"label": "value", "pattern": [{"LOWER": {"IN": ["bit", "little", "notch"]}}]},
    {"label": "value", "pattern": [{"LIKE_NUM": True}]},

    # Explicit action state
    {"label": "action", "pattern": [{"LOWER": {"IN": ["off", "on", "toggle"]}}]},

    # Quoted targets (strong signal)
    {"label": "target", "pattern": [{"ORTH": "\""}, {"OP": "+"}, {"ORTH": "\""}]},

    # Multi-word proper nouns
    {"label": "target", "pattern": [{"POS": "PROPN", "OP": "+"}]},

    # Fallback single noun (lowest priority)
    {"label": "target", "pattern": [{"POS": "NOUN"}]},
]
