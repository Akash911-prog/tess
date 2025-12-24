EXECUTABLE_INTENTS = [
    # File & Application Management
    "open",
    "close",

    # Clipboard
    "copy",
    "paste",
    "cut",

    # File operations
    "delete",
    "save",
    "save as",

    # Edit
    "undo",
    "redo",
    "select all",

    # Tabs
    "new tab",
    "close tab",
    "next tab",
    "previous tab",
    "switch tab",
    "open tab",

    # Windows
    "minimize window",
    "maximize window",
    "restore window",
    "close all windows",
    "switch window",

    # System controls
    "volume up",
    "volume down",
    "mute",
    "unmute",

    # Connectivity
    "wifi",
    "bluetooth",

    # Power
    "shutdown",
    "restart",
    "sleep",
    "lock",

    # Navigation
    "scroll up",
    "scroll down",
    "go back",
    "go forward",
    "refresh",

    # Editing / input
    "find and replace",
    "click",
    "type",

    # Utility
    "screenshot"
]


REQUIRED_PARAMS = {
    # App / file targeting
    "open": ["target"],
    "close": ["target"],
    "delete": ["target"],

    # Tab / window targeting
    "switch tab": ["target"],
    "open tab": ["target"],
    "switch window": ["target"],

    # Input
    "click": ["btn"],          # button / UI element
    "type": ["btn"],

    # Text operations
    "find and replace": ["find", "replace"],

    # Save-as needs destination
    "save as": ["target"],

    # Connectivity (optional but recommended)
    "wifi": ["action"],            # on/off
    "bluetooth": ["action"],       # on/off
}

NO_PARAM_INTENTS = [
    "copy",
    "paste",
    "cut",
    "undo",
    "redo",
    "select all",
    "new tab",
    "close tab",
    "next tab",
    "previous tab",
    "minimize window",
    "maximize window",
    "restore window",
    "close all windows",
    "volume up",
    "volume down",
    "mute",
    "unmute",
    "shutdown",
    "restart",
    "sleep",
    "lock",
    "scroll up",
    "scroll down",
    "go back",
    "go forward",
    "refresh",
    "screenshot",
]
