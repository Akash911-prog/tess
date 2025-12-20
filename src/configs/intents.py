INTENTS = {
    # File & Application Management
    "open": [
        "open", "launch", "start", "run", "execute",
        "open file", "open folder", "open settings", "open app",
        "open window", "open application", "open program",
        "launch app", "launch application", "launch program",
        "start app", "start application", "start program",
        "run app", "run application", "run program",
        "open chrome", "open firefox", "open vscode",
        "open whatsapp", "open slack", "open notepad"
    ],

    "close": [
        "close", "exit", "quit", "shut", "end", "terminate",
        "close file", "close app", "close application",
        "close window", "close program",
        "shut window", "shut app", "shut application",
        "exit app", "exit application", "exit program",
        "quit app", "quit application",
        "terminate app", "terminate application",
        "end app", "end program"
    ],

    # Clipboard Operations
    "copy": [
        "copy", "duplicate", "replicate",
        "copy text", "copy selection", "copy file",
        "copy this", "copy that", "copy content",
        "make copy", "duplicate this", "duplicate selection",
        "copy to clipboard"
    ],

    "paste": [
        "paste", "insert",
        "paste here", "paste content", "paste text",
        "paste clipboard", "insert clipboard",
        "paste from clipboard", "insert content",
        "put clipboard", "add clipboard"
    ],

    "cut": [
        "cut", "remove and copy",
        "cut text", "cut this", "cut selection",
        "cut and copy", "remove selected",
        "cut to clipboard", "extract selection"
    ],

    # File Operations
    "delete": [
        "delete", "remove", "erase", "trash",
        "delete file", "delete this", "delete item",
        "remove file", "remove this", "remove item",
        "erase file", "erase this", "erase item",
        "trash file", "move to trash",
        "delete folder", "remove folder"
    ],

    "save": [
        "save", "store", "write",
        "save file", "save document", "save progress",
        "save work", "save changes", "store file",
        "write file", "commit changes"
    ],

    "save as": [
        "save as", "save with name", "save copy",
        "save with new name", "export as",
        "save as new file", "create copy as",
        "save to new location"
    ],

    # Edit Operations
    "undo": [
        "undo", "revert", "go back",
        "undo action", "undo last change",
        "undo last action", "revert change",
        "go back one step", "reverse action",
        "cancel last action"
    ],

    "redo": [
        "redo", "redo action", "redo last",
        "redo change", "repeat action",
        "go forward", "restore action"
    ],

    "select all": [
        "select all", "highlight all", "select everything",
        "highlight everything", "select all text",
        "highlight all text", "mark all",
        "choose all", "select entire document"
    ],

    # Tab Management
    "new tab": [
        "new tab", "open new tab", "create tab",
        "create new tab", "start new tab",
        "open tab", "add tab", "make new tab",
        "add new tab"
    ],

    "close tab": [
        "close tab", "close this tab", "close current tab",
        "close active tab", "shut tab", "remove tab",
        "end tab", "kill tab"
    ],

    "next tab": [
        "next tab", "go next tab", "move next tab",
        "switch next tab", "go to next tab",
        "forward tab", "right tab", "tab right"
    ],

    "previous tab": [
        "previous tab", "prev tab", "go previous tab",
        "back tab", "go back tab", "left tab",
        "tab left", "prior tab"
    ],

    "switch tab": [
        "switch tab", "change tab", "go to tab",
        "move to tab", "jump to tab", "select tab",
        "navigate to tab"
    ],

    "open tab": [
        "open tab", "go to tab number",
        "switch to tab", "select tab number",
        "jump to tab number", "tab number"
    ],

    # Window Management
    "minimize window": [
        "minimize", "minimize window", "minimize this",
        "reduce window", "shrink window",
        "hide window", "minimize app",
        "collapse window"
    ],

    "maximize window": [
        "maximize", "maximize window", "full screen",
        "fullscreen window", "expand window",
        "enlarge window", "make fullscreen",
        "maximize app", "full screen window"
    ],

    "restore window": [
        "restore window", "restore size",
        "unmaximize", "normal size",
        "default size", "restore normal"
    ],

    "close all windows": [
        "close all windows", "close everything",
        "close all apps", "close all applications",
        "shut all windows", "exit all",
        "quit all", "close all programs"
    ],

    "switch window": [
        "switch window", "change window", "next window",
        "move to next window", "go to next window",
        "alternate window", "toggle window",
        "switch application"
    ],

    # System Controls
    "volume up": [
        "volume up", "increase volume", "raise volume",
        "make louder", "turn up volume", "boost volume",
        "louder", "raise sound", "increase sound",
        "up volume", "turn up sound", "boost sound"
    ],

    "volume down": [
        "volume down", "decrease volume", "lower volume",
        "make quieter", "turn down volume", "reduce volume",
        "quieter", "lower sound", "decrease sound",
        "down volume", "turn down sound", "reduce sound"
    ],

    "mute": [
        "mute", "silence", "mute volume",
        "silence sound", "turn off sound",
        "mute audio", "no sound"
    ],

    "unmute": [
        "unmute", "unmute volume", "restore sound",
        "turn on sound", "enable sound",
        "unmute audio"
    ],

    # Connectivity
    "wifi": [
        "Wi-Fi", "wi-fi", "wireless",
        "toggle Wi-Fi", "Wi-Fi settings",
        "turn Wi-Fi on", "turn wifi off",
        "enable wifi", "disable wifi",
        "wifi on", "wifi off",
        "connect wifi", "disconnect wifi"
    ],

    "bluetooth": [
        "bluetooth", "toggle bluetooth",
        "bluetooth settings", "turn bluetooth on",
        "turn bluetooth off", "enable bluetooth",
        "disable bluetooth", "bluetooth on",
        "bluetooth off"
    ],

    # Power Management
    "shutdown": [
        "shutdown", "shut down", "power off",
        "turn off", "power down",
        "shut down computer", "turn off computer",
        "turn off pc", "shutdown system",
        "power off system", "shut down pc"
    ],

    "restart": [
        "restart", "reboot", "restart computer",
        "reboot computer", "restart system",
        "reboot system", "restart pc",
        "reboot pc", "reset computer"
    ],

    "sleep": [
        "sleep", "hibernate", "suspend",
        "sleep computer", "put to sleep",
        "suspend computer", "hibernate computer",
        "sleep mode", "standby"
    ],

    "lock": [
        "lock", "lock screen", "lock computer",
        "lock pc", "lock system",
        "screen lock", "secure screen"
    ],

    # Navigation
    "scroll up": [
        "scroll up", "scroll upward", "page up",
        "move up", "go up", "up",
        "scroll to top"
    ],

    "scroll down": [
        "scroll down", "scroll downward", "page down",
        "move down", "go down", "down",
        "scroll to bottom"
    ],

    "go back": [
        "go back", "back", "navigate back",
        "previous page", "back page",
        "return", "go to previous"
    ],

    "go forward": [
        "go forward", "forward", "navigate forward",
        "next page", "forward page",
        "advance"
    ],

    "refresh": [
        "refresh", "reload", "refresh page",
        "reload page", "update page",
        "refresh browser", "reload browser"
    ],

    # Search & Find
    "search": [
        "search", "find", "look for",
        "search for", "find text",
        "locate", "look up", "seek",
        "what is", "what are", "who is",
        "who are", "when is", "when was",
        "where is", "where are", "how do",
        "how does", "why is", "why are",
        "tell me about", "show me", "give me",
        "what's the", "who's the", "where's the",
        "tell me the", "show the", "find out",
        "look up the", "search the web",
        "google", "search google", "search online"
    ],

    "find and replace": [
        "find and replace", "replace text",
        "find replace", "search and replace",
        "substitute text"
    ],

    # Input Actions
    "click": [
        "click", "press", "hit", "tap",
        "press button", "click button",
        "press key", "hit key",
        "press enter", "press escape",
        "press shift", "press control",
        "press delete", "press backspace",
        "click enter", "hit enter"
    ],

    "type": [
        "type", "write", "enter text",
        "input text", "type text",
        "write text", "insert text"
    ],

    # Screenshot
    "screenshot": [
        "screenshot", "screen capture", "capture screen",
        "take screenshot", "snap screen",
        "print screen", "screen shot",
        "capture window"
    ],

    # AI/Code Generation
    "code": [
        # Direct coding requests
        "write code for me",
        "generate code for this",
        "create a program",
        "build a script",
        "develop an application",
        "make a function",
        "write a function for me",
        "create a function that",

        # Article / explanation style (important)
        "write an article on",
        "write an article about programming",
        "explain how to code",
        "explain this code",
        "teach me how to program",
        "show me how to build",

        # Task-based requests
        "build a program that",
        "create a script that",
        "make a program that",
        "write a script to",
        "generate a function to",

        # Web / app specific
        "build a web application",
        "create a website",
        "generate a login page",
        "write backend code",
        "write frontend code",

        # AI-style phrasing
        "can you write code",
        "can you generate code",
        "help me write code",
        "i want to build a program",
        "i need a script that",

        # General dev intent
        "program this for me",
        "develop this feature",
        "implement this logic"
    ],

    # Help
    "help": [
        "help", "assist", "support",
        "help me", "need help",
        "assistance", "guide me",
        "show help", "get help"
    ],
}