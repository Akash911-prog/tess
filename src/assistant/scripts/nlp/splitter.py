# assistant/nlp/splitter.py
def split_commands(text: str):
    connectors = [" and ", " then ", " after that "]
    for conn in connectors:
        if conn in text:
            return [cmd.strip() for cmd in text.split(conn)]
    return [text.strip()]
