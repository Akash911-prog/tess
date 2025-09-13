# assistant/worker.py
import sys, json, joblib, os
from scripts.nlp.splitter import split_commands
from scripts.nlp.chatbot import generate_reply

# Add src/assistant/ to sys.path

# Load classifier
clf = joblib.load(r"src\assistant\scripts\classifier\model.pkl")
vectorizer = joblib.load("src/assistant/scripts/classifier/vectorizer.pkl")

def classify_command(cmd: str):
    X = vectorizer.transform([cmd])
    label = clf.predict(X)[0]
    return label

for line in sys.stdin:
    user_input = line.strip()
    responses = []

    # Split into multiple commands
    commands = split_commands(user_input)

    for cmd in commands:
        try:
            label = classify_command(cmd)
            if label == "open_app":
                responses.append({"intent": "open_app", "app": cmd.replace("open", "").strip()})
            elif label == "search_web":
                responses.append({"intent": "search_web", "query": cmd.replace("search", "").strip()})
            elif label == "chat":
                reply = generate_reply(cmd)
                responses.append({"intent": "chat", "reply": reply})
            else:
                responses.append({"intent": label})
        except Exception:
            # If no intent, fallback to chatbot
            reply = generate_reply(cmd)
            responses.append({"intent": "chat", "reply": reply})

    print(json.dumps(responses))
    sys.stdout.flush()
