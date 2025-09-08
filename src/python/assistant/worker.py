import joblib

clf = joblib.load("src/python/assistant/classifier/model.pkl")
vectorizer = joblib.load("src/python/assistant/classifier/vectorizer.pkl")

def predict(text):
    X = vectorizer.transform([text])
    label = clf.predict(X)[0]
    return label

while True:
    text = input(">>> ")
    print(predict(text))