from joblib import load
from configs.bin_classifier import model_path, vectorizer_path


def predict(text):
    model = load(model_path)
    vectorizer = load(vectorizer_path)
    X = vectorizer.transform([text])
    return model.predict(X)[0]

if __name__ == "__main__":
    p = predict("can you open settings")
    d = {
        0 : 'chat',
        1 : 'command'
    }
    print(d[p])
