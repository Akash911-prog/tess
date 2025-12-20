from pathlib import Path

current_path = Path(__file__).parent.parent
model_path = current_path.rglob("classifier_model.pkl").__next__().resolve()
vectorizer_path = current_path.rglob("tfidf_vectorizer.pkl").__next__().resolve()

