from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from tira.rest_api_client import Client
from joblib import dump
from pathlib import Path

if __name__ == "__main__":
    ti = Client()
    # Assuming inputs and truths are correctly fetched with the sentences' pairs
    data = ti.pd.inputs("nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training")
    labels = ti.pd.truths("nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training")

    # Combining sentences into a single feature for simplicity
    data["combined"] = data["sentence1"] + " " + data["sentence2"]

    # Define logistic regression pipeline
    mod = Pipeline([
        ("vectorizer", TfidfVectorizer()),
        ("classifier", LogisticRegression(solver='liblinear'))  # Suitable for binary classification
    ]).fit(data["combined"], labels["label"])

    dump(mod, Path(__file__).parent / "model.joblib")
