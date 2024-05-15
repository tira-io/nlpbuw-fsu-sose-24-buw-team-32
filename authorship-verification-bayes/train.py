from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB  # Import Multinomial Naive Bayes classifier
from tira.rest_api_client import Client
from joblib import dump
from pathlib import Path

if __name__ == "__main__":
    ti = Client()
    dataForTraining = ti.pd.inputs("nlpbuw-fsu-sose-24", "language-identification-train-20240408-training").set_index("id")
    labels = ti.pd.truths("nlpbuw-fsu-sose-24", "language-identification-train-20240408-training")
    dataf = dataForTraining.join(labels.set_index("id"))

    # Define Multinomial Naive Bayes classifier pipeline
    mod = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", MultinomialNB())  # Using Multinomial Naive Bayes classifier
    ]).fit(dataf["text"], dataf["lang"])

    dump(mod, Path(__file__).parent / "model.joblib")
