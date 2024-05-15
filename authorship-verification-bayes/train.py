from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from tira.rest_api_client import Client
from joblib import dump
from pathlib import Path

if __name__ == "__main__":
    # Load data
    ti = Client()
    dataForTraining = ti.pd.inputs("nlpbuw-fsu-sose-24", "language-identification-train-20240408-training").set_index("id")
    labels = ti.pd.truths("nlpbuw-fsu-sose-24", "language-identification-train-20240408-training")
    dataf = dataForTraining.join(labels.set_index("id"))

    # Define classifier pipeline with TF-IDF Vectorizer and Multinomial Naive Bayes
    mod = Pipeline([
        ("vectorizer", TfidfVectorizer()),
        ("classifier", MultinomialNB())
    ])

    # Train classifier using cross-validation
    scores = cross_val_score(mod, dataf["text"], dataf["lang"], cv=5, scoring="f1_micro")
    print("Cross-validation F1 scores:", scores)
    print("Mean F1 score:", scores.mean())

    # Fit model on full training data
    mod.fit(dataf["text"], dataf["lang"])

    # Save trained model
    dump(mod, Path(__file__).parent / "model.joblib")
