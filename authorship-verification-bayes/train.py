from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from tira.rest_api_client import Client

from joblib import dump
from pathlib import Path

if __name__ == "__main__":
    ti = Client()
    dataForTraining = ti.pd.inputs("nlpbuw-fsu-sose-24", "language-identification-train-20240408-training").set_index("id")
    labels = ti.pd.truths("nlpbuw-fsu-sose-24", "language-identification-train-20240408-training")
    dataf = dataForTraining.join(labels.set_index("id"))

    # Define SVM classifier pipeline
    mod = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", SVC(kernel='rbf', C=1.0, gamma='scale'))  # Using linear kernel for SVM
    ]).fit(dataf["text"], dataf["lang"])

    dump(mod, Path(__file__).parent / "model.joblib")
