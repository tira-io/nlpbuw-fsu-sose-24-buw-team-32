# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import Pipeline
# from tira.rest_api_client import Client
# from pathlib import Path
# import json
# from joblib import dump
# from sklearn.feature_extraction.text import CountVectorizer


# if __name__ == "__main__":
#     ti = Client()
#     dataFortraining = ti.pd.inputs("nlpbuw-fsu-sose-24", "language-identification-train-20240408-training").set_index("id") #this to fetch the training data and making the 'id' column the index
#     labels = ti.pd.truths("nlpbuw-fsu-sose-24", "language-identification-train-20240408-training")
#     dataf = dataFortraining.join(labels.set_index("id"))

#     # now the training
#     mod = Pipeline([("vectorizer", CountVectorizer()),("classifier", MultinomialNB())]).fit(dataf["text"], dataf["lang"])

   
#     dump(mod, Path(__file__).parent / "model.joblib")


from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from tira.rest_api_client import Client
from pathlib import Path
from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == "__main__":
    ti = Client()
    dataFortraining = ti.pd.inputs("nlpbuw-fsu-sose-24", "language-identification-train-20240408-training").set_index("id")
    labels = ti.pd.truths("nlpbuw-fsu-sose-24", "language-identification-train-20240408-training")
    dataf = dataFortraining.join(labels.set_index("id"))

    # Define SVM classifier pipeline
    mod = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", SVC(kernel='linear'))  # Using linear kernel for SVM
    ]).fit(dataf["text"], dataf["lang"])

    dump(mod, Path(__file__).parent / "model.joblib")
