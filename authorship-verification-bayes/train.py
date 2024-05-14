# from pathlib import Path
# import json
# from joblib import load

# from joblib import dump
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import Pipeline
# from tira.rest_api_client import Client

# if __name__ == "__main__":

#     # Load the data
#     model = load(Path(__file__).parent / "model.joblib")
#     tira = Client()
#     test_text = tira.pd.inputs(
#         "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
#     )
#     test_text = test_text.set_index("id")

#     # Make predictions
#     predictions = model.predict(test_text["text"])

#      # Save predictions to JSONL file
#     with open("predictions.jsonl", "w") as f:
#         for idx, prediction in zip(test_text.index, predictions):
#             json.dump({"id": idx, "generated": int(prediction)}, f)
#             f.write("\n")
#     # labels = tira.pd.truths(
#     #     "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
#     # )
#     # df = text.join(labels.set_index("id"))

#     # # Train the model
#     # model = Pipeline(
#     #     [("vectorizer", CountVectorizer()), ("classifier", MultinomialNB())]
#     # )
#     # model.fit(df["text"], df["generated"])

#     # # Save the model
#     # dump(model, Path(__file__).parent / "model.joblib")
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Load your training data
# Example: train_data = [{'id': 1, 'text': 'Bonjour'}, {'id': 2, 'text': 'Hello'}]
# Example: train_labels = [{'id': 1, 'lang': 'fr'}, {'id': 2, 'lang': 'en'}]

def load_training_data():
    # Replace with your actual data loading logic
    pass

train_data, train_labels = load_training_data()

texts = [entry['text'] for entry in train_data]
labels = [entry['lang'] for entry in train_labels]

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(texts, labels)

joblib.dump(model, 'model.joblib')
