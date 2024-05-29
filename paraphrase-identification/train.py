import pandas as pd
from tira.rest_api_client import Client
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from joblib import dump

if __name__ == "__main__":
    # Load the data
    tira = Client()
    text = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
    ).set_index("id")
    labels = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
    ).set_index("id")
    
    df = text.join(labels)

    # Prepare the data
    X = df['sentence1'] + ' ' + df['sentence2']
    y = df['label']

    # Create a pipeline with TF-IDF and Logistic Regression
    pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression())

    # Train the model
    pipeline.fit(X, y)

    # Save the model
    dump(pipeline, 'model.joblib')
