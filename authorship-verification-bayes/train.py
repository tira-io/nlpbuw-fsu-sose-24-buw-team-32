from sklearn.feature_extraction.text import CountVectorizer

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from tira.rest_api_client import Client

from joblib import dump
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
#from sklearn.externals import joblib,,
import joblib 


import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

def load_data(text_file, labels_file):
    with open(text_file, 'r') as f:
        texts = [json.loads(line) for line in f]
    with open(labels_file, 'r') as f:
        labels = [json.loads(line) for line in f]
    return texts, labels

def preprocess_and_train(texts, labels):
    sentences = [f"{text['sentence1']} {text['sentence2']}" for text in texts]
    y = [label['label'] for label in labels]
    # Create a pipeline that vectorizes the text and then trains a classifier
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(sentences, y)
    return model

if __name__ == "__main__":
    texts, labels = load_data('text.jsonl', 'labels.jsonl')
    model = preprocess_and_train(texts, labels)
    joblib.dump(model, 'naive_bayes_model.joblib')
