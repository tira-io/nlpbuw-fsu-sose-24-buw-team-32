import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import joblib 


def load_data(text_file, labels_file):
    with open(text_file, 'r') as file:
        texts = [json.loads(line) for line in file]
    with open(labels_file, 'r') as file:
        labels = [json.loads(line) for line in file]
    return texts, labels

def preprocess_data(texts, labels):
    sentences = ["{} {}".format(text['sentence1'], text['sentence2']) for text in texts]
    y = [label['label'] for label in labels]
    return sentences, y

def train_model(X_train, y_train):
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    return vectorizer, model

if __name__ == "__main__":
    texts, labels = load_data('text.jsonl', 'labels.jsonl')
    X, y = preprocess_data(texts, labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer, model = train_model(X_train, y_train)
    joblib.dump(model, 'model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')
