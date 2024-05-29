import json
import joblib
from pathlib import Path
import logging

import json
import joblib

def load_model(model_path):
    return joblib.load(model_path)

def make_predictions(model, texts):
    sentences = [f"{text['sentence1']} {text['sentence2']}" for text in texts]
    predictions = model.predict(sentences)
    return predictions

def write_predictions(predictions, output_file):
    with open(output_file, 'w') as f:
        for i, prediction in enumerate(predictions):
            result = {"id": i, "label": int(prediction)}
            json.dump(result, f)
            f.write('\n')

if __name__ == "__main__":
    model = load_model('naive_bayes_model.joblib')
    with open('text.jsonl', 'r') as file:
        texts = [json.loads(line) for line in file]
    predictions = make_predictions(model, texts)
    write_predictions(predictions, 'predictions.jsonl')
