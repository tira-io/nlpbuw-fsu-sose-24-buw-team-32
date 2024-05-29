import json
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def load_model(model_path, vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def make_predictions(model, vectorizer, texts):
    sentences = ["{} {}".format(text['sentence1'], text['sentence2']) for text in texts]
    X = vectorizer.transform(sentences)
    return model.predict(X)

def write_predictions(predictions, output_file):
    if predictions:
        with open(output_file, 'w') as file:
            for i, pred in enumerate(predictions):
                result = {"id": i, "label": int(pred)}
                json.dump(result, file)
                file.write('\n')
        logging.info("Predictions written successfully.")
    else:
        logging.error("No predictions to write.")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / 'model.joblib'
    vectorizer_path = base_dir / 'vectorizer.joblib'
    text_file_path = base_dir / 'text.jsonl'
    
    try:
        model, vectorizer = load_model(model_path, vectorizer_path)
        with open(text_file_path, 'r') as file:
            texts = [json.loads(line) for line in file]
        predictions = make_predictions(model, vectorizer, texts)
        write_predictions(predictions, base_dir / 'predictions.jsonl')
    except Exception as e:
        logging.error(f"An error occurred: {e}")
