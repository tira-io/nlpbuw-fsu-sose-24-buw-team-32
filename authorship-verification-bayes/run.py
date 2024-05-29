import json
import joblib 

from sklearn.externals import joblib

def load_model():
    model = joblib.load('model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
    return model, vectorizer

def make_predictions(model, vectorizer, texts):
    sentences = ["{} {}".format(text['sentence1'], text['sentence2']) for text in texts]
    X = vectorizer.transform(sentences)
    predictions = model.predict(X)
    return predictions

def write_predictions(predictions, output_file):
    with open(output_file, 'w') as file:
        for i, pred in enumerate(predictions):
            result = {"id": i, "label": int(pred)}
            json.dump(result, file)
            file.write('\n')

if __name__ == "__main__":
    model, vectorizer = load_model()
    with open('text.jsonl', 'r') as file:
        texts = [json.loads(line) for line in file]
    predictions = make_predictions(model, vectorizer, texts)
    write_predictions(predictions, 'predictions.jsonl')
