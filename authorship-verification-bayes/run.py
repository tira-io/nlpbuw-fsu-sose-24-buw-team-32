import json
from pathlib import Path
from joblib import load

if __name__ == "__main__":
    # Load the trained model
    model = load(Path(__file__).parent / "model.joblib")
    
    # Load the test data
    test_file = Path(__file__).parent / "text.jsonl"
    texts = []
    with open(test_file, 'r') as f:
        for line in f:
            texts.append(json.loads(line))
    
    # Make predictions
    predictions = []
    for entry in texts:
        text = entry['text']
        pred_lang = model.predict([text])[0]
        predictions.append({'id': entry['id'], 'lang': pred_lang})
    
    # Save predictions to predictions.jsonl file
    with open(Path(__file__).parent / "predictions.jsonl", 'w') as f:
        for pred in predictions:
            json.dump(pred, f)
            f.write("\n")
