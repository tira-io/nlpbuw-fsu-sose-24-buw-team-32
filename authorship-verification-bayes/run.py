# from pathlib import Path

# from joblib import load
# from tira.rest_api_client import Client
# from tira.third_party_integrations import get_output_directory

# if __name__ == "__main__":

#     # Load the data
#     tira = Client()
#     df = tira.pd.inputs(
#         "nlpbuw-fsu-sose-24", f"authorship-verification-validation-20240408-training"
#     )

#     # Load the model and make predictions
#     model = load(Path(__file__).parent / "model.joblib")
#     predictions = model.predict(df["text"])
#     df["generated"] = predictions
#     df = df[["id", "generated"]]

#     # Save the predictions
#     output_directory = get_output_directory(str(Path(__file__).parent))
#     df.to_json(
#         Path(output_directory) / "predictions.jsonl", orient="records", lines=True
#     )
import joblib
import json

# Load the trained model
model = joblib.load('model.joblib')

# Load the text data to make predictions
with open('text.jsonl', 'r') as f:
    texts = [json.loads(line) for line in f]

predictions = []
for entry in texts:
    text = entry['text']
    pred_lang = model.predict([text])[0]
    predictions.append({'id': entry['id'], 'lang': pred_lang})

# Save the predictions to predictions.jsonl
with open('predictions.jsonl', 'w') as f:
    for pred in predictions:
        f.write(json.dumps(pred) + '\n')
