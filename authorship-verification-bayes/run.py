from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from pathlib import Path
from joblib import load

if __name__ == "__main__":
    # Load the data
    ti = Client()
    data = ti.pd.inputs("nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-validation")

    # Load trained model
    mod = load(Path(__file__).parent / "model.joblib")

    # Combine sentences into one feature as in training
    data["combined"] = data["sentence1"] + " " + data["sentence2"]

    # Predict whether the pairs are paraphrases
    predictions = mod.predict(data["combined"])

    # Update dataframe with predictions
    data["label"] = predictions
    data = data[["id", "label"]]

    # Save predictions
    outputDir = get_output_directory(str(Path(__file__).parent))
    output_file = Path(outputDir) / "predictions.jsonl"
    data.to_json(output_file, orient="records", lines=True)
