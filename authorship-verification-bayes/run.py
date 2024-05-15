
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from pathlib import Path
from joblib import load

if __name__ == "__main__":
    # Load the data
    dataf = Client().pd.inputs("nlpbuw-fsu-sose-24", "language-identification-validation-20240408-validation")

    mod = load(Path(__file__).parent / "model.joblib")

    # Predict language using the trained SVM classifier
    pred = mod.predict(dataf["text"])

    # Update dataframe with predictions
    dataf["lang"] = pred
    dataf = dataf[["id", "lang"]]

    # Save predictions
    outputDir = get_output_directory(str(Path(__file__).parent))
    output_file = Path(outputDir) / "predictions.jsonl"
    dataf.to_json(output_file, orient="records", lines=True)
