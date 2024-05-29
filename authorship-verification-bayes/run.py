from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from pathlib import Path
from joblib import load
import pandas as pd

if __name__ == "__main__":
    try:
        # Load the data
        ti = Client()
        data = ti.pd.inputs("nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-validation")

        # Verify required columns are present
        if "sentence1" not in data.columns or "sentence2" not in data.columns:
            print("Data does not contain the expected 'sentence1' or 'sentence2' columns.")
            exit()

        # Load trained model
        model_path = Path(__file__).parent / "model.joblib"
        if not model_path.exists():
            print(f"Model file not found at {model_path}.")
            exit()
        else:
            mod = load(model_path)

            # Combine sentences into one feature as in training
            data["combined"] = data["sentence1"] + " " + data["sentence2"]

            # Predict whether the pairs are paraphrases
            predictions = mod.predict(data["combined"])

            # Update dataframe with predictionsss
            data["label"] = predictions
            data = data[["id", "label"]]

            # Save predictions
            outputDir = get_output_directory(str(Path(__file__).parent))
            output_file = Path(outputDir) / "predictions.jsonl"
            data.to_json(output_file, orient="records", lines=True)

            print(f"Output successfully saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")
