from pathlib import Path
from levenshtein import levenshtein_distance
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import os

if __name__ == "__main__":
    try:
        # Load the best threshold from train.py
        print("Loading best threshold from best_threshold.txt...")
        with open('best_threshold.txt', 'r') as f:
            best_threshold = float(f.read().strip())
        print(f"Best threshold loaded: {best_threshold}")

        # Load the data
        print("Loading validation data...")
        tira = Client()
        df = tira.pd.inputs(
            "nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training"
        ).set_index("id")
        print("Validation data loaded.")
        print("Columns in the validation data:", df.columns.tolist())

        # Ensure required columns are present
        if 'sentence1' not in df.columns or 'sentence2' not in df.columns:
            raise KeyError("Columns 'sentence1' and 'sentence2' must be present in the data.")

        # Compute the Levenshtein distance between sentence pairs
        print("Computing Levenshtein distances for validation data...")
        df["distance"] = df.apply(lambda row: levenshtein_distance(row["sentence1"], row["sentence2"]), axis=1)
        df["label"] = (df["distance"] <= best_threshold).astype(int)
        df = df.drop(columns=["distance", "sentence1", "sentence2"]).reset_index()
        print("Levenshtein distances computed.")

        # Save the predictionss
        output_directory = get_output_directory(str(Path(__file__).parent))
        print(f"Saving predictions to {output_directory}/predictions.jsonl...")
        df.to_json(
            Path(output_directory) / "predictions.jsonl", orient="records", lines=True
        )
        print("Predictions saved successfully.")
    except FileNotFoundError:
        print("Error: best_threshold.txt file not found. Ensure that train.py generates the file before running run.py.")
    except KeyError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error occurred: {e}")
