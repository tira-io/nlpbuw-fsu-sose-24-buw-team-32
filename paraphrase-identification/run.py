from pathlib import Path
from levenshtein import levenshtein_distance
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import os
import json
import pandas as pd

def extract_sentences(text):
    try:
        data = json.loads(text)
        return pd.Series([data['sentence1'], data['sentence2']])
    except json.JSONDecodeError:
        # Assume text is plain text with two sentences separated by some delimiter
        # This is a placeholder: replace with the actual logic for your data format
        sentences = text.split('\n')[:2]  # Example: split by newline and take the first two lines
        if len(sentences) == 2:
            return pd.Series(sentences)
        else:
            raise ValueError(f"Unable to extract two sentences from text: {text}")

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

        # Ensure the 'text' column is present if sentence1 and sentence2 are not
        if 'sentence1' not in df.columns or 'sentence2' not in df.columns:
            if 'text' not in df.columns:
                raise KeyError("Column 'text' must be present in the data if 'sentence1' and 'sentence2' are not.")
            # Extract 'sentence1' and 'sentence2' from 'text' column
            print("Extracting 'sentence1' and 'sentence2' from 'text' column...")
            df[['sentence1', 'sentence2']] = df['text'].apply(extract_sentences)
            print("Extraction complete. Columns now:", df.columns.tolist())

        # Compute the Levenshtein distance between sentence pairs
        print("Computing Levenshtein distances for validation data...")
        df["distance"] = df.apply(lambda row: levenshtein_distance(row["sentence1"], row["sentence2"]), axis=1)
        df["label"] = (df["distance"] <= best_threshold).astype(int)
        
        # Drop columns that exist in the DataFrame
        columns_to_drop = ["distance", "sentence1", "sentence2"]
        if 'text' in df.columns:
            columns_to_drop.append('text')
        df = df.drop(columns=columns_to_drop).reset_index()
        
        print("Levenshtein distances computed.")

        # Save the predictions
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
    except ValueError as e:
        print(f"Error processing text: {e}")
    except Exception as e:
        print(f"Error occurred: {e}")
