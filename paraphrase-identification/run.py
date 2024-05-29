from pathlib import Path
from levenshtein import levenshtein_distance
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import os

if __name__ == "__main__":
    # Load the best threshold from train.py
    with open('best_threshold.txt', 'r') as f:
        best_threshold = float(f.read().strip())

    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training"
    ).set_index("id")

    # Compute the Levenshtein distance between sentence pairs
    df["distance"] = df.apply(lambda row: levenshtein_distance(row["sentence1"], row["sentence2"]), axis=1)
    df["label"] = (df["distance"] <= best_threshold).astype(int)
    df = df.drop(columns=["distance", "sentence1", "sentence2"]).reset_index()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Combine with the filename to get the absolute path of best_threshold.txt
    threshold_file = os.path.join(dir_path, 'best_threshold.txt')
    # Now open the file
    with open(threshold_file, 'r') as f:
        best_threshold = float(f.read().strip())