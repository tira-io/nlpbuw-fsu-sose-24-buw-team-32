from pathlib import Path
from levenshtein import levenshtein_distance
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

if __name__ == "__main__":
    import os
    # Load the best threshold from train.py
    with open('/workspaces/nlpbuw-fsu-sose-24-buw-team-32/paraphrase-identification/best_threshold.txt', 'r') as f:
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

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
