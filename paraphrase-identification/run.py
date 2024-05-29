from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from levenshtein import levenshtein_distance

if __name__ == "__main__":
    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training"
    ).set_index("id")

    # Ensure 'sentence1' and 'sentence2' columns are present
    if 'sentence1' not in df.columns or 'sentence2' not in df.columns:
        raise ValueError("DataFrame does not contain 'sentence1' and 'sentence2' columns.")

    # Compute the Levenshtein distance
    df["distance"] = levenshtein_distance(df)
    df["label"] = (df["distance"] <= 10).astype(int)
    df = df.drop(columns=["distance", "sentence1", "sentence2"]).reset_index()

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
