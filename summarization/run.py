from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

if __name__ == "__main__":
    # Initialize TIRA Client
    tira = Client()

    # Load the dataset
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "summarization-validation-20240530-training"
    ).set_index("id")

    # Generate summary by grabbing the first two sentences of the story
    df["summary"] = df["story"].str.split("\n").str[:2].str.join("\n")
    df = df.drop(columns=["story"]).reset_index()

    # Save the predictions to JSONL file
    output_directory = get_output_directory(str(Path(__file__).parent))
    output_path = Path(output_directory) / "predictions.jsonl"
    df.to_json(output_path, orient="records", lines=True)

    print(f"Predictions saved to {output_path}")
