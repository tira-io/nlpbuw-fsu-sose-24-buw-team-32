from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import json

if __name__ == "__main__":
    # Initialize TIRA Client
    tira = Client()

    # Load the dataset
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "summarization-validation-20240530-training"
    ).set_index("id")

    # Print the columns of the dataframe
    print("Columns in the dataframe:", df.columns)

    # Print the first few rows of the dataframe to inspect its structure
    print("First few rows of the dataframe:")
    print(df.head())

    # Check if 'story' column exists and handle accordingly
    if 'story' in df.columns:
        # Generate summary by grabbing the first two sentences of the story
        df["summary"] = df["story"].str.split("\n").str[:2].str.join("\n")
        df = df.drop(columns=["story"]).reset_index()

        # Save the predictions to JSONL file
        output_directory = get_output_directory(str(Path(__file__).parent))
        output_path = Path(output_directory) / "predictions.jsonl"
        df.to_json(output_path, orient="records", lines=True)

        print(f"Predictions saved to {output_path}")

        # Verify the content of the predictions file
        with open(output_path, 'r') as f:
            for _ in range(5):
                print(json.loads(f.readline()))

    else:
        print("Error: 'story' column not found in the dataframe")
