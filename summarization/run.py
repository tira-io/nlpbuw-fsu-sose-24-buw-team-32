from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import json

if __name__ == "__main__":
    # Initialize the TIRA Client to interact with the TIRA platform
    tira_client = Client()

    # Load the dataset using the TIRA Client
    # The dataset is identified by task name "nlpbuw-fsu-sose-24" and dataset name "summarization-validation-20240530-training"
    # Set the dataframe's index to the 'id' column for easy reference
    dataset_df = tira_client.pd.inputs(
        "nlpbuw-fsu-sose-24", "summarization-validation-20240530-training"
    ).set_index("id")

    # Print the column names in the dataframe to understand its structure
    print("Columns in the dataframe:", dataset_df.columns)

    # Print the first few rows of the dataframe to get an overview of the data
    print("First few rows of the dataframe:")
    print(dataset_df.head())

    # Check if the 'text' column exists in the dataframe
    # The 'text' column is assumed to contain the data that needs to be summarized
    if 'text' in dataset_df.columns:
        # Create a new column 'summary' in the dataframe
        # The 'summary' is generated by taking the first two sentences from the 'text' column
        # The 'text' column is split into sentences using newline character as the delimiter
        # Only the first two sentences are kept and joined back with newline characters
        dataset_df["summary"] = dataset_df["text"].str.split("\n").str[:2].str.join("\n")
        
        # Drop the original 'text' column from the dataframe as it's no longer needed
        dataset_df = dataset_df.drop(columns=["text"]).reset_index()

        # Get the output directory to save the predictions file
        # The output directory is determined relative to the script's location
        output_dir = get_output_directory(str(Path(__file__).parent))
        # Define the path for the output file where predictions will be saved
        output_file_path = Path(output_dir) / "predictions.jsonl"
        
        # Save the modified dataframe to a JSON Lines file
        # Each row in the dataframe is written as a separate JSON object in the file
        dataset_df.to_json(output_file_path, orient="records", lines=True)

        # Print a message indicating the file location where predictions are saved
        print(f"Predictions saved to {output_file_path}")

        # Open the saved JSON Lines file to verify its content
        # Read and print the first five lines to check the saved predictions
        with open(output_file_path, 'r') as json_file:
            for _ in range(5):
                # Load each line as a JSON object and print it
                print(json.loads(json_file.readline()))
    else:
        # Print an error message if the 'text' column is not found in the dataframe
        print("Error: 'text' column not found in the dataframe")
