from pathlib import Path
import pandas as pd
from joblib import load
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

if __name__ == "__main__":
    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training"
    ).set_index("id")

    # Prepare the data
    X = df['sentence1'] + ' ' + df['sentence2']

    # Load the trained model
    model = load('model.joblib')

    # Make predictions
    df['label'] = model.predict(X)
    df = df.drop(columns=["sentence1", "sentence2"]).reset_index()

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
