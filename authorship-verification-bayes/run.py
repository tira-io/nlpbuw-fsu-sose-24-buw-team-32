# from pathlib import Path

# from joblib import load
# from tira.rest_api_client import Client
# from tira.third_party_integrations import get_output_directory

# if __name__ == "__main__":

#     # Load the data
#     tira = Client()
#     df = tira.pd.inputs(
#         "nlpbuw-fsu-sose-24", f"authorship-verification-validation-20240408-training"
#     )

#     # Load the model and make predictions
#     model = load(Path(__file__).parent / "model.joblib")
#     predictions = model.predict(df["text"])
#     df["generated"] = predictions
#     df = df[["id", "generated"]]

#     # Save the predictions
#     output_directory = get_output_directory(str(Path(__file__).parent))
#     df.to_json(
#         Path(output_directory) / "predictions.jsonl", orient="records", lines=True
#     )











# from pathlib import Path
# from joblib import load
# from tira.rest_api_client import Client
# from tira.third_party_integrations import get_output_directory

# if __name__ == "_main_":
#     # Load the data
#     tira = Client()
#     df = tira.pd.inputs(
#         "nlpbuw-fsu-sose-24", "language-identification-validation-20240408-validation"
#     )

#     # Load the model and make predictions
#     model = load(Path(__file__).parent / "model.joblib")
#     predictions = model.predict(df["text"])
#     df["lang"] = predictions
#     df = df[["id", "lang"]]

#     # Save the predictions
#     output_directory = get_output_directory(str(Path(__file__).parent))
#     df.to_json(
#         Path(output_directory) / "predictions.jsonl", orient="records", lines=True
#     )


from pathlib import Path
from joblib import load
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

if __name__ == "__main__":
    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240408-validation"
    )
    print("Loaded data:", df.head())

    # Load the model
    model_path = Path(__file__).parent / "model.joblib"
    print("Model path:", model_path)
    model = load(model_path)

    # Make predictions
    predictions = model.predict(df["text"])
    print("Predictions:", predictions[:5])  # Print first 5 predictions

    # Update dataframe with predictions
    df["lang"] = predictions
    df = df[["id", "lang"]]

    # Save predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    print("Output directory:", output_directory)
    output_file = Path(output_directory) / "predictions.jsonl"
    print("Output file:", output_file)
    df.to_json(output_file, orient="records", lines=True)
