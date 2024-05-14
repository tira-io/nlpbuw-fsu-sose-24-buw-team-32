from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from pathlib import Path
from joblib import load
#.
if __name__ == "__main__":
    # Load the data
    dataf = Client().pd.inputs("nlpbuw-fsu-sose-24", "language-identification-validation-20240408-validation")
    print("our loaded data:", dataf.head())
    

    
    mod = load(Path(__file__).parent / "model.joblib")
    print("Model path:", Path(__file__).parent / "model.joblib")
    # now for pred
    pred = mod.predict(dataf["text"])
    print("the Predictions are:", pred[:5])  # Print first 5 predictions

    # we need tp update our df
    dataf["lang"] = pred
    dataf = dataf[["id", "lang"]]

    # Save predictions
    outputDir = get_output_directory(str(Path(__file__).parent))
    print("Output directory:", outputDir)
    output_file = Path(outputDir) / "predictions.jsonl"
    print("Output file:", output_file)
    dataf.to_json(output_file, orient="records", lines=True)
