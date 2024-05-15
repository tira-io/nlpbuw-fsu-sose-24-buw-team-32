from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB  # Import Multinomial Naive Bayes classifier
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from pathlib import Path
from joblib import load

if __name__ == "__main__":
    # Load the data
    dataf = Client().pd.inputs("nlpbuw-fsu-sose-24", "language-identification-validation-20240408-validation")

    # Define the classifier pipeline with CountVectorizer and Multinomial Naive Bayes
    mod = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", MultinomialNB())  # Using Multinomial Naive Bayes classifier
    ])

    # Load the trained model
    mod = load(Path(__file__).parent / "model.joblib")

    # Predict language using the trained classifier
    pred = mod.predict(dataf["text"])

    # Update dataframe with predictions
    dataf["lang"] = pred
    dataf = dataf[["id", "lang"]]

    # Save predictions
    outputDir = get_output_directory(str(Path(__file__).parent))
    output_file = Path(outputDir) / "predictions.jsonl"
    dataf.to_json(output_file, orient="records", lines=True)
