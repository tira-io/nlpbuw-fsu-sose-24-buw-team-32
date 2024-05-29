from tira.rest_api_client import Client
from levenshtein import levenshtein_distance
import os

if __name__ == "__main__":
    try:
        # Load the data
        print("Loading data...")
        tira = Client()
        text = tira.pd.inputs(
            "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
        ).set_index("id")
        labels = tira.pd.truths(
            "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
        ).set_index("id")
        print("Data loaded successfully.")

        # Calculate Levenshtein distance between sentence pairs
        print("Calculating Levenshtein distances...")
        text["distance"] = text.apply(lambda row: levenshtein_distance(row["sentence1"], row["sentence2"]), axis=1)
        df = text.join(labels)
        print("Levenshtein distances calculated.")

        mccs = {}
        for threshold in sorted(text["distance"].unique()):
            tp = df[(df["distance"] <= threshold) & (df["label"] == 1)].shape[0]
            fp = df[(df["distance"] <= threshold) & (df["label"] == 0)].shape[0]
            tn = df[(df["distance"] > threshold) & (df["label"] == 0)].shape[0]
            fn = df[(df["distance"] > threshold) & (df["label"] == 1)].shape[0]
            try:
                mcc = (tp * tn - fp * fn) / (
                    (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
                ) ** 0.5
            except ZeroDivisionError:
                mcc = 0
            mccs[threshold] = mcc

        best_threshold = max(mccs, key=mccs.get)
        print(f"Best threshold: {best_threshold}")

        # Save the best threshold for use in run.py
        print("Saving best threshold to best_threshold.txt...")
        with open('best_threshold.txt', 'w') as f:
            f.write(str(best_threshold))

        # Verify the file has been created
        if os.path.isfile('best_threshold.txt'):
            print("best_threshold.txt created successfully.")
        else:
            print("Failed to create best_threshold.txt.")
    except Exception as e:
        print(f"Error occurred: {e}")
