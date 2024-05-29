from tira.rest_api_client import Client
from levenshtein import levenshtein_distance

if __name__ == "__main__":
    # Load the data
    tira = Client()
    text = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
    ).set_index("id")
    labels = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
    ).set_index("id")

    # Calculate Levenshtein distance between sentence pairs
    text["distance"] = text.apply(lambda row: levenshtein_distance(row["sentence1"], row["sentence2"]), axis=1)
    df = text.join(labels)

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
    with open('/workspaces/nlpbuw-fsu-sose-24-buw-team-32/paraphrase-identification/best_threshold.txt', 'w') as f:
        f.write(str(best_threshold))
