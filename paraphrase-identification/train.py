from tira.rest_api_client import Client
import numpy as np

def wer(reference, hypothesis):
    r = reference.split()
    h = hypothesis.split()
    d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8).reshape((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        for j in range(len(h) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    return d[len(r)][len(h)] / float(len(r))

if __name__ == "__main__":
    # Load the data
    tira = Client()
    text = tira.pd.inputs("nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training").set_index("id")
    labels = tira.pd.truths("nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training").set_index("id")
    
    text["wer"] = text.apply(lambda row: wer(row['sentence1'], row['sentence2']), axis=1)
    df = text.join(labels)

    mccs = {}
    for threshold in sorted(text["wer"].unique()):
        tp = df[(df["wer"] <= threshold) & (df["label"] == 1)].shape[0]
        fp = df[(df["wer"] <= threshold) & (df["label"] == 0)].shape[0]
        tn = df[(df["wer"] > threshold) & (df["label"] == 0)].shape[0]
        fn = df[(df["wer"] > threshold) & (df["label"] == 1)].shape[0]
        try:
            mcc = (tp * tn - fp * fn) / (
                (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
            ) ** 0.5
        except ZeroDivisionError:
            mcc = 0
        mccs[threshold] = mcc
    best_threshold = max(mccs, key=mccs.get)
    print(f"Best threshold: {best_threshold}")
