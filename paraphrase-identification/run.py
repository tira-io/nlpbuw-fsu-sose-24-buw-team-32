from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
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
    df = tira.pd.inputs("nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training").set_index("id")

    # Compute the WER
    df["wer"] = df.apply(lambda row: wer(row['sentence1'], row['sentence2']), axis=1)
    df["label"] = (df["wer"] <= 10).astype(int)
    df = df.drop(columns=["wer", "sentence1", "sentence2"]).reset_index()

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True)
