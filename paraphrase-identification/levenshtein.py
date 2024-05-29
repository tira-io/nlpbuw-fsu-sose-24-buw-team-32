import nltk
import pandas as pd

def levenshtein_distance(df: pd.DataFrame):
    distance = df.apply(lambda row: nltk.edit_distance(row["sentence1"], row["sentence2"]), axis=1)
    return distance
