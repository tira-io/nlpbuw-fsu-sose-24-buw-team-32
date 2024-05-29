import nltk
import pandas as pd

import nltk
# C
def levenshtein_distance(df):
    distance = df.apply(lambda row: nltk.edit_distance(row["sentence1"], row["sentence2"]), axis=1)
    return distance

