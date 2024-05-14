import json
from pathlib import Path
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Function to load training data
def load_training_data(filepath):
    texts, labels = [], []
    with open(filepath, 'r') as f:
        for line in f:
            entry = json.loads(line)
            texts.append(entry['text'])
            labels.append(entry['lang'])
    return texts, labels

if __name__ == "__main__":
    # Path to the training data file
    training_data_file = Path(__file__).parent / "train_data.jsonl"
    
    # Load the training data
    texts, labels = load_training_data(training_data_file)
    
    # Create a model pipeline
    model = Pipeline([
        ("vectorizer", TfidfVectorizer()),
        ("classifier", MultinomialNB())
    ])
    
    # Train the model
    model.fit(texts, labels)
    
    # Save the trained model
    dump(model, Path(__file__).parent / "model.joblib")
