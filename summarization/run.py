from pathlib import Path
import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

def preprocess_text(text):
    # Preprocess the text: clean the text, remove stop words, tokenize the text, etc.
    # Example of basic preprocessing
    text = text.replace("\n", " ")
    return text

def load_model():
    # Load the pre-trained BART model and tokenizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    return tokenizer, model

def generate_summary(tokenizer, model, text, max_length=130, min_length=30, num_beams=4):
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(
        inputs['input_ids'], 
        num_beams=num_beams, 
        max_length=max_length, 
        min_length=min_length, 
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

if __name__ == "__main__":

    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "summarization-validation-20240530-training"
    ).set_index("id")

    # Preprocess the data
    df["story"] = df["story"].apply(preprocess_text)

    # Load the model
    tokenizer, model = load_model()

    # Generate summaries
    df["summary"] = df["story"].apply(lambda x: generate_summary(tokenizer, model, x))

    # Save the predictions
    df = df.drop(columns=["story"]).reset_index()
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
