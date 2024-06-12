from pathlib import Path
import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

def preprocess_text(text):
    # Basic preprocessing to clean the text data
    return text.replace("\n", " ")

def load_model():
    # Load the pre-trained BART model and tokenizer from Hugging Face
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    return tokenizer, model

def generate_summary(tokenizer, model, text, max_length=130, min_length=30, num_beams=4):
    # Generate a summary for a given text using the BART model
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

    # Load the data from TIRA
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "summarization-validation-20240530-training"
    ).set_index("id")

    # Print the first few rows of the DataFrame to view the data
    print("Loaded DataFrame:")
    print(df.head())

    # Print the content of the 'story' column for the first few rows
    print("\nStory column of the loaded DataFrame:")
    print(df["story"].head())

    # Preprocess the data
    df["story"] = df["story"].apply(preprocess_text)

    # Load the pre-trained BART model
    tokenizer, model = load_model()

    # Generate summaries for each story in the DataFrame
    df["summary"] = df["story"].apply(lambda x: generate_summary(tokenizer, model, x))

    # Print the modified DataFrame to view the summaries
    print("\nModified DataFrame with summaries:")
    print(df.head())

    # Print the content of the 'summary' column for the first few rows
    print("\nSummary column of the modified DataFrame:")
    print(df["summary"].head())

    # Save the predictions to a JSONL file
    df = df.drop(columns=["story"]).reset_index()
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
