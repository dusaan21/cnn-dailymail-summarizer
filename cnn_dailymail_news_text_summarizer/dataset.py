from pathlib import Path
import pandas as pd
from transformers import BartTokenizer
from datasets import Dataset, concatenate_datasets, load_from_disk
import numpy as np
import os

def load_data(file_path):
    return pd.read_csv(file_path)

def load_datasets(train_path, test_path, val_path):
    train_data = load_data(train_path)
    test_data = load_data(test_path)
    val_data = load_data(val_path)
    train_data.drop('id', axis=1, inplace=True)
    test_data.drop('id', axis=1, inplace=True)
    val_data.drop('id', axis=1, inplace=True)
    return train_data, test_data, val_data

def remove_punctuation(text):
    import re
    return re.sub(r'[^\w\s]', '', text)

def preprocess_text(data, text_column):
    return data[text_column].str.lower().apply(remove_punctuation)

def tokenization(data, tokenizer, max_length=1024):
    inputs = ["summarize: " + doc for doc in data['article']]
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding='max_length')
    labels = tokenizer(text_target=data['highlights'], max_length=128, truncation=True, padding='max_length')
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def convert_to_dataset(df):
    return Dataset.from_pandas(df)

def tokenize(df, tokenizer, chunk_size=10000):
    num_chunks = len(df) // chunk_size + 1
    data_chunks = np.array_split(df, num_chunks)
    datasets = [convert_to_dataset(chunk) for chunk in data_chunks]
    data = concatenate_datasets(datasets)
    return data.map(tokenization, batched=True, fn_kwargs={'tokenizer': tokenizer})

def save_tokenized_datasets(train_data, test_data, val_data, output_dir):
    train_data.save_to_disk(os.path.join(output_dir, "train"))
    test_data.save_to_disk(os.path.join(output_dir, "test"))
    val_data.save_to_disk(os.path.join(output_dir, "validation"))

def load_tokenized_datasets(output_dir):
    train_data = load_from_disk(os.path.join(output_dir, "train"))
    test_data = load_from_disk(os.path.join(output_dir, "test"))
    val_data = load_from_disk(os.path.join(output_dir, "validation"))
    return train_data, test_data, val_data

if __name__ == "__main__":
    train_path = Path('data/raw/cnn_dailymail/train.csv')
    test_path = Path('data/raw/cnn_dailymail/test.csv')
    val_path = Path('data/raw/cnn_dailymail/validation.csv')
    train_data, test_data, val_data = load_datasets(train_path, test_path, val_path)


