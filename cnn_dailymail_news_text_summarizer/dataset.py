from pathlib import Path
import pandas as pd


def load_data(file_path):
    return pd.read_csv(file_path)

def load_datasets(train_path, test_path, val_path):
    train_data = load_data(train_path)
    test_data = load_data(test_path)
    val_data = load_data(val_path)
    return train_data, test_data, val_data
    

if __name__ == "__main__":
    train_path = Path('data/raw/cnn_dailymail/train.csv')
    test_path = Path('data/raw/cnn_dailymail/test.csv')
    val_path = Path('data/raw/cnn_dailymail/validation.csv')
    train_data, test_data, val_data = load_datasets(train_path, test_path, val_path)


