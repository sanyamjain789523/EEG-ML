import pandas as pd
from pathlib import Path
import os
from io import BytesIO

from src.data_processing.data_processor import session_data_aggregation

current_dir = Path.cwd()

train_folder = current_dir / "src" / "data" / "training_data"
test_folder = current_dir / "src" /"data" / "test_data"


def prepare_data_for_prediction(csv_file):
    contents = csv_file.read()
    data = BytesIO(contents)
    df = pd.read_excel(data, sheet_name=[f"Session {i}" for i in range(1, 8)], index_col=None)
    aggregated_df = session_data_aggregation(df)
    return aggregated_df
    

def add_training_data_to_folder(csv_file, filename, is_adhd):
    contents = csv_file.read()
    data = BytesIO(contents)
    df = pd.read_excel(data, sheet_name=[f"Session {i}" for i in range(1, 8)], index_col=None)
    aggregated_df = session_data_aggregation(df)
    data.close()
    aggregated_df["label"] = is_adhd
    aggregated_df.to_csv(f"{train_folder}/{filename}", index=False)


def add_test_data_to_folder(csv_file, filename, is_adhd):
    contents = csv_file.read()
    data = BytesIO(contents)
    df = pd.read_excel(data, sheet_name=[f"Session {i}" for i in range(1, 8)], index_col=None)
    aggregated_df = session_data_aggregation(df)
    data.close()
    aggregated_df["label"] = is_adhd
    aggregated_df.to_csv(f"{test_folder}/{filename}", index=False)


def delete_file(filename, is_train):
    if is_train:
        os.remove(f"{train_folder}/{filename}")
    else:
        os.remove(f"{test_folder}/{filename}")


def list_files(is_train):
    if is_train:
        return os.listdir(f"{train_folder}")
    else:
        return os.listdir(f"{test_folder}")