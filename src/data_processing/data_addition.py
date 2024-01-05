import pandas as pd
# from pathlib import Path
import os
from io import BytesIO
# from src.main import train_folder, test_folder
# from src.data_processing.data_processor import session_data_aggregation

# current_dir = Path.cwd()

# train_folder = current_dir / "src" / "data" / "training_data"
# test_folder = current_dir / "src" / "data" / "test_data"
# train_folder = current_dir / "data" / "training_data"
# test_folder = current_dir / "data" / "test_data"


def prepare_data_for_prediction(csv_file, agg_func):
    contents = csv_file.read()
    data = BytesIO(contents)
    df = pd.read_excel(data, sheet_name=[f"Session {i}" for i in range(1, 8)], index_col=None)
    aggregated_df = agg_func(df, "pred")
    return aggregated_df
    

def add_training_data_to_folder(csv_file, filename, is_adhd, train_folder, agg_func):
    contents = csv_file.read()
    data = BytesIO(contents)
    df = pd.read_excel(data, sheet_name=[f"Session {i}" for i in range(1, 8)], index_col=None)
    aggregated_df = agg_func(df, filename)
    data.close()
    aggregated_df["label"] = is_adhd
    aggregated_df.to_csv(f"{train_folder}/{filename}", index=False)


def add_test_data_to_folder(csv_file, filename, is_adhd, test_folder, agg_func):
    contents = csv_file.read()
    data = BytesIO(contents)
    df = pd.read_excel(data, sheet_name=[f"Session {i}" for i in range(1, 8)], index_col=None)
    aggregated_df = agg_func(df, filename)
    data.close()
    aggregated_df["label"] = is_adhd
    aggregated_df.to_csv(f"{test_folder}/{filename}", index=False)


def delete_file(filename, is_train, train_folder, test_folder):
    if is_train:
        os.remove(f"{train_folder}/{filename}")
    else:
        os.remove(f"{test_folder}/{filename}")


def list_files(is_train, train_folder, test_folder):
    if is_train:
        print("train_folder::", train_folder)
        return os.listdir(f"{train_folder}")
    else:
        return os.listdir(f"{test_folder}")