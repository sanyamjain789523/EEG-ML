import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle

current_dir = Path.cwd()

train_folder = current_dir / "src" / "data" / "training_data"
test_folder = current_dir / "src" /"data" / "test_data"
modelling = current_dir / "src" / "modelling"


def obtain_data(path):
    files = os.listdir(path)

    data = []
    for file in files:
        if (file.endswith('.csv')):
            df = pd.read_csv(f"{path}/{file}")
            # if file.filename == "Dave Wed 18th Oct 2023 7X2 Template with Reports  (1).csv":
            #     print(df.info())
            # df["file"] = file
            data.append(df)
    
    df = pd.concat(data, ignore_index=True)
    
    return df


def train_models(algorithm):
    training_data = obtain_data(train_folder)
    test_data = obtain_data(test_folder)

    if algorithm == "rf":
        clf = RandomForestClassifier()
    elif algorithm == "lg":
        clf = LogisticRegression()
    elif algorithm == "xgb":
        clf = GradientBoostingClassifier()
    
    training_data.to_csv(f"{modelling}/training_data.csv", index=False)
    test_data.to_csv(f"{modelling}/test_data.csv", index=False)
    print("training_data.shape: ", training_data.shape)
    training_data = training_data.dropna()
    clf.fit(training_data.drop(["label"], axis = 1), training_data["label"])
    filename = f'{modelling}/{algorithm}.pkl'
    pickle.dump(clf, open(filename, 'wb'))
    
    # load the model from disk
    # clf = pickle.load(open(filename, 'rb'))
    
    print("test_data.shape: ", test_data.shape)
    test_data = test_data.dropna()
    y_pred = clf.predict(test_data.drop(["label"], axis = 1))
    # cm = confusion_matrix(training_data["label"], y_pred)
    acc = accuracy_score(test_data["label"], y_pred)
    # cm = confusion_matrix(training_data["label"], y_pred)
    tn, fp, fn, tp = confusion_matrix(test_data["label"], y_pred).ravel()
    # print(tp.item())
    metrics = {
        "accurancy":acc,
        "true_positive": tp.item(),
        "false_positive": fp.item(),
        "true_negative": tn.item(),
        "false_negative": fn.item(),
    }
    return metrics


def predict_file(test_file, algorithm):
    # load the model from disk
    loaded_model = pickle.load(open(f'{modelling}/{algorithm}.pkl', 'rb'))
    test_file = test_file.dropna()
    if test_file.shape[0] == 0:
        return -1
    y_pred_proba = loaded_model.predict_proba(test_file)
    y_pred = loaded_model.predict(test_file)
    return {y_pred.item(): y_pred_proba.max()}
