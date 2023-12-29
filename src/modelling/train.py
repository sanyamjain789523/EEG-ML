import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
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
        # training_data.to_csv(f"{modelling}/training_data.csv", index=False)
        training_data = training_data.dropna()
        clf.fit(training_data.drop(["label"], axis = 1), training_data["label"])
        print("Model trained")
        filename = f'{modelling}/rf.pkl'
        pickle.dump(clf, open(filename, 'wb'))
        
        # load the model from disk
        # clf = pickle.load(open(filename, 'rb'))
        test_data = test_data.dropna()
        y_pred = clf.predict(test_data.drop(["label"], axis = 1))
        print(y_pred)
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


def predict_using_rf(test_file):
    # load the model from disk
    loaded_model = pickle.load(open(f'{modelling}/rf.pkl', 'rb'))
    test_file = test_file.dropna()
    if test_file.shape[0] == 0:
        return -1
    y_pred = loaded_model.predict(test_file)
    return y_pred.item()
