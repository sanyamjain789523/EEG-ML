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
    for file in files[:3]:
        if (file.endswith('.csv')) and (file not in ["Dave Wed 18th Oct 2023 7X2 Template with Reports_.csv"]):
            df = pd.read_csv(f"{path}/{file}")
            data.append(df)
    
    df = pd.concat(data, ignore_index=True)
    
    return df


def train_models(algorithm):
    training_data = obtain_data(train_folder)
    test_data = obtain_data(test_folder)

    if algorithm == "rf":
        clf = RandomForestClassifier()
        clf.fit(training_data.drop(["label"], axis = 1), training_data["label"])
        filename = f'{modelling}/rf.pkl'
        pickle.dump(clf, open(filename, 'wb'))
        
        # some time later...
        
        # load the model from disk
        # clf = pickle.load(open(filename, 'rb'))
        y_pred = clf.predict(test_data.drop(["label"], axis = 1))
        # cm = confusion_matrix(training_data["label"], y_pred)
        acc = accuracy_score(training_data["label"], y_pred)
        return acc

def predict_using_rf(test_file):
    # load the model from disk
    loaded_model = pickle.load(open(f'{modelling}/rf.pkl', 'rb'))
    y_pred = loaded_model.predict(test_file)
    return y_pred

# clf = LogisticRegression(max_iter=1000)
# # scores = cross_validate(clf, data_train.dropna().drop(["label"], axis = 1), 
# #                          data_train.dropna()["label"], cv=5, scoring=confusion_matrix_scorer)

# # print("Logistic regression scores: ")
# # display(scores)
# # X_train, X_test, y_train, y_test = train_test_split(data_train.dropna().drop(["label"], axis = 1), 
# #                          data_train.dropna()["label"], test_size = 0.3, random_state = 42)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# cm = confusion_matrix(y_test, y_pred)

# disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
# disp.plot()
# plt.show()
    

# clf = RandomForestClassifier()
# # scores = cross_validate(clf, data_train.dropna().drop(["label"], axis = 1), 
# #                          data_train.dropna()["label"], cv=5, scoring=confusion_matrix_scorer)

# # print("rf scores: ")
# # display(scores)
# # X_train, X_test, y_train, y_test = train_test_split(data_train.dropna().drop(["label"], axis = 1), 
# #                          data_train.dropna()["label"], test_size = 0.3, random_state = 42)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# cm = confusion_matrix(y_test, y_pred)

# disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
# disp.plot()
# plt.show()

