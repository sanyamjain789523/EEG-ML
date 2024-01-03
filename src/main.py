from src.modelling.train import predict_file, train_models
from src.data_processing.data_addition import add_test_data_to_folder, add_training_data_to_folder, delete_file, list_files, prepare_data_for_prediction
from fastapi import FastAPI, UploadFile
from pathlib import Path



# current_dir = os.getcwd()
# print("current_dir: ", current_dir)
current_dir = Path.cwd()
print("current_dir: ", current_dir)


train_folder = current_dir / "src" / "data" / "training_data"
test_folder = current_dir / "src" / "data" / "test_data"

print("train_folder: ", train_folder)
app = FastAPI()


@app.get("/")
def root():
    return {"message": "NEUBRAi MVP"}

@app.post("/delete_data")
def remove_file(filenames: list[str], is_train: bool):
    try:
        for file in filenames:
            delete_file(file, is_train, train_folder, test_folder)
    except Exception as e:
        return {f"Error in file: {file.filename}": str(e)}
    return {"message": "files removed"}

@app.get("/list_data")
def list_file(is_train: bool):
    try:
        files = list_files(is_train, train_folder, test_folder)
    except Exception as e:
        return {"Error": str(e)}
    return {"files": files}

@app.post("/upload_adhd_training_files")
async def upload_adhd_training_files(files: list[UploadFile]):
    try:
        for file in files:
            file.filename = file.filename.replace("xlsx", "csv")
            add_training_data_to_folder(file.file, file.filename, 1, train_folder)
    except Exception as e:
        return {f"Error in file: {file.filename}": str(e)}
    return {"filenames": [file.filename for file in files]}

@app.post("/upload_non_adhd_training_files")
async def upload_non_adhd_training_files(files: list[UploadFile]):
    try:
        for file in files:
            file.filename = file.filename.replace("xlsx", "csv")
            add_training_data_to_folder(file.file, file.filename, 0, train_folder)
    except Exception as e:
        return {f"Error in file: {file.filename}": str(e)}
    return {"filenames": [file.filename for file in files]}

@app.post("/upload_adhd_test_files")
async def upload_adhd_test_files(files: list[UploadFile]):
    try:
        for file in files:
            file.filename = file.filename.replace("xlsx", "csv")
            add_test_data_to_folder(file.file, file.filename, 1, test_folder)
    except Exception as e:
        return {f"Error in file: {file.filename}": str(e)}
    return {"filenames": [file.filename for file in files]}

@app.post("/upload_non_adhd_test_files")
async def upload_non_adhd_test_files(files: list[UploadFile]):
    try:
        for file in files:
            file.filename = file.filename.replace("xlsx", "csv")
            add_test_data_to_folder(file.file, file.filename, 0, test_folder)
    except Exception as e:
        return {f"Error in file: {file.filename}": str(e)}
    return {"filenames": [file.filename for file in files]}

@app.post("/train_model")
async def train_model(algorithm: str):
    try:
        metrics = train_models(algorithm)
        # metrics = jsonable_encoder(metrics)
    except Exception as e:
        return {"Error": str(e)}
    return {"metrics": {algorithm: metrics}}
 
@app.post("/predict")
async def predict(files: list[UploadFile], algorithm: str):
    predictions = {}
    try:
        for file in files:
            file.filename = file.filename.replace("xlsx", "csv")
            pred_df = prepare_data_for_prediction(file.file)
            pred = predict_file(pred_df, algorithm)
            
            predictions[file.filename] = pred
    except Exception as e:
        return {f"Error in file: {file.filename}": str(e)}
    return predictions
    
# uvicorn src.main:app --reload