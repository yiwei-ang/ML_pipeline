import pandas as pd
import uvicorn
from fastapi import FastAPI
from typing import Dict
from pydantic import BaseModel

from engine.model.model import SupervisedModels

app = FastAPI()


class InputData(BaseModel):
    file: Dict


@app.get("/")
def home():
    return {"message": "Welcome from the API"}


# @app.post("/")
# def create_job(training_data: TrainingData):
#     return {"Data": 1}


@app.post("/")
def get_outputs(file: InputData):
    df = pd.DataFrame.from_dict(file.file)
    self = SupervisedModels(input_data=df)
    result = self.run_pipeline()
    return result


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)