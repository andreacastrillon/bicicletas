import sys
import typing as t
from datetime import datetime
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, Depends, Body  # type: ignore # noqa: E402
from pydantic import BaseSettings, PositiveFloat

from service.entities import ModelInput

app = FastAPI(title="API to make inference with my great model", version="0.0.1")


class Settings(BaseSettings):
    serialized_model_path: str
    model_lib_dir: str


@lru_cache(None)
def get_settings():
    return Settings()


@lru_cache(None)
def load_estimator():
    sys.path.append(get_settings().model_lib_dir)
    path_serialized_model_path = get_settings().serialized_model_path
    estimator = joblib.load(path_serialized_model_path)
    return estimator

class Logger:
    def __init__(self, file: t.TextIO = sys.stdout):
        self.file = file

    def log(self, inputs: t.List[ModelInput]):
        for row in inputs:
            record = {"datetime": datetime.now(), "input": row.dict()}
            print(record, file=self.file)


def get_logger():
    return Logger()


@app.post("/prediction", response_model=t.List[float])
async def post_prediction(
    inputs: t.List[ModelInput] = Body(...),
    estimator=Depends(load_estimator),
    logger=Depends(get_logger),
):
    logger.log(inputs)
    X = pd.DataFrame([row.dict() for row in inputs])
    prediction = estimator.predict(X).tolist()
    return prediction


@app.get("/")
async def service_status():
    """Check the status of the service"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
