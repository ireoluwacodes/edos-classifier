from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model_loader import model_service


class PredictRequest(BaseModel):
    text: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...")
    model_service.load()
    print("Model loaded.")
    yield


app = FastAPI(
    title="EDOS Classifier API",
    lifespan=lifespan
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictRequest):
    try:
        return model_service.predict(payload.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))