import os
import sys
import asyncio
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import Body
from pydantic import BaseModel
import uvicorn

import spam_detector as sd


app = FastAPI(title="Spam Detector API", version="1.0.0")

# Allow all origins for simplicity in local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    message: str
    model_path: Optional[str] = sd.DEFAULT_MODEL_PATH


class PredictResponse(BaseModel):
    label: str
    spam_probability: float


class TrainRequest(BaseModel):
    data_path: Optional[str] = sd.DEFAULT_DATA_PATH
    model_path: Optional[str] = sd.DEFAULT_MODEL_PATH
    seed: int = 42


class TrainResponse(BaseModel):
    model_path: str
    accuracy: float
    f1_spam: float
    auc: float


class AddTrainDataRequest(BaseModel):
    message: str
    label: str  # 'spam' or 'ham'

class AddTrainDataResponse(BaseModel):
    success: bool
    message: str


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    label, prob = sd.predict_text(req.message, model_path=req.model_path or sd.DEFAULT_MODEL_PATH)
    return PredictResponse(label=label, spam_probability=prob)


@app.post("/api/train", response_model=TrainResponse)
def train(req: TrainRequest) -> TrainResponse:
    result = sd.train_and_save(
        data_path=req.data_path or sd.DEFAULT_DATA_PATH,
        model_path=req.model_path or sd.DEFAULT_MODEL_PATH,
        seed=req.seed,
    )
    return TrainResponse(
        model_path=result.model_path,
        accuracy=result.test_accuracy,
        f1_spam=result.test_f1_spam,
        auc=result.test_auc,
    )


@app.post("/api/add_train_data", response_model=AddTrainDataResponse)
def add_train_data(req: AddTrainDataRequest = Body(...)) -> AddTrainDataResponse:
    # Append to spam.csv in the same format as the dataset
    import csv
    csv_path = sd.DEFAULT_DATA_PATH
    # Clean label and message
    label = req.label.strip().lower()
    msg = req.message.strip()
    if label not in ("spam", "ham") or not msg:
        return AddTrainDataResponse(success=False, message="Invalid label or message.")
    # Append to CSV (assume columns: v1,v2,,,) for compatibility
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label, msg, '', '', ''])
    return AddTrainDataResponse(success=True, message="Training data added.")


# Serve the React (CDN) single-file app from ./web
WEB_DIR = os.path.join(os.path.dirname(__file__), "web")
if not os.path.exists(WEB_DIR):
    os.makedirs(WEB_DIR, exist_ok=True)

app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="static")


if __name__ == "__main__":
    # Workaround for Windows WinError 10014 with ProactorEventLoop by switching to SelectorPolicy
    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
        except Exception:
            pass
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=False)

