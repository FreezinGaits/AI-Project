"""
FastAPI backend for Next-Location Prediction.
Run: uvicorn app:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(title="Next-Location Prediction API", version="1.0.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model artifacts ──
ARTIFACTS = "model_artifacts"

if not os.path.exists(ARTIFACTS):
    raise RuntimeError(
        "model_artifacts/ not found. Run the notebook first to export the model."
    )

model = joblib.load(f"{ARTIFACTS}/best_model.pkl")
le_dict = joblib.load(f"{ARTIFACTS}/label_encoders.pkl")
le_target = joblib.load(f"{ARTIFACTS}/target_encoder.pkl")
feature_cols = joblib.load(f"{ARTIFACTS}/feature_cols.pkl")
markov_matrix = joblib.load(f"{ARTIFACTS}/markov_matrix.pkl")
model_accuracies = joblib.load(f"{ARTIFACTS}/model_accuracies.pkl")

HOUR_MAP = {"morning": 9, "afternoon": 14, "evening": 18}


# ── Request / Response schemas ──
class PredictionRequest(BaseModel):
    current_location: str   # stage, canteen, lab
    time_of_day: str        # morning, afternoon, evening
    event_type: str         # technical, cultural, sports
    crowd_density: str      # low, medium, high


class PredictionResponse(BaseModel):
    predicted_location: str
    markov_prediction: str
    confidence: float
    all_probabilities: dict
    markov_probabilities: dict


# ── Endpoints ──
@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    # Encode inputs
    enc_current = le_dict["current_location"].transform([req.current_location])[0]
    enc_event = le_dict["event_type"].transform([req.event_type])[0]
    enc_crowd = le_dict["crowd_density"].transform([req.crowd_density])[0]
    enc_tod = le_dict["time_of_day"].transform([req.time_of_day])[0]

    hour = HOUR_MAP[req.time_of_day]
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    X_input = pd.DataFrame(
        [[enc_current, enc_event, enc_crowd, enc_tod, hour_sin, hour_cos]],
        columns=feature_cols,
    )

    # ML prediction with probabilities
    pred_encoded = model.predict(X_input)[0]
    pred_label = le_target.inverse_transform([pred_encoded])[0]

    probas = model.predict_proba(X_input)[0]
    class_names = le_target.inverse_transform(range(len(probas)))
    all_probs = {name: round(float(p), 4) for name, p in zip(class_names, probas)}
    confidence = round(float(max(probas)), 4)

    # Markov prediction
    markov_row = markov_matrix.loc[req.current_location]
    markov_pred = markov_row.idxmax()
    markov_probs = {k: round(float(v), 4) for k, v in markov_row.items()}

    return PredictionResponse(
        predicted_location=pred_label,
        markov_prediction=markov_pred,
        confidence=confidence,
        all_probabilities=all_probs,
        markov_probabilities=markov_probs,
    )


@app.get("/api/model-info")
def model_info():
    return {
        "model_type": type(model).__name__,
        "features": feature_cols,
        "locations": list(le_target.classes_),
        "model_accuracies": {k: round(v, 4) for k, v in model_accuracies.items()},
    }


# ── Serve frontend ──
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse("frontend/index.html")
