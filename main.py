from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Train ETA Delay Prediction API")

bundle = joblib.load("eta_delay_model.pkl")
model = bundle["model"]
features = bundle["features"]
le_station = bundle["le_station"]
le_delay_type = bundle["le_delay_type"]


class PredictionRequest(BaseModel):
    speed_kmh: float
    distance_to_station_km: float
    distance_to_destination_km: float
    current_delay_minutes: float
    historical_average_delay_minutes: float
    scheduled_arrival_minutes: float
    time_of_day: int
    day_of_week: int
    station_code: str
    delay_type: str


@app.get("/")
def root():
    return {"status": "ETA Delay Prediction API is running"}


@app.post("/predict")
def predict_delay(req: PredictionRequest):
    if req.station_code in le_station.classes_:
        station_enc = le_station.transform([req.station_code])[0]
    else:
        station_enc = -1

    if req.delay_type in le_delay_type.classes_:
        delay_type_enc = le_delay_type.transform([req.delay_type])[0]
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown delay_type '{req.delay_type}'. Must be one of {list(le_delay_type.classes_)}"
        )

    row = {
        "speed_kmh": req.speed_kmh,
        "distance_to_station_km": req.distance_to_station_km,
        "distance_to_destination_km": req.distance_to_destination_km,
        "current_delay_minutes": req.current_delay_minutes,
        "historical_average_delay_minutes": req.historical_average_delay_minutes,
        "scheduled_arrival_minutes": req.scheduled_arrival_minutes,
        "time_of_day": req.time_of_day,
        "day_of_week": req.day_of_week,
        "Station_Enc": station_enc,
        "Delay_Type_Enc": delay_type_enc,
    }

    X = pd.DataFrame([row])[features]
    prediction = model.predict(X)[0]

    return {
        "predicted_delay_minutes": round(float(prediction), 2),
        "model_used": bundle.get("model_name", "unknown"),
    }