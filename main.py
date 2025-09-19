from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials, db
import os
import logging

# =========================
# Safe imports for heavy libs
# =========================
try:
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    import matplotlib
    matplotlib.use("Agg")  # Safe backend for servers
    import matplotlib.pyplot as plt
except ImportError as e:
    logging.warning(f"Optional dependency missing: {e}")

# =========================
# FastAPI setup
# =========================
app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Firebase setup
# =========================
if not firebase_admin._apps:
    
# service_account_info = json.loads(os.environ['FIREBASE_SERVICE_ACCOUNT'])
# cred = credentials.Certificate(service_account_info)
    cred_path =  json.loads(os.environ['FIREBASE_SERVICE_ACCOUNT'])
    if not os.path.exists(cred_path):
        raise RuntimeError(f"Firebase credential file not found at {cred_path}")
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {
        "databaseURL": os.getenv(
            "FIREBASE_DB_URL",
            "https://gasmonitoring-ec511-default-rtdb.asia-southeast1.firebasedatabase.app/"
        )
    })

# =========================
# Helper: Fetch history by sensorID
# =========================
def fetch_sensor_history(sensor_id: str, limit: int = 30):
    ref = db.reference("history")
    query = ref.order_by_child("sensorID").equal_to(sensor_id).limit_to_last(limit)
    data = query.get()
    if not data:
        return []
    return list(data.values())

# =========================
# Routes
# =========================
@app.get("/")
def root():
    return {"status": "ok", "message": "Gas Monitoring API running"}

@app.get("/forecast/{sensor_id}")
def forecast(sensor_id: str):
    history = fetch_sensor_history(sensor_id)
    if not history:
        raise HTTPException(status_code=404, detail="No history found for sensor")

    df = pd.DataFrame(history)
    if "timestamp" not in df or "co2" not in df:
        raise HTTPException(status_code=500, detail="Missing required fields")

    # Prepare simple forecast (CO₂ vs time index)
    df["ts_index"] = range(len(df))
    X = df[["ts_index"]]
    y = df["co2"]
    model = LinearRegression().fit(X, y)
    future_index = [[len(df) + i] for i in range(5)]
    forecast_values = model.predict(future_index).tolist()

    return {"sensorID": sensor_id, "forecast": forecast_values}

@app.get("/compare/{sensor_a}/{sensor_b}")
def compare(sensor_a: str, sensor_b: str):
    hist_a = fetch_sensor_history(sensor_a)
    hist_b = fetch_sensor_history(sensor_b)
    if not hist_a or not hist_b:
        raise HTTPException(status_code=404, detail="No data for one or both sensors")
    return {
        "sensorA": sensor_a,
        "recordsA": len(hist_a),
        "sensorB": sensor_b,
        "recordsB": len(hist_b),
    }
