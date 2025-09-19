
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, StreamingResponse
import os, json, io
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import firebase_admin
from firebase_admin import credentials, db

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Firebase setup
service_account_info = json.loads(os.environ['FIREBASE_SERVICE_ACCOUNT'])
cred = credentials.Certificate(service_account_info)
firebase_admin.initialize_app(cred, { 
databaseURL: "https://gasmonitoring-ec511-default-rtdb.asia-southeast1.firebasedatabase.app",

 
})

# ✅ Safe matplotlib import
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# 🔹 Helper: Fetch history
def fetch_sensor_history(sensor_id: str):
    ref = db.reference(f"history/{sensor_id}")
    data = ref.get()
    if not data:
        return []
    records = []
    for _, entry in data.items():
        try:
            dt = datetime.strptime(entry.get("timestamp"), "%Y-%m-%d %H:%M:%S")
        except Exception:
            dt = datetime.now()
        records.append({
            "timestamp": dt,
            "methane": entry.get("methane", 0),
            "co2": entry.get("co2", 0),
            "ammonia": entry.get("ammonia", 0),
            "humidity": entry.get("humidity", 0),
            "temperature": entry.get("temperature", 0),
        })
    return sorted(records, key=lambda x: x["timestamp"])


# 🔹 Forecast with XGBoost
@app.get("/forecast/{sensor_id}")
def forecast(sensor_id: str, steps: int = 7):
    records = fetch_sensor_history(sensor_id)
    if not records:
        return JSONResponse({"error": "No data found"}, status_code=404)

    df = pd.DataFrame(records)
    values = df["methane"].values
    X = np.arange(len(values)).reshape(-1, 1)
    y = values

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X, y)

    future_X = np.arange(len(values), len(values) + steps).reshape(-1, 1)
    preds = model.predict(future_X)

    last_date = df["timestamp"].iloc[-1]
    forecast_dates = [last_date + timedelta(days=i + 1) for i in range(steps)]
    forecast_data = [{"date": d.strftime("%Y-%m-%d"), "forecast": float(v)} for d, v in zip(forecast_dates, preds)]

    return {"sensor_id": sensor_id, "forecast": forecast_data}


# 🔹 Compare 3 models
@app.get("/compare/{sensor_id}")
def compare(sensor_id: str, steps: int = 7):
    records = fetch_sensor_history(sensor_id)
    if not records:
        return JSONResponse({"error": "No data found"}, status_code=404)

    df = pd.DataFrame(records)
    values = df["methane"].values
    X = np.arange(len(values)).reshape(-1, 1)
    y = values

    models = {
        "xgboost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3),
        "random_forest": RandomForestRegressor(n_estimators=100),
        "neural_network": MLPRegressor(hidden_layer_sizes=(50,), max_iter=500),
    }

    results = {}
    future_X = np.arange(len(values), len(values) + steps).reshape(-1, 1)
    last_date = df["timestamp"].iloc[-1]
    forecast_dates = [last_date + timedelta(days=i + 1) for i in range(steps)]

    for name, model in models.items():
        try:
            model.fit(X, y)
            preds = model.predict(future_X)
            results[name] = [{"date": d.strftime("%Y-%m-%d"), "forecast": float(v)} for d, v in zip(forecast_dates, preds)]
        except Exception as e:
            results[name] = {"error": str(e)}

    return {"sensor_id": sensor_id, "comparison": results}
