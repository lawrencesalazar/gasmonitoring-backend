from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, StreamingResponse
import os
import json
import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ML Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# Firebase
import firebase_admin
from firebase_admin import credentials, db

# Safe matplotlib import
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-GUI backend for servers
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# ---------------------------------------------------
# FastAPI App
# ---------------------------------------------------
app = FastAPI()


# ---------------------------------------------------
# ✅ Firebase Setup (from ENV)
# ---------------------------------------------------
service_account_info = json.loads(os.environ["FIREBASE_SERVICE_ACCOUNT"])
database_url = os.getenv(
    "FIREBASE_DB_URL",
    "https://gasmonitoring-ec511-default-rtdb.asia-southeast1.firebasedatabase.app"
)

if not firebase_admin._apps:
    cred = credentials.Certificate(service_account_info)
    firebase_admin.initialize_app(cred, {"databaseURL": database_url})


# ---------------------------------------------------
# Helper: Fetch Sensor History
# ---------------------------------------------------
def fetch_sensor_history(sensor_id: str):
    ref = db.reference(f"history/{sensor_id}")
    data = ref.get()
    if not data:
        return []

    records = []
    for _, entry in data.items():
        try:
            ts = entry.get("timestamp") or entry.get("time")
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except Exception:
            dt = datetime.now()

        records.append({
            "timestamp": dt,
            "methane": float(entry.get("methane", 0)),
            "co2": float(entry.get("co2", 0)),
            "ammonia": float(entry.get("ammonia", 0)),
            "humidity": float(entry.get("humidity", 0)),
            "temperature": float(entry.get("temperature", 0)),
        })

    return sorted(records, key=lambda x: x["timestamp"])


# ---------------------------------------------------
# Forecast (XGBoost)
# ---------------------------------------------------
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
    forecast_data = [
        {"date": d.strftime("%Y-%m-%d"), "forecast": float(v)}
        for d, v in zip(forecast_dates, preds)
    ]

    return {"sensor_id": sensor_id, "forecast": forecast_data}
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------
# Compare Models (XGBoost, RF, NN) with MSE
# ---------------------------------------------------
@app.get("/compare/{sensor_id}")
def compare(sensor_id: str, sensor: str = "methane", steps: int = 7):
    """
    Compare forecasting models for a given sensor_id and sensor type.
    Example: /compare/321?sensor=ammonia&steps=7
    """
    records = fetch_sensor_history(sensor_id)
    if not records:
        return JSONResponse({"error": "No data found"}, status_code=404)

    df = pd.DataFrame(records)

    # ✅ Ensure the sensor exists in data
    if sensor not in df.columns:
        return JSONResponse(
            {"error": f"Sensor '{sensor}' not found. Available: {list(df.columns)}"},
            status_code=400,
        )

    values = df[sensor].values
    X = np.arange(len(values)).reshape(-1, 1)
    y = values

    # Train/test split (last 20% as test set)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

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
            # Train model
            model.fit(X_train, y_train)

            # Predict test set and compute MSE
            test_preds = model.predict(X_test)
            mse = mean_squared_error(y_test, test_preds)

            # Forecast future values
            future_preds = model.predict(future_X)

            results[name] = {
                "mse": mse,
                "forecasts": [
                    {"date": d.strftime("%Y-%m-%d"), "forecast": float(v)}
                    for d, v in zip(forecast_dates, future_preds)
                ],
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    return {
        "sensor_id": sensor_id,
        "sensor_type": sensor,
        "comparison": results
    }
# ---------------------------------------------------
# Predict Forecast for a given sensor
# ---------------------------------------------------
@app.get("/predict/{sensor_id}")
def predict(sensor_id: str, sensor: str = "methane", steps: int = 7):
    """
    Forecast values for a given sensor type (e.g., methane, ammonia, co2, humidity, temperature, riskIndex).
    Example: /forecast/321?sensor=riskIndex&steps=7
    """
    records = fetch_sensor_history(sensor_id)
    if not records:
        return JSONResponse({"error": "No data found"}, status_code=404)

    df = pd.DataFrame(records)

    # ✅ Ensure the sensor exists
    if sensor not in df.columns:
        return JSONResponse(
            {"error": f"Sensor '{sensor}' not found. Available: {list(df.columns)}"},
            status_code=400,
        )

    values = df[sensor].values
    X = np.arange(len(values)).reshape(-1, 1)
    y = values

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)

    try:
        model.fit(X, y)

        # Forecast future steps
        future_X = np.arange(len(values), len(values) + steps).reshape(-1, 1)
        future_preds = model.predict(future_X)

        last_date = pd.to_datetime(df["timestamp"].iloc[-1])
        forecast_dates = [last_date + timedelta(days=i + 1) for i in range(steps)]

        forecast_result = [
            {"date": d.strftime("%Y-%m-%d"), "forecast": float(v)}
            for d, v in zip(forecast_dates, future_preds)
        ]

        # ✅ Special handling: also predict forecast riskIndex if sensor != riskIndex
        riskIndex_forecast = None
        if sensor != "riskIndex" and "riskIndex" in df.columns:
            risk_values = df["riskIndex"].values
            X_risk = np.arange(len(risk_values)).reshape(-1, 1)
            y_risk = risk_values

            risk_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
            risk_model.fit(X_risk, y_risk)

            risk_preds = risk_model.predict(
                np.arange(len(risk_values), len(risk_values) + 1).reshape(-1, 1)
            )
            riskIndex_forecast = float(risk_preds[0])

        return {
            "sensor_id": sensor_id,
            "sensor_type": sensor,
            "forecasts": forecast_result,
            "riskIndex_next_day": riskIndex_forecast,
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
# ---------------------------------------------------
# Health Check
# ---------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
