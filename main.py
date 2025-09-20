from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
import os
import json
from datetime import datetime, timedelta

from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import pandas as pd

# ML Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

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

# ✅ Standard CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend URL in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ✅ Global middleware to force CORS headers on all responses
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# ✅ Handle preflight OPTIONS requests explicitly
@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str):
    return JSONResponse(
        content={"message": "CORS preflight ok"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
    )

# ---------------------------------------------------
# Firebase Setup
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

# ---------------------------------------------------
# Compare Models (XGBoost, RF, NN) with MSE
# ---------------------------------------------------
@app.get("/compare/{sensor_id}")
def compare(sensor_id: str, sensor: str = "methane", steps: int = 7):
    records = fetch_sensor_history(sensor_id)
    if not records:
        return JSONResponse({"error": "No data found"}, status_code=404)

    df = pd.DataFrame(records)

    if sensor not in df.columns:
        return JSONResponse(
            {"error": f"Sensor '{sensor}' not found. Available: {list(df.columns)}"},
            status_code=400,
        )

    values = df[sensor].values
    X = np.arange(len(values)).reshape(-1, 1)
    y = values

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
            model.fit(X_train, y_train)
            test_preds = model.predict(X_test)
            mse = mean_squared_error(y_test, test_preds)
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
def predict(
    sensor_id: str,
    sensor: str = Query(
        "methane",
        enum=["methane", "co2", "ammonia", "humidity", "temperature", "riskIndex"]
    ),
    steps: int = 7
):
    records = fetch_sensor_history(sensor_id)
    if not records:
        return JSONResponse({"error": f"No history data found for sensor {sensor_id}"}, status_code=404)

    df = pd.DataFrame(records)

    # Compute riskIndex if missing
    if "riskIndex" not in df.columns and all(c in df.columns for c in ["methane", "co2", "ammonia"]):
        df["riskIndex"] = 0.4 * df["methane"] + 0.35 * df["co2"] + 0.25 * df["ammonia"]

    if sensor not in df.columns:
        return JSONResponse(
            {"error": f"Sensor '{sensor}' not found. Available: {list(df.columns)}"},
            status_code=400,
        )

    values = df[sensor].dropna().values
    if len(values) < 10:
        return JSONResponse({"error": f"Not enough data for forecasting {sensor}"}, status_code=400)

    # Build lag features
    lags = 3
    X, y = [], []
    for i in range(lags, len(values)):
        X.append(values[i-lags:i])
        y.append(values[i])
    X, y = np.array(X), np.array(y)

    # Train XGBoost
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3)
    model.fit(X, y)

    # Forecast future values iteratively
    last_known = values[-lags:].tolist()
    preds = []
    for _ in range(steps):
        next_val = model.predict(np.array(last_known[-lags:]).reshape(1, -1))[0]
        preds.append(next_val)
        last_known.append(next_val)

    # Build forecast dates
    last_date = pd.to_datetime(df["timestamp"].iloc[-1])
    forecast_dates = [last_date + timedelta(days=i + 1) for i in range(steps)]

    forecast_result = [
        {"date": d.strftime("%Y-%m-%d"), "forecast": float(v)}
        for d, v in zip(forecast_dates, preds)
    ]

    # Forecast riskIndex (next day only)
    riskIndex_forecast = None
    if "riskIndex" in df.columns:
        risk_vals = df["riskIndex"].dropna().values
        if len(risk_vals) >= lags:
            X_risk, y_risk = [], []
            for i in range(lags, len(risk_vals)):
                X_risk.append(risk_vals[i-lags:i])
                y_risk.append(risk_vals[i])
            X_risk, y_risk = np.array(X_risk), np.array(y_risk)

            risk_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3)
            risk_model.fit(X_risk, y_risk)
            riskIndex_forecast = float(risk_model.predict(np.array(risk_vals[-lags:]).reshape(1, -1))[0])

    return {
        "sensor_id": sensor_id,
        "sensor_type": sensor,
        "forecasts": forecast_result,
        "riskIndex_next_day": riskIndex_forecast,
    }

# ---------------------------------------------------
# Health Check
# ---------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
