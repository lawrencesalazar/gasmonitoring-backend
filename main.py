from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import os
import json
import io
import base64
from datetime import timedelta

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")  # For headless servers (Render)
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb

# Firebase
import firebase_admin
from firebase_admin import credentials, db

# ---------------------------------------------------
# FastAPI App
# ---------------------------------------------------
app = FastAPI()

# Standard CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with frontend URL in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Global middleware to force CORS headers
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

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

# ----------------------
# Helper Functions
# ----------------------
def fetch_sensor_history(sensor_id: str):
    """Fetch sensor history from Firebase and return as list of dicts"""
    ref = db.reference(f"history/{sensor_id}")
    snapshot = ref.get()
    if not snapshot:
        return []

    records = []
    for _, value in snapshot.items():
        row = value.copy()
        if "timestamp" in row:
            try:
                row["timestamp"] = pd.to_datetime(row["timestamp"])
            except Exception:
                row["timestamp"] = None
        records.append(row)
    return records


def preprocess_dataframe(records, sensor: str):
    """Preprocess raw Firebase records into a clean DataFrame for ML"""
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp")

    if sensor not in df.columns:
        return pd.DataFrame()

    df = df[["timestamp", sensor]].dropna()
    df = df.rename(columns={sensor: "value"})
    return df


def make_lag_features(df: pd.DataFrame):
    """Add lag features for time-series forecasting"""
    df["lag1"] = df["value"].shift(1)
    df["lag2"] = df["value"].shift(2)
    df["lag3"] = df["value"].shift(3)
    return df.dropna()


def train_xgboost(df: pd.DataFrame, steps: int = 7):
    """Train XGBoost model and forecast future values"""
    df = make_lag_features(df)
    if df.empty:
        return []

    X = df[["lag1", "lag2", "lag3"]].values
    y = df["value"].values

    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X, y)

    # Prepare last known lags
    last_lags = X[-1].copy()
    last_date = df["timestamp"].iloc[-1]

    predictions = []
    for i in range(steps):
        pred = model.predict(last_lags.reshape(1, -1))[0]
        next_date = last_date + timedelta(days=1)
        predictions.append({
            "date": next_date.strftime("%Y-%m-%d"),
            "forecast": float(pred)
        })
        # Update lags
        last_lags = np.roll(last_lags, -1)
        last_lags[-1] = pred
        last_date = next_date

    return predictions


# ----------------------
# Endpoints
# ----------------------
@app.get("/predict/{sensor_id}")
def predict(sensor_id: str, sensor: str = Query(...), steps: int = 7):
    records = fetch_sensor_history(sensor_id)
    df = preprocess_dataframe(records, sensor)

    if df.empty:
        return {"sensor_id": sensor_id, "sensor_type": sensor, "forecasts": []}

    forecasts = train_xgboost(df, steps=steps)
    return {
        "sensor_id": sensor_id,
        "sensor_type": sensor,
        "forecasts": forecasts
    }


@app.get("/dataframe/{sensor_id}")
def dataframe(sensor_id: str, sensor: str = Query(...)):
    records = fetch_sensor_history(sensor_id)
    df = preprocess_dataframe(records, sensor)

    if df.empty:
        return {"sensor_id": sensor_id, "sensor_type": sensor, "data": []}

    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return {
        "sensor_id": sensor_id,
        "sensor_type": sensor,
        "data": df.to_dict(orient="records")
    }


@app.get("/explain/{sensor_id}")
def explain(sensor_id: str, sensor: str = Query(...)):
    """Return SHAP explanation plot as PNG"""
    records = fetch_sensor_history(sensor_id)
    df = preprocess_dataframe(records, sensor)
    if df.empty or len(df) < 20:
        return {"sensor_id": sensor_id, "sensor_type": sensor, "note": "Not enough data for SHAP"}

    df = make_lag_features(df)
    X = df[["lag1", "lag2", "lag3"]].values
    y = df["value"].values

    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y.reshape(-1, 1)).ravel()

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train = y_scaled[:split_idx]

    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:20])
    shap_values = np.array(shap_values)

    feature_names = ["lag1", "lag2", "lag3"]

    plt.figure()
    shap.summary_plot(shap_values, X_test[:len(shap_values)], feature_names=feature_names, show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")

from fastapi.responses import StreamingResponse

@app.get("/plot/{sensor_id}")
def plot(
    sensor_id: str,
    sensor: str = Query(..., description="Sensor type"),
    chart: str = Query("summary", description="Chart type: summary or scatter")
):
    """
    SHAP explanation endpoint.
    chart = "summary" -> SHAP summary plot
    chart = "scatter" -> SHAP scatter plot (sensor_lag1 vs SHAP values)
    """
    records = fetch_sensor_history(sensor_id)
    df = preprocess_dataframe(records, sensor)

    if df.empty or len(df) < 20:
        return JSONResponse(
            {"error": "Not enough data for SHAP analysis"},
            status_code=400
        )

    # Create lag features
    df = make_lag_features(df)
    X = df[["lag1", "lag2", "lag3"]].values
    y = df["value"].values

    # Normalize target
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y.reshape(-1, 1)).ravel()

    # Train/test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train = y_scaled[:split_idx]

    # Train XGBoost model
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X_train, y_train)

    # SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:50])

    # Feature names (use sensor name instead of generic)
    feature_names = [f"{sensor}_lag1", f"{sensor}_lag2", f"{sensor}_lag3"]

    # Choose chart type
    buf = io.BytesIO()
    if chart == "scatter":
        plt.figure()
        plt.scatter(X_test[:50, 0], shap_values[:50, 0], alpha=0.6)
        plt.xlabel(f"{sensor}_lag1 values")
        plt.ylabel("SHAP value")
        plt.title(f"SHAP Scatter for {sensor}_lag1")
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
    else:
        plt.figure()
        shap.summary_plot(shap_values, X_test[:50], feature_names=feature_names, show=False)
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()

    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.get("/health")
def health():
    return {"status": "ok"}
