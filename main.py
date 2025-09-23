from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import logging
from datetime import timedelta
import io
import base64

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ML
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import shap

# Firebase
import firebase_admin
from firebase_admin import credentials, db

# ---------------------------------------------------
# FastAPI App
# ---------------------------------------------------
app = FastAPI()
logging.basicConfig(level=logging.INFO)

# Standard CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" in production with frontend URL
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

# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def fetch_sensor_history(sensor_id: str):
    """Fetch sensor history from Firebase and return as list of dicts"""
    ref = db.reference(f"history/{sensor_id}")
    snapshot = ref.get()

    if not snapshot:
        return []

    records = []
    for _, value in snapshot.items():
        row = value.copy()
        # Convert timestamp
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
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

    if sensor not in df.columns:
        return pd.DataFrame()

    df = df[["timestamp", sensor]].dropna()
    df = df.rename(columns={sensor: "value"})
    return df


def make_lag_features(df: pd.DataFrame, lags: int = 3):
    """Add lag features for forecasting"""
    for lag in range(1, lags + 1):
        df[f"lag{lag}"] = df["value"].shift(lag)
    return df.dropna()


def select_best_model(X_train, y_train, X_test, y_test):
    """Train multiple models and select the best one using MSE"""
    models = {
        "xgboost": xgb.XGBRegressor(objective="reg:squarederror", n_estimators=200),
        "random_forest": RandomForestRegressor(n_estimators=100),
        "neural_network": MLPRegressor(hidden_layer_sizes=(50,), max_iter=500),
    }

    best_model, best_name, best_mse = None, None, float("inf")

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            if mse < best_mse:
                best_model, best_name, best_mse = model, name, mse
        except Exception as e:
            logging.error(f"Model {name} failed: {e}")

    return best_model, best_name, best_mse


def forecast_with_model(model, df: pd.DataFrame, scaler, steps: int = 7):
    """Iterative forecasting using the selected model"""
    last_lags = df[["lag1", "lag2", "lag3"]].iloc[-1].values
    last_date = df["timestamp"].iloc[-1]

    forecasts = []
    for _ in range(steps):
        pred_scaled = model.predict(last_lags.reshape(1, -1))[0]
        pred = float(scaler.inverse_transform([[pred_scaled]])[0][0])

        next_date = last_date + timedelta(days=1)
        forecasts.append({"date": next_date.strftime("%Y-%m-%d"), "forecast": pred})

        last_lags = np.roll(last_lags, -1)
        last_lags[-1] = pred_scaled
        last_date = next_date
    return forecasts

# ---------------------------------------------------
# Endpoints
# ---------------------------------------------------
@app.get("/predict/{sensor_id}")
def predict(sensor_id: str, sensor: str = Query(..., description="Sensor type"), steps: int = 7):
    records = fetch_sensor_history(sensor_id)
    df = preprocess_dataframe(records, sensor)
    if df.empty or len(df) < 20:
        return {"sensor_id": sensor_id, "sensor_type": sensor, "forecasts": []}

    df = make_lag_features(df)
    X = df[["lag1", "lag2", "lag3"]].values
    y = df["value"].values

    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y.reshape(-1, 1)).ravel()

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

    best_model, best_name, best_mse = select_best_model(X_train, y_train, X_test, y_test)
    if not best_model:
        return {"error": "No model trained successfully"}

    forecasts = forecast_with_model(best_model, df, scaler, steps)
    return {
        "sensor_id": sensor_id,
        "sensor_type": sensor,
        "selected_model": best_name,
        "mse": best_mse,
        "forecasts": forecasts,
    }


@app.get("/dataframe/{sensor_id}")
def dataframe(sensor_id: str, sensor: str = Query(...)):
    records = fetch_sensor_history(sensor_id)
    df = preprocess_dataframe(records, sensor)
    if df.empty:
        return {"sensor_id": sensor_id, "sensor_type": sensor, "data": []}
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return {"sensor_id": sensor_id, "sensor_type": sensor, "data": df.to_dict(orient="records")}


@app.get("/explain/{sensor_id}")
def explain(sensor_id: str, sensor: str = Query(..., description="Sensor type")):
    records = fetch_sensor_history(sensor_id)
    df = preprocess_dataframe(records, sensor)
    if df.empty or len(df) < 20:
        return {"sensor_id": sensor_id, "sensor_type": sensor, "explanations": []}

    df = make_lag_features(df)
    X = df[["lag1", "lag2", "lag3"]].values
    y = df["value"].values

    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y.reshape(-1, 1)).ravel()

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

    best_model, best_name, best_mse = select_best_model(X_train, y_train, X_test, y_test)
    if not best_model:
        return {"error": "No model trained successfully"}

    if best_name in ["xgboost", "random_forest"]:
        explainer = shap.TreeExplainer(best_model)
    else:
        explainer = shap.KernelExplainer(best_model.predict, X_train[:50])

    shap_values = explainer.shap_values(X_test[:20])
    feature_names = ["lag1", "lag2", "lag3"]

    # SHAP plot to base64
    plt.figure()
    shap.summary_plot(shap_values, X_test[:20], feature_names=feature_names, show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    shap_img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "sensor_id": sensor_id,
        "sensor_type": sensor,
        "selected_model": best_name,
        "mse": best_mse,
        "shap_values": np.array(shap_values).tolist(),
        "shap_summary_plot": f"data:image/png;base64,{shap_img_b64}",
    }


@app.get("/health")
def health():
    return {"status": "ok"}
