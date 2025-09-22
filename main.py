from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Firebase
import firebase_admin
from firebase_admin import credentials, db

# ---------------------------------------------------
# FastAPI App + Logging
# ---------------------------------------------------
app = FastAPI()
logging.basicConfig(level=logging.INFO)

# Standard CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

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
# Helpers
# ----------------------
def fetch_sensor_history(sensor_id: str):
    ref = db.reference(f"history/{sensor_id}")
    snapshot = ref.get()
    if not snapshot:
        return []

    records = []
    for _, entry in snapshot.items():
        row = entry.copy()
        ts = row.get("timestamp") or row.get("time")
        if ts:
            try:
                row["timestamp"] = pd.to_datetime(ts)
            except Exception:
                row["timestamp"] = None
        records.append(row)
    return records


def preprocess_dataframe(records, sensor: str):
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if sensor not in df.columns:
        return pd.DataFrame()
    df = df[["timestamp", sensor]].dropna()
    df = df.rename(columns={sensor: "value"})
    return df


def make_lag_features(df, lags=3):
    for lag in range(1, lags+1):
        df[f"lag{lag}"] = df["value"].shift(lag)
    return df.dropna()

# ----------------------
# Endpoints
# ----------------------
@app.get("/predict/{sensor_id}")
def predict(sensor_id: str, sensor: str = Query(..., description="Sensor type"), steps: int = 7):
    try:
        records = fetch_sensor_history(sensor_id)
        df = preprocess_dataframe(records, sensor)
        if df.empty or len(df) < 10:
            return {"sensor_id": sensor_id, "sensor_type": sensor, "forecasts": []}

        df = make_lag_features(df)
        X = df[["lag1", "lag2", "lag3"]].values
        y = df["value"].values.reshape(-1, 1)

        scaler = MinMaxScaler()
        y_scaled = scaler.fit_transform(y).ravel()

        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=200)
        model.fit(X, y_scaled)

        last_lags = df[["lag1", "lag2", "lag3"]].iloc[-1].values.tolist()
        last_date = df["timestamp"].iloc[-1]

        forecasts = []
        for _ in range(steps):
            lag_scaled = scaler.transform(np.array(last_lags).reshape(-1, 1)).ravel()
            pred_scaled = model.predict(lag_scaled.reshape(1, -1))[0]
            pred = float(scaler.inverse_transform([[pred_scaled]])[0][0])
            next_date = last_date + timedelta(days=1)
            forecasts.append({"date": next_date.strftime("%Y-%m-%d"), "forecast": pred})
            last_lags = [pred] + last_lags[:2]
            last_date = next_date

        return {"sensor_id": sensor_id, "sensor_type": sensor, "forecasts": forecasts}
    except Exception as e:
        logging.error(f"Prediction failed: {e}", exc_info=True)
        return {"error": str(e), "sensor_id": sensor_id, "sensor_type": sensor}


@app.get("/compare/{sensor_id}")
def compare(sensor_id: str, sensor: str = Query(..., description="Sensor type"), steps: int = 7):
    try:
        records = fetch_sensor_history(sensor_id)
        df = preprocess_dataframe(records, sensor)
        if df.empty or len(df) < 20:
            return {"error": "Not enough data for comparison"}

        df = make_lag_features(df)
        X = df[["lag1", "lag2", "lag3"]].values
        y = df["value"].values

        # Normalize
        scaler = MinMaxScaler()
        y_scaled = scaler.fit_transform(y.reshape(-1, 1)).ravel()

        # Train/test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

        models = {
            "xgboost": xgb.XGBRegressor(objective="reg:squarederror", n_estimators=200),
            "random_forest": RandomForestRegressor(n_estimators=100),
            "neural_network": MLPRegressor(hidden_layer_sizes=(50,), max_iter=500),
        }

        results = {}
        last_lags = df[["lag1", "lag2", "lag3"]].iloc[-1].values.tolist()
        last_date = df["timestamp"].iloc[-1]

        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                test_preds = model.predict(X_test)
                mse = float(mean_squared_error(y_test, test_preds))

                # Forecast steps
                preds = []
                tmp_lags = last_lags.copy()
                tmp_date = last_date
                for _ in range(steps):
                    lag_scaled = scaler.transform(np.array(tmp_lags).reshape(-1, 1)).ravel()
                    pred_scaled = model.predict(lag_scaled.reshape(1, -1))[0]
                    pred = float(scaler.inverse_transform([[pred_scaled]])[0][0])
                    tmp_date = tmp_date + timedelta(days=1)
                    preds.append({"date": tmp_date.strftime("%Y-%m-%d"), "forecast": pred})
                    tmp_lags = [pred] + tmp_lags[:2]

                results[name] = {"mse": mse, "forecasts": preds}
            except Exception as e:
                results[name] = {"error": str(e)}

        return {"sensor_id": sensor_id, "sensor_type": sensor, "comparison": results}
    except Exception as e:
        logging.error(f"Comparison failed: {e}", exc_info=True)
        return {"error": str(e), "sensor_id": sensor_id, "sensor_type": sensor}


@app.get("/dataframe/{sensor_id}")
def dataframe(sensor_id: str, sensor: str = Query(..., description="Sensor type")):
    records = fetch_sensor_history(sensor_id)
    df = preprocess_dataframe(records, sensor)
    if df.empty:
        return {"sensor_id": sensor_id, "sensor_type": sensor, "data": []}
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return {"sensor_id": sensor_id, "sensor_type": sensor, "data": df.to_dict(orient="records")}


@app.get("/health")
def health():
    return {"status": "ok"}
