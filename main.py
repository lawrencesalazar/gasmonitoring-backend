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
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
        if df.empty or len(df) < 20:
            return {"sensor_id": sensor_id, "sensor_type": sensor, "forecasts": []}

        df = make_lag_features(df)
        X = df[["lag1", "lag2", "lag3"]].values
        y = df["value"].values

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

        best_model = None
        best_name = None
        best_mse = float("inf")

        # Train + evaluate all models
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                test_preds = model.predict(X_test)
                mse = mean_squared_error(y_test, test_preds)
                if mse < best_mse:
                    best_mse = mse
                    best_model = model
                    best_name = name
            except Exception as e:
                logging.error(f"Model {name} failed: {e}")

        if best_model is None:
            return {"error": "No model trained successfully"}

        # Forecast future steps
        last_lags = df[["lag1", "lag2", "lag3"]].iloc[-1].values.tolist()
        last_date = df["timestamp"].iloc[-1]
        forecasts = []

        for _ in range(steps):
            lag_scaled = scaler.transform(np.array(last_lags).reshape(-1, 1)).ravel()
            pred_scaled = best_model.predict(lag_scaled.reshape(1, -1))[0]
            pred = float(scaler.inverse_transform([[pred_scaled]])[0][0])
            next_date = last_date + timedelta(days=1)
            forecasts.append({"date": next_date.strftime("%Y-%m-%d"), "forecast": pred})
            last_lags = [pred] + last_lags[:2]
            last_date = next_date

        return {
            "sensor_id": sensor_id,
            "sensor_type": sensor,
            "selected_model": best_name,
            "mse": best_mse,
            "forecasts": forecasts
        }
    except Exception as e:
        logging.error(f"Prediction failed: {e}", exc_info=True)
        return {"error": str(e), "sensor_id": sensor_id, "sensor_type": sensor}
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