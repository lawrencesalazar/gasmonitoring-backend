from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import os
import json
import io
import base64
from datetime import timedelta

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")  # Headless servers (Render)
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Firebase
import firebase_admin
from firebase_admin import credentials, db

# ---------------------------------------------------
# FastAPI App
# ---------------------------------------------------
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: replace "*" with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add global CORS headers
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
    for _ in range(steps):
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
    """Return SHAP summary plot + stats"""
    records = fetch_sensor_history(sensor_id)
    df = preprocess_dataframe(records, sensor)

    if df.empty or len(df) < 50:
        return {"note": "Not enough data for SHAP"}

    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    X = df[["hour", "value"]]
    y = df["value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Compute SHAP stats
    shap_stats = {}
    for i, feature in enumerate(X.columns):
        vals = X_test.iloc[:, i].values
        shap_vals = shap_values[:, i]
        shap_stats[feature] = {
            "mean_abs_shap": float(np.abs(shap_vals).mean()),
            "impact_range": [float(shap_vals.min()), float(shap_vals.max())],
            "correlation": float(np.corrcoef(vals, shap_vals)[0, 1]),
        }

    # SHAP summary plot
    buf = io.BytesIO()
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
    plt.title(f"SHAP Summary for {sensor}")
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)

    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {
        "sensor_id": sensor_id,
        "sensor_type": sensor,
        "shap_stats": shap_stats,
        "plot": f"data:image/png;base64,{img_b64}"
    }


@app.get("/plot/{sensor_id}")
def plot(sensor_id: str, sensor: str = Query(...), chart: str = Query("summary")):
    """SHAP summary or scatter plot"""
    records = fetch_sensor_history(sensor_id)
    df = preprocess_dataframe(records, sensor)

    if df.empty or len(df) < 50:
        return JSONResponse({"error": "Not enough data"}, status_code=400)

    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    X = df[["hour", "value"]]
    y = df["value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    buf = io.BytesIO()
    if chart == "scatter":
        plt.figure(figsize=(8, 6))
        plt.scatter(X_test["hour"], shap_values[:, 0], alpha=0.6, c=X_test["hour"], cmap="viridis")
        plt.xlabel("Hour of Day")
        plt.ylabel("SHAP value (Hour)")
        plt.title(f"SHAP Scatter for {sensor}")
        plt.colorbar(label="Hour")
        plt.axhline(y=0, color="black", linestyle="--")
        plt.tight_layout()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
    else:
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
        plt.title(f"SHAP Summary Plot for {sensor}")
        plt.tight_layout()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()

    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/shap_hour/{sensor_id}")
def shap_hour(sensor_id: str, sensor: str = Query(...)):
    """Detailed SHAP stats for hour feature"""
    records = fetch_sensor_history(sensor_id)
    df = preprocess_dataframe(records, sensor)
    if df.empty or len(df) < 50:
        return {"error": "Not enough data"}

    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    X = df[["hour", "value"]]
    y = df["value"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    hour_idx = list(X.columns).index("hour")
    hour_values = X_test["hour"].values
    hour_shap = shap_values[:, hour_idx]

    stats = {
        "mean_abs_shap": float(np.abs(hour_shap).mean()),
        "impact_range": [float(hour_shap.min()), float(hour_shap.max())],
        "correlation": float(np.corrcoef(hour_values, hour_shap)[0, 1]),
    }

    plt.figure(figsize=(10, 6))
    plt.scatter(hour_values, hour_shap, alpha=0.6, s=30, edgecolors="k")
    plt.xlabel("Hour of Day")
    plt.ylabel("SHAP value (Hour)")
    plt.axhline(y=0, color="black", linestyle="--")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "sensor_id": sensor_id,
        "sensor_type": sensor,
        "stats": stats,
        "shap_plot": f"data:image/png;base64,{img_b64}"
    }


@app.get("/health")
def health():
    return {"status": "ok"}
