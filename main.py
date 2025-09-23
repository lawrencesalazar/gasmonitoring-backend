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

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
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
    """
    Return SHAP explanation plot (summary) as PNG + print stats.
    Features: hour + sensor value
    """
    # 1. Fetch data
    records = fetch_sensor_history(sensor_id)
    df = preprocess_dataframe(records, sensor)

    if df.empty or len(df) < 50:
        return {
            "sensor_id": sensor_id,
            "sensor_type": sensor,
            "note": "Not enough data for SHAP"
        }

    # 2. Extract hour feature
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour

    # Features: hour + sensor value
    X = df[["hour", "value"]]
    y = df["value"]

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Train XGBoost model
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X_train, y_train)

    # 5. SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # 6. Compute SHAP stats for each feature
    shap_stats = {}
    for i, feature in enumerate(X.columns):
        vals = X_test.iloc[:, i].values
        shap_vals = shap_values[:, i]
        shap_stats[feature] = {
            "mean_abs_shap": float(np.abs(shap_vals).mean()),
            "impact_range": [float(shap_vals.min()), float(shap_vals.max())],
            "correlation": float(np.corrcoef(vals, shap_vals)[0, 1]),
        }

    # 7. Plot SHAP summary
    buf = io.BytesIO()
    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=["hour", f"{sensor}_value"],
        show=False
    )
    plt.title(f"SHAP Summary for {sensor}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()

    buf.seek(0)

    # 8. Return both stats + image
    return {
        "sensor_id": sensor_id,
        "sensor_type": sensor,
        "shap_stats": shap_stats,
        "plot": StreamingResponse(buf, media_type="image/png")
    }

@app.get("/plot/{sensor_id}")
def plot(
    sensor_id: str,
    sensor: str = Query(..., description="Sensor type"),
    chart: str = Query("summary", description="Chart type: summary or scatter")
):
    """
    SHAP explanation endpoint (hour + sensor value).
    chart = "summary" -> SHAP summary plot
    chart = "scatter" -> SHAP scatter plot (hour vs SHAP values)
    """
    # 1. Fetch data
    records = fetch_sensor_history(sensor_id)
    df = preprocess_dataframe(records, sensor)

    if df.empty or len(df) < 50:
        return JSONResponse(
            {"error": "Not enough data for SHAP analysis"},
            status_code=400
        )

    # 2. Extract hour from timestamp
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour

    # Features: hour + sensor value
    X = df[["hour", "value"]]
    y = df["value"]

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Train XGBoost
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X_train, y_train)

    # 5. SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Feature names (use actual features, not lags)
    feature_names = ["hour", f"{sensor}_value"]

    # 6. Choose chart type
    buf = io.BytesIO()
    if chart == "scatter":
        # Scatter plot: SHAP value vs hour
        plt.figure(figsize=(8, 6))
        hour_idx = list(X.columns).index("hour")
        plt.scatter(
            X_test["hour"].values,
            shap_values[:, hour_idx],
            alpha=0.6,
            c=X_test["hour"].values,
            cmap="viridis",
            edgecolors="k",
            linewidth=0.3
        )
        plt.xlabel("Hour of Day", fontsize=12, fontweight="bold")
        plt.ylabel("SHAP value for Hour", fontsize=12, fontweight="bold")
        plt.title(f"SHAP Scatter for Hour ({sensor})", fontsize=14, fontweight="bold")
        plt.colorbar(label="Hour")
        plt.axhline(y=0, color="black", linestyle="--", alpha=0.6)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
    else:
        # SHAP summary plot
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.title(f"SHAP Summary Plot for {sensor}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()

    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

 
@app.get("/shap_hour/{sensor_id}")
def shap_hour(sensor_id: str, sensor: str = Query(..., description="Sensor type")):
    # 1. Fetch sensor history
    records = fetch_sensor_history(sensor_id)
    df = preprocess_dataframe(records, sensor)
    if df.empty or len(df) < 50:
        return {"error": "Not enough data for SHAP hour analysis"}

    # 2. Extract hour feature from timestamp
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour

    # Use hour + sensor value as features
    X = df[["hour", "value"]]
    y = df["value"]

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 5. SHAP analysis
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Focus on "hour"
    hour_idx = list(X.columns).index("hour")
    hour_values = X_test["hour"].values
    hour_shap = shap_values[:, hour_idx]

    # === SHAP stats ===
    mean_abs_shap = float(np.abs(hour_shap).mean())
    shap_min, shap_max = float(hour_shap.min()), float(hour_shap.max())
    correlation = float(np.corrcoef(hour_values, hour_shap)[0, 1])

    hour_stats = (
        pd.DataFrame({"hour": hour_values, "shap_value": hour_shap})
        .groupby("hour")["shap_value"]
        .agg(["mean", "std", "count"])
        .round(3)
        .reset_index()
        .to_dict(orient="records")
    )

    stats = {
        "mean_abs_shap": mean_abs_shap,
        "impact_range": [shap_min, shap_max],
        "correlation": correlation,
        "hourly_stats": hour_stats,
    }

    # 6. Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(
        hour_values,
        hour_shap,
        alpha=0.6,
        s=30,
        color='#1f77b4',
        edgecolors='black',
        linewidth=0.3
    )
    plt.ylim(-7.5, 10.0)
    plt.yticks([-7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0])
    plt.xlim(0, 23)
    plt.xticks(range(0, 24, 3))
    plt.ylabel('SHAP value for hour', fontsize=12, fontweight='bold')
    plt.xlabel('Hour of Day', fontsize=12, fontweight='bold')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
    plt.grid(True, axis='y', alpha=0.3, linestyle='--')
    plt.grid(False, axis='x')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()

    # 7. Save plot to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)

    # 8. Encode image in base64 for JSON
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
