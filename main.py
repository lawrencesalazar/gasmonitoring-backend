from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

import os
import json
import io
import base64
from datetime import datetime, timedelta
import math
import gc

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import xgboost as xgb

# Firebase
import firebase_admin
from firebase_admin import credentials, db

# -----------------------------
# FastAPI App & CORS
# -----------------------------
app = FastAPI(title="Gas Monitoring API")

origins = [
    "http://localhost:3000",
    "https://gasmonitoring-ec511.web.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# -----------------------------
# Firebase Setup
# -----------------------------
service_account_info = json.loads(os.environ["FIREBASE_SERVICE_ACCOUNT"])
database_url = os.getenv(
    "FIREBASE_DB_URL",
    "https://gasmonitoring-ec511-default-rtdb.asia-southeast1.firebasedatabase.app"
)

if not firebase_admin._apps:
    cred = credentials.Certificate(service_account_info)
    firebase_admin.initialize_app(cred, {"databaseURL": database_url})

# -----------------------------
# Helper Functions
# -----------------------------
def fetch_sensor_history(sensor_id: str):
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
    df["lag1"] = df["value"].shift(1)
    df["lag2"] = df["value"].shift(2)
    df["lag3"] = df["value"].shift(3)
    return df.dropna()

def train_xgboost(df: pd.DataFrame, steps: int = 7):
    df = make_lag_features(df)
    if df.empty:
        return []
    X = df[["lag1", "lag2", "lag3"]].values
    y = df["value"].values
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X, y)

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
        last_lags = np.roll(last_lags, -1)
        last_lags[-1] = pred
        last_date = next_date

    del model
    gc.collect()
    return predictions

def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else None
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2_score": r2,
        "explained_variance": evs,
        "y_true_sample": y_true[:10].tolist(),
        "y_pred_sample": y_pred[:10].tolist()
    }

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

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

@app.get("/algorithm/{sensor_id}")
def algorithm_analysis(sensor_id: str, sensor: str = Query(...)):
    history = fetch_sensor_history(sensor_id)
    rows = [{"timestamp": r.get("timestamp"), "value": float(r.get(sensor))} for r in history if sensor in r]
    df = pd.DataFrame(rows)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data for this sensor type")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["lag1"] = df["value"].shift(1)
    df["lag2"] = df["value"].shift(2)
    df = df.dropna()
    X = df[["lag1", "lag2"]]
    y = df["value"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    results["LinearRegression"] = compute_metrics(y_test, lr.predict(X_test))
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    results["RandomForest"] = compute_metrics(y_test, rf.predict(X_test))
    # Decision Tree
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    results["DecisionTree"] = compute_metrics(y_test, dt.predict(X_test))
    # SVR
    svr = SVR(kernel="rbf")
    svr.fit(X_train_scaled, y_train)
    results["SVR"] = compute_metrics(y_test, svr.predict(X_test_scaled))
    # XGBoost
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    xgb_model.fit(X_train, y_train)
    results["XGBoost"] = compute_metrics(y_test, xgb_model.predict(X_test))

    del lr, rf, dt, svr, xgb_model, X_train, X_test, y_train, y_test
    gc.collect()

    return {
        "sensor_id": sensor_id,
        "sensor_type": sensor,
        "sample_size": len(df),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "algorithms": results
    }

# -----------------------------
# SHAP & Plot Endpoints
# -----------------------------
@app.get("/explain/{sensor_id}")
def explain(sensor_id: str, sensor: str = Query(...), format: str = Query("json")):
    records = fetch_sensor_history(sensor_id)
    df = preprocess_dataframe(records, sensor)
    if df.empty or len(df) < 50:
        return {"sensor_id": sensor_id, "sensor_type": sensor, "note": "Not enough data for SHAP"}

    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    X = df[["hour", "value"]]
    y = df["value"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X_train, y_train)
    last_row = X.iloc[[-1]]
    prediction = float(model.predict(last_row)[0])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap_stats = {}
    for i, feature in enumerate(X.columns):
        vals = X_test.iloc[:, i].values
        shap_vals = shap_values[:, i]
        shap_stats[feature] = {
            "mean_abs_shap": float(np.abs(shap_vals).mean()),
            "impact_range": [float(shap_vals.min()), float(shap_vals.max())],
            "correlation": float(np.corrcoef(vals, shap_vals)[0, 1]),
        }

    buf = io.BytesIO()
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
    plt.title(f"SHAP Summary for {sensor}")
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close("all")
    buf.seek(0)

    if format == "png":
        return StreamingResponse(buf, media_type="image/png")
    else:
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {
            "sensor_id": sensor_id,
            "sensor_type": sensor,
            "prediction": prediction,
            "shap_stats": shap_stats,
            "shap_plot": f"data:image/png;base64,{img_b64}"
        }

# -----------------------------
# SHAP Hour Endpoint
# -----------------------------
@app.get("/shap_hour/{sensor_id}")
def shap_hour(sensor_id: str, sensor: str = Query(...)):
    records = fetch_sensor_history(sensor_id)
    df = preprocess_dataframe(records, sensor)
    if df.empty or len(df) < 50:
        return {"error": "Not enough data"}
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    X = df[["hour", "value"]]
    y = df["value"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(n_estimators=100)
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
    buf = io.BytesIO()
    plt.figure(figsize=(10, 6))
    plt.scatter(hour_values, hour_shap, alpha=0.6, s=30, edgecolors="k")
    plt.xlabel("Hour of Day")
    plt.ylabel(f"{sensor} value (Hour)")
    plt.axhline(y=0, color="black", linestyle="--")
    plt.tight_layout()
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
