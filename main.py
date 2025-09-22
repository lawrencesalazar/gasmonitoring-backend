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

# Standard CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], #["https://gasmonitoring-ec511.web.app", "http://localhost:3000"],  # Replace "*" with your frontend URL in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

#  Global middleware to force CORS headers on all responses
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# Handle preflight OPTIONS requests explicitly
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

def make_features(df: pd.DataFrame, target_col: str, lags: int = 7):
    """Generate lag features for time-series forecasting"""
    for i in range(1, lags+1):
        df[f"{target_col}_lag{i}"] = df[target_col].shift(i)
    df = df.dropna()
    return df

def train_xgboost(df: pd.DataFrame, target_col: str, lags: int = 7):
    """Train XGBoost on historical sensor data"""
    df = make_features(df, target_col, lags)

    X = df.drop(columns=[target_col, "timestamp"], errors="ignore")
    y = df[target_col]

    if len(X) < 10:  # not enough data
        return None, None

    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror"
    )
    model.fit(X, y, verbose=False)
    return model, df

# ---------------------------------------------------
# PreProcess DataFrame
# ---------------------------------------------------
def preprocess_sensor_data(records, resample_freq: str = "D"):
    """
    Preprocess sensor records:
      - Convert timestamps
      - Handle missing values
      - Resample (default: daily)
      - Scale numeric values if needed
    """

    df = pd.DataFrame(records)

    if df.empty:
        return df

    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # if "timestamp" in df.columns:
        # df["timestamp"] = df["timestamp"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if not isinstance(x, str) else x)

    # Set timestamp as index
    df = df.set_index("timestamp").sort_index()

    # Resample (e.g. daily averages)
    df = df.resample(resample_freq).mean()

    # Fill missing values (forward-fill, then back-fill as fallback)
    df = df.ffill().bfill()
 
    # Example feature engineering: rolling average
    df["methane_rolling"] = df["methane"].rolling(window=3, min_periods=1).mean()

    return df.reset_index()

# ---------------------------------------------------
# Display Dataframe 
# ---------------------------------------------------
@app.get("/dataframe/{sensor_id}")
def dataframe(sensor_id: str, steps: int = 7):
    records = fetch_sensor_history(sensor_id)
    
    if not records:
        return JSONResponse({"error": f"No history found for {sensor_id}"}, status_code=404)
  
    # Convert to DataFrame
    df = preprocess_sensor_data(records, resample_freq="D")  
    # print(df)
     
    # Convert DataFrame to JSON-safe dict
    return JSONResponse(content=df.to_dict(orient="records"))

# ---------------------------------------------------
# Forecast (XGBoost)
# ---------------------------------------------------
@app.get("/forecast/{sensor_id}")
def forecast(sensor_id: str, steps: int = 7):
    records = fetch_sensor_history(sensor_id)
    if not records:
        return JSONResponse({"error": "No data found"}, status_code=404)

    df = preprocess_sensor_data(records, resample_freq="D") 
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

    df = preprocess_sensor_data(records, resample_freq="D") 

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
    sensor: str = Query(..., description="Sensor type: methane, co2, ammonia, temperature, humidity"),
    steps: int = 7
):
    # Fetch history from Firebase
    df = fetch_sensor_history(sensor_id, sensor)
    if df.empty:
        return {"error": f"No data found for sensor {sensor_id} ({sensor})"}

    model, processed_df = train_xgboost(df.copy(), sensor)
    if model is None:
        return {"error": "Not enough data to train XGBoost"}

    # Prepare last known data for forecasting
    features_df = make_features(df.copy(), sensor)
    last_row = features_df.iloc[-1:].drop(columns=[sensor, "timestamp"], errors="ignore")

    preds: List[float] = []
    for _ in range(steps):
        y_pred = model.predict(last_row)[0]
        preds.append(float(y_pred))

        # Roll forward: simulate feeding prediction as new lag
        new_row = last_row.copy()
        for i in range(2, len(new_row.columns) + 1):
            if f"{sensor}_lag{i}" in new_row.columns:
                new_row[f"{sensor}_lag{i}"] = new_row[f"{sensor}_lag{i-1}"]
        new_row[f"{sensor}_lag1"] = y_pred
        last_row = new_row

    forecast_result = [
        {"date": str(pd.Timestamp.now().normalize() + pd.Timedelta(days=i+1)), "forecast": preds[i]}
        for i in range(steps)
    ]

    return {
        "sensor_id": sensor_id,
        "sensor_type": sensor,
        "forecasts": forecast_result
    }
#---------------------------
# Predict Methane
# -----------------------------
def make_features(df, target_col="methane", lags=7):
    # Add lag features (last N readings)
    for i in range(1, lags+1):
        df[f"{target_col}_lag{i}"] = df[target_col].shift(i)
    df = df.dropna()
    return df

import xgboost as xgb
from sklearn.model_selection import train_test_split
def train_xgboost(df, target_col="methane"):
    df = make_features(df, target_col)
    
    X = df.drop(columns=[target_col, "timestamp"])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    return model

@app.get("/methane-forecast/{sensor_id}")
def methane_forecast(sensor_id: str, steps: int = 7):
    records = fetch_sensor_history(sensor_id)
    df = preprocess_sensor_data(records)

    model = train_xgboost(df, target_col="methane")

    # Prepare last known data as input for forecasting
    last_data = make_features(df).iloc[-1:].drop(columns=["methane", "timestamp"])

    preds = []
    for _ in range(steps):
        y_pred = model.predict(last_data)[0]
        preds.append(float(y_pred))

        # Roll forward (simulate future lags)
        new_row = last_data.copy()
        new_row.iloc[0, 0] = y_pred  # inject prediction
        last_data = new_row

    return {"sensor_id": sensor_id, "forecasts": preds}

# ---------------------------------------------------
# Health Check
# ---------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
