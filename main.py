from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import xgboost as xgb
import numpy as np
from datetime import datetime, timedelta

# ----------------------
# FastAPI app setup
# ----------------------
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
# Helper function
# ----------------------
def fetch_sensor_history(sensor_id: str):
    """Fetch sensor history from Firebase and return as list of dicts"""
    ref = db.reference(f"history/{sensor_id}")
    snapshot = ref.get()

    if not snapshot:
        return []

    records = []
    for key, value in snapshot.items():
        row = value.copy()
        # Convert timestamp safely
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

    # Keep only timestamp + sensor column
    if sensor not in df.columns:
        return pd.DataFrame()

    df = df[["timestamp", sensor]].dropna()
    df = df.rename(columns={sensor: "value"})

    return df


def train_xgboost(df: pd.DataFrame, steps: int = 7):
    """Train an XGBoost model on the given sensor data and forecast future values"""
    if df.empty or len(df) < 5:
        return []

    # Create lag features
    df["lag1"] = df["value"].shift(1)
    df["lag2"] = df["value"].shift(2)
    df["lag3"] = df["value"].shift(3)
    df = df.dropna()

    X = df[["lag1", "lag2", "lag3"]]
    y = df["value"]

    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X, y)

    # Prepare last known lags
    last_lags = df[["lag1", "lag2", "lag3"]].iloc[-1].values

    predictions = []
    last_date = df["timestamp"].iloc[-1]

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
def predict(sensor_id: str, sensor: str = Query(..., description="Sensor type (humidity, temperature, methane, co2, ammonia)"), steps: int = 7):
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
def dataframe(sensor_id: str, sensor: str = Query(..., description="Sensor type")):
    records = fetch_sensor_history(sensor_id)
    df = preprocess_dataframe(records, sensor)

    if df.empty:
        return {"sensor_id": sensor_id, "sensor_type": sensor, "data": []}

    # Convert Timestamp to ISO strings
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return {
        "sensor_id": sensor_id,
        "sensor_type": sensor,
        "data": df.to_dict(orient="records")
    }
