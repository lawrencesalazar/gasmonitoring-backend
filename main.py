from fastapi import FastAPI
import firebase_admin
from firebase_admin import credentials, db
import os, json
import numpy as np
import xgboost as xgb

app = FastAPI()

# Firebase Admin setup
service_account_info = json.loads(os.environ['FIREBASE_SERVICE_ACCOUNT'])
cred = credentials.Certificate(service_account_info)
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://gasmonitoring-ec511.firebaseio.com/"
})

@app.get("/")
def home():
    return {"message": "Python API is running on Render"}

@app.get("/readings")
def get_readings():
    ref = db.reference("sensorReadings")
    return ref.get()

@app.post("/readings")
def add_reading(data: dict):
    ref = db.reference("sensorReadings")
    ref.push(data)
    return {"status": "success"}

# -------------------------------------------------
# 🆕 Forecast Endpoint (XGBoost placeholder)
# -------------------------------------------------
@app.get("/forecast/{sensor_id}")
def forecast(sensor_id: str, steps: int = 7):
    """
    Train a dummy XGBoost regressor on historical riskIndex data
    and predict 'steps' future values.
    """
    # Get historical readings
    ref = db.reference(f"history/{sensor_id}")
    data = ref.get()
    if not data:
        return {"forecast": [], "message": "No data found for this sensor"}

    # Extract riskIndex = weighted average
    records = []
    for rec in data.values():
        try:
            methane = rec.get("methane", 0)
            co2 = rec.get("co2", 0)
            ammonia = rec.get("ammonia", 0)
            humidity = rec.get("humidity", 0)
            temperature = rec.get("temperature", 0)
            riskIndex = (
                methane * 0.3 + co2 * 0.25 + ammonia * 0.25 +
                humidity * 0.1 + temperature * 0.1
            )
            records.append(riskIndex)
        except Exception:
            continue

    if len(records) < 5:
        return {"forecast": [], "message": "Not enough data for forecasting"}

    # Prepare training data
    y = np.array(records, dtype=float)
    X = np.arange(len(y)).reshape(-1, 1)

    dtrain = xgb.DMatrix(X, label=y)

    params = {
        "objective": "reg:squarederror",
        "max_depth": 3,
        "eta": 0.1,
        "verbosity": 0
    }

    model = xgb.train(params, dtrain, num_boost_round=50)

    # Forecast next N steps
    future_X = np.arange(len(y), len(y) + steps).reshape(-1, 1)
    dfuture = xgb.DMatrix(future_X)
    forecast = model.predict(dfuture).tolist()

    return {"forecast": forecast}
