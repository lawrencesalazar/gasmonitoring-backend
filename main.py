from fastapi import FastAPI
import firebase_admin
from firebase_admin import credentials, db
import os, json
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

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
# Forecast Endpoint (XGBoost)
# -------------------------------------------------
@app.get("/forecast/{sensor_id}")
def forecast(sensor_id: str, steps: int = 7):
    ref = db.reference(f"history/{sensor_id}")
    data = ref.get()
    if not data:
        return {"forecast": [], "message": "No data found"}

    records = []
    for rec in data.values():
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

    if len(records) < 5:
        return {"forecast": [], "message": "Not enough data"}

    y = np.array(records, dtype=float)
    X = np.arange(len(y)).reshape(-1, 1)

    dtrain = xgb.DMatrix(X, label=y)
    params = {"objective": "reg:squarederror", "max_depth": 3, "eta": 0.1}
    model = xgb.train(params, dtrain, num_boost_round=50)

    future_X = np.arange(len(y), len(y) + steps).reshape(-1, 1)
    forecast = model.predict(xgb.DMatrix(future_X)).tolist()

    return {"forecast": forecast}

# -------------------------------------------------
# Compare Models Endpoint (XGBoost, Random Forest, Neural Net)
# -------------------------------------------------
@app.get("/compare/{sensor_id}")
def compare_models(sensor_id: str, steps: int = 7):
    ref = db.reference(f"history/{sensor_id}")
    data = ref.get()
    if not data:
        return {"xgboost": [], "random_forest": [], "neural_net": [], "message": "No data found"}

    records = []
    for rec in data.values():
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

    if len(records) < 5:
        return {"xgboost": [], "random_forest": [], "neural_net": [], "message": "Not enough data"}

    y = np.array(records, dtype=float)
    X = np.arange(len(y)).reshape(-1, 1)

    # ---- XGBoost ----
    dtrain = xgb.DMatrix(X, label=y)
    params = {"objective": "reg:squarederror", "max_depth": 3, "eta": 0.1}
    model_xgb = xgb.train(params, dtrain, num_boost_round=50)
    future_X = np.arange(len(y), len(y) + steps).reshape(-1, 1)
    forecast_xgb = model_xgb.predict(xgb.DMatrix(future_X)).tolist()

    # ---- Random Forest ----
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    forecast_rf = rf.predict(future_X).tolist()

    # ---- Neural Network ----
    nn = MLPRegressor(hidden_layer_sizes=(50, 20), max_iter=1000, random_state=42)
    nn.fit(X, y)
    forecast_nn = nn.predict(future_X).tolist()

    return {
        "xgboost": forecast_xgb,
        "random_forest": forecast_rf,
        "neural_net": forecast_nn
    }
