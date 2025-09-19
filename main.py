from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, StreamingResponse
import firebase_admin
from firebase_admin import credentials, db
import io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

app = FastAPI()

# ✅ Firebase setup
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://gasmonitoring-ec511-default-rtdb.firebaseio.com/"
})


# 🔹 Helper: Fetch history
def fetch_sensor_history(sensor_id: str):
    ref = db.reference(f"history/{sensor_id}")
    data = ref.get()
    if not data:
        return []
    records = []
    for ts, entry in data.items():
        try:
            dt = datetime.fromtimestamp(int(ts))  # ✅ Fix timestamp parsing
        except:
            dt = datetime.now()
        records.append({
            "timestamp": dt,
            "methane": entry.get("methane", 0),
            "co2": entry.get("co2", 0),
            "ammonia": entry.get("ammonia", 0),
            "humidity": entry.get("humidity", 0),
            "temperature": entry.get("temperature", 0),
        })
    return sorted(records, key=lambda x: x["timestamp"])


# 🔹 Forecast with XGBoost
@app.get("/forecast/{sensor_id}")
def forecast(sensor_id: str, steps: int = 7):
    records = fetch_sensor_history(sensor_id)
    if not records:
        return JSONResponse({"error": "No data found"}, status_code=404)

    df = pd.DataFrame(records)
    values = df["methane"].values  # example metric
    X = np.arange(len(values)).reshape(-1, 1)
    y = values

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X, y)

    future_X = np.arange(len(values), len(values) + steps).reshape(-1, 1)
    preds = model.predict(future_X)

    last_date = df["timestamp"].iloc[-1]
    forecast_dates = [last_date + timedelta(days=i + 1) for i in range(steps)]
    forecast_data = [{"date": d.strftime("%Y-%m-%d"), "forecast": float(v)} for d, v in zip(forecast_dates, preds)]

    return {"sensor_id": sensor_id, "forecast": forecast_data}


# 🔹 Compare 3 models
@app.get("/compare/{sensor_id}")
def compare(sensor_id: str, steps: int = 7):
    records = fetch_sensor_history(sensor_id)
    if not records:
        return JSONResponse({"error": "No data found"}, status_code=404)

    df = pd.DataFrame(records)
    values = df["methane"].values
    X = np.arange(len(values)).reshape(-1, 1)
    y = values

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
            model.fit(X, y)
            preds = model.predict(future_X)
            results[name] = [{"date": d.strftime("%Y-%m-%d"), "forecast": float(v)} for d, v in zip(forecast_dates, preds)]
        except Exception as e:
            results[name] = {"error": str(e)}

    return {"sensor_id": sensor_id, "comparison": results}


# 🔹 Visualization
@app.get("/visualize/{sensor_id}")
def visualize(sensor_id: str,
              metric: str = Query("methane", enum=["methane", "co2", "ammonia", "humidity", "temperature"])):
    records = fetch_sensor_history(sensor_id)
    if not records:
        return JSONResponse({"error": "No data found"}, status_code=404)

    df = pd.DataFrame(records)
    plt.figure(figsize=(8, 4))
    plt.plot(df["timestamp"], df[metric], marker="o", linestyle="-", label=f"Historical {metric}")

    # optional: add forecast preview (7 days)
    X = np.arange(len(df)).reshape(-1, 1)
    y = df[metric].values
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    model.fit(X, y)
    future_X = np.arange(len(df), len(df) + 7).reshape(-1, 1)
    preds = model.predict(future_X)
    future_dates = [df["timestamp"].iloc[-1] + timedelta(days=i + 1) for i in range(7)]
    plt.plot(future_dates, preds, linestyle="--", color="orange", label="Forecast (XGBoost)")

    plt.xlabel("Time")
    plt.ylabel(metric.capitalize())
    plt.title(f"Sensor {sensor_id} - {metric} history & forecast")
    plt.legend()
    plt.grid(True)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    return StreamingResponse(buf, media_type="image/png")
