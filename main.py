from fastapi import FastAPI
import firebase_admin
from firebase_admin import credentials, db
import os
import json
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://gasmonitoring-ec511.web.app",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Firebase Admin setup
service_account_info = json.loads(os.environ["FIREBASE_SERVICE_ACCOUNT"])
cred = credentials.Certificate(service_account_info)

firebase_admin.initialize_app(cred, {
    "databaseURL": "https://gasmonitoring-ec511.firebaseio.com/"
})

@app.get("/")
def home():
    return {"message": "Python API with forecasting is running on Render"}

@app.get("/forecast/{sensor_id}")
def forecast(sensor_id: str, steps: int = 10):
    """
    Forecast future readings for a given sensor_id using XGBoost.
    steps = number of future predictions
    """

    # Fetch history from Firebase
    ref = db.reference(f"history/{sensor_id}")
    history = ref.get()

    if not history:
        return {"error": "No history found for this sensor"}

    # Convert history (dict) into DataFrame
    df = pd.DataFrame(history).T  # history is dict with timestamps as keys
    df = df.sort_index()

    # Assume readings are in "value" field
    if "value" not in df.columns:
        return {"error": "History does not contain 'value' field"}

    values = df["value"].astype(float).values

    # Prepare supervised learning dataset (lag features)
    X, y = [], []
    window = 5  # use last 5 points to predict next
    for i in range(len(values) - window):
        X.append(values[i:i+window])
        y.append(values[i+window])

    X, y = np.array(X), np.array(y)

    if len(X) < 10:
        return {"error": "Not enough data for forecasting"}

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train XGBoost
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X_train, y_train)

    # Forecast next N steps
    forecast_values = []
    last_window = values[-window:].tolist()
    for _ in range(steps):
        pred = model.predict(np.array(last_window).reshape(1, -1))[0]
        forecast_values.append(float(pred))
        last_window.pop(0)
        last_window.append(pred)

    return {
        "sensor_id": sensor_id,
        "forecast": forecast_values
    }
