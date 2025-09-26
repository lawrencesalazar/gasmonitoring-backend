from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi import HTTPException


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
# ML & preprocessingfrom fastapi import HTTPException
import math
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
 

# Firebase
import firebase_admin
from firebase_admin import credentials, db
import gc

# ---------------------------------------------------
# FastAPI App
# ---------------------------------------------------
app = FastAPI()
# Allow your frontend domains
origins = [
    "http://localhost:3000",    # Vite dev
    "https://gasmonitoring-ec511.web.app",  # Render frontend
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # You can restrict to your frontend domain later
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

# ---------------------------
# Utility to generate error images
# ---------------------------
def error_image(message: str):
    buf = io.BytesIO()
    plt.figure(figsize=(6, 3))
    plt.text(0.5, 0.5, f"⚠️ {message}", ha="center", va="center", fontsize=12, color="red")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
    )
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
import gc
from fastapi import Query
from fastapi.responses import StreamingResponse
import io
import matplotlib.pyplot as plt
import shap
import numpy as np
import pandas as pd
import xgboost as xgb

# ---------------------------
# Utility to generate error images
# ---------------------------
def error_image(message: str):
    buf = io.BytesIO()
    plt.figure(figsize=(6, 3))
    plt.text(0.5, 0.5, f"⚠️ {message}", ha="center", va="center", fontsize=12, color="red")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
    )

# ---------------------------
# EXPLAIN endpoint (SHAP summary)
# ---------------------------
@app.get("/explain/{sensor_id}")
def explain(sensor_id: str, sensor: str = Query(...)):
    try:
        records = fetch_sensor_history(sensor_id)
        df = preprocess_dataframe(records, sensor)

        if df.empty or len(df) < 50:
            return error_image("Not enough data for SHAP")

        df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
        X = df[["hour", "value"]]
        y = df["value"]

        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
        model.fit(X, y)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        buf = io.BytesIO()
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, X, feature_names=X.columns, show=False)
        plt.title(f"SHAP Summary for {sensor}")
        plt.tight_layout()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close("all")
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            },
        )

    except Exception as e:
        return error_image(str(e))


# ---------------------------
# PLOT endpoint (scatter or summary)
# ---------------------------
@app.get("/plot/{sensor_id}")
def plot(sensor_id: str, sensor: str = Query(...), chart: str = Query("scatter")):
    try:
        history = fetch_history(sensor_id, sensor, days=3)
        if not history or len(history) < 20:
            return error_image("Not enough data for plot")

        df = pd.DataFrame(history)
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d_%H%M%S")
        df["date"] = df["timestamp"].dt.date
        df["hour"] = df["timestamp"].dt.hour
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna()

        agg = (
            df.groupby(["date", "hour"])["value"]
            .mean()
            .reset_index()
            .sort_values(["date", "hour"])
        )

        if agg.empty or len(agg) < 10:
            return error_image("Not enough consolidated data")

        # Train simple model
        X = agg[["hour"]]
        y = agg["value"]
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
        model.fit(X, y)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        buf = io.BytesIO()
        if chart == "scatter":
            plt.figure(figsize=(9, 6))
            scatter = plt.scatter(
                shap_values[:, 0],
                agg["hour"],
                c=pd.factorize(agg["date"])[0],
                cmap="tab10",
                alpha=0.7
            )
            plt.xlabel("SHAP Value (Impact)")
            plt.ylabel("Hour of Day")
            plt.title(f"SHAP Scatter (3-day hourly) - {sensor}")
            cbar = plt.colorbar(scatter, ticks=range(len(agg["date"].unique())))
            cbar.ax.set_yticklabels([str(d) for d in agg["date"].unique()])
            plt.axvline(x=0, color="black", linestyle="--")
            plt.tight_layout()
            plt.savefig(buf, format="png", bbox_inches="tight")
            plt.close()
        else:
            plt.figure(figsize=(9, 6))
            shap.summary_plot(shap_values, X, feature_names=["hour"], show=False)
            plt.title(f"SHAP Summary (3-day hourly) - {sensor}")
            plt.tight_layout()
            plt.savefig(buf, format="png", bbox_inches="tight")
            plt.close()

        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            },
        )

    except Exception as e:
        return error_image(str(e))
        
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
    plt.ylabel("{sensor} value (Hour)")
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

@app.get("/recommendation/{sensor_id}")
def recommendation(sensor_id: str, sensor: str = Query(...)):
    """
    Generate 1-day ahead forecast + OSH recommendation
    using the last 3 days of data for the selected sensor.
    """
    records = fetch_sensor_history(sensor_id)
    df = preprocess_dataframe(records, sensor)

    if df.empty:
        return {"sensor_id": sensor_id, "sensor_type": sensor, "recommendation": "No data available"}

    # Filter last 3 days of history
    cutoff = df["timestamp"].max() - pd.Timedelta(days=3)
    df_recent = df[df["timestamp"] >= cutoff]

    if len(df_recent) < 5:
        return {"sensor_id": sensor_id, "sensor_type": sensor, "recommendation": "Not enough recent data"}

    # Train XGBoost using lag features
    df_recent = make_lag_features(df_recent)
    if df_recent.empty:
        return {"sensor_id": sensor_id, "sensor_type": sensor, "recommendation": "Insufficient lag data"}

    X = df_recent[["lag1", "lag2", "lag3"]].values
    y = df_recent["value"].values

    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X, y)

    # Predict 1 day ahead
    last_lags = X[-1].copy()
    last_date = df_recent["timestamp"].iloc[-1]
    pred = model.predict(last_lags.reshape(1, -1))[0]
    next_date = last_date + timedelta(days=1)

    # OSH Recommendation logic
    forecast_val = float(pred)
    recommendation = "No recommendation"

    if sensor.lower() == "co2":
        if forecast_val > 1000:
            recommendation = "⚠️ High CO₂: Improve ventilation and reduce occupancy."
        else:
            recommendation = "✅ CO₂ levels are safe."
    elif sensor.lower() == "methane":
        if forecast_val > 1000:
            recommendation = "⚠️ Methane leak risk: Check for gas leaks, ensure proper ventilation."
        else:
            recommendation = "✅ Methane levels are safe."
    elif sensor.lower() == "ammonia":
        if forecast_val > 50:
            recommendation = "⚠️ Ammonia hazard: Use protective equipment and ventilate area."
        else:
            recommendation = "✅ Ammonia levels are safe."
    elif sensor.lower() == "temperature":
        if forecast_val > 35:
            recommendation = "⚠️ High temperature: Risk of heat stress, ensure hydration and cooling."
        elif forecast_val < 10:
            recommendation = "⚠️ Low temperature: Risk of cold stress, ensure heating and PPE."
        else:
            recommendation = "✅ Temperature within safe range."
    elif sensor.lower() == "humidity":
        if forecast_val > 70:
            recommendation = "⚠️ High humidity: Risk of mold growth, improve dehumidification."
        elif forecast_val < 30:
            recommendation = "⚠️ Low humidity: Risk of dehydration and discomfort."
        else:
            recommendation = "✅ Humidity within safe range."

    return {
        "sensor_id": sensor_id,
        "sensor_type": sensor,
        "date": next_date.strftime("%Y-%m-%d"),
        "forecast": forecast_val,
        "recommendation": recommendation
    }
    
# Utility: fetch last N days of history for a sensor
def fetch_history(sensor_id: str, sensor_type: str, days: int = 3):
    ref = db.reference(f"history")
    snapshot = ref.get()
    if not snapshot:
        return []

    now = datetime.now()
    start_date = (now - timedelta(days=days)).strftime("%Y%m%d")

    data_points = []
    for ts, record in snapshot.items():
        date_key = ts.split("_")[0]  # format: yyyyMMdd_HHmmss
        if date_key >= start_date:
            if record.get("sensor_id") == sensor_id and record.get("sensor_type") == sensor_type:
                data_points.append({
                    "timestamp": ts,
                    "value": record.get("value")
                })
    data_points.sort(key=lambda x: x["timestamp"])
    return data_points


# ENDPOINT: Compute XGBoost metrics
@app.get("/xgboost_compute/{sensor_id}")
async def xgboost_compute(sensor_id: str, sensor: str):
    history = fetch_history(sensor_id, sensor, days=3)
    if not history or len(history) < 5:
        return {"error": "Not enough data to compute XGBoost analysis."}

    # Prepare dataset
    df = pd.DataFrame(history)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna()

    if len(df) < 5:
        return {"error": "Insufficient clean data for training."}

    # Create features: lag values
    df["lag1"] = df["value"].shift(1)
    df["lag2"] = df["value"].shift(2)
    df["lag3"] = df["value"].shift(3)
    df = df.dropna()

    X = df[["lag1", "lag2", "lag3"]]
    y = df["value"]

    # Train-test split: last 1 as test
    X_train, X_test = X.iloc[:-1], X.iloc[-1:]
    y_train, y_test = y.iloc[:-1], y.iloc[-1:]

    # XGBoost model
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=50)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_train, y_pred_train)

    # Return analysis
    return {
        "status": "ok",
        "sensor_id": sensor_id,
        "sensor_type": sensor,
        "latest_actual": float(y_test.iloc[0]),
        "predicted_next": float(y_pred[0]),
        "metrics": {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        },
        "history_points": len(df)
    }
    from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score
)

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
        "y_true_sample": y_true[:10].tolist(),   # include sample actual values
        "y_pred_sample": y_pred[:10].tolist()    # include sample predictions
    }

@app.get("/algorithm/{sensor_id}")
def alogrithm(sensor_id: str, sensor:  str = Query(...)):
    try:
        # Fetch history data
        ref = db.reference(f"history/{sensor_id}")
        history = ref.get()

        if not history:
            raise HTTPException(status_code=404, detail="No history found for this sensor")

        # Flatten history JSON
        rows = []
        for _, record in history.items():
            if sensor in record:
                rows.append({
                    "timestamp": record.get("timestamp"),
                    "value": float(record.get(sensor))
                })

        df = pd.DataFrame(rows)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for sensor type: {sensor}")

        # Sort and preprocess
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Feature engineering: lags
        df["lag1"] = df["value"].shift(1)
        df["lag2"] = df["value"].shift(2)
        df = df.dropna()

        X = df[["lag1", "lag2"]]
        y = df["value"]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if len(X_test) == 0:
            raise HTTPException(status_code=400, detail="Not enough data for testing")

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

        # Support Vector Regression
        svr = SVR(kernel="rbf")
        svr.fit(X_train_scaled, y_train)
        results["SVR"] = compute_metrics(y_test, svr.predict(X_test_scaled))

        # XGBoost
        xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
        xgb_model.fit(X_train, y_train)
        results["XGBoost"] = compute_metrics(y_test, xgb_model.predict(X_test))

        return {
            "sensor_id": sensor_id,
            "sensor_type": sensor,
            "sample_size": len(df),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "algorithms": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
        
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>Gas Monitoring API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 2em; line-height: 1.6; }
            h1 { color: #2c3e50; }
            h2 { margin-top: 1.5em; color: #34495e; }
            code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }
            pre { background: #f4f4f4; padding: 1em; border-radius: 5px; overflow-x: auto; }
            ul { margin-left: 1.2em; }
        </style>
    </head>
    <body>
        <h1>🚀 Gas Monitoring API</h1>
        <p>Welcome to the Gas Monitoring Backend. This service provides endpoints for forecasting, SHAP explanations, and analytics for your sensors.</p>

        <h2>📖 API Documentation</h2>
        <ul>
            <li><a href="/docs">Swagger UI</a> (interactive API docs)</li>
            <li><a href="/redoc">ReDoc</a> (alternative docs)</li>
        </ul>

        <h2>📘 How to Use</h2>
        <p>Below is a quick reference on how to use the API with your React.js dashboard.</p>

        <h3>1. Forecast Endpoint</h3>
        <p>Get predictions for a specific sensor.</p>
        <pre><code>GET /predict/{sensor_id}?sensor=temperature</code></pre>
        <p><b>Response (JSON):</b></p>
        <pre><code>{
  "sensor_id": "123",
  "sensor_type": "temperature",
  "forecasts": [
    {"date": "2025-09-20", "forecast": 25.7},
    {"date": "2025-09-21", "forecast": 26.1}
  ]
}</code></pre>

        <h3>2. SHAP Explanation Endpoint</h3>
        <p>Visualize feature importance with SHAP.</p>
        <pre><code>GET /explain/{sensor_id}?sensor=temperature</code></pre>
        <p><b>Response:</b> PNG image (SHAP summary plot).</p>

        <h3>3. SHAP Hour Analysis</h3>
        <p>Analyze SHAP values by hour.</p>
        <pre><code>GET /shap_hour/{sensor_id}?sensor=temperature</code></pre>
        <p><b>Response (JSON + Base64 Image):</b></p>
        <pre><code>{
  "stats": {
    "mean_abs_shap": 2.34,
    "shap_range": [-5.1, 6.7],
    "correlation": 0.452,
    "mse": 0.12
  },
  "plot_base64": "iVBORw0KGgoAAAANSUhEUg..."
}</code></pre>

        <h2>💻 Example React.js Integration</h2>
        <pre><code>{`useEffect(() => {
  fetch(\`https://gasmonitoring-backend.onrender.com/predict/\${sensorID}?sensor=\${sensor}\`)
    .then(res => res.json())
    .then(data => setForecast(data.forecasts || []));
}, [sensorID, sensor]);`}</code></pre>

        <p>See the <b>SensorAnalytics</b> React component for a full example.</p>

        <hr />
        <p style="font-size: 0.9em; color: #666;">
            Powered by FastAPI & XGBoost | Gas Monitoring Project
        </p>
    </body>
    </html>
    """