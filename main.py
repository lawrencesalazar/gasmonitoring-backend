# --- main.py ---
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io

# CORS Middleware
origins = [
    "http://localhost:3000",
    "https://gasmonitoring-ec511.web.app",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

# Global CORS headers middleware
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# Utility functions
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

def compute_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2_score": r2}

def error_image(msg: str):
    buf = io.BytesIO()
    plt.figure(figsize=(8, 2))
    plt.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12, wrap=True)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png", headers={"Access-Control-Allow-Origin": "*"})

# Endpoints
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
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/dataframe/{sensor_id}")
def dataframe(sensor_id: str, sensor: str = Query(...)):
    records = fetch_sensor_history(sensor_id)
    df = preprocess_dataframe(records, sensor)
    return df.to_dict(orient="records")

@app.get("/plot/{sensor_id}")
def plot(sensor_id: str, sensor: str = Query(...), chart: str = Query("scatter")):
    try:
        records = fetch_sensor_history(sensor_id)
        if not records:
            return error_image("No data found for this sensor")
        df = preprocess_dataframe(records, sensor)
        if df.empty or len(df) < 10:
            return error_image(f"Not enough data for plot. Found {len(df)} records.")
        cutoff = datetime.now() - timedelta(days=3)
        df_recent = df[df["timestamp"] >= cutoff]
        if df_recent.empty or len(df_recent) < 5:
            return error_image("Not enough recent data for plotting")
        df_recent["hour"] = df_recent["timestamp"].dt.hour
        df_recent["date"] = df_recent["timestamp"].dt.date
        agg = df_recent.groupby(["date", "hour"])["value"].mean().reset_index()
        X = agg[["hour"]]; y = agg["value"]
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=50, random_state=42)
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        buf = io.BytesIO()
        if chart == "scatter":
            plt.figure(figsize=(10, 6))
            plt.scatter(shap_values, agg["hour"], c=pd.factorize(agg["date"])[0], cmap="tab10", alpha=0.7)
            plt.xlabel("SHAP Value"); plt.ylabel("Hour"); plt.title(f"SHAP Scatter {sensor}")
        else:
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X, feature_names=["Hour"], show=False)
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(); buf.seek(0)
        return StreamingResponse(buf, media_type="image/png", headers={"Access-Control-Allow-Origin": "*"})
    except Exception as e:
        return error_image(f"Error: {str(e)}")

@app.get("/algorithm/{sensor_id}")
def algorithm(sensor_id: str, sensor: str = Query(...)):
    try:
        records = fetch_sensor_history(sensor_id)
        df = preprocess_dataframe(records, sensor)
        if df.empty or len(df) < 5:
            raise HTTPException(status_code=404, detail="Not enough data")
        df["lag1"] = df["value"].shift(1); df["lag2"] = df["value"].shift(2)
        df = df.dropna()
        X, y = df[["lag1", "lag2"]], df["value"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        scaler = StandardScaler(); X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)
        results = {}
        lr = LinearRegression().fit(X_train, y_train)
        results["LinearRegression"] = compute_metrics(y_test, lr.predict(X_test))
        rf = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)
        results["RandomForest"] = compute_metrics(y_test, rf.predict(X_test))
        dt = DecisionTreeRegressor().fit(X_train, y_train)
        results["DecisionTree"] = compute_metrics(y_test, dt.predict(X_test))
        svr = SVR().fit(X_train_s, y_train)
        results["SVR"] = compute_metrics(y_test, svr.predict(X_test_s))
        xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100).fit(X_train, y_train)
        results["XGBoost"] = compute_metrics(y_test, xgb_model.predict(X_test))
        last_row = df.iloc[-1]
        pred_next = xgb_model.predict([[last_row["lag1"], last_row["lag2"]]])[0]
        forecast_val = float(pred_next); next_date = df["timestamp"].iloc[-1] + timedelta(days=1)
        recommendation = "No recommendation"
        if sensor.lower() == "co2":
            recommendation = "⚠️ High CO₂: Improve ventilation." if forecast_val > 1000 else "✅ CO₂ safe."
        elif sensor.lower() == "methane":
            recommendation = "⚠️ Methane leak risk." if forecast_val > 1000 else "✅ Methane safe."
        elif sensor.lower() == "ammonia":
            recommendation = "⚠️ Ammonia hazard." if forecast_val > 50 else "✅ Ammonia safe."
        elif sensor.lower() == "temperature":
            if forecast_val > 35: recommendation = "⚠️ High temperature risk."
            elif forecast_val < 10: recommendation = "⚠️ Low temperature risk."
            else: recommendation = "✅ Temperature safe."
        elif sensor.lower() == "humidity":
            if forecast_val > 70: recommendation = "⚠️ High humidity risk."
            elif forecast_val < 30: recommendation = "⚠️ Low humidity risk."
            else: recommendation = "✅ Humidity safe."
        return {
            "sensor_id": sensor_id,
            "sensor_type": sensor,
            "algorithms": results,
            "forecast": {
                "date": next_date.strftime("%Y-%m-%d"),
                "predicted_next": forecast_val,
                "recommendation": recommendation
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

    # Filter last 10 days of history
    cutoff = df["timestamp"].max() - pd.Timedelta(days=10)
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
