# --- main.py ---
import json
import os
import math
from fastapi import FastAPI, HTTPException, Query, Request
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Gas Monitoring API", 
    version="1.0.0",
    description="API for gas sensor monitoring, forecasting, and SHAP explanations"
)

# CORS Middleware
origins = [
    "http://localhost:3000",
    "https://gasmonitoring-ec511.web.app",
    "http://localhost:3001",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# Firebase Setup
# ---------------------------------------------------
try:
    service_account_info = json.loads(os.environ.get("FIREBASE_SERVICE_ACCOUNT", "{}"))
    database_url = os.getenv(
        "FIREBASE_DB_URL",
        "https://gasmonitoring-ec511-default-rtdb.asia-southeast1.firebasedatabase.app"
    )

    if not firebase_admin._apps:
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred, {"databaseURL": database_url})
        logger.info("Firebase initialized successfully")
except Exception as e:
    logger.error(f"Firebase initialization failed: {e}")

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
    """Fetch sensor history from Firebase"""
    try:
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
        logger.info(f"Fetched {len(records)} records for sensor {sensor_id}")
        return records
    except Exception as e:
        logger.error(f"Error fetching sensor history: {e}")
        return []

def preprocess_dataframe(records, sensor: str):
    """Preprocess dataframe for analysis"""
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

def make_lag_features(df, lags=3):
    """Create lag features for time series forecasting"""
    for i in range(1, lags + 1):
        df[f"lag{i}"] = df["value"].shift(i)
    return df.dropna()

def compute_metrics(y_true, y_pred):
    """Compute regression metrics"""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    if len(y_true) > 1:
        r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
    else:
        r2 = 0
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2_score": r2}

def error_image(msg: str):
    """Generate error image with message"""
    buf = io.BytesIO()
    plt.figure(figsize=(8, 2))
    plt.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12, wrap=True)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=100)
    plt.close()
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

def generate_recommendation(sensor_type: str, value: float):
    """Generate OSH recommendations based on sensor values"""
    sensor_type = sensor_type.lower()
    
    if sensor_type == "co2":
        if value > 1000:
            return "⚠️ HIGH RISK: CO₂ levels critical. Evacuate area immediately and improve ventilation."
        elif value > 800:
            return "⚠️ MODERATE RISK: Elevated CO₂ levels. Increase ventilation and reduce occupancy."
        else:
            return "✅ SAFE: CO₂ levels within acceptable range."
    
    elif sensor_type == "methane":
        if value > 1000:
            return "⚠️ HIGH RISK: Methane concentration dangerous. Potential explosion hazard. Evacuate immediately."
        elif value > 500:
            return "⚠️ MODERATE RISK: Elevated methane levels. Check for leaks and increase ventilation."
        else:
            return "✅ SAFE: Methane levels within safe limits."
    
    elif sensor_type == "ammonia":
        if value > 50:
            return "⚠️ HIGH RISK: Ammonia levels hazardous. Use respiratory protection and evacuate area."
        elif value > 25:
            return "⚠️ MODERATE RISK: Elevated ammonia levels. Ensure proper ventilation and monitoring."
        else:
            return "✅ SAFE: Ammonia levels within acceptable range."
    
    elif sensor_type == "temperature":
        if value > 35:
            return "⚠️ HIGH RISK: Extreme temperature. Risk of heat stress. Implement cooling measures."
        elif value > 30:
            return "⚠️ MODERATE RISK: High temperature. Ensure hydration and adequate breaks."
        elif value < 10:
            return "⚠️ HIGH RISK: Low temperature risk. Risk of hypothermia. Provide heating."
        elif value < 15:
            return "⚠️ MODERATE RISK: Low temperature. Ensure proper insulation and warm clothing."
        else:
            return "✅ SAFE: Temperature within comfortable range."
    
    elif sensor_type == "humidity":
        if value > 70:
            return "⚠️ HIGH RISK: High humidity. Risk of mold growth and discomfort. Use dehumidifiers."
        elif value > 60:
            return "⚠️ MODERATE RISK: Elevated humidity. Improve ventilation."
        elif value < 30:
            return "⚠️ HIGH RISK: Low humidity. Risk of dehydration and respiratory issues. Use humidifiers."
        elif value < 40:
            return "⚠️ MODERATE RISK: Low humidity. Ensure adequate hydration."
        else:
            return "✅ SAFE: Humidity within comfortable range."
    
    else:
        return "ℹ️ No specific recommendations available for this sensor type."

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
            .endpoint { background: #e8f4fd; padding: 1em; border-radius: 5px; margin: 1em 0; }
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

        <h2>🔧 Available Endpoints</h2>
        
        <div class="endpoint">
            <h3>📊 Data Endpoints</h3>
            <ul>
                <li><code>GET /health</code> - API status check</li>
                <li><code>GET /dataframe/{sensor_id}?sensor=type</code> - Raw sensor data</li>
                <li><code>GET /plot/{sensor_id}?sensor=type&chart=scatter</code> - Visualization plots</li>
            </ul>
        </div>

        <div class="endpoint">
            <h3>🤖 AI/ML Endpoints</h3>
            <ul>
                <li><code>GET /algorithm/{sensor_id}?sensor=type</code> - Multi-algorithm comparison</li>
                <li><code>GET /xgboost_compute/{sensor_id}?sensor=type</code> - XGBoost analysis</li>
                <li><code>GET /explain/{sensor_id}?sensor=type</code> - SHAP explanations</li>
                <li><code>GET /shap_hour/{sensor_id}?sensor=type</code> - Hourly SHAP analysis</li>
            </ul>
        </div>

        <div class="endpoint">
            <h3>📈 Forecasting Endpoints</h3>
            <ul>
                <li><code>GET /predict/{sensor_id}?sensor=type</code> - Future predictions</li>
                <li><code>GET /recommendation/{sensor_id}?sensor=type</code> - OSH recommendations</li>
            </ul>
        </div>

        <h2>💻 Example React.js Integration</h2>
        <pre><code>{`useEffect(() => {
  fetch(\`/api/predict/\${sensorID}?sensor=\${sensorType}\`)
    .then(res => res.json())
    .then(data => setForecast(data.forecasts || []));
}, [sensorID, sensorType]);`}</code></pre>

        <hr />
        <p style="font-size: 0.9em; color: #666;">
            Powered by FastAPI & XGBoost | Gas Monitoring Project
        </p>
    </body>
    </html>
    """

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.get("/dataframe/{sensor_id}")
def get_dataframe(sensor_id: str, sensor: str = Query(..., description="Sensor type (co2, temperature, etc.)")):
    """Get raw sensor data as JSON"""
    records = fetch_sensor_history(sensor_id)
    df = preprocess_dataframe(records, sensor)
    return df.to_dict(orient="records")

@app.get("/plot/{sensor_id}")
def plot_sensor_data(
    sensor_id: str, 
    sensor: str = Query(..., description="Sensor type"),
    chart: str = Query("scatter", description="Chart type: scatter or summary")
):
    """Generate visualization plots for sensor data"""
    try:
        records = fetch_sensor_history(sensor_id)
        if not records:
            return error_image("No data found for this sensor")
        
        df = preprocess_dataframe(records, sensor)
        if df.empty or len(df) < 10:
            return error_image(f"Not enough data for plot. Found {len(df)} records.")
        
        # Use recent data (last 3 days)
        cutoff = datetime.now() - timedelta(days=3)
        df_recent = df[df["timestamp"] >= cutoff]
        
        if df_recent.empty or len(df_recent) < 5:
            return error_image("Not enough recent data for plotting")
        
        # Prepare features
        df_recent["hour"] = df_recent["timestamp"].dt.hour
        df_recent["date"] = df_recent["timestamp"].dt.date
        agg = df_recent.groupby(["date", "hour"])["value"].mean().reset_index()
        
        if len(agg) < 2:
            return error_image("Insufficient data for analysis")
        
        X = agg[["hour"]]
        y = agg["value"]
        
        # Train simple model for SHAP
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=50, random_state=42)
        model.fit(X, y)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Generate plot
        buf = io.BytesIO()
        plt.figure(figsize=(10, 6))
        
        if chart == "scatter":
            plt.scatter(agg["hour"], y, c=shap_values, cmap="coolwarm", alpha=0.7)
            plt.colorbar(label="SHAP Value")
            plt.xlabel("Hour of Day")
            plt.ylabel("Sensor Value")
            plt.title(f"SHAP Scatter Plot - {sensor}")
        else:
            shap.summary_plot(shap_values, X, feature_names=["Hour"], show=False)
            plt.title(f"SHAP Summary - {sensor}")
        
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="image/png")
        
    except Exception as e:
        logger.error(f"Plot generation error: {e}")
        return error_image(f"Error generating plot: {str(e)}")

@app.get("/algorithm/{sensor_id}")
def compare_algorithms(
    sensor_id: str, 
    sensor: str = Query(..., description="Sensor type")
):
    """Compare multiple ML algorithms and provide forecasts"""
    try:
        records = fetch_sensor_history(sensor_id)
        df = preprocess_dataframe(records, sensor)
        
        if df.empty or len(df) < 5:
            raise HTTPException(status_code=404, detail="Not enough data for analysis")
        
        # Create lag features
        df = make_lag_features(df, lags=2)
        if df.empty:
            raise HTTPException(status_code=404, detail="Insufficient data after feature engineering")
        
        X, y = df[["lag1", "lag2"]], df["value"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        if len(X_test) == 0:
            X_test = X_train.iloc[-1:].copy()
            y_test = y_train.iloc[-1:].copy()
        
        # Scale data for SVR
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        results = {}
        
        # Linear Regression
        lr = LinearRegression().fit(X_train, y_train)
        results["LinearRegression"] = compute_metrics(y_test, lr.predict(X_test))
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
        results["RandomForest"] = compute_metrics(y_test, rf.predict(X_test))
        
        # Decision Tree
        dt = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
        results["DecisionTree"] = compute_metrics(y_test, dt.predict(X_test))
        
        # SVR
        svr = SVR().fit(X_train_s, y_train)
        results["SVR"] = compute_metrics(y_test, svr.predict(X_test_s))
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
        xgb_model.fit(X_train, y_train)
        results["XGBoost"] = compute_metrics(y_test, xgb_model.predict(X_test))
        
        # Generate forecast
        last_row = df.iloc[-1]
        pred_next = xgb_model.predict([[last_row["lag1"], last_row["lag2"]]])[0]
        forecast_val = float(pred_next)
        next_date = df["timestamp"].iloc[-1] + timedelta(days=1)
        
        # Generate recommendation
        recommendation = generate_recommendation(sensor, forecast_val)
        
        return {
            "sensor_id": sensor_id,
            "sensor_type": sensor,
            "algorithms": results,
            "best_algorithm": min(results, key=lambda x: results[x]["rmse"]),
            "forecast": {
                "date": next_date.strftime("%Y-%m-%d"),
                "predicted_value": forecast_val,
                "recommendation": recommendation
            },
            "data_points": len(df)
        }
        
    except Exception as e:
        logger.error(f"Algorithm comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommendation/{sensor_id}")
def get_recommendation(
    sensor_id: str, 
    sensor: str = Query(..., description="Sensor type")
):
    """Generate 1-day ahead forecast with OSH recommendation"""
    try:
        records = fetch_sensor_history(sensor_id)
        df = preprocess_dataframe(records, sensor)

        if df.empty:
            return {
                "sensor_id": sensor_id, 
                "sensor_type": sensor, 
                "recommendation": "No data available",
                "status": "error"
            }

        # Use recent data (last 10 days)
        cutoff = df["timestamp"].max() - timedelta(days=10)
        df_recent = df[df["timestamp"] >= cutoff]

        if len(df_recent) < 5:
            return {
                "sensor_id": sensor_id,
                "sensor_type": sensor,
                "recommendation": "Not enough recent data for forecasting",
                "status": "insufficient_data"
            }

        # Create lag features and train model
        df_recent = make_lag_features(df_recent, lags=3)
        if df_recent.empty:
            return {
                "sensor_id": sensor_id,
                "sensor_type": sensor,
                "recommendation": "Insufficient data for forecasting",
                "status": "insufficient_data"
            }

        X = df_recent[["lag1", "lag2", "lag3"]].values
        y = df_recent["value"].values

        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
        model.fit(X, y)

        # Generate forecast
        last_lags = X[-1].copy()
        last_date = df_recent["timestamp"].iloc[-1]
        pred = model.predict(last_lags.reshape(1, -1))[0]
        next_date = last_date + timedelta(days=1)
        forecast_val = float(pred)

        # Generate recommendation
        recommendation = generate_recommendation(sensor, forecast_val)

        return {
            "sensor_id": sensor_id,
            "sensor_type": sensor,
            "forecast_date": next_date.strftime("%Y-%m-%d"),
            "predicted_value": forecast_val,
            "recommendation": recommendation,
            "current_value": float(df_recent["value"].iloc[-1]),
            "data_points": len(df_recent),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        return {
            "sensor_id": sensor_id,
            "sensor_type": sensor,
            "recommendation": f"Error generating recommendation: {str(e)}",
            "status": "error"
        }

@app.get("/xgboost_compute/{sensor_id}")
async def xgboost_analysis(
    sensor_id: str, 
    sensor: str = Query(..., description="Sensor type")
):
    """XGBoost analysis with detailed metrics"""
    try:
        records = fetch_sensor_history(sensor_id)
        if not records:
            return {"error": "No data found for sensor", "status": "error"}
            
        df = pd.DataFrame(records)
        if sensor not in df.columns:
            return {"error": f"Sensor type '{sensor}' not found in data", "status": "error"}
            
        df["value"] = pd.to_numeric(df[sensor], errors="coerce")
        df = df.dropna(subset=["value"])
        
        if len(df) < 5:
            return {"error": "Insufficient clean data for training", "status": "error"}

        # Create lag features
        df = make_lag_features(df, lags=3)
        if len(df) < 2:
            return {"error": "Insufficient data after feature engineering", "status": "error"}

        X = df[["lag1", "lag2", "lag3"]]
        y = df["value"]

        # Train-test split
        X_train, X_test = X.iloc[:-1], X.iloc[-1:]
        y_train, y_test = y.iloc[:-1], y.iloc[-1:]

        # XGBoost model
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_train, y_pred_train)

        # Generate recommendation
        recommendation = generate_recommendation(sensor, float(y_pred[0]))

        return {
            "status": "success",
            "sensor_id": sensor_id,
            "sensor_type": sensor,
            "latest_actual": float(y_test.iloc[0]) if len(y_test) > 0 else None,
            "predicted_next": float(y_pred[0]),
            "recommendation": recommendation,
            "metrics": {
                "MSE": round(mse, 4),
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
                "R2": round(r2, 4)
            },
            "history_points": len(df)
        }
        
    except Exception as e:
        logger.error(f"XGBoost analysis error: {e}")
        return {"error": str(e), "status": "error"}

@app.get("/predict/{sensor_id}")
def predict_future(
    sensor_id: str,
    sensor: str = Query(..., description="Sensor type"),
    days: int = Query(7, description="Forecast horizon in days")
):
    """Generate multi-day forecasts"""
    try:
        records = fetch_sensor_history(sensor_id)
        df = preprocess_dataframe(records, sensor)
        
        if df.empty or len(df) < 10:
            raise HTTPException(status_code=404, detail="Not enough historical data")
        
        # Create features and train model
        df_lags = make_lag_features(df, lags=3)
        if df_lags.empty:
            raise HTTPException(status_code=404, detail="Insufficient data for forecasting")
        
        X = df_lags[["lag1", "lag2", "lag3"]].values
        y = df_lags["value"].values
        
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Generate forecasts
        forecasts = []
        current_features = X[-1].copy()
        current_date = df["timestamp"].iloc[-1]
        
        for i in range(min(days, 30)):  # Limit to 30 days max
            next_pred = model.predict(current_features.reshape(1, -1))[0]
            current_date += timedelta(days=1)
            
            forecasts.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "forecast": float(next_pred),
                "recommendation": generate_recommendation(sensor, float(next_pred))
            })
            
            # Update features for next prediction (simple approach)
            current_features = np.roll(current_features, -1)
            current_features[-1] = next_pred
        
        return {
            "sensor_id": sensor_id,
            "sensor_type": sensor,
            "forecast_horizon_days": days,
            "forecasts": forecasts,
            "last_actual_value": float(y[-1]),
            "last_actual_date": df["timestamp"].iloc[-1].strftime("%Y-%m-%d")
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/explain/{sensor_id}")
def explain_shap(
    sensor_id: str,
    sensor: str = Query(..., description="Sensor type")
):
    """Generate SHAP explanation plot"""
    try:
        records = fetch_sensor_history(sensor_id)
        df = preprocess_dataframe(records, sensor)
        
        if df.empty or len(df) < 10:
            return error_image("Not enough data for SHAP analysis")
        
        # Prepare features
        df_lags = make_lag_features(df, lags=3)
        if df_lags.empty:
            return error_image("Insufficient data for feature engineering")
        
        X = df_lags[["lag1", "lag2", "lag3"]]
        y = df_lags["value"]
        
        # Train model
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # SHAP explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Create plot
        buf = io.BytesIO()
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=["Lag-1", "Lag-2", "Lag-3"], show=False)
        plt.title(f"SHAP Feature Importance - {sensor}")
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="image/png")
        
    except Exception as e:
        logger.error(f"SHAP explanation error: {e}")
        return error_image(f"SHAP analysis error: {str(e)}")

@app.get("/shap_hour/{sensor_id}")
def shap_hour_analysis(
    sensor_id: str,
    sensor: str = Query(..., description="Sensor type")
):
    """SHAP analysis by hour with detailed statistics"""
    try:
        records = fetch_sensor_history(sensor_id)
        df = preprocess_dataframe(records, sensor)
        
        if df.empty or len(df) < 10:
            return {"error": "Not enough data for analysis", "status": "error"}
        
        # Extract hour and prepare features
        df["hour"] = df["timestamp"].dt.hour
        df_lags = make_lag_features(df, lags=2)
        
        if df_lags.empty:
            return {"error": "Insufficient data after feature engineering", "status": "error"}
        
        X = df_lags[["lag1", "lag2", "hour"]]
        y = df_lags["value"]
        
        # Train model
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # SHAP analysis
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Hourly analysis
        hourly_shap = pd.DataFrame({
            'hour': X['hour'],
            'shap': shap_values[:, 2]  # SHAP values for hour feature
        })
        
        stats = hourly_shap.groupby('hour')['shap'].agg(['mean', 'std', 'count']).reset_index()
        
        # Create plot
        buf = io.BytesIO()
        plt.figure(figsize=(10, 6))
        plt.bar(stats['hour'], stats['mean'], yerr=stats['std'], capsize=5)
        plt.xlabel('Hour of Day')
        plt.ylabel('Mean SHAP Value')
        plt.title(f'SHAP Values by Hour - {sensor}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=150)
        plt.close()
        
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        return {
            "status": "success",
            "sensor_id": sensor_id,
            "sensor_type": sensor,
            "stats": {
                "mean_abs_shap": float(np.mean(np.abs(shap_values))),
                "shap_range": [float(np.min(shap_values)), float(np.max(shap_values))],
                "hourly_correlation": float(X['hour'].corr(y)),
                "data_points": len(X)
            },
            "hourly_analysis": stats.to_dict('records'),
            "plot_base64": plot_base64
        }
        
    except Exception as e:
        logger.error(f"SHAP hour analysis error: {e}")
        return {"error": str(e), "status": "error"}

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)