# --- main.py ---
import json
import os
import math
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
import logging
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app FIRST
app = FastAPI(
    title="Gas Monitoring API", 
    version="1.0.0",
    description="API for gas sensor monitoring, forecasting, and SHAP explanations"
)

# CORS Middleware - Place this RIGHT AFTER app initialization
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Enhanced OPTIONS handler
@app.options("/{path:path}")
async def preflight_handler(request: Request, path: str):
    return JSONResponse(
        content={"status": "ok", "message": "CORS preflight successful"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "3600",
        }
    )

# ---------------------------------------------------
# Firebase Setup
# ---------------------------------------------------
try:
    firebase_config = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
    if firebase_config:
        service_account_info = json.loads(firebase_config)
    else:
        service_account_info = {
            "type": "service_account",
            "project_id": os.getenv("FIREBASE_PROJECT_ID", "gasmonitoring-ec511"),
            "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID", ""),
            "private_key": os.getenv("FIREBASE_PRIVATE_KEY", "").replace('\\n', '\n'),
            "client_email": os.getenv("FIREBASE_CLIENT_EMAIL", ""),
            "client_id": os.getenv("FIREBASE_CLIENT_ID", ""),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_CERT_URL", "")
        }
    
    database_url = os.getenv(
        "FIREBASE_DB_URL",
        "https://gasmonitoring-ec511-default-rtdb.asia-southeast1.firebasedatabase.app"
    )

    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(service_account_info)
            firebase_admin.initialize_app(cred, {"databaseURL": database_url})
            logger.info("Firebase initialized successfully")
    except Exception as e:
        logger.warning(f"Firebase initialization note: {e}")
except Exception as e:
    logger.error(f"Firebase initialization failed: {e}")

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def fetch_sensor_history(sensor_id: str) -> List[Dict[str, Any]]:
    """Fetch sensor history from Firebase"""
    try:
        logger.info(f"Fetching sensor history for {sensor_id}")
        ref = db.reference(f"history/{sensor_id}")
        snapshot = ref.get()
        
        if not snapshot:
            logger.warning(f"No data found for sensor {sensor_id}")
            return []
        
        records = []
        for key, value in snapshot.items():
            if isinstance(value, dict):
                row = value.copy()
                if "timestamp" in row:
                    try:
                        # Ensure timestamp is properly parsed
                        row["timestamp"] = pd.to_datetime(row["timestamp"])
                    except Exception as e:
                        logger.warning(f"Failed to parse timestamp for record {key}: {e}")
                        row["timestamp"] = None
                records.append(row)
        
        logger.info(f"Fetched {len(records)} records for sensor {sensor_id}")
        return records
        
    except Exception as e:
        logger.error(f"Error fetching sensor history for {sensor_id}: {e}")
        return []
        
def get_current_sensor_reading(sensor_id: str) -> Optional[Dict[str, Any]]:
    """Get current sensor reading from sensorReadings"""
    try:
        ref = db.reference(f"sensorReadings/{sensor_id}")
        snapshot = ref.get()
        if snapshot and isinstance(snapshot, dict):
            if "timestamp" in snapshot:
                try:
                    snapshot["timestamp"] = pd.to_datetime(snapshot["timestamp"])
                except Exception:
                    snapshot["timestamp"] = None
            return snapshot
        return None
    except Exception as e:
        logger.error(f"Error fetching current sensor reading: {e}")
        return None

def preprocess_dataframe(records: List[Dict[str, Any]], sensor: str) -> pd.DataFrame:
    """Preprocess dataframe for analysis"""
    if not records:
        logger.warning("No records to preprocess")
        return pd.DataFrame()
    
    try:
        df = pd.DataFrame(records)
        logger.info(f"Created dataframe with columns: {df.columns.tolist()}")
        
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp")
            logger.info(f"Sorted by timestamp. Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        if sensor not in df.columns:
            logger.error(f"Sensor column '{sensor}' not found in dataframe. Available columns: {df.columns.tolist()}")
            return pd.DataFrame()
        
        # Select only necessary columns and drop NaN
        df = df[["timestamp", sensor]].dropna()
        df = df.rename(columns={sensor: "value"})
        
        logger.info(f"Final preprocessed dataframe shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error preprocessing dataframe: {e}")
        return pd.DataFrame()
  
def filter_by_date_range(df: pd.DataFrame, date_range: str) -> pd.DataFrame:
    """Filter dataframe by date range"""
    if df.empty or "timestamp" not in df.columns:
        logger.warning("Cannot filter empty dataframe or dataframe without timestamp")
        return df
    
    try:
        now = datetime.now()
        
        date_filters = {
            "1week": now - timedelta(weeks=1),
            "1month": now - timedelta(days=30),
            "3months": now - timedelta(days=90),
            "6months": now - timedelta(days=180),
            "1year": now - timedelta(days=365),
            "all": datetime.min
        }
        
        cutoff_date = date_filters.get(date_range, date_filters["1month"])
        logger.info(f"Filtering data from {cutoff_date} for range {date_range}")
        
        if cutoff_date != datetime.min:
            filtered_df = df[df["timestamp"] >= cutoff_date]
            logger.info(f"After filtering: {len(filtered_df)} records")
            return filtered_df
        
        return df
        
    except Exception as e:
        logger.error(f"Error filtering by date range: {e}")
        return df
        
def make_lag_features(df: pd.DataFrame, lags: int = 3) -> pd.DataFrame:
    """Create lag features for time series forecasting"""
    if df.empty:
        return df
    
    df_copy = df.copy()
    for i in range(1, lags + 1):
        df_copy[f"lag{i}"] = df_copy["value"].shift(i)
    return df_copy.dropna()

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics"""
    if len(y_true) == 0 or len(y_pred) == 0:
        return {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "r2_score": 0.0}
    
    try:
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        if len(y_true) > 1 and np.var(y_true) > 0:
            r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
        else:
            r2 = 0.0
        return {"mse": float(mse), "rmse": float(rmse), "mae": float(mae), "r2_score": float(r2)}
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        return {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "r2_score": 0.0}

def generate_recommendation(sensor_type: str, value: float) -> str:
    """Generate OSH recommendations based on sensor values"""
    sensor_type = sensor_type.lower()
    
    try:
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
    except Exception as e:
        logger.error(f"Error generating recommendation: {e}")
        return "ℹ️ Error generating recommendation."

def error_image(msg: str) -> StreamingResponse:
    """Generate error image with message"""
    try:
        buf = io.BytesIO()
        plt.figure(figsize=(8, 2))
        plt.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12, wrap=True)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=100)
        plt.close()
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        logger.error(f"Error generating error image: {e}")
        # Fallback to JSON response
        return JSONResponse(
            content={"error": msg},
            status_code=500
        )

def calculate_risk_index(data: Dict[str, Any]) -> float:
    """Calculate risk index based on sensor readings"""
    try:
        # Normalize values and calculate weighted risk
        co2_risk = min(float(data['co2']) / 1000, 1.0)  # Normalize to 0-1
        methane_risk = min(float(data['methane']) / 1000, 1.0)
        ammonia_risk = min(float(data['ammonia']) / 50, 1.0)
        
        # Temperature risk (ideal range 18-28°C)
        temp = float(data['temperature'])
        if 18 <= temp <= 28:
            temp_risk = 0.0
        else:
            temp_risk = min(abs(temp - 23) / 20, 1.0)  # Normalize based on deviation from ideal
        
        # Humidity risk (ideal range 40-60%)
        humidity = float(data['humidity'])
        if 40 <= humidity <= 60:
            humidity_risk = 0.0
        else:
            humidity_risk = min(abs(humidity - 50) / 50, 1.0)
        
        # Weighted risk calculation (gas sensors have higher weight)
        total_risk = (
            co2_risk * 0.3 +
            methane_risk * 0.3 +
            ammonia_risk * 0.2 +
            temp_risk * 0.1 +
            humidity_risk * 0.1
        )
        
        return round(total_risk, 5)
    
    except (ValueError, TypeError) as e:
        logger.error(f"Error calculating risk index: {e}")
        return 0.0

# =============================================================================
# SENSOR ANALYTICS ENDPOINTS
# =============================================================================

@app.get("/health")
def health():
    """Health check endpoint with detailed CORS info"""    
    return {
        "status": "ok", 
        "timestamp": datetime.now().isoformat(), 
        "service": "Gas Monitoring API",
        "cors_enabled": True,
        "cors_config": {
            "allow_origins": "all (*)",
            "allow_methods": "GET, POST, PUT, DELETE, OPTIONS, PATCH",
            "allow_headers": "all",
            "max_age": "3600"
        }
    }

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
            .new { border-left: 4px solid #4CAF50; }
        </style>
    </head>
    <body>
        <h1>Gas Monitoring API</h1>
        <p>Welcome to the Gas Monitoring Backend. This service provides endpoints for forecasting, SHAP explanations, and analytics for your sensors.</p>
        <p><strong>CORS Status:</strong> ✅ Enabled for all origins</p>

        <h2>API Documentation</h2>
        <ul>
            <li><a href="/docs">Swagger UI</a> (interactive API docs)</li>
            <li><a href="/redoc">ReDoc</a> (alternative docs)</li>
        </ul>

        <h2>Available Endpoints</h2>
        
        <div class="endpoint new">
            <h3>📱 ESP32 Sensor Endpoints (NEW)</h3>
            <ul>
                <li><code>POST /api/sensor/data</code> - Receive sensor data from ESP32 (JSON)</li>
                <li><code>GET /api/sensor/current/{sensor_id}</code> - Get current sensor reading</li>
                <li><code>GET /api/sensor/{sensor_id}/stats</code> - Get sensor statistics</li>
            </ul>
        </div>

        <div class="endpoint">
            <h3>Data Endpoints</h3>
            <ul>
                <li><code>GET /health</code> - API status check</li>
                <li><code>GET /dataframe/{sensor_id}?sensor=type</code> - Raw sensor data</li>
                <li><code>GET /plot/{sensor_id}?sensor=type&chart=scatter</code> - Visualization plots</li>
            </ul>
        </div>

        <div class="endpoint">
            <h3>AI/ML Endpoints</h3>
            <ul>
                <li><code>GET /algorithm/{sensor_id}?sensor=type</code> - Multi-algorithm comparison</li>
                <li><code>GET /xgboost_compute/{sensor_id}?sensor=type</code> - XGBoost analysis</li>
                <li><code>GET /explain/{sensor_id}?sensor=type</code> - SHAP explanations</li>
                <li><code>GET /shap_hour/{sensor_id}?sensor=type</code> - Hourly SHAP analysis</li>
            </ul>
        </div>

        <div class="endpoint">
            <h3>Forecasting Endpoints</h3>
            <ul>
                <li><code>GET /predict/{sensor_id}?sensor=type</code> - Future predictions</li>
                <li><code>GET /recommendation/{sensor_id}?sensor=type</code> - OSH recommendations</li>
            </ul>
        </div>

        <div class="endpoint">
            <h3>Analytics Endpoints</h3>
            <ul>
                <li><code>GET /performance/{sensor_id}?sensor=type</code> - Performance metrics</li>
                <li><code>GET /confusion_matrix/{sensor_id}?sensor=type</code> - Confusion matrix</li>
            </ul>
        </div>

        <hr />
        <p style="font-size: 0.9em; color: #666;">
            Powered by FastAPI & XGBoost | Gas Monitoring Project
        </p>
    </body>
    </html>
    """

# =============================================================================
# ESP32 SENSOR ENDPOINTS
# =============================================================================

@app.post("/api/sensor/data")
async def receive_sensor_data(request: Request):
    """
    Receive sensor data from ESP32 devices and store in Firebase (JSON format)
    """
    try:
        data = await request.json()
        
        # Validate required fields
        required_fields = ['sensorID', 'timestamp', 'co2', 'methane', 'ammonia', 'temperature', 'humidity']
        for field in required_fields:
            if field not in data:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "message": f"Missing required field: {field}"}
                )
        
        # Calculate risk index
        risk_index = calculate_risk_index(data)
        data['riskIndex'] = risk_index
        
        # Ensure time field exists for compatibility
        if 'time' not in data:
            data['time'] = data['timestamp']
        
        # Get current timestamp for unique key
        current_timestamp = int(datetime.now().timestamp())
        
        # Store in history
        history_ref = db.reference(f"history/{data['sensorID']}/{current_timestamp}")
        history_ref.set(data)
        
        # Update current reading
        current_ref = db.reference(f"sensorReadings/{data['sensorID']}")
        current_ref.set(data)
        
        logger.info(f"Successfully stored data for sensor {data['sensorID']} at {data['timestamp']}")
        
        return {
            "status": "success",
            "message": "Sensor data stored successfully",
            "sensorID": data['sensorID'],
            "timestamp": data['timestamp'],
            "riskIndex": risk_index,
            "storage": {
                "history_key": current_timestamp,
                "current_reading_updated": True
            }
        }
        
    except Exception as e:
        logger.error(f"Error storing sensor data: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Internal server error: {str(e)}"}
        )

@app.get("/api/sensor/current/{sensor_id}")
async def get_current_sensor_data(sensor_id: str):
    """
    Get current sensor reading for a specific sensor
    """
    try:
        current_reading = get_current_sensor_reading(sensor_id)
        
        if not current_reading:
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": f"No current reading found for sensor {sensor_id}"}
            )
        
        return {
            "status": "success",
            "sensorID": sensor_id,
            "current_reading": current_reading,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching current sensor data: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Internal server error: {str(e)}"}
        )

@app.get("/api/sensor/{sensor_id}/stats")
async def get_sensor_stats(sensor_id: str):
    """
    Get statistics for a specific sensor
    """
    try:
        records = fetch_sensor_history(sensor_id)
        
        if not records:
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": f"No data found for sensor {sensor_id}"}
            )
        
        df = pd.DataFrame(records)
        
        # Calculate basic statistics for each sensor type
        sensor_types = ['co2', 'methane', 'ammonia', 'temperature', 'humidity']
        stats = {}
        
        for sensor_type in sensor_types:
            if sensor_type in df.columns:
                values = pd.to_numeric(df[sensor_type], errors='coerce').dropna()
                if len(values) > 0:
                    stats[sensor_type] = {
                        "count": int(len(values)),
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "latest": float(values.iloc[-1]) if len(values) > 0 else 0
                    }
        
        current_reading = get_current_sensor_reading(sensor_id)
        
        return {
            "status": "success",
            "sensorID": sensor_id,
            "total_records": len(records),
            "statistics": stats,
            "current_reading": current_reading,
            "time_range": {
                "first_record": df['timestamp'].min().isoformat() if 'timestamp' in df.columns and not df.empty else None,
                "last_record": df['timestamp'].max().isoformat() if 'timestamp' in df.columns and not df.empty else None
            }
        }
        
    except Exception as e:
        logger.error(f"Error calculating sensor stats: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Internal server error: {str(e)}"}
        )

# =============================================================================
# DATA ENDPOINTS
# =============================================================================
@app.get("/explain/{sensor_id}")
def explain(
    sensor_id: str, 
    sensor: str = Query(...),
    range: str = Query("1month", description="Date range: 1week, 1month, 3months, 6months, 1year, all")
):
    """SHAP explanation with date range filtering"""
    try:
        records = fetch_sensor_history(sensor_id)
        df = preprocess_dataframe(records, sensor)
        
        if df.empty or len(df) < 10:
            return JSONResponse({"error": "Not enough data"})
        
        df_filtered = filter_by_date_range(df, range)
        
        if df_filtered.empty or len(df_filtered) < 5:
            return JSONResponse({"error": f"Not enough data after applying {range} filter"})
        
        df_filtered["hour"] = df_filtered["timestamp"].dt.hour
        agg = df_filtered.groupby("hour")["value"].mean().reset_index()
        
        if len(agg) < 3:
            return JSONResponse({"error": "Insufficient hourly data after filtering"})
        
        X = agg[["hour"]]
        y = agg["value"]
        
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
        model.fit(X, y)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        buf = io.BytesIO()
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, show=False)
        plt.title(f"SHAP Summary - {sensor} (Sensor {sensor_id})\nDate Range: {range}")
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        
        return {
            "sensor_id": sensor_id,
            "sensor_type": sensor,
            "date_range": range,
            "shap_values": shap_values.tolist(),
            "features": X.to_dict(orient="records"),
            "image_data": base64.b64encode(buf.getvalue()).decode('utf-8')
        }
        
    except Exception as e:
        logger.error(f"SHAP explanation error: {e}")
        return JSONResponse({"error": str(e)})
@app.get("/shap_hour/{sensor_id}")
def shap_hour(
    sensor_id: str, 
    sensor: str = Query(...),
    range: str = Query("1month", description="Date range: 1week, 1month, 3months, 6months, 1year, all")
):
    """SHAP hourly analysis with date range filtering"""
    try:
        logger.info(f"Starting SHAP hour analysis for sensor {sensor_id}, type {sensor}, range {range}")
        
        # Step 1: Fetch data
        records = fetch_sensor_history(sensor_id)
        logger.info(f"Fetched {len(records)} records for sensor {sensor_id}")
        
        if not records:
            return {
                "error": "No data found for sensor", 
                "sensor_id": sensor_id, 
                "sensor_type": sensor,
                "status": "error"
            }
        
        # Step 2: Preprocess dataframe
        df = preprocess_dataframe(records, sensor)
        logger.info(f"Preprocessed dataframe shape: {df.shape}")
        
        if df.empty:
            return {
                "error": "No valid data after preprocessing", 
                "sensor_id": sensor_id, 
                "sensor_type": sensor,
                "status": "error"
            }
        
        # Step 3: Filter by date range
        df_filtered = filter_by_date_range(df, range)
        logger.info(f"After date filtering shape: {df_filtered.shape}")
        
        if df_filtered.empty:
            return {
                "error": f"No data after applying {range} filter", 
                "sensor_id": sensor_id, 
                "sensor_type": sensor,
                "status": "error"
            }
        
        # Check if we have enough data after filtering
        if len(df_filtered) < 3:
            return {
                "error": f"Not enough data after filtering. Need at least 3 records, got {len(df_filtered)}", 
                "sensor_id": sensor_id, 
                "sensor_type": sensor,
                "status": "error"
            }
        
        # Step 4: Create hour column and aggregate
        try:
            df_filtered["hour"] = df_filtered["timestamp"].dt.hour
            agg = df_filtered.groupby("hour")["value"].agg(['mean', 'std', 'count']).reset_index()
            agg = agg.rename(columns={'mean': 'value'})
            logger.info(f"Aggregated data shape: {agg.shape}")
        except Exception as e:
            logger.error(f"Error in aggregation: {e}")
            return {
                "error": f"Data aggregation failed: {str(e)}", 
                "sensor_id": sensor_id, 
                "sensor_type": sensor,
                "status": "error"
            }
        
        # Check if we have any aggregated data
        if agg.empty:
            return {
                "error": "No hourly data available after aggregation", 
                "sensor_id": sensor_id, 
                "sensor_type": sensor,
                "status": "error"
            }
        
        # Step 5: Generate plot
        try:
            buf = io.BytesIO()
            plt.figure(figsize=(10, 6))
            
            # Plot main line
            plt.plot(agg["hour"], agg["value"], marker='o', linewidth=2, label='Average Value')
            
            # Add standard deviation shading if available
            if 'std' in agg.columns and not agg['std'].isna().all():
                valid_hours = agg[~agg['std'].isna()]
                if not valid_hours.empty:
                    plt.fill_between(
                        valid_hours["hour"], 
                        valid_hours["value"] - valid_hours["std"], 
                        valid_hours["value"] + valid_hours["std"], 
                        alpha=0.2, 
                        label='Standard Deviation'
                    )
            
            plt.xlabel("Hour of Day")
            plt.ylabel("Sensor Value")
            plt.title(f"Hourly Analysis - {sensor} (Sensor {sensor_id})\nDate Range: {range}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(range(0, 24))
            plt.tight_layout()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            plt.close()
            buf.seek(0)
            
            # Convert to base64
            image_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            logger.info("Plot generated successfully")
            
        except Exception as e:
            logger.error(f"Plot generation failed: {e}")
            return {
                "error": f"Plot generation failed: {str(e)}", 
                "sensor_id": sensor_id, 
                "sensor_type": sensor,
                "status": "error"
            }
        
        # Step 6: Prepare response
        hourly_data = agg.to_dict(orient="records")
        
        response = {
            "sensor_id": sensor_id,
            "sensor_type": sensor,
            "date_range": range,
            "hourly_data": hourly_data,
            "image_data": image_data,
            "status": "success",
            "data_points": len(df_filtered),
            "hourly_points": len(hourly_data)
        }
        
        logger.info(f"SHAP hour analysis completed successfully. Data points: {len(df_filtered)}")
        return response
        
    except Exception as e:
        logger.error(f"SHAP hour analysis error: {str(e)}", exc_info=True)
        return {
            "error": f"Internal server error: {str(e)}", 
            "sensor_id": sensor_id, 
            "sensor_type": sensor,
            "status": "error"
        }

@app.get("/dataframe/{sensor_id}")
def get_dataframe(
    sensor_id: str, 
    sensor: str = Query(..., description="Sensor type (co2, temperature, etc.)"),
    range: str = Query("1month", description="Date range: 1week, 1month, 3months, 6months, 1year, all")
):
    """Get raw sensor data as JSON with date range filtering"""
    try:
        records = fetch_sensor_history(sensor_id)
        df = preprocess_dataframe(records, sensor)
        
        df_filtered = filter_by_date_range(df, range)
        
        return {
            "sensor_id": sensor_id,
            "sensor_type": sensor,
            "date_range": range,
            "records": df_filtered.to_dict(orient="records"),
            "count": len(df_filtered),
            "date_range_applied": True
        }
    except Exception as e:
        logger.error(f"Dataframe error: {e}")
        return {"error": str(e), "sensor_id": sensor_id, "sensor_type": sensor}

@app.get("/plot/{sensor_id}")
def plot_sensor_data(
    sensor_id: str, 
    sensor: str = Query(..., description="Sensor type"),
    chart: str = Query("scatter", description="Chart type: scatter or summary"),
    range: str = Query("1month", description="Date range: 1week, 1month, 3months, 6months, 1year, all")
):
    """Generate visualization plots for sensor data with date range"""
    try:
        records = fetch_sensor_history(sensor_id)
        if not records:
            return error_image("No data found for this sensor")
        
        df = preprocess_dataframe(records, sensor)
        if df.empty or len(df) < 5:
            return error_image(f"Not enough data for plot. Found {len(df)} records.")
        
        df_filtered = filter_by_date_range(df, range)
        
        if df_filtered.empty or len(df_filtered) < 3:
            return error_image(f"Not enough data after applying {range} filter. Found {len(df_filtered)} records.")
        
        buf = io.BytesIO()
        plt.figure(figsize=(10, 6))
        
        if chart == "scatter":
            plt.scatter(df_filtered["timestamp"], df_filtered["value"], alpha=0.7)
            plt.xlabel("Timestamp")
            plt.ylabel("Sensor Value")
            plt.title(f"Scatter Plot - {sensor} (Sensor {sensor_id})\nDate Range: {range}")
        elif chart == "line":
            plt.plot(df_filtered["timestamp"], df_filtered["value"], marker='o', linewidth=2, markersize=4)
            plt.xlabel("Timestamp")
            plt.ylabel("Sensor Value")
            plt.title(f"Line Plot - {sensor} (Sensor {sensor_id})\nDate Range: {range}")
        else:
            plt.plot(df_filtered["timestamp"], df_filtered["value"], marker='o')
            plt.xlabel("Timestamp")
            plt.ylabel("Sensor Value")
            plt.title(f"Time Series - {sensor} (Sensor {sensor_id})\nDate Range: {range}")
        
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="image/png")
        
    except Exception as e:
        logger.error(f"Plot generation error: {e}")
        return error_image(f"Error generating plot: {str(e)}")

# =============================================================================
# AI/ML ENDPOINTS
# =============================================================================

@app.get("/algorithm/{sensor_id}")
def compare_algorithms(
    sensor_id: str, 
    sensor: str = Query(..., description="Sensor type"),
    range: str = Query("1month", description="Date range: 1week, 1month, 3months, 6months, 1year, all")
):
    """Compare multiple ML algorithms with date range filtering"""
    try:
        records = fetch_sensor_history(sensor_id)
        df = preprocess_dataframe(records, sensor)
        
        if df.empty or len(df) < 5:
            return {"error": "Not enough data for analysis", "sensor_id": sensor_id, "sensor_type": sensor}
        
        df_filtered = filter_by_date_range(df, range)
        
        if df_filtered.empty or len(df_filtered) < 5:
            return {"error": f"Not enough data after applying {range} filter", "sensor_id": sensor_id, "sensor_type": sensor}
        
        df_lags = make_lag_features(df_filtered, lags=2)
        if df_lags.empty:
            return {"error": "Insufficient data after feature engineering", "sensor_id": sensor_id, "sensor_type": sensor}
        
        X, y = df_lags[["lag1", "lag2"]], df_lags["value"]
        
        test_size = min(0.2, 1.0 / len(X))
        if len(X) <= 5:
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        
        results = {}
        
        lr = LinearRegression().fit(X_train, y_train)
        results["LinearRegression"] = compute_metrics(y_test, lr.predict(X_test))
        
        rf = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_train, y_train)
        results["RandomForest"] = compute_metrics(y_test, rf.predict(X_test))
        
        dt = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
        results["DecisionTree"] = compute_metrics(y_test, dt.predict(X_test))
        
        xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=50, random_state=42)
        xgb_model.fit(X_train, y_train)
        results["XGBoost"] = compute_metrics(y_test, xgb_model.predict(X_test))
        
        last_row = df_lags.iloc[-1]
        pred_next = xgb_model.predict([[last_row["lag1"], last_row["lag2"]]])[0]
        forecast_val = float(pred_next)
        next_date = df_lags["timestamp"].iloc[-1] + timedelta(days=1)
        
        recommendation = generate_recommendation(sensor, forecast_val)
        
        return {
            "sensor_id": sensor_id,
            "sensor_type": sensor,
            "date_range": range,
            "algorithms": results,
            "best_algorithm": min(results.keys(), key=lambda x: results[x]["rmse"]),
            "forecast": {
                "date": next_date.strftime("%Y-%m-%d"),
                "predicted_value": forecast_val,
                "recommendation": recommendation
            },
            "data_points": len(df_lags),
            "date_range_applied": True
        }
        
    except Exception as e:
        logger.error(f"Algorithm comparison error: {e}")
        return {"error": str(e), "sensor_id": sensor_id, "sensor_type": sensor}

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
            
        df = preprocess_dataframe(records, sensor)
        if df.empty:
            return {"error": "No valid data after preprocessing", "status": "error"}
        
        if len(df) < 3:
            return {"error": "Insufficient clean data for training", "status": "error"}

        df_lags = make_lag_features(df, lags=2)
        if len(df_lags) < 2:
            return {"error": "Insufficient data after feature engineering", "status": "error"}

        X = df_lags[["lag1", "lag2"]]
        y = df_lags["value"]

        if len(X) <= 3:
            X_train, X_test, y_train, y_test = X, X.iloc[-1:], y, y.iloc[-1:]
        else:
            X_train, X_test = X.iloc[:-1], X.iloc[-1:]
            y_train, y_test = y.iloc[:-1], y.iloc[-1:]

        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=30, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        mse = mean_squared_error(y_test, y_pred) if len(y_test) > 0 else 0
        rmse = math.sqrt(mse) if mse > 0 else 0
        mae = mean_absolute_error(y_test, y_pred) if len(y_test) > 0 else 0
        r2 = r2_score(y_train, y_pred_train) if len(y_train) > 1 else 0

        recommendation = generate_recommendation(sensor, float(y_pred[0]))

        return {
            "status": "success",
            "sensor_id": sensor_id,
            "sensor_type": sensor,
            "latest_actual": float(y_test.iloc[0]) if len(y_test) > 0 else float(y_train.iloc[-1]),
            "predicted_next": float(y_pred[0]),
            "recommendation": recommendation,
            "metrics": {
                "MSE": round(mse, 4),
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
                "R2": round(r2, 4)
            },
            "history_points": len(df_lags)
        }
        
    except Exception as e:
        logger.error(f"XGBoost analysis error: {e}")
        return {"error": str(e), "status": "error"}

# =============================================================================
# FORECASTING ENDPOINTS
# =============================================================================

@app.get("/predict/{sensor_id}")
def predict_future(
    sensor_id: str,
    sensor: str = Query(..., description="Sensor type"),
    days: int = Query(3, description="Forecast horizon in days")
):
    """Generate multi-day forecasts"""
    try:
        records = fetch_sensor_history(sensor_id)
        df = preprocess_dataframe(records, sensor)
        
        if df.empty or len(df) < 3:
            return {"error": "Not enough historical data", "sensor_id": sensor_id, "sensor_type": sensor}
        
        df_lags = make_lag_features(df, lags=2)
        if df_lags.empty:
            return {"error": "Insufficient data for forecasting", "sensor_id": sensor_id, "sensor_type": sensor}
        
        X = df_lags[["lag1", "lag2"]].values
        y = df_lags["value"].values
        
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=50, random_state=42)
        model.fit(X, y)
        
        forecasts = []
        current_features = X[-1].copy()
        current_date = df_lags["timestamp"].iloc[-1]
        
        for i in range(min(days, 7)):
            next_pred = model.predict(current_features.reshape(1, -1))[0]
            current_date += timedelta(days=1)
            
            forecasts.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "forecast": float(next_pred),
                "recommendation": generate_recommendation(sensor, float(next_pred))
            })
            
            current_features = np.roll(current_features, -1)
            current_features[-1] = next_pred
        
        return {
            "sensor_id": sensor_id,
            "sensor_type": sensor,
            "forecast_horizon_days": len(forecasts),
            "forecasts": forecasts,
            "last_actual_value": float(y[-1]),
            "last_actual_date": df_lags["timestamp"].iloc[-1].strftime("%Y-%m-%d")
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e), "sensor_id": sensor_id, "sensor_type": sensor}

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

        cutoff = datetime.now() - timedelta(days=30)
        df_recent = df[df["timestamp"] >= cutoff]

        if len(df_recent) < 3:
            current_value = df["value"].iloc[-1] if not df.empty else 0
            recommendation = generate_recommendation(sensor, float(current_value))
            
            return {
                "sensor_id": sensor_id,
                "sensor_type": sensor,
                "current_value": float(current_value),
                "recommendation": recommendation,
                "status": "current_value_only"
            }

        df_lags = make_lag_features(df_recent, lags=2)
        if df_lags.empty:
            current_value = df_recent["value"].iloc[-1]
            recommendation = generate_recommendation(sensor, float(current_value))
            return {
                "sensor_id": sensor_id,
                "sensor_type": sensor,
                "current_value": float(current_value),
                "recommendation": recommendation,
                "status": "current_value_only"
            }

        X = df_lags[["lag1", "lag2"]].values
        y = df_lags["value"].values

        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=50, random_state=42)
        model.fit(X, y)

        last_lags = X[-1].copy()
        last_date = df_lags["timestamp"].iloc[-1]
        pred = model.predict(last_lags.reshape(1, -1))[0]
        next_date = last_date + timedelta(days=1)
        forecast_val = float(pred)

        recommendation = generate_recommendation(sensor, forecast_val)

        return {
            "sensor_id": sensor_id,
            "sensor_type": sensor,
            "forecast_date": next_date.strftime("%Y-%m-%d"),
            "predicted_value": forecast_val,
            "recommendation": recommendation,
            "current_value": float(df_lags["value"].iloc[-1]),
            "data_points": len(df_lags),
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

# =============================================================================
# ANALYTICS ENDPOINTS
# =============================================================================

def create_classification_labels(df: pd.DataFrame, sensor_type: str) -> pd.DataFrame:
    """Create classification labels based on sensor thresholds with better class distribution"""
    if df.empty or len(df) < 5:
        return pd.DataFrame()
    
    try:
        df_class = df.copy()
        
        # Use quantile-based approach to ensure we have multiple classes
        if len(df_class) >= 10:
            q1 = df_class['value'].quantile(0.25)
            q2 = df_class['value'].quantile(0.5)
            q3 = df_class['value'].quantile(0.75)
            
            conditions = [
                df_class['value'] < q1,
                (df_class['value'] >= q1) & (df_class['value'] < q2),
                (df_class['value'] >= q2) & (df_class['value'] < q3),
                df_class['value'] >= q3
            ]
            choices = ['low', 'medium_low', 'medium_high', 'high']
        else:
            median_val = df_class['value'].median()
            conditions = [
                df_class['value'] <= median_val,
                df_class['value'] > median_val
            ]
            choices = ['below_median', 'above_median']
        
        df_class['class'] = np.select(conditions, choices, default='unknown')
        df_class = df_class[df_class['class'] != 'unknown']
        
        class_counts = df_class['class'].value_counts()
        valid_classes = class_counts[class_counts >= 1].index.tolist()
        
        if len(valid_classes) < 2:
            median_val = df_class['value'].median()
            df_class['class'] = np.where(df_class['value'] > median_val, 'high', 'low')
            class_counts = df_class['class'].value_counts()
            valid_classes = class_counts[class_counts >= 1].index.tolist()
            
            if len(valid_classes) < 2:
                return pd.DataFrame()
        
        df_class = df_class[df_class['class'].isin(valid_classes)]
        return df_class
    except Exception as e:
        logger.error(f"Error creating classification labels: {e}")
        return pd.DataFrame()

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str]) -> Dict[str, Any]:
    """Compute comprehensive classification metrics"""
    try:
        if len(y_true) == 0 or len(y_pred) == 0:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "class_report": {},
                "confusion_matrix": [],
                "classes": classes
            }
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "class_report": class_report,
            "confusion_matrix": conf_matrix.tolist(),
            "classes": classes
        }
    except Exception as e:
        logger.error(f"Error computing classification metrics: {e}")
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "class_report": {},
            "confusion_matrix": [],
            "classes": classes
        }

@app.get("/performance/{sensor_id}")
def performance_metrics(
    sensor_id: str,
    sensor: str = Query(..., description="Sensor type"),
    test_size: float = Query(0.2, description="Test set size ratio"),
    cv_folds: int = Query(3, description="Cross-validation folds"),
    range: str = Query("1month", description="Date range: 1week, 1month, 3months, 6months, 1year, all")
):
    """Get performance metrics for different classifiers with hyperparameter tuning and date range filtering"""
    try:
        logger.info(f"Starting performance metrics for sensor {sensor_id}, type {sensor}, range {range}")
        
        # Step 1: Fetch and preprocess data
        records = fetch_sensor_history(sensor_id)
        logger.info(f"Fetched {len(records)} records for sensor {sensor_id}")
        
        if not records:
            return {
                "error": f"No data found for sensor {sensor_id}", 
                "sensor_id": sensor_id, 
                "sensor_type": sensor,
                "date_range": range,
                "status": "error"
            }
        
        df = preprocess_dataframe(records, sensor)
        logger.info(f"Preprocessed dataframe shape: {df.shape}")
        
        if df.empty:
            return {
                "error": f"No valid data for sensor {sensor_id} and type {sensor}", 
                "sensor_id": sensor_id, 
                "sensor_type": sensor,
                "date_range": range,
                "status": "error"
            }
        
        # Step 2: Filter by date range
        df_filtered = filter_by_date_range(df, range)
        logger.info(f"After date filtering shape: {df_filtered.shape}")
        
        if df_filtered.empty or len(df_filtered) < 10:
            return {
                "error": f"Not enough data for performance analysis after {range} filter. Need at least 10 samples, got {len(df_filtered)}", 
                "sensor_id": sensor_id, 
                "sensor_type": sensor,
                "date_range": range,
                "status": "error"
            }
        
        # Step 3: Create classification labels
        df_class = create_classification_labels(df_filtered, sensor)
        logger.info(f"Classification labels created. Shape: {df_class.shape if not df_class.empty else 'empty'}")
        
        if df_class.empty:
            return {
                "error": f"Failed to create classification labels - insufficient class diversity after {range} filter", 
                "sensor_id": sensor_id, 
                "sensor_type": sensor,
                "date_range": range,
                "status": "error"
            }
        
        # Step 4: Check class distribution
        class_distribution = df_class['class'].value_counts()
        logger.info(f"Class distribution: {class_distribution.to_dict()}")
        
        if len(class_distribution) < 2:
            return {
                "error": f"Need at least 2 classes for classification after {range} filter. Found only: {class_distribution.to_dict()}", 
                "sensor_id": sensor_id, 
                "sensor_type": sensor,
                "date_range": range,
                "status": "error"
            }
        
        min_samples_per_class = 2
        valid_classes = class_distribution[class_distribution >= min_samples_per_class].index.tolist()
        logger.info(f"Valid classes with min {min_samples_per_class} samples: {valid_classes}")
        
        if len(valid_classes) < 2:
            return {
                "error": f"Need at least 2 classes with minimum {min_samples_per_class} samples each after {range} filter. Current distribution: {class_distribution.to_dict()}", 
                "sensor_id": sensor_id, 
                "sensor_type": sensor,
                "date_range": range,
                "status": "error"
            }
        
        df_class = df_class[df_class['class'].isin(valid_classes)]
        logger.info(f"After filtering valid classes shape: {df_class.shape}")
        
        # Step 5: Create lag features
        df_lags = make_lag_features(df_class, lags=2)
        logger.info(f"After creating lag features shape: {df_lags.shape}")
        
        if df_lags.empty or len(df_lags) < 5:
            return {
                "error": f"Insufficient data after feature engineering with {range} filter. Need at least 5 samples, got {len(df_lags)}", 
                "sensor_id": sensor_id, 
                "sensor_type": sensor,
                "date_range": range,
                "status": "error"
            }
        
        # Step 6: Prepare features and target
        feature_cols = [col for col in df_lags.columns if col.startswith('lag')]
        X = df_lags[feature_cols]
        y = df_lags['class']
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Step 7: Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        classes = le.classes_.tolist()
        
        logger.info(f"Encoded classes: {classes}")
        
        if len(classes) < 2:
            return {
                "error": f"Need at least 2 classes for classification after {range} filter. Found: {classes}", 
                "sensor_id": sensor_id, 
                "sensor_type": sensor,
                "date_range": range,
                "status": "error"
            }
        
        # Step 8: Split data
        n_samples = len(X)
        actual_test_size = min(test_size, 0.3)
        actual_cv_folds = min(cv_folds, max(2, n_samples // 3))
        
        logger.info(f"Data split - total: {n_samples}, test_size: {actual_test_size}, cv_folds: {actual_cv_folds}")
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=actual_test_size, random_state=42, stratify=y_encoded
            )
        except ValueError as e:
            logger.error(f"Train-test split failed: {e}")
            return {
                "error": f"Data splitting failed: {str(e)}", 
                "sensor_id": sensor_id, 
                "sensor_type": sensor,
                "date_range": range,
                "status": "error"
            }
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        if len(X_train) < 5:
            return {
                "error": f"Insufficient training data after {range} filter: {len(X_train)} samples. Need at least 5.", 
                "sensor_id": sensor_id, 
                "sensor_type": sensor,
                "date_range": range,
                "status": "error"
            }
        
        # Step 9: Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        
        # Step 10: Train and evaluate models
        models_to_try = [
            ('XGBoost', xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=50)),
            ('RandomForest', RandomForestClassifier(random_state=42, n_estimators=50)),
            ('LogisticRegression', LogisticRegression(random_state=42, max_iter=1000))
        ]
        
        for model_name, model in models_to_try:
            try:
                logger.info(f"Training {model_name}...")
                
                if model_name == 'LogisticRegression':
                    # Use scaled features for Logistic Regression
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    # Use original features for tree-based models
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                metrics = compute_classification_metrics(y_test, y_pred, classes)
                
                results[model_name] = {
                    'metrics': metrics
                }
                
                logger.info(f"{model_name} trained successfully. Accuracy: {metrics['accuracy']:.3f}")
                
            except Exception as e:
                logger.error(f"{model_name} failed: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        # Step 11: Determine best algorithm
        successful_models = {k: v for k, v in results.items() if 'metrics' in v and 'error' not in v}
        
        if successful_models:
            best_algorithm = max(successful_models.keys(), 
                               key=lambda x: successful_models[x]['metrics']['f1_score'])
            best_score = successful_models[best_algorithm]['metrics']['f1_score']
            logger.info(f"Best algorithm: {best_algorithm} with F1 score: {best_score:.3f}")
        else:
            best_algorithm = "No successful models"
            best_score = 0
            logger.warning("No models trained successfully")
        
        # Step 12: Prepare final response
        response = {
            "sensor_id": sensor_id,
            "sensor_type": sensor,
            "date_range": range,
            "dataset_info": {
                "total_samples": len(df_lags),
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "feature_count": len(feature_cols),
                "classes": classes,
                "class_distribution": dict(df_lags['class'].value_counts()),
                "original_data_points": len(df),
                "filtered_data_points": len(df_filtered),
                "date_range_applied": range
            },
            "algorithms": results,
            "best_algorithm": best_algorithm,
            "best_score": float(best_score),
            "test_size_used": actual_test_size,
            "cv_folds_used": actual_cv_folds,
            "status": "success"
        }
        
        logger.info(f"Performance metrics completed successfully for sensor {sensor_id}")
        return response
        
    except Exception as e:
        logger.error(f"Performance metrics error: {str(e)}", exc_info=True)
        return {
            "error": f"Internal server error: {str(e)}", 
            "sensor_id": sensor_id, 
            "sensor_type": sensor,
            "date_range": range,
            "status": "error"
        }
        
@app.get("/confusion_matrix/{sensor_id}")
def confusion_matrix_chart(
    sensor_id: str,
    sensor: str = Query(..., description="Sensor type"),
    test_size: float = Query(0.2, description="Test size"),
    cv_folds: int = Query(3, description="CV folds"),
    range: str = Query("1month", description="Date range: 1week, 1month, 3months, 6months, 1year, all")
):
    """Generate confusion matrix chart for best performing algorithm with date range filtering"""
    try:
        perf_response = performance_metrics(sensor_id, sensor, test_size, cv_folds, range)
        
        if "error" in perf_response:
            error_msg = perf_response["error"]
            return error_image(f"Performance metrics error: {error_msg}")
        
        if "algorithms" not in perf_response:
            return error_image("No algorithms data available")
        
        best_algo = None
        for algo_name, algo_data in perf_response["algorithms"].items():
            if "metrics" in algo_data and "error" not in algo_data:
                best_algo = algo_name
                break
        
        if not best_algo:
            return error_image("No valid model with metrics available for confusion matrix")
        
        metrics = perf_response["algorithms"][best_algo]["metrics"]
        
        if "confusion_matrix" not in metrics or "classes" not in metrics:
            return error_image("No confusion matrix data available")
        
        conf_matrix = np.array(metrics["confusion_matrix"])
        classes = metrics["classes"]
        
        if len(classes) < 2:
            return error_image(f"Need at least 2 classes for confusion matrix. Found: {classes}")
        
        if conf_matrix.size == 0 or conf_matrix.shape[0] != len(classes):
            return error_image("Invalid confusion matrix dimensions")
        
        buf = io.BytesIO()
        plt.figure(figsize=(8, 6))
        
        plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues, alpha=0.7)
        plt.title(f"Confusion Matrix - {best_algo}\n(Sensor: {sensor}, Date Range: {range})", fontsize=14, pad=20)
        plt.colorbar()

        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, ha='right')
        plt.yticks(tick_marks, classes)

        cm_normalized = conf_matrix.astype("float") / np.maximum(conf_matrix.sum(axis=1)[:, np.newaxis], 1)
        thresh = conf_matrix.max() / 2.
        
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(
                    j, i,
                    f"{conf_matrix[i, j]}\n({cm_normalized[i, j]:.1%})",
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="white" if conf_matrix[i, j] > thresh else "black",
                    fontsize=10
                )

        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.tight_layout()
        
        accuracy = metrics.get("accuracy", 0)
        plt.figtext(0.5, 0.01, f"Accuracy: {accuracy:.2%} | Date Range: {range}", ha="center", fontsize=10, 
                   bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        logger.error(f"Confusion matrix error: {e}")
        return error_image(f"Error generating confusion matrix: {str(e)}")

# =============================================================================
# TEST ENDPOINT
# =============================================================================

@app.get("/test/{sensor_id}")
def test_endpoint(sensor_id: str, sensor: str = Query("temperature")):
    """Test endpoint to verify basic functionality"""
    return {
        "message": "API is working",
        "sensor_id": sensor_id,
        "sensor_type": sensor,
        "timestamp": datetime.now().isoformat()
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)