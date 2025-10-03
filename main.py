# --- main.py ---
import json
import gc
import psutil
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
from sklearn.model_selection import  GridSearchCV, train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import logging
from typing import Dict, Any, Optional, List
  


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")

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

def log_memory_usage(prefix=""):
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"{prefix} Memory usage: {memory_mb:.2f} MB")
    return memory_mb
    
def fetch_sensor_history(sensor_id: str) -> List[Dict[str, Any]]:
    """Fetch sensor history from Firebase with robust timestamp parsing"""
    try:
        logger.info(f"Fetching sensor history for {sensor_id}")
        ref = db.reference(f"history/{sensor_id}")
        snapshot = ref.get()
        
        if not snapshot:
            logger.warning(f"No data found for sensor {sensor_id}")
            return []
        
        records = []
        valid_records = 0
        error_records = 0
        
        for key, value in snapshot.items():
            if isinstance(value, dict):
                row = value.copy()
                
                # Robust timestamp parsing
                if "timestamp" in row:
                    timestamp_str = str(row["timestamp"]).strip()
                    
                    # Skip records with "Time Error" or other invalid timestamp strings
                    if timestamp_str.lower() in ["time error", "error", "none", "null", "nan", ""]:
                        error_records += 1
                        continue
                    
                    try:
                        # Try multiple datetime formats
                        if timestamp_str.isdigit() and len(timestamp_str) == 10:
                            # Unix timestamp (seconds)
                            row["timestamp"] = datetime.fromtimestamp(int(timestamp_str))
                        elif timestamp_str.isdigit() and len(timestamp_str) == 13:
                            # Unix timestamp (milliseconds)
                            row["timestamp"] = datetime.fromtimestamp(int(timestamp_str) / 1000)
                        else:
                            # Try ISO format or other common formats
                            row["timestamp"] = pd.to_datetime(timestamp_str, errors='coerce')
                            
                            # Check if parsing was successful
                            if pd.isna(row["timestamp"]):
                                error_records += 1
                                continue
                                
                        valid_records += 1
                        records.append(row)
                        
                    except (ValueError, TypeError, OSError) as e:
                        error_records += 1
                        logger.debug(f"Failed to parse timestamp '{timestamp_str}' for record {key}: {e}")
                        continue
                else:
                    # Record has no timestamp, skip it
                    error_records += 1
                    continue
        
        logger.info(f"Fetched {valid_records} valid records and skipped {error_records} invalid records for sensor {sensor_id}")
        
        # Sort by timestamp
        if records:
            records.sort(key=lambda x: x["timestamp"])
            logger.info(f"Date range: {records[0]['timestamp']} to {records[-1]['timestamp']}")
        
        return records
        
    except Exception as e:
        logger.error(f"Error fetching sensor history for {sensor_id}: {e}")
        return []

def get_current_sensor_reading(sensor_id: str) -> Optional[Dict[str, Any]]:
    """Get current sensor reading from sensorReadings with robust timestamp parsing"""
    try:
        ref = db.reference(f"sensorReadings/{sensor_id}")
        snapshot = ref.get()
        if snapshot and isinstance(snapshot, dict):
            # Robust timestamp parsing for current reading
            if "timestamp" in snapshot:
                timestamp_str = str(snapshot["timestamp"]).strip()
                
                # Skip if timestamp is invalid
                if timestamp_str.lower() in ["time error", "error", "none", "null", "nan", ""]:
                    logger.warning(f"Invalid timestamp in current reading for sensor {sensor_id}: {timestamp_str}")
                    snapshot["timestamp"] = None
                else:
                    try:
                        if timestamp_str.isdigit() and len(timestamp_str) == 10:
                            snapshot["timestamp"] = datetime.fromtimestamp(int(timestamp_str))
                        elif timestamp_str.isdigit() and len(timestamp_str) == 13:
                            snapshot["timestamp"] = datetime.fromtimestamp(int(timestamp_str) / 1000)
                        else:
                            parsed_time = pd.to_datetime(timestamp_str, errors='coerce')
                            if pd.isna(parsed_time):
                                snapshot["timestamp"] = None
                            else:
                                snapshot["timestamp"] = parsed_time
                    except (ValueError, TypeError, OSError):
                        snapshot["timestamp"] = None
            return snapshot
        return None
    except Exception as e:
        logger.error(f"Error fetching current sensor reading: {e}")
        return None

def preprocess_dataframe(records: List[Dict[str, Any]], sensor: str) -> pd.DataFrame:
    """Preprocess dataframe for analysis with robust data validation"""
    if not records:
        logger.warning("No records to preprocess")
        return pd.DataFrame()
    
    try:
        # Filter out records with invalid timestamps or missing sensor data
        valid_records = []
        for record in records:
            # Check if timestamp exists and is valid
            if "timestamp" not in record or record["timestamp"] is None:
                continue
                
            # Check if sensor data exists and is numeric
            if sensor not in record:
                continue
                
            try:
                sensor_value = float(record[sensor])
                # Skip NaN or infinite values
                if not np.isfinite(sensor_value):
                    continue
                    
                valid_records.append({
                    "timestamp": record["timestamp"],
                    "value": sensor_value
                })
            except (ValueError, TypeError):
                continue
        
        if not valid_records:
            logger.warning("No valid records after filtering")
            return pd.DataFrame()
            
        df = pd.DataFrame(valid_records)
        df = df.sort_values("timestamp")
        
        logger.info(f"Final preprocessed dataframe: {len(df)} valid records")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"Value range: {df['value'].min():.2f} to {df['value'].max():.2f}")
        
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

# Also update the ESP32 data reception endpoint to validate timestamps
@app.post("/api/sensor/data")
async def receive_sensor_data(request: Request):
    """
    Receive sensor data from ESP32 devices and store in Firebase with timestamp validation
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
        
        # Validate and parse timestamp
        timestamp_str = str(data['timestamp']).strip()
        valid_timestamp = None
        
        try:
            if timestamp_str.isdigit() and len(timestamp_str) == 10:
                valid_timestamp = datetime.fromtimestamp(int(timestamp_str))
            elif timestamp_str.isdigit() and len(timestamp_str) == 13:
                valid_timestamp = datetime.fromtimestamp(int(timestamp_str) / 1000)
            else:
                valid_timestamp = pd.to_datetime(timestamp_str, errors='coerce')
                if pd.isna(valid_timestamp):
                    valid_timestamp = datetime.now()  # Fallback to current time
        except (ValueError, TypeError, OSError):
            valid_timestamp = datetime.now()  # Fallback to current time
        
        # Replace the timestamp with validated one
        data['timestamp'] = valid_timestamp.isoformat()
        
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
    """SHAP explanation with date range filtering and memory optimization"""
    try:
        log_memory_usage("Before SHAP explanation")
        
        records = fetch_sensor_history(sensor_id)
        df = preprocess_dataframe(records, sensor)
        
        if df.empty or len(df) < 10:
            return JSONResponse({"error": "Not enough data"})
        
        df_filtered = filter_by_date_range(df, range)
        
        if df_filtered.empty or len(df_filtered) < 5:
            return JSONResponse({"error": f"Not enough data after applying {range} filter"})
        
        # Limit data size for SHAP analysis
        max_shap_samples = 1000
        if len(df_filtered) > max_shap_samples:
            df_filtered = df_filtered.sample(max_shap_samples, random_state=42)
            logger.info(f"Sampled {max_shap_samples} records for SHAP analysis")
        
        df_filtered["hour"] = df_filtered["timestamp"].dt.hour
        agg = df_filtered.groupby("hour")["value"].mean().reset_index()
        
        if len(agg) < 3:
            return JSONResponse({"error": "Insufficient hourly data after filtering"})
        
        X = agg[["hour"]]
        y = agg["value"]
        
        # Use smaller model for SHAP
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=50, random_state=42)
        model.fit(X, y)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        buf = io.BytesIO()
        plt.figure(figsize=(8, 5))  # Smaller figure size
        shap.summary_plot(shap_values, X, show=False)
        plt.title(f"SHAP Summary - {sensor} (Sensor {sensor_id})\nDate Range: {range}")
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")  # Lower DPI
        plt.close('all')  # Explicitly close plot
        buf.seek(0)
        
        # Force garbage collection
        del model, explainer, shap_values, df, df_filtered, agg
        gc.collect()
        
        log_memory_usage("After SHAP explanation")
        
        return {
            "sensor_id": sensor_id,
            "sensor_type": sensor,
            "date_range": range,
            "shap_values": shap_values.tolist(),
            "features": X.to_dict(orient="records"),
            "feature_importance": {
                "hour": float(np.abs(shap_values).mean(axis=0)[0])  # Mean absolute SHAP value
            },
            "summary_stats": {
                "total_samples": len(df_filtered),
                "shap_samples": len(agg),
                "mean_shap_magnitude": float(np.mean(np.abs(shap_values)))
            }
        }
        
    except Exception as e:
        logger.error(f"SHAP explanation error: {e}")
        return JSONResponse({"error": str(e)})

@app.get("/plot/{sensor_id}")
def plot_sensor_data(
    sensor_id: str, 
    sensor: str = Query(..., description="Sensor type"),
    chart: str = Query("scatter", description="Chart type: scatter, line, summary, distribution"),
    range: str = Query("1month", description="Date range: 1week, 1month, 3months, 6months, 1year, all")
):
    """Generate visualization plots with memory optimization"""
    try:
        log_memory_usage("Before plot generation")
        
        records = fetch_sensor_history(sensor_id)
        if not records:
            return error_image("No data found for this sensor")
        
        df = preprocess_dataframe(records, sensor)
        if df.empty or len(df) < 5:
            return error_image(f"Not enough data for plot. Found {len(df)} records.")
        
        df_filtered = filter_by_date_range(df, range)
        
        if df_filtered.empty or len(df_filtered) < 3:
            return error_image(f"Not enough data after applying {range} filter. Found {len(df_filtered)} records.")
        
        # Limit data size for plotting
        max_plot_samples = 2000
        if len(df_filtered) > max_plot_samples:
            df_filtered = df_filtered.sample(max_plot_samples, random_state=42)
            logger.info(f"Sampled {max_plot_samples} records for plotting")
        
        buf = io.BytesIO()
        
        if chart == "distribution":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))  # Smaller figure
            
            sns.histplot(data=df_filtered, x="value", kde=True, ax=ax1, color='skyblue', alpha=0.7)
            ax1.set_xlabel(f"{sensor.upper()} Value")
            ax1.set_ylabel("Frequency")
            ax1.set_title(f"Distribution - {sensor.upper()}")
            
            sns.boxplot(data=df_filtered, y="value", ax=ax2, color='lightcoral')
            ax2.set_ylabel(f"{sensor.upper()} Value")
            ax2.set_title(f"Box Plot - {sensor.upper()}")
            
        elif chart == "scatter":
            plt.figure(figsize=(10, 6))  # Smaller figure
            
            # Sample data for scatter plot
            if len(df_filtered) > 500:
                plot_data = df_filtered.sample(500, random_state=42)
            else:
                plot_data = df_filtered
                
            plt.scatter(plot_data["timestamp"], plot_data["value"], alpha=0.6, s=10)
            plt.xlabel("Timestamp")
            plt.ylabel(f"{sensor.upper()} Value")
            plt.title(f"Scatter Plot - {sensor.upper()} (Sensor {sensor_id})")
            plt.xticks(rotation=45)
            
        elif chart == "line":
            plt.figure(figsize=(10, 6))
            
            # Resample for line plot to reduce points
            if len(df_filtered) > 1000:
                df_resampled = df_filtered.set_index('timestamp').resample('12H').mean().reset_index()
            else:
                df_resampled = df_filtered
                
            plt.plot(df_resampled["timestamp"], df_resampled["value"], linewidth=1)
            plt.xlabel("Timestamp")
            plt.ylabel(f"{sensor.upper()} Value")
            plt.title(f"Time Series - {sensor.upper()} (Sensor {sensor_id})")
            plt.xticks(rotation=45)
            
        else:  # summary plot - simplified
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            # Time series
            ax1.plot(df_filtered["timestamp"], df_filtered["value"], linewidth=0.5)
            ax1.set_title("Time Series")
            ax1.tick_params(axis='x', rotation=45)
            
            # Distribution
            ax2.hist(df_filtered["value"], bins=20, alpha=0.7, color='green')
            ax2.set_title("Distribution")
            
            # Rolling mean with sampling
            if len(df_filtered) > 100:
                sample_df = df_filtered.sample(100, random_state=42).sort_values('timestamp')
                ax3.plot(sample_df["timestamp"], sample_df["value"], alpha=0.7)
            else:
                ax3.plot(df_filtered["timestamp"], df_filtered["value"], alpha=0.7)
            ax3.set_title("Sample Values")
            ax3.tick_params(axis='x', rotation=45)
            
            # Simple stats
            stats_text = f"Mean: {df_filtered['value'].mean():.2f}\nStd: {df_filtered['value'].std():.2f}\nCount: {len(df_filtered)}"
            ax4.text(0.1, 0.5, stats_text, fontsize=12, va='center')
            ax4.set_title("Statistics")
            ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=80, bbox_inches="tight")  # Lower DPI
        plt.close('all')  # Explicitly close all figures
        
        # Clean up memory
        del df, df_filtered, records
        if 'plot_data' in locals():
            del plot_data
        if 'df_resampled' in locals():
            del df_resampled
        gc.collect()
        
        log_memory_usage("After plot generation")
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="image/png")
        
    except Exception as e:
        logger.error(f"Plot generation error: {e}")
        return error_image(f"Error generating plot: {str(e)}")


@app.get("/shap_hour_minimal/{sensor_id}")
def shap_hour_minimal(
    sensor_id: str, 
    sensor: str = Query(...),
    range: str = Query("1month")
):
    """Minimal version of SHAP hourly analysis"""
    try:
        # Fetch and process data
        records = fetch_sensor_history(sensor_id)
        if not records:
            return {"error": "No data", "status": "error"}
        
        df = preprocess_dataframe(records, sensor)
        if df.empty:
            return {"error": "No valid data", "status": "error"}
        
        df_filtered = filter_by_date_range(df, range)
        if df_filtered.empty:
            return {"error": "No data after filtering", "status": "error"}
        
        # Simple aggregation
        df_filtered["hour"] = pd.to_datetime(df_filtered["timestamp"]).dt.hour
        agg = df_filtered.groupby("hour")["value"].mean().reset_index()
        
        # Create enhanced Seaborn plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=agg, x="hour", y="value", ax=ax, marker="o", linewidth=2.5, markersize=8)
        ax.set_xlabel("Hour of Day", fontsize=12)
        ax.set_ylabel(f"{sensor.upper()} Value", fontsize=12)
        ax.set_title(f"Hourly {sensor.upper()} Pattern - Sensor {sensor_id}\nDate Range: {range}", fontsize=14, pad=20)
        ax.grid(True, alpha=0.3)
        
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        
        image_data = base64.b64encode(buf.read()).decode('utf-8')
        
        return {
            "sensor_id": sensor_id,
            "sensor_type": sensor,
            "hourly_data": agg.to_dict("records"),
            "image_data": image_data,
            "status": "success"
        }
        
    except Exception as e:
        return {"error": str(e), "status": "error"}
        
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
    
def preprocess_dataframe_simple(records, sensor):
    """Simplified dataframe preprocessing for performance metrics"""
    try:
        if not records:
            return pd.DataFrame()
        
        # Create DataFrame with only necessary columns
        data = []
        for record in records:
            if sensor in record and 'timestamp' in record:
                try:
                    value = float(record[sensor])
                    timestamp = pd.to_datetime(record['timestamp'])
                    data.append({'timestamp': timestamp, 'value': value})
                except (ValueError, TypeError):
                    continue
        
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        df = df.sort_values('timestamp')
        df = df.dropna()
        
        logger.info(f"Preprocessed {len(df)} records for sensor {sensor}")
        return df
        
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return pd.DataFrame()

def filter_by_date_range_simple(df, date_range):
    """Simplified date filtering for performance metrics"""
    try:
        if df.empty:
            return df
            
        now = pd.Timestamp.now()
        
        date_filters = {
            "1week": now - pd.Timedelta(weeks=1),
            "1month": now - pd.Timedelta(days=30),
            "3months": now - pd.Timedelta(days=90),
            "6months": now - pd.Timedelta(days=180),
            "1year": now - pd.Timedelta(days=365),
            "all": pd.Timestamp.min
        }
        
        cutoff = date_filters.get(date_range, date_filters["1month"])
        
        if cutoff != pd.Timestamp.min:
            filtered_df = df[df['timestamp'] >= cutoff]
            logger.info(f"Filtered from {len(df)} to {len(filtered_df)} records for range {date_range}")
            return filtered_df
            
        return df
        
    except Exception as e:
        logger.error(f"Date filtering error: {e}")
        return df

def create_simple_binary_labels(df):
    """Create simple binary labels for classification"""
    try:
        if df.empty or len(df) < 5:
            return pd.DataFrame()
            
        df_copy = df.copy()
        
        # Use median for binary classification
        median_val = df_copy['value'].median()
        df_copy['class_binary'] = (df_copy['value'] > median_val).astype(int)
        
        # Check if we have both classes
        class_counts = df_copy['class_binary'].value_counts()
        logger.info(f"Class distribution: {dict(class_counts)}")
        
        if len(class_counts) < 2:
            logger.warning(f"Only one class found: {class_counts.index[0]}")
            return pd.DataFrame()
            
        return df_copy
        
    except Exception as e:
        logger.error(f"Binary labels error: {e}")
        return pd.DataFrame()

def create_enhanced_features(df_binary, sensor_col='value'):
    """Create more predictive features for sensor data"""
    if df_binary is None or len(df_binary) < 10:
        return df_binary
    
    df = df_binary.copy()
    
    # Basic lag features
    df['lag_1'] = df[sensor_col].shift(1)
    df['lag_2'] = df[sensor_col].shift(2)
    df['lag_3'] = df[sensor_col].shift(3)
    
    # Enhanced: Rolling statistics
    df['rolling_mean_3'] = df[sensor_col].rolling(window=3).mean()
    df['rolling_std_3'] = df[sensor_col].rolling(window=3).std()
    df['rolling_mean_5'] = df[sensor_col].rolling(window=5).mean()
    df['rolling_std_5'] = df[sensor_col].rolling(window=5).std()
    
    # Enhanced: Rate of change and momentum
    df['momentum_3'] = df[sensor_col] - df[sensor_col].shift(3)
    df['momentum_5'] = df[sensor_col] - df[sensor_col].shift(5)
    
    # Enhanced: Percent changes
    df['pct_change_1'] = df[sensor_col].pct_change(periods=1)
    df['pct_change_3'] = df[sensor_col].pct_change(periods=3)
    
    # Enhanced: Volatility features
    df['volatility_5'] = df[sensor_col].rolling(window=5).std()
    df['volatility_10'] = df[sensor_col].rolling(window=10).std()
    
    # Enhanced: Statistical features
    overall_mean = df[sensor_col].mean()
    overall_std = df[sensor_col].std()
    if overall_std > 0:
        df['z_score'] = (df[sensor_col] - overall_mean) / overall_std
    else:
        df['z_score'] = 0
    
    # Enhanced: Binary features for spikes/drops
    df['is_spike'] = ((df[sensor_col] - df[sensor_col].shift(1)) > (2 * overall_std)).astype(int) if overall_std > 0 else 0
    df['is_drop'] = ((df[sensor_col].shift(1) - df[sensor_col]) > (2 * overall_std)).astype(int) if overall_std > 0 else 0
    
    # Enhanced: Time-based features
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['month'] = df['timestamp'].dt.month
        except:
            pass
    
    # Remove rows with NaN values
    df = df.dropna()
    
    return df
    
@app.get("/performance/{sensor_id}")
def performance_metrics(
    sensor_id: str,
    sensor: str = Query(..., description="Sensor type"),
    test_size: float = Query(0.2, description="Test set size ratio"),
    cv_folds: int = Query(3, description="Cross-validation folds"),
    date_range: str = Query("1month", description="Date range: 1week, 1month, 3months, 6months, 1year, all"),
    use_grid_search: bool = Query(False, description="Enable Grid Search for hyperparameters (slower)"),
    grid_n_estimators: str = Query("50,100", description="Comma-separated values for n_estimators"),
    grid_max_depth: str = Query("3,5", description="Comma-separated values for max_depth"),
    grid_learning_rate: str = Query("0.1,0.05", description="Comma-separated values for learning_rate"),
    grid_subsample: str = Query("0.8,1.0", description="Comma-separated values for subsample"),
    grid_colsample_bytree: str = Query("0.8,1.0", description="Comma-separated values for colsample_bytree")
):
    """Get XGBoost performance metrics with enhanced features and diagnostics"""
    try:
        log_memory_usage("Before performance metrics")
        
        # Limit data size for performance analysis
        max_training_samples = 5000
        
        records = fetch_sensor_history(sensor_id)
        if not records:
            return JSONResponse(
                content={
                    "error": f"No data found for sensor {sensor_id}",
                    "sensor_id": sensor_id,
                    "sensor_type": sensor,
                    "status": "error"
                },
                status_code=404,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, OPTIONS",
                    "Access-Control-Allow-Headers": "*"
                }
            )

        # Use the simple preprocessing
        df = preprocess_dataframe_simple(records, sensor)
        if df is None or df.empty:
            return JSONResponse(
                content={
                    "error": "No valid data after preprocessing",
                    "sensor_id": sensor_id,
                    "sensor_type": sensor,
                    "status": "error"
                },
                status_code=400,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, OPTIONS",
                    "Access-Control-Allow-Headers": "*"
                }
            )

        # Use simple date filtering
        df_filtered = filter_by_date_range_simple(df, date_range)
        if df_filtered is None or len(df_filtered) < 10:
            return JSONResponse(
                content={
                    "error": f"Not enough data after filtering: {0 if df_filtered is None else len(df_filtered)} records",
                    "sensor_id": sensor_id,
                    "sensor_type": sensor,
                    "status": "error"
                },
                status_code=400,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, OPTIONS",
                    "Access-Control-Allow-Headers": "*"
                }
            )

        # Sample data if too large
        if len(df_filtered) > max_training_samples:
            df_filtered = df_filtered.sample(max_training_samples, random_state=42)
            logger.info(f"Sampled {max_training_samples} records for performance analysis")

        # Create binary labels
        df_binary = create_simple_binary_labels(df_filtered)
        if df_binary is None or df_binary.empty:
            return JSONResponse(
                content={
                    "error": "Could not create binary labels (possibly single-class).",
                    "sensor_id": sensor_id,
                    "sensor_type": sensor,
                    "status": "error"
                },
                status_code=400,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, OPTIONS",
                    "Access-Control-Allow-Headers": "*"
                }
            )

        # Enhanced feature creation
        df_features = create_enhanced_features(df_binary, sensor_col='value')
        if df_features is None or df_features.empty:
            return JSONResponse(
                content={
                    "error": "Could not create features (likely because of insufficient rows for feature engineering).",
                    "sensor_id": sensor_id,
                    "sensor_type": sensor,
                    "status": "error"
                },
                status_code=400,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, OPTIONS",
                    "Access-Control-Allow-Headers": "*"
                }
            )

        # Get ALL feature columns (not just lag)
        feature_cols = [col for col in df_features.columns if col not in ['class_binary', 'timestamp', 'value']]
        
        if not feature_cols:
            return JSONResponse(
                content={
                    "error": "No features generated for training",
                    "sensor_id": sensor_id,
                    "sensor_type": sensor,
                    "status": "error"
                },
                status_code=400,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, OPTIONS",
                    "Access-Control-Allow-Headers": "*"
                }
            )

        X = df_features[feature_cols].values
        y = df_features['class_binary'].values

        # Calculate baseline accuracy (majority class)
        if len(y) > 0:
            # FIX: Handle numpy types properly
            unique_classes, class_counts = np.unique(y, return_counts=True)
            baseline_accuracy = float(max(class_counts) / len(y))
        else:
            baseline_accuracy = 0.5

        # Use smaller test size for large datasets
        actual_test_size = min(test_size, 0.2)
        if len(X) < 10:
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=actual_test_size, random_state=42,
                stratify=y if len(np.unique(y)) > 1 else None
            )

        # Decide if grid search should be used
        allow_grid = use_grid_search and len(X_train) <= 2000
        if allow_grid:
            param_grid = {
                "n_estimators": [int(v) for v in grid_n_estimators.split(",")],
                "max_depth": [int(v) for v in grid_max_depth.split(",")],
                "learning_rate": [float(v) for v in grid_learning_rate.split(",")],
                "subsample": [float(v) for v in grid_subsample.split(",")],
                "colsample_bytree": [float(v) for v in grid_colsample_bytree.split(",")],
            }
            base_model = xgb.XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42
            )
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                scoring="accuracy",
                cv=cv_folds,
                n_jobs=1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            # Better default parameters based on data size
            if len(X_train) < 1000:
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=42
                )
            else:
                model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    use_label_encoder=False,
                    eval_metric="logloss", 
                    random_state=42
                )
            model.fit(X_train, y_train)
            best_params = None

        y_pred = model.predict(X_test)

        accuracy = float(accuracy_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        recall = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))

        # Calculate feature importance - FIXED: Handle numpy types
        feature_importance = []
        if hasattr(model, 'feature_importances_'):
            for i, importance in enumerate(model.feature_importances_):
                feature_importance.append({
                    "feature": feature_cols[i] if i < len(feature_cols) else f"feature_{i}",
                    "importance": float(importance)  # Convert numpy float to Python float
                })
            # Sort by importance
            feature_importance.sort(key=lambda x: x['importance'], reverse=True)

        # Get class distribution - FIXED: Handle numpy types properly
        try:
            class_counts = pd.Series(y).value_counts()
            class_distribution = {}
            class_percentages = {}
            
            for class_val, count in class_counts.items():
                class_distribution[str(int(class_val))] = int(count)  # Convert numpy to native types
                class_percentages[str(int(class_val))] = float(count / len(y))
                
        except Exception as e:
            logger.warning(f"Error processing class distribution: {e}")
            class_distribution = {"0": len(y)} if len(y) > 0 else {"0": 0}
            class_percentages = {"0": 1.0} if len(y) > 0 else {"0": 0.0}

        # ✅ CORRECTED: Proper response structure with CORS headers
        response_data = {
            "sensor_id": sensor_id,
            "sensor_type": sensor,
            "date_range": date_range,
            "algorithm": "XGBoost (GridSearch)" if allow_grid else "XGBoost (Enhanced)",
            "best_params": best_params,
            "performance_metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "baseline_accuracy": float(baseline_accuracy),
                "improvement_over_baseline": float(accuracy - baseline_accuracy),
                "test_samples": int(len(y_test))
            },
            "data_info": {
                "total_samples": int(len(df_features)),
                "training_samples": int(len(X_train)),
                "test_samples": int(len(y_test)),
                "class_distribution": class_distribution,
                "class_percentages": class_percentages,
                "features_used": feature_cols,
                "feature_count": len(feature_cols)
            },
            "feature_importance": feature_importance[:10],
            "diagnostics": {
                "baseline_interpretation": "Accuracy if always predicting majority class",
                "model_interpretation": "Positive improvement means model is learning patterns",
                "feature_quality": "High importance > 0.1 suggests good features"
            },
            "status": "success"
        }

        # Clean up memory
        del records, df, df_filtered, df_binary, df_features, model
        del X, y, X_train, X_test, y_train, y_test, y_pred
        gc.collect()
        
        log_memory_usage("After performance metrics")
        
        # Return with explicit CORS headers
        return JSONResponse(
            content=response_data,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "*"
            }
        )

    except Exception as e:
        logger.exception("PERFORMANCE ENDPOINT CRASH")
        return JSONResponse(
            content={
                "error": f"Internal server error: {str(e)}",
                "sensor_id": sensor_id,
                "sensor_type": sensor,
                "status": "error"
            },
            status_code=500,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "*"
            }
        )    

def create_simple_lag_features(df, lags=2):
    """Create simple lag features"""
    try:
        if df.empty:
            return pd.DataFrame()
            
        df_copy = df.copy()
        
        # Create lag features
        for i in range(1, lags + 1):
            df_copy[f'lag{i}'] = df_copy['value'].shift(i)
        
        # Drop rows with NaN
        df_copy = df_copy.dropna()
        
        return df_copy
        
    except Exception as e:
        logger.error(f"Lag features error: {e}")
        return pd.DataFrame()


def get_accuracy_assessment(accuracy):
    """Get accuracy assessment"""
    if accuracy >= 0.9:
        level = "Excellent"
        confidence = "Very High"
    elif accuracy >= 0.8:
        level = "Very Good" 
        confidence = "High"
    elif accuracy >= 0.7:
        level = "Good"
        confidence = "Medium"
    elif accuracy >= 0.6:
        level = "Fair"
        confidence = "Low-Medium"
    else:
        level = "Needs Improvement"
        confidence = "Low"
    
    return {
        "accuracy_level": level,
        "confidence": confidence,
        "interpretation": f"XGBoost shows {level.lower()} performance with {confidence.lower()} confidence"
    }

# ---------------------------------------------------
# Correlation Plots (Single or Multiple Sensors)
# ---------------------------------------------------

@app.get("/correlation")
def correlation(
    sensor_ids: str = Query(..., description="Comma-separated sensor IDs"),
    plot_type: str = Query("heatmap", enum=["heatmap", "scatter", "pairplot"]),
    output: str = Query("img", enum=["img", "json"]),
    limit: int = Query(500, description="Maximum number of records"),  # Reduced default limit
):
    """Correlation analysis with memory optimization"""
    try:
        log_memory_usage("Before correlation analysis")
        
        sensor_list = [s.strip() for s in sensor_ids.split(",") if s.strip()]
        if not sensor_list:
            return JSONResponse({"error": "No sensor IDs provided"}, status_code=400)

        all_data = []
        for sid in sensor_list:
            records = fetch_sensor_history(sid)
            if records:
                df = pd.DataFrame(records)
                df["sensor_id"] = sid
                all_data.append(df)

        if not all_data:
            return JSONResponse({"error": "No data found for given sensors"}, status_code=404)

        df = pd.concat(all_data, ignore_index=True)

        # Keep only numeric fields
        numeric_fields = ["methane", "co2", "ammonia", "humidity", "temperature", "riskIndex"]
        df = df[[col for col in numeric_fields if col in df.columns]].dropna()

        # Aggressive downsampling
        max_correlation_samples = 1000
        if len(df) > max_correlation_samples:
            df = df.sample(max_correlation_samples, random_state=42)
            logger.info(f"Sampled {max_correlation_samples} records for correlation analysis")

        if output == "json":
            corr = df.corr().to_dict()
            
            # Clean up
            del all_data, df
            gc.collect()
            log_memory_usage("After correlation JSON")
            
            return {
                "sensors": sensor_list,
                "correlation": corr
            }

        # IMG output with memory optimization
        plt.figure(figsize=(8, 6))  # Smaller figure
        
        if plot_type == "heatmap":
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", 
                       cbar_kws={"shrink": .8})
            plt.title(f"Correlation Heatmap", fontsize=14)
            
        elif plot_type == "pairplot":
            # Use only 3 columns for pairplot to reduce memory
            if len(df.columns) > 3:
                df_small = df[df.columns[:3]]
            else:
                df_small = df
            sns.pairplot(df_small, diag_kind='hist', plot_kws={'alpha': 0.6})
            plt.suptitle(f"Pair Plot", y=1.02)
            
        else:  # scatter
            if len(df.columns) >= 2:
                plt.scatter(df.iloc[:, 0], df.iloc[:, 1], alpha=0.6)
                plt.xlabel(df.columns[0])
                plt.ylabel(df.columns[1])
                plt.title(f"Scatter Plot: {df.columns[0]} vs {df.columns[1]}")

        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=80, bbox_inches="tight")  # Lower DPI
        plt.close('all')
        buf.seek(0)
        
        # Clean up memory
        del all_data, df
        if 'df_small' in locals():
            del df_small
        gc.collect()
        
        log_memory_usage("After correlation image")
        
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        logger.error(f"Correlation analysis error: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
# ===========
#      confusion matrix
#==============
@app.get("/confusion_matrix/{sensor_id}")
def enhanced_confusion_matrix(
    sensor_id: str,
    sensor: str = Query(..., description="Sensor type (co2, temperature, etc.)"),
    output_format: str = Query("image", enum=["image", "json", "both"]),
    date_range: str = Query("1month", description="Date range: 1week, 1month, 3months, 6months, 1year, all"),
    color_scheme: str = Query("viridis", enum=["viridis", "plasma", "magma", "inferno", "cividis"]),
    show_predictions: bool = Query(True, description="Show next day prediction probabilities"),
    test_size: float = Query(0.2, description="Test set size ratio")
):
    """
    Enhanced Confusion Matrix with Next-Day Prediction Probabilities
    - Modern color schemes for professional visualization
    - Precision, Recall, F1-score metrics
    - Next-day prediction confidence scores
    - Multiple output formats (Image, JSON, Both)
    """
    try:
        # Fetch and preprocess data
        records = fetch_sensor_history(sensor_id)
        if not records:
            return JSONResponse(
                content={"error": f"No data found for sensor {sensor_id}"},
                status_code=404
            )

        df = preprocess_dataframe(records, sensor)
        if df.empty or len(df) < 10:
            return JSONResponse(
                content={"error": "Insufficient data for analysis"},
                status_code=400
            )

        # Filter by date range
        df_filtered = filter_by_date_range(df, date_range)
        if len(df_filtered) < 5:
            return JSONResponse(
                content={"error": f"Not enough data after {date_range} filter"},
                status_code=400
            )

        # Create binary classification labels (high/low based on median)
        df_binary = create_simple_binary_labels(df_filtered)
        if df_binary.empty:
            return JSONResponse(
                content={"error": "Could not create classification labels"},
                status_code=400
            )

        # Create enhanced features for better prediction
        df_features = create_enhanced_features(df_binary, sensor_col='value')
        if df_features.empty:
            return JSONResponse(
                content={"error": "Feature engineering failed"},
                status_code=400
            )

        # Prepare features and target
        feature_cols = [col for col in df_features.columns if col not in ['class_binary', 'timestamp', 'value']]
        X = df_features[feature_cols].values
        y = df_features['class_binary'].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Train XGBoost model with probability calibration
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        )
        model.fit(X_train, y_train)

        # Get predictions and probabilities
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Calculate comprehensive metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_names = ['Low', 'High']  # Based on binary classification

        # Calculate next-day prediction probabilities
        next_day_prediction = None
        if show_predictions and len(df_features) > 0:
            # Use the most recent data point for next-day prediction
            latest_features = df_features[feature_cols].iloc[-1:].values
            next_day_proba = model.predict_proba(latest_features)[0]
            next_day_pred = model.predict(latest_features)[0]
            
            next_day_prediction = {
                "predicted_class": "High" if next_day_pred == 1 else "Low",
                "confidence_scores": {
                    "low_probability": float(next_day_proba[0]),
                    "high_probability": float(next_day_proba[1])
                },
                "prediction_confidence": float(max(next_day_proba)),
                "prediction_date": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            }

        # Class-wise metrics
        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Prepare response data
        response_data = {
            "sensor_id": sensor_id,
            "sensor_type": sensor,
            "date_range": date_range,
            "model_performance": {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "test_samples": len(y_test),
                "training_samples": len(y_train)
            },
            "confusion_matrix": {
                "matrix": conf_matrix.tolist(),
                "classes": class_names,
                "normalized_matrix": (conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]).tolist()
            },
            "class_metrics": class_report,
            "next_day_prediction": next_day_prediction,
            "feature_importance": [
                {"feature": feature_cols[i], "importance": float(imp)}
                for i, imp in enumerate(model.feature_importances_)
            ][:5]  # Top 5 features
        }

        # Return based on output format
        if output_format == "json":
            return JSONResponse(content=response_data)

        elif output_format == "image":
            return generate_confusion_matrix_image(
                conf_matrix, class_names, response_data, sensor, color_scheme
            )

        else:  # "both"
            image_response = generate_confusion_matrix_image(
                conf_matrix, class_names, response_data, sensor, color_scheme
            )
            
            # For both format, we need to handle this differently
            # Since we can't return two responses, we'll return JSON with image as base64
            buf = io.BytesIO()
            plt.figure(figsize=(14, 10))
            create_enhanced_confusion_matrix_plot(
                conf_matrix, class_names, response_data, sensor, color_scheme
            )
            plt.tight_layout()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            plt.close()
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            
            response_data["confusion_matrix_image"] = f"data:image/png;base64,{image_base64}"
            return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Enhanced confusion matrix error: {e}")
        return JSONResponse(
            content={"error": f"Analysis failed: {str(e)}"},
            status_code=500
        )


def generate_confusion_matrix_image(conf_matrix, class_names, response_data, sensor, color_scheme):
    """Generate enhanced confusion matrix visualization"""
    try:
        buf = io.BytesIO()
        plt.figure(figsize=(14, 10))
        
        # Create the main plot
        create_enhanced_confusion_matrix_plot(conf_matrix, class_names, response_data, sensor, color_scheme)
        
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close()
        buf.seek(0)
        
        return StreamingResponse(buf, media_type="image/png")
        
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        return error_image(f"Error generating visualization: {str(e)}")


def create_enhanced_confusion_matrix_plot(conf_matrix, class_names, response_data, sensor, color_scheme):
    """Create modern confusion matrix visualization with metrics"""
    
    # Create subplot grid
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1])
    
    # Main confusion matrix
    ax1 = plt.subplot(gs[0, 0])
    
    # Use modern color scheme
    cmap = plt.get_cmap(color_scheme)
    
    # Plot confusion matrix
    im = ax1.imshow(conf_matrix, interpolation='nearest', cmap=cmap, alpha=0.8)
    
    # Add text annotations
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax1.text(j, i, f"{conf_matrix[i, j]}\n({conf_matrix[i, j]/conf_matrix.sum():.1%})",
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black",
                    fontsize=14, fontweight='bold')
    
    # Customize the plot
    ax1.set_xticks(np.arange(len(class_names)))
    ax1.set_yticks(np.arange(len(class_names)))
    ax1.set_xticklabels([f'Predicted\n{name}' for name in class_names], fontsize=12)
    ax1.set_yticklabels([f'Actual\n{name}' for name in class_names], fontsize=12)
    ax1.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    
    # Add colorbar
    plt.colorbar(im, ax=ax1, shrink=0.8)
    
    # Metrics panel
    ax2 = plt.subplot(gs[0, 1])
    ax2.axis('off')
    
    # Display metrics
    metrics = response_data["model_performance"]
    next_pred = response_data.get("next_day_prediction")
    
    metrics_text = [
        "MODEL PERFORMANCE METRICS",
        "=" * 25,
        f"Accuracy:  {metrics['accuracy']:.1%}",
        f"Precision: {metrics['precision']:.1%}",
        f"Recall:    {metrics['recall']:.1%}",
        f"F1-Score:  {metrics['f1_score']:.1%}",
        "",
        f"Test Samples: {metrics['test_samples']}",
        f"Train Samples: {metrics['training_samples']}",
        "",
        f"Sensor: {sensor.upper()}",
        f"Date Range: {response_data['date_range']}"
    ]
    
    if next_pred:
        metrics_text.extend([
            "",
            "NEXT DAY PREDICTION",
            "=" * 20,
            f"Class: {next_pred['predicted_class']}",
            f"Confidence: {next_pred['prediction_confidence']:.1%}",
            f"Date: {next_pred['prediction_date']}",
            "",
            "Probability Breakdown:",
            f"  High: {next_pred['confidence_scores']['high_probability']:.1%}",
            f"  Low:  {next_pred['confidence_scores']['low_probability']:.1%}"
        ])
    
    ax2.text(0.1, 0.95, "\n".join(metrics_text), transform=ax2.transAxes,
            fontsize=11, fontfamily='monospace', va='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.7))
    
    # Feature importance (bottom left)
    ax3 = plt.subplot(gs[1, 0])
    features = response_data["feature_importance"]
    feature_names = [f['feature'] for f in features]
    importance_scores = [f['importance'] for f in features]
    
    y_pos = np.arange(len(feature_names))
    ax3.barh(y_pos, importance_scores, align='center', color=cmap(0.6), alpha=0.8)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(feature_names)
    ax3.invert_yaxis()
    ax3.set_xlabel('Feature Importance', fontweight='bold')
    ax3.set_title('Top Predictive Features', fontweight='bold')
    ax3.grid(True, axis='x', alpha=0.3)
    
    # Class distribution (bottom right)
    ax4 = plt.subplot(gs[1, 1])
    class_totals = conf_matrix.sum(axis=1)
    colors = [cmap(0.3), cmap(0.7)]
    wedges, texts, autotexts = ax4.pie(class_totals, labels=class_names, autopct='%1.1f%%',
                                      colors=colors, startangle=90)
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax4.set_title('Class Distribution', fontweight='bold')
    
    # Main title
    plt.suptitle(
        f'Enhanced Confusion Matrix - {sensor.upper()} Sensor\n'
        f'XGBoost Classification with Next-Day Predictions',
        fontsize=16, fontweight='bold', y=0.98
    )


# Helper function to create enhanced features for better prediction
def create_enhanced_prediction_features(df, sensor_col='value'):
    """Create features specifically optimized for next-day prediction"""
    if df is None or len(df) < 10:
        return df
    
    df_enhanced = df.copy()
    
    # Time-based features
    if 'timestamp' in df_enhanced.columns:
        try:
            df_enhanced['timestamp'] = pd.to_datetime(df_enhanced['timestamp'])
            df_enhanced['hour'] = df_enhanced['timestamp'].dt.hour
            df_enhanced['day_of_week'] = df_enhanced['timestamp'].dt.dayofweek
            df_enhanced['is_weekend'] = df_enhanced['day_of_week'].isin([5, 6]).astype(int)
            df_enhanced['day_of_month'] = df_enhanced['timestamp'].dt.day
            df_enhanced['month'] = df_enhanced['timestamp'].dt.month
        except:
            pass
    
    # Enhanced lag features with different windows
    for lag in [1, 2, 3, 5, 7]:  # Multiple time windows
        df_enhanced[f'lag_{lag}'] = df_enhanced[sensor_col].shift(lag)
    
    # Rolling statistics with multiple windows
    for window in [3, 5, 7]:
        df_enhanced[f'rolling_mean_{window}'] = df_enhanced[sensor_col].rolling(window=window).mean()
        df_enhanced[f'rolling_std_{window}'] = df_enhanced[sensor_col].rolling(window=window).std()
        df_enhanced[f'rolling_min_{window}'] = df_enhanced[sensor_col].rolling(window=window).min()
        df_enhanced[f'rolling_max_{window}'] = df_enhanced[sensor_col].rolling(window=window).max()
    
    # Rate of change features
    df_enhanced['momentum_1'] = df_enhanced[sensor_col].diff(1)
    df_enhanced['momentum_3'] = df_enhanced[sensor_col].diff(3)
    df_enhanced['acceleration'] = df_enhanced['momentum_1'].diff(1)
    
    # Volatility and trend features
    df_enhanced['volatility_5'] = df_enhanced[sensor_col].rolling(window=5).std()
    df_enhanced['trend_5'] = df_enhanced[sensor_col].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True
    )
    
    # Remove rows with NaN values
    df_enhanced = df_enhanced.dropna()
    
    return df_enhanced
    
# Run the application
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)