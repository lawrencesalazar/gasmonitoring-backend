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
import matplotlib.gridspec as gridspec   


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

def standard_error_response(message: str, status_code: int = 500, details: Dict = None):
    """Return standardized error response"""
    response = {
        "status": "error",
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    if details:
        response["details"] = details
    return JSONResponse(content=response, status_code=status_code)
    
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

def fetch_and_preprocess_data(sensor_id: str, sensor: str, date_range: str = "1month") -> pd.DataFrame:
    """Common data fetching and preprocessing workflow"""
    records = fetch_sensor_history(sensor_id)
    df = preprocess_dataframe(records, sensor)
    if not df.empty:
        df = filter_by_date_range(df, date_range)
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

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {str(key): convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def create_simple_binary_labels(df):
    """Create simple binary labels for classification with robust error handling"""
    try:
        if df is None or df.empty or len(df) < 5:
            logger.warning("Cannot create binary labels: insufficient data")
            return pd.DataFrame()
            
        df_copy = df.copy()
        
        # Check if 'value' column exists
        if 'value' not in df_copy.columns:
            logger.error("Missing 'value' column for binary classification")
            return pd.DataFrame()
        
        # Use median for binary classification
        median_val = df_copy['value'].median()
        logger.info(f"Creating binary labels using median: {median_val}")
        
        df_copy['class_binary'] = (df_copy['value'] > median_val).astype(int)
        
        # Check if we have both classes
        class_counts = df_copy['class_binary'].value_counts()
        logger.info(f"Binary class distribution: {dict(class_counts)}")
        
        if len(class_counts) < 2:
            logger.warning(f"Only one class found: {class_counts.index[0]}")
            return pd.DataFrame()
            
        return df_copy
        
    except Exception as e:
        logger.error(f"Error creating binary labels: {e}")
        return pd.DataFrame()

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

def create_safe_features(df_binary, sensor_col='value'):
    """Create features without data leakage"""
    if df_binary is None or len(df_binary) < 10:
        return pd.DataFrame()
    
    df = df_binary.copy()
    
    # Safe lag features (only past data)
    df['lag_1'] = df[sensor_col].shift(1)
    df['lag_2'] = df[sensor_col].shift(2)
    df['lag_3'] = df[sensor_col].shift(3)
    
    # Safe rolling statistics (using only past data)
    df['rolling_mean_3'] = df[sensor_col].shift(1).rolling(window=3).mean()
    df['rolling_std_3'] = df[sensor_col].shift(1).rolling(window=3).std()
    
    # Remove rows with NaN values (from shifting)
    df = df.dropna()
    
    # Remove any features that might leak future information
    safe_cols = [col for col in df.columns if not any(x in str(col) for x in ['future', 'target', 'class'])]
    df = df[safe_cols]
    
    return df

def check_data_leakage(df, feature_cols, target_col):
    """Check for potential data leakage"""
    try:
        # Check if target is in features
        if target_col in feature_cols:
            return True
        
        # Check for perfect correlation with target
        for col in feature_cols:
            if col != target_col:
                correlation = abs(df[col].corr(df[target_col]))
                if correlation > 0.95:  # Nearly perfect correlation
                    logger.warning(f"High correlation between {col} and target: {correlation}")
                    return True
        
        return False
    except:
        return False

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_type='xgb'):
    """Standardized model training and evaluation"""
    try:
        if model_type == 'xgb':
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42
            )
        elif model_type == 'rf':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42
            )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        return {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred
        }
    except Exception as e:
        logger.error(f"Model training error: {e}")
        raise

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
    Receive sensor data from ESP32 devices and store in Firebase with timestamp validation
    """
    try:
        data = await request.json()
        
        # Validate required fields
        required_fields = ['sensorID', 'timestamp', 'co2', 'methane', 'ammonia', 'temperature', 'humidity']
        for field in required_fields:
            if field not in data:
                return standard_error_response(f"Missing required field: {field}", 400)
        
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
        return standard_error_response(f"Internal server error: {str(e)}", 500)
        
@app.get("/api/sensor/current/{sensor_id}")
async def get_current_sensor_data(sensor_id: str):
    """
    Get current sensor reading for a specific sensor
    """
    try:
        current_reading = get_current_sensor_reading(sensor_id)
        
        if not current_reading:
            return standard_error_response(f"No current reading found for sensor {sensor_id}", 404)
        
        return {
            "status": "success",
            "sensorID": sensor_id,
            "current_reading": current_reading,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching current sensor data: {e}")
        return standard_error_response(f"Internal server error: {str(e)}", 500)

@app.get("/api/sensor/{sensor_id}/stats")
async def get_sensor_stats(sensor_id: str):
    """
    Get statistics for a specific sensor
    """
    try:
        records = fetch_sensor_history(sensor_id)
        
        if not records:
            return standard_error_response(f"No data found for sensor {sensor_id}", 404)
        
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
        return standard_error_response(f"Internal server error: {str(e)}", 500)

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
        
        df = fetch_and_preprocess_data(sensor_id, sensor, range)
        
        if df.empty or len(df) < 10:
            return standard_error_response("Not enough data for SHAP analysis", 400)
        
        # Limit data size for SHAP analysis
        max_shap_samples = 1000
        if len(df) > max_shap_samples:
            df = df.sample(max_shap_samples, random_state=42)
            logger.info(f"Sampled {max_shap_samples} records for SHAP analysis")
        
        df["hour"] = df["timestamp"].dt.hour
        agg = df.groupby("hour")["value"].mean().reset_index()
        
        if len(agg) < 3:
            return standard_error_response("Insufficient hourly data after filtering", 400)
        
        X = agg[["hour"]]
        y = agg["value"]
        
        # Use smaller model for SHAP
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=50, random_state=42)
        model.fit(X, y)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        buf = io.BytesIO()
        plt.figure(figsize=(8, 5))
        shap.summary_plot(shap_values, X, show=False)
        plt.title(f"SHAP Summary - {sensor} (Sensor {sensor_id})\nDate Range: {range}")
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close('all')
        buf.seek(0)
        
        # Force garbage collection
        del model, explainer, shap_values, df, agg
        gc.collect()
        
        log_memory_usage("After SHAP explanation")
        
        return {
            "sensor_id": sensor_id,
            "sensor_type": sensor,
            "date_range": range,
            "shap_values": shap_values.tolist(),
            "features": X.to_dict(orient="records"),
            "feature_importance": {
                "hour": float(np.abs(shap_values).mean(axis=0)[0])
            },
            "summary_stats": {
                "total_samples": len(df),
                "shap_samples": len(agg),
                "mean_shap_magnitude": float(np.mean(np.abs(shap_values)))
            }
        }
        
    except Exception as e:
        logger.error(f"SHAP explanation error: {e}")
        return standard_error_response(f"SHAP analysis failed: {str(e)}", 500)

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
        
        df = fetch_and_preprocess_data(sensor_id, sensor, range)
        
        if df.empty or len(df) < 5:
            return error_image(f"Not enough data for plot. Found {len(df)} records.")
        
        # Limit data size for plotting
        max_plot_samples = 2000
        if len(df) > max_plot_samples:
            df = df.sample(max_plot_samples, random_state=42)
            logger.info(f"Sampled {max_plot_samples} records for plotting")
        
        buf = io.BytesIO()
        
        if chart == "distribution":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            sns.histplot(data=df, x="value", kde=True, ax=ax1, color='skyblue', alpha=0.7)
            ax1.set_xlabel(f"{sensor.upper()} Value")
            ax1.set_ylabel("Frequency")
            ax1.set_title(f"Distribution - {sensor.upper()}")
            
            sns.boxplot(data=df, y="value", ax=ax2, color='lightcoral')
            ax2.set_ylabel(f"{sensor.upper()} Value")
            ax2.set_title(f"Box Plot - {sensor.upper()}")
            
        elif chart == "scatter":
            plt.figure(figsize=(10, 6))
            
            # Sample data for scatter plot
            if len(df) > 500:
                plot_data = df.sample(500, random_state=42)
            else:
                plot_data = df
                
            plt.scatter(plot_data["timestamp"], plot_data["value"], alpha=0.6, s=10)
            plt.xlabel("Timestamp")
            plt.ylabel(f"{sensor.upper()} Value")
            plt.title(f"Scatter Plot - {sensor.upper()} (Sensor {sensor_id})")
            plt.xticks(rotation=45)
            
        elif chart == "line":
            plt.figure(figsize=(10, 6))
            
            # Resample for line plot to reduce points
            if len(df) > 1000:
                df_resampled = df.set_index('timestamp').resample('12H').mean().reset_index()
            else:
                df_resampled = df
                
            plt.plot(df_resampled["timestamp"], df_resampled["value"], linewidth=1)
            plt.xlabel("Timestamp")
            plt.ylabel(f"{sensor.upper()} Value")
            plt.title(f"Time Series - {sensor.upper()} (Sensor {sensor_id})")
            plt.xticks(rotation=45)
            
        else:  # summary plot - simplified
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            # Time series
            ax1.plot(df["timestamp"], df["value"], linewidth=0.5)
            ax1.set_title("Time Series")
            ax1.tick_params(axis='x', rotation=45)
            
            # Distribution
            ax2.hist(df["value"], bins=20, alpha=0.7, color='green')
            ax2.set_title("Distribution")
            
            # Rolling mean with sampling
            if len(df) > 100:
                sample_df = df.sample(100, random_state=42).sort_values('timestamp')
                ax3.plot(sample_df["timestamp"], sample_df["value"], alpha=0.7)
            else:
                ax3.plot(df["timestamp"], df["value"], alpha=0.7)
            ax3.set_title("Sample Values")
            ax3.tick_params(axis='x', rotation=45)
            
            # Simple stats
            stats_text = f"Mean: {df['value'].mean():.2f}\nStd: {df['value'].std():.2f}\nCount: {len(df)}"
            ax4.text(0.1, 0.5, stats_text, fontsize=12, va='center')
            ax4.set_title("Statistics")
            ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=80, bbox_inches="tight")
        plt.close('all')
        
        # Clean up memory
        del df
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
        
@app.get("/shap_hourly_analysis/{sensor_id}")
def shap_hourly_analysis(
    sensor_id: str,
    sensor: str = Query(..., description="Sensor type (co2, methane, temperature, humidity, ammonia)"),
    date_range: str = Query("1month", description="Date range: 1week, 1month, 3months, 6months, 1year, all"),
    confidence_level: float = Query(0.95, description="Confidence level for intervals (0.90, 0.95, 0.99)"),
    anomaly_threshold: float = Query(2.0, description="Z-score threshold for anomaly detection")
):
    """
    Enhanced SHAP-based hourly analysis with statistical significance testing,
    anomaly detection, and actionable recommendations for gas monitoring.
    """
    try:
        logger.info(f"Starting enhanced SHAP analysis for {sensor_id}, sensor: {sensor}")
        
        # Data collection & preprocessing
        df = fetch_and_preprocess_data(sensor_id, sensor, date_range)
        if df.empty:
            return standard_error_response("No valid sensor data", 400)
        
        # Feature engineering for temporal analysis
        df_enhanced = df.copy()
        df_enhanced['timestamp'] = pd.to_datetime(df_enhanced['timestamp'])
        df_enhanced['hour'] = df_enhanced['timestamp'].dt.hour
        df_enhanced['day_of_week'] = df_enhanced['timestamp'].dt.dayofweek
        df_enhanced['is_weekend'] = df_enhanced['day_of_week'].isin([5, 6]).astype(int)
        df_enhanced['day_of_month'] = df_enhanced['timestamp'].dt.day
        df_enhanced['month'] = df_enhanced['timestamp'].dt.month
        
        # Rolling statistics
        df_enhanced['rolling_mean_6h'] = df_enhanced['value'].rolling(window=6, min_periods=1).mean()
        df_enhanced['rolling_std_6h'] = df_enhanced['value'].rolling(window=6, min_periods=1).std()
        df_enhanced['prev_hour_value'] = df_enhanced['value'].shift(1)
        
        # Remove rows with NaN values
        df_enhanced = df_enhanced.dropna()
        
        if df_enhanced.empty:
            return standard_error_response("Insufficient data for analysis after feature engineering", 400)
        
        # Statistical significance of hourly patterns
        hourly_stats = []
        hourly_groups = df_enhanced.groupby('hour')['value']
        
        for hour in range(24):
            hour_data = hourly_groups.get_group(hour) if hour in hourly_groups.groups else []
            if len(hour_data) > 1:
                # Basic statistics
                mean_val = np.mean(hour_data)
                std_val = np.std(hour_data)
                n = len(hour_data)
                
                # Confidence interval
                from scipy import stats
                t_critical = stats.t.ppf((1 + confidence_level) / 2, n-1)
                margin_error = t_critical * (std_val / np.sqrt(n))
                ci_lower = mean_val - margin_error
                ci_upper = mean_val + margin_error
                
                # Compare with overall mean (t-test)
                overall_mean = df_enhanced['value'].mean()
                t_stat, p_value = stats.ttest_1samp(hour_data, overall_mean)
                
                # Effect size (Cohen's d)
                cohen_d = (mean_val - overall_mean) / (df_enhanced['value'].std() + 1e-8)
                
                hourly_stats.append({
                    'hour': hour,
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'sample_size': n,
                    'confidence_interval': [float(ci_lower), float(ci_upper)],
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'cohen_d': float(cohen_d),
                    'significant': p_value < 0.05,
                    'effect_size': 'large' if abs(cohen_d) > 0.8 else 'medium' if abs(cohen_d) > 0.5 else 'small'
                })
            else:
                hourly_stats.append({
                    'hour': hour,
                    'mean': None,
                    'sample_size': len(hour_data),
                    'significant': False,
                    'error': 'Insufficient data'
                })
        
        # Anomaly detection in hourly behavior
        anomalies = []
        hourly_anomaly_analysis = []
        
        # Z-score based anomaly detection per hour
        for hour in range(24):
            if hour in hourly_groups.groups:
                hour_data = hourly_groups.get_group(hour)
                if len(hour_data) > 5:
                    z_scores = np.abs(stats.zscore(hour_data))
                    hour_anomalies = hour_data[z_scores > anomaly_threshold]
                    
                    for idx, value in hour_anomalies.items():
                        original_idx = df_enhanced.index[df_enhanced['hour'] == hour][list(hour_anomalies.index).index(idx)]
                        timestamp = df_enhanced.loc[original_idx, 'timestamp']
                        
                        anomalies.append({
                            'hour': hour,
                            'timestamp': timestamp.isoformat(),
                            'value': float(value),
                            'z_score': float(z_scores[hour_anomalies.index.get_loc(idx)]),
                            'deviation': f"{(value - hour_data.mean()) / hour_data.std():.1f}σ"
                        })
                    
                    hourly_anomaly_analysis.append({
                        'hour': hour,
                        'total_readings': len(hour_data),
                        'anomalies_detected': len(hour_anomalies),
                        'anomaly_rate': len(hour_anomalies) / len(hour_data),
                        'mean_value': float(hour_data.mean()),
                        'std_value': float(hour_data.std())
                    })
        
        # SHAP analysis for predictive insights
        feature_columns = ['hour', 'day_of_week', 'is_weekend', 'rolling_mean_6h', 'rolling_std_6h', 'prev_hour_value']
        X = df_enhanced[feature_columns]
        y = df_enhanced['value']
        
        # Train model for SHAP analysis
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X, y)
        
        # Compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Feature importance from SHAP
        feature_importance = []
        for i, feature in enumerate(feature_columns):
            feature_importance.append({
                'feature': feature,
                'importance': float(np.mean(np.abs(shap_values[:, i]))),
                'direction': 'positive' if np.mean(shap_values[:, i]) > 0 else 'negative'
            })
        
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        # Hourly impact from SHAP
        hourly_impact = []
        for hour in range(24):
            hour_mask = X['hour'] == hour
            if hour_mask.sum() > 0:
                hour_shap = shap_values[hour_mask, 0]
                hourly_impact.append({
                    'hour': hour,
                    'mean_impact': float(np.mean(hour_shap)),
                    'impact_std': float(np.std(hour_shap)),
                    'impact_samples': int(hour_mask.sum())
                })
        
        # Actionable recommendations
        recommendations = generate_shap_based_recommendations(
            hourly_stats, feature_importance, anomalies, sensor
        )
        
        # Visual explanations
        visualization_data = create_three_section_visualizations(
            df_enhanced, hourly_stats, shap_values, X, feature_columns, 
            anomalies, sensor, sensor_id, date_range
        )
        
        # Final response
        response = {
            "sensor_id": sensor_id,
            "sensor_type": sensor,
            "date_range": date_range,
            "analysis_timestamp": datetime.now().isoformat(),
            
            "statistical_significance": {
                "hourly_patterns": hourly_stats,
                "confidence_level": confidence_level,
                "significant_hours": [h for h in hourly_stats if h.get('significant', False)],
                "overall_pattern_strength": len([h for h in hourly_stats if h.get('significant', False)]) / 24
            },
            
            "anomaly_detection": {
                "anomalies_found": len(anomalies),
                "anomaly_threshold": f"{anomaly_threshold}σ",
                "hourly_anomaly_analysis": hourly_anomaly_analysis,
                "anomaly_details": anomalies[:10]
            },
            
            "predictive_insights": {
                "feature_importance": feature_importance,
                "hourly_impact_analysis": hourly_impact,
                "model_performance": {
                    "r_squared": float(model.score(X, y)),
                    "features_used": feature_columns,
                    "samples_analyzed": len(X)
                }
            },
            
            "actionable_recommendations": recommendations,
            
            "visual_explanations": visualization_data,
            
            "status": "success"
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Enhanced SHAP analysis failed: {str(e)}", exc_info=True)
        return standard_error_response(f"Analysis failed: {str(e)}", 500)


def generate_shap_based_recommendations(hourly_stats, feature_importance, anomalies, sensor_type):
    """Generate actionable recommendations based on SHAP and statistical analysis"""
    recommendations = []
    
    # Find peak hours with statistical significance
    significant_hours = [h for h in hourly_stats if h.get('significant', False) and h.get('mean') is not None]
    if significant_hours:
        peak_hours = sorted(significant_hours, key=lambda x: x['mean'], reverse=True)[:3]
        
        recommendations.append({
            "type": "peak_hours",
            "priority": "high",
            "message": f"Statistically significant peaks detected at hours: {[h['hour'] for h in peak_hours]}",
            "action": f"Increase monitoring and ventilation during hours {[h['hour'] for h in peak_hours]}"
        })
    
    # Feature-based recommendations
    top_features = feature_importance[:3]
    if top_features:
        feature_msg = ", ".join([f"{f['feature']} ({f['importance']:.3f})" for f in top_features])
        recommendations.append({
            "type": "key_drivers",
            "priority": "medium",
            "message": f"Primary factors affecting {sensor_type}: {feature_msg}",
            "action": "Focus on controlling these key influencing factors"
        })
    
    # Anomaly-based recommendations
    if anomalies:
        recommendations.append({
            "type": "anomaly_alert",
            "priority": "high" if len(anomalies) > 5 else "medium",
            "message": f"Found {len(anomalies)} anomalous readings requiring investigation",
            "action": "Review sensor calibration and investigate environmental conditions during anomaly periods"
        })
    
    # Safety thresholds based on sensor type
    safety_info = {
        "co2": {"warning": 800, "danger": 1000},
        "methane": {"warning": 500, "danger": 1000},
        "ammonia": {"warning": 25, "danger": 50},
        "temperature": {"warning_low": 15, "warning_high": 30, "danger_low": 10, "danger_high": 35},
        "humidity": {"warning_low": 30, "warning_high": 60, "danger_low": 20, "danger_high": 70}
    }
    
    if sensor_type in safety_info:
        recommendations.append({
            "type": "safety_benchmark",
            "priority": "info",
            "message": f"Current analysis relative to {sensor_type.upper()} safety thresholds",
            "action": "Compare hourly patterns with established safety limits"
        })
    
    return recommendations


def create_three_section_visualizations(df, hourly_stats, shap_values, X, feature_columns, anomalies, sensor, sensor_id, date_range):
    """Create three distinct visualization sections"""
    import base64
    from io import BytesIO
    
    visualizations = {}
    
    # SECTION 1: SHAP Summary Plot
    try:
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        shap.summary_plot(shap_values, X, feature_names=feature_columns, show=False)
        plt.title(f"SHAP Feature Importance - {sensor.upper()}", fontsize=14, fontweight='bold')
        
        plt.subplot(2, 2, 2)
        # SHAP dependence plot for hour feature
        hour_idx = feature_columns.index('hour')
        shap.dependence_plot(hour_idx, shap_values, X, feature_names=feature_columns, show=False)
        plt.title("SHAP Hourly Dependence", fontweight='bold')
        
        buf1 = BytesIO()
        plt.tight_layout()
        plt.savefig(buf1, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf1.seek(0)
        visualizations['shap_analysis'] = base64.b64encode(buf1.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"SHAP visualization failed: {e}")
        visualizations['shap_analysis'] = None
    
    # SECTION 2: Hourly Pattern Analysis with Confidence Intervals
    try:
        plt.figure(figsize=(14, 10))
        
        # Subplot 1: Hourly means with confidence intervals
        plt.subplot(2, 2, 1)
        hours = [h['hour'] for h in hourly_stats if h.get('mean') is not None]
        means = [h['mean'] for h in hourly_stats if h.get('mean') is not None]
        ci_lower = [h['confidence_interval'][0] for h in hourly_stats if h.get('mean') is not None]
        ci_upper = [h['confidence_interval'][1] for h in hourly_stats if h.get('mean') is not None]
        
        plt.plot(hours, means, 'b-', linewidth=2, label='Hourly Mean')
        plt.fill_between(hours, ci_lower, ci_upper, alpha=0.2, label='95% CI')
        plt.xlabel('Hour of Day')
        plt.ylabel(f'{sensor.upper()} Value')
        plt.title('Hourly Patterns with Confidence Intervals', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Statistical significance
        plt.subplot(2, 2, 2)
        significant_hours = [h['hour'] for h in hourly_stats if h.get('significant', False)]
        p_values = [h['p_value'] for h in hourly_stats if h.get('p_value') is not None]
        
        plt.bar(range(len(p_values)), -np.log10(p_values))
        plt.axhline(y=-np.log10(0.05), color='r', linestyle='--', label='p=0.05 threshold')
        plt.xlabel('Hour of Day')
        plt.ylabel('-log10(p-value)')
        plt.title('Statistical Significance by Hour', fontweight='bold')
        plt.legend()
        
        buf2 = BytesIO()
        plt.tight_layout()
        plt.savefig(buf2, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        visualizations['hourly_patterns'] = base64.b64encode(buf2.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Hourly pattern visualization failed: {e}")
        visualizations['hourly_patterns'] = None
    
    # SECTION 3: Anomaly Detection Visualization
    try:
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Anomalies over time
        plt.subplot(2, 2, 1)
        plt.plot(df['timestamp'], df['value'], 'b-', alpha=0.7, label='Normal readings')
        
        if anomalies:
            anomaly_times = [pd.to_datetime(a['timestamp']) for a in anomalies]
            anomaly_values = [a['value'] for a in anomalies]
            plt.scatter(anomaly_times, anomaly_values, color='red', s=50, label='Anomalies')
        
        plt.xlabel('Time')
        plt.ylabel(f'{sensor.upper()} Value')
        plt.title('Anomaly Detection Timeline', fontweight='bold')
        plt.legend()
        plt.xticks(rotation=45)
        
        # Subplot 2: Hourly anomaly distribution
        plt.subplot(2, 2, 2)
        if anomalies:
            anomaly_hours = [a['hour'] for a in anomalies]
            plt.hist(anomaly_hours, bins=24, range=(0, 24), alpha=0.7, color='red')
            plt.xlabel('Hour of Day')
            plt.ylabel('Anomaly Count')
            plt.title('Anomaly Distribution by Hour', fontweight='bold')
        
        buf3 = BytesIO()
        plt.tight_layout()
        plt.savefig(buf3, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        visualizations['anomaly_detection'] = base64.b64encode(buf3.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Anomaly visualization failed: {e}")
        visualizations['anomaly_detection'] = None
    
    return visualizations
    
@app.get("/debug/hourly-data/{sensor_id}")
def debug_hourly_data(
    sensor_id: str,
    sensor: str = Query(...),
    range: str = Query("1month")
):
    """Debug endpoint to check hourly data availability"""
    df = fetch_and_preprocess_data(sensor_id, sensor, range)
    
    response = {
        "sensor_id": sensor_id,
        "sensor_type": sensor,
        "range": range,
        "data_availability": {
            "after_preprocessing": len(df),
            "preview": df.head(3).to_dict('records') if not df.empty else []
        }
    }
    
    if not df.empty:
        response["data_availability"]["date_range_original"] = {
            "start": df['timestamp'].min().isoformat(),
            "end": df['timestamp'].max().isoformat()
        }
        
        if not df.empty:
            df["hour"] = df["timestamp"].dt.hour
            hourly_stats = df.groupby("hour").size().reset_index(name='count')
            response["hourly_distribution"] = hourly_stats.to_dict('records')
    
    return response
        
@app.get("/dataframe/{sensor_id}")
def get_dataframe(
    sensor_id: str, 
    sensor: str = Query(..., description="Sensor type (co2, temperature, etc.)"),
    range: str = Query("1month", description="Date range: 1week, 1month, 3months, 6months, 1year, all")
):
    """Get raw sensor data as JSON with date range filtering"""
    try:
        df = fetch_and_preprocess_data(sensor_id, sensor, range)
        
        return {
            "sensor_id": sensor_id,
            "sensor_type": sensor,
            "date_range": range,
            "records": df.to_dict(orient="records"),
            "count": len(df),
            "date_range_applied": True
        }
    except Exception as e:
        logger.error(f"Dataframe error: {e}")
        return standard_error_response(f"Data retrieval failed: {str(e)}", 500)

 
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
        df = fetch_and_preprocess_data(sensor_id, sensor, range)
        
        if df.empty or len(df) < 5:
            return standard_error_response("Not enough data for analysis", 400)
        
        df_lags = make_lag_features(df, lags=2)
        if df_lags.empty:
            return standard_error_response("Insufficient data after feature engineering", 400)
        
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
        return standard_error_response(f"Algorithm comparison failed: {str(e)}", 500)

@app.get("/xgboost_compute/{sensor_id}")
async def xgboost_analysis(
    sensor_id: str, 
    sensor: str = Query(..., description="Sensor type")
):
    """XGBoost analysis with detailed metrics"""
    try:
        df = fetch_and_preprocess_data(sensor_id, sensor, "1month")
        if df.empty:
            return standard_error_response("No valid data for analysis", 400)
        
        if len(df) < 3:
            return standard_error_response("Insufficient clean data for training", 400)

        df_lags = make_lag_features(df, lags=2)
        if len(df_lags) < 2:
            return standard_error_response("Insufficient data after feature engineering", 400)

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
        return standard_error_response(f"XGBoost analysis failed: {str(e)}", 500)

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
        df = fetch_and_preprocess_data(sensor_id, sensor, "3months")
        
        if df.empty or len(df) < 3:
            return standard_error_response("Not enough historical data", 400)
        
        df_lags = make_lag_features(df, lags=2)
        if df_lags.empty:
            return standard_error_response("Insufficient data for forecasting", 400)
        
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
        return standard_error_response(f"Prediction failed: {str(e)}", 500)

@app.get("/recommendation/{sensor_id}")
def get_recommendation(
    sensor_id: str, 
    sensor: str = Query(..., description="Sensor type")
):
    """Generate 1-day ahead forecast with OSH recommendation"""
    try:
        df = fetch_and_preprocess_data(sensor_id, sensor, "1month")

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

        model = xgb.XGBoostRegressor(objective="reg:squarederror", n_estimators=50, random_state=42)
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
    """Get XGBoost performance metrics - OPTIMIZED VERSION"""
    try:
        logger.info(f"Starting performance analysis for {sensor_id}, sensor: {sensor}")
        
        # Fetch and preprocess data using common utility
        df = fetch_and_preprocess_data(sensor_id, sensor, date_range)
        if df.empty or len(df) < 5:
            return standard_error_response("Not enough data for analysis", 400)

        # Create binary labels using median
        median_val = float(df['value'].median())
        df_binary = df.copy()
        df_binary['class_binary'] = (df_binary['value'] > median_val).astype(int)
        
        # Check class distribution
        class_counts = df_binary['class_binary'].value_counts()
        class_distribution = {
            str(int(k)): int(v) for k, v in class_counts.items()
        }
        logger.info(f"Class distribution: {class_distribution}")
        
        if len(class_counts) < 2:
            return standard_error_response(
                "Only one class found in data - cannot perform classification", 
                400,
                {"class_distribution": class_distribution}
            )

        # Create basic features
        df_features = df_binary.copy()
        df_features['lag_1'] = df_features['value'].shift(1)
        df_features['lag_2'] = df_features['value'].shift(2)
        
        # Remove rows with NaN
        df_features = df_features.dropna()
        
        if df_features.empty or len(df_features) < 5:
            return standard_error_response("Not enough data after feature engineering", 400)

        # Prepare features and target
        feature_cols = ['lag_1', 'lag_2']
        X = df_features[feature_cols].values
        y = df_features['class_binary'].values

        logger.info(f"Final dataset - X: {X.shape}, y: {y.shape}")

        # Calculate baseline accuracy
        y_counts = np.bincount(y)
        baseline_accuracy = float(max(y_counts) / len(y)) if len(y_counts) > 0 else 0.0
        logger.info(f"Baseline accuracy: {baseline_accuracy:.3f}")

        # Simple train-test split
        if len(X) > 10:
            split_point = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_point], X[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]
        else:
            # For very small datasets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

        # Validate test set
        if len(X_test) == 0 or len(np.unique(y_test)) < 2:
            return standard_error_response(
                "Test set too small or only one class",
                400,
                {
                    "train_samples": int(len(X_train)),
                    "test_samples": int(len(X_test))
                }
            )

        # Use standardized model training
        model_result = train_and_evaluate_model(X_train, X_test, y_train, y_test, 'xgb')
        
        accuracy = model_result['accuracy']
        precision = model_result['precision']
        recall = model_result['recall']
        f1 = model_result['f1']

        logger.info(f"Metrics - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

        # Feature importance
        feature_importance = []
        if hasattr(model_result['model'], 'feature_importances_'):
            for i, importance in enumerate(model_result['model'].feature_importances_):
                feature_importance.append({
                    "feature": str(feature_cols[i]) if i < len(feature_cols) else f"feature_{i}",
                    "importance": float(importance)
                })
            feature_importance.sort(key=lambda x: x['importance'], reverse=True)

        # Get unique class counts for distribution
        unique_classes, class_counts = np.unique(y, return_counts=True)
        final_class_distribution = {
            str(int(cls)): int(count) for cls, count in zip(unique_classes, class_counts)
        }

        # Prepare response
        response_data = {
            "sensor_id": str(sensor_id),
            "sensor_type": str(sensor),
            "date_range": str(date_range),
            "algorithm": "XGBoost",
            
            "performance_metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "baseline_accuracy": baseline_accuracy,
                "improvement_over_baseline": float(accuracy - baseline_accuracy),
                "test_samples": int(len(y_test)),
                "train_samples": int(len(X_train))
            },
            
            "data_info": {
                "total_samples": int(len(df_features)),
                "training_samples": int(len(X_train)),
                "test_samples": int(len(y_test)),
                "class_distribution": final_class_distribution,
                "features_used": [str(f) for f in feature_cols],
                "data_quality": "good",
                "median_threshold": median_val
            },
            
            "feature_importance": feature_importance,
            "status": "success"
        }

        # Final validation using json.dumps to catch any serialization issues
        try:
            json.dumps(response_data)
            return JSONResponse(content=response_data)
        except TypeError as e:
            logger.error(f"JSON serialization error: {e}")
            return standard_error_response("Data serialization error", 500)

    except Exception as e:
        logger.exception(f"Performance metrics error: {str(e)}")
        return standard_error_response(f"Analysis failed: {str(e)}", 500)
        
@app.get("/debug/performance/{sensor_id}")
def debug_performance_metrics(
    sensor_id: str,
    sensor: str = Query(...)
):
    """Debug endpoint to check data quality before running analysis"""
    df = fetch_and_preprocess_data(sensor_id, sensor, "1month")
    
    if df.empty:
        return standard_error_response("No data after preprocessing", 400)
    
    df_binary = create_simple_binary_labels(df)
    
    analysis = {
        "sensor_id": sensor_id,
        "sensor_type": sensor,
        "after_preprocessing": len(df),
        "after_binary_labels": len(df_binary) if df_binary is not None else 0,
        "data_quality_checks": {}
    }
    
    if df_binary is not None and not df_binary.empty:
        # Class distribution analysis
        class_counts = df_binary['class_binary'].value_counts()
        analysis["class_distribution"] = dict(class_counts)
        analysis["data_quality_checks"]["class_balance"] = len(class_counts) >= 2
        analysis["data_quality_checks"]["minority_class_ratio"] = min(class_counts) / len(df_binary)
        
        # Value range analysis
        analysis["value_statistics"] = {
            "min": float(df['value'].min()),
            "max": float(df['value'].max()),
            "mean": float(df['value'].mean()),
            "std": float(df['value'].std())
        }
    
    return analysis

# ---------------------------------------------------
# Correlation Plots (Single or Multiple Sensors)
# ---------------------------------------------------

@app.get("/correlation")
def correlation(
    sensor_ids: str = Query(..., description="Comma-separated sensor IDs"),
    plot_type: str = Query("heatmap", enum=["heatmap", "scatter", "pairplot"]),
    output: str = Query("img", enum=["img", "json"]),
    limit: int = Query(500, description="Maximum number of records"),
):
    """Correlation analysis with memory optimization"""
    try:
        log_memory_usage("Before correlation analysis")
        
        sensor_list = [s.strip() for s in sensor_ids.split(",") if s.strip()]
        if not sensor_list:
            return standard_error_response("No sensor IDs provided", 400)

        all_data = []
        for sid in sensor_list:
            records = fetch_sensor_history(sid)
            if records:
                df = pd.DataFrame(records)
                df["sensor_id"] = sid
                all_data.append(df)

        if not all_data:
            return standard_error_response("No data found for given sensors", 404)

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
        plt.figure(figsize=(8, 6))
        
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
        plt.savefig(buf, format="png", dpi=80, bbox_inches="tight")
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
        return standard_error_response(f"Correlation analysis failed: {str(e)}", 500)

# ===========
# Confusion Matrix
# ===========

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
    """
    try:
        # Fetch and preprocess data using common utility
        df = fetch_and_preprocess_data(sensor_id, sensor, date_range)
        if df.empty or len(df) < 10:
            return standard_error_response("Insufficient data for analysis", 400)

        # Create binary classification labels
        df_binary = create_simple_binary_labels(df)
        if df_binary.empty:
            return standard_error_response("Could not create classification labels", 400)

        # Create enhanced features for better prediction
        df_features = create_enhanced_features(df_binary, sensor_col='value')
        if df_features.empty:
            return standard_error_response("Feature engineering failed", 400)

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
        class_names = ['Low', 'High']

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
        return standard_error_response(f"Confusion matrix analysis failed: {str(e)}", 500)


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
    for lag in [1, 2, 3, 5, 7]:
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