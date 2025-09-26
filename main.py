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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app FIRST
app = FastAPI(
    title="Gas Monitoring API", 
    version="1.0.0",
    description="API for gas sensor monitoring, forecasting, and SHAP explanations"
)
# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[  "*"    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------------------------------------------
# Firebase Setup
# ---------------------------------------------------
try:
    # For Render.com deployment, use environment variables
    firebase_config = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
    if firebase_config:
        service_account_info = json.loads(firebase_config)
    else:
        # Fallback for local development
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

    if not firebase_admin._apps:
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred, {"databaseURL": database_url})
        logger.info("Firebase initialized successfully")
except Exception as e:
    logger.error(f"Firebase initialization failed: {e}")

# Add explicit CORS headers to key endpoints as backup
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# Add OPTIONS handler for preflight requests
@app.options("/{path:path}")
async def preflight_handler():
    return JSONResponse(
        content={"status": "ok"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )
    
# Utility functions
    # Utility functions
def fetch_sensor_history(sensor_id: str):
    """Fetch sensor history from Firebase"""
    try:
        ref = db.reference(f"history/{sensor_id}")
        snapshot = ref.get()
        if not snapshot:
            return []
        records = []
        for key, value in snapshot.items():
            if isinstance(value, dict):
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
    df_copy = df.copy()
    for i in range(1, lags + 1):
        df_copy[f"lag{i}"] = df_copy["value"].shift(i)
    return df_copy.dropna()

def compute_metrics(y_true, y_pred):
    """Compute regression metrics"""
    if len(y_true) == 0:
        return {"mse": 0, "rmse": 0, "mae": 0, "r2_score": 0}
    
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    if len(y_true) > 1 and np.var(y_true) > 0:
        r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
    else:
        r2 = 0
    return {"mse": float(mse), "rmse": float(rmse), "mae": float(mae), "r2_score": float(r2)}


def compute_classification_metrics(y_true, y_pred, classes):
    """Compute comprehensive classification metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
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
def create_classification_labels(df, sensor_type):
    """Create classification labels based on sensor thresholds"""
    if df.empty:
        return pd.DataFrame()
    
    df_class = df.copy()
    
    # Define thresholds for different sensor types
    thresholds = {
        'co2': {'low': 400, 'medium': 800, 'high': 1000},
        'methane': {'low': 200, 'medium': 500, 'high': 1000},
        'ammonia': {'low': 10, 'medium': 25, 'high': 50},
        'temperature': {'low': 15, 'medium_low': 20, 'medium_high': 30, 'high': 35},
        'humidity': {'low': 30, 'medium_low': 40, 'medium_high': 60, 'high': 70}
    }
    
    sensor_type_lower = sensor_type.lower()
    
    if sensor_type_lower in thresholds:
        thresholds_config = thresholds[sensor_type_lower]
        
        if sensor_type_lower in ['co2', 'methane', 'ammonia']:
            # For gas sensors: low, medium, high
            conditions = [
                df_class['value'] <= thresholds_config['low'],
                (df_class['value'] > thresholds_config['low']) & (df_class['value'] <= thresholds_config['medium']),
                (df_class['value'] > thresholds_config['medium']) & (df_class['value'] <= thresholds_config['high']),
                df_class['value'] > thresholds_config['high']
            ]
            choices = ['very_low', 'low', 'medium', 'high']
            
        else:  # temperature and humidity
            # For environmental sensors: very_low, low, normal, high, very_high
            conditions = [
                df_class['value'] < thresholds_config['low'],
                (df_class['value'] >= thresholds_config['low']) & (df_class['value'] < thresholds_config.get('medium_low', 25)),
                (df_class['value'] >= thresholds_config.get('medium_low', 25)) & (df_class['value'] <= thresholds_config.get('medium_high', 30)),
                (df_class['value'] > thresholds_config.get('medium_high', 30)) & (df_class['value'] <= thresholds_config['high']),
                df_class['value'] > thresholds_config['high']
            ]
            choices = ['very_low', 'low', 'normal', 'high', 'very_high']
        
        df_class['class'] = np.select(conditions, choices, default='unknown')
        
    else:
        # Default binary classification based on median
        median_val = df_class['value'].median()
        df_class['class'] = np.where(df_class['value'] > median_val, 'above_median', 'below_median')
    
    # Remove any rows with 'unknown' class
    df_class = df_class[df_class['class'] != 'unknown']
    
    return df_class
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
        <h1>Gas Monitoring API</h1>
        <p>Welcome to the Gas Monitoring Backend. This service provides endpoints for forecasting, SHAP explanations, and analytics for your sensors.</p>
        <p><strong>CORS Status:</strong> ✅ Enabled for all origins</p>

        <h2>API Documentation</h2>
        <ul>
            <li><a href="/docs">Swagger UI</a> (interactive API docs)</li>
            <li><a href="/redoc">ReDoc</a> (alternative docs)</li>
        </ul>

        <h2>Available Endpoints</h2>
        
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

        <h2>CORS Test</h2>
        <pre><code>// Test from your frontend:
fetch('https://gasmonitoring-backend-1.onrender.com/health')
  .then(response => response.json())
  .then(data => console.log(data));</code></pre>

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
    return {
        "status": "ok", 
        "timestamp": datetime.now().isoformat(), 
        "service": "Gas Monitoring API",
        "cors_enabled": True,
        "allowed_origins": "all (*)"
    }

@app.get("/dataframe/{sensor_id}")
def get_dataframe(sensor_id: str, sensor: str = Query(..., description="Sensor type (co2, temperature, etc.)")):
    """Get raw sensor data as JSON"""
    try:
        records = fetch_sensor_history(sensor_id)
        df = preprocess_dataframe(records, sensor)
        return {
            "sensor_id": sensor_id,
            "sensor_type": sensor,
            "records": df.to_dict(orient="records"),
            "count": len(df)
        }
    except Exception as e:
        logger.error(f"Dataframe error: {e}")
        return {"error": str(e), "sensor_id": sensor_id, "sensor_type": sensor}

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
        if df.empty or len(df) < 5:
            return error_image(f"Not enough data for plot. Found {len(df)} records.")
        
        # Use recent data (last 7 days)
        cutoff = datetime.now() - timedelta(days=7)
        df_recent = df[df["timestamp"] >= cutoff]
        
        if df_recent.empty or len(df_recent) < 3:
            return error_image("Not enough recent data for plotting")
        
        # Create simple time series plot
        buf = io.BytesIO()
        plt.figure(figsize=(10, 6))
        
        if chart == "scatter":
            plt.scatter(df_recent["timestamp"], df_recent["value"], alpha=0.7)
            plt.xlabel("Timestamp")
            plt.ylabel("Sensor Value")
            plt.title(f"Scatter Plot - {sensor} (Sensor {sensor_id})")
        else:
            plt.plot(df_recent["timestamp"], df_recent["value"], marker='o')
            plt.xlabel("Timestamp")
            plt.ylabel("Sensor Value")
            plt.title(f"Time Series - {sensor} (Sensor {sensor_id})")
        
        plt.xticks(rotation=45)
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
            return {"error": "Not enough data for analysis", "sensor_id": sensor_id, "sensor_type": sensor}
        
        # Create lag features
        df_lags = make_lag_features(df, lags=2)
        if df_lags.empty:
            return {"error": "Insufficient data after feature engineering", "sensor_id": sensor_id, "sensor_type": sensor}
        
        X, y = df_lags[["lag1", "lag2"]], df_lags["value"]
        
        # Use last 20% for testing, but ensure at least 1 sample
        test_size = min(0.2, 1.0 / len(X))
        if len(X) <= 5:
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        
        results = {}
        
        # Linear Regression
        lr = LinearRegression().fit(X_train, y_train)
        results["LinearRegression"] = compute_metrics(y_test, lr.predict(X_test))
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=50, random_state=42).fit(X_train, y_train)
        results["RandomForest"] = compute_metrics(y_test, rf.predict(X_test))
        
        # Decision Tree
        dt = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
        results["DecisionTree"] = compute_metrics(y_test, dt.predict(X_test))
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=50, random_state=42)
        xgb_model.fit(X_train, y_train)
        results["XGBoost"] = compute_metrics(y_test, xgb_model.predict(X_test))
        
        # Generate forecast
        last_row = df_lags.iloc[-1]
        pred_next = xgb_model.predict([[last_row["lag1"], last_row["lag2"]]])[0]
        forecast_val = float(pred_next)
        next_date = df_lags["timestamp"].iloc[-1] + timedelta(days=1)
        
        # Generate recommendation
        recommendation = generate_recommendation(sensor, forecast_val)
        
        return {
            "sensor_id": sensor_id,
            "sensor_type": sensor,
            "algorithms": results,
            "best_algorithm": min(results.keys(), key=lambda x: results[x]["rmse"]),
            "forecast": {
                "date": next_date.strftime("%Y-%m-%d"),
                "predicted_value": forecast_val,
                "recommendation": recommendation
            },
            "data_points": len(df_lags)
        }
        
    except Exception as e:
        logger.error(f"Algorithm comparison error: {e}")
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

        # Use recent data (last 30 days)
        cutoff = datetime.now() - timedelta(days=30)
        df_recent = df[df["timestamp"] >= cutoff]

        if len(df_recent) < 3:
            # Use current value for recommendation
            current_value = df["value"].iloc[-1] if not df.empty else 0
            recommendation = generate_recommendation(sensor, float(current_value))
            
            return {
                "sensor_id": sensor_id,
                "sensor_type": sensor,
                "current_value": float(current_value),
                "recommendation": recommendation,
                "status": "current_value_only"
            }

        # Create lag features and train model
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

        # Generate forecast
        last_lags = X[-1].copy()
        last_date = df_lags["timestamp"].iloc[-1]
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

        # Create lag features
        df_lags = make_lag_features(df, lags=2)
        if len(df_lags) < 2:
            return {"error": "Insufficient data after feature engineering", "status": "error"}

        X = df_lags[["lag1", "lag2"]]
        y = df_lags["value"]

        # Use simple validation approach for small datasets
        if len(X) <= 3:
            X_train, X_test, y_train, y_test = X, X.iloc[-1:], y, y.iloc[-1:]
        else:
            X_train, X_test = X.iloc[:-1], X.iloc[-1:]
            y_train, y_test = y.iloc[:-1], y.iloc[-1:]

        # XGBoost model
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=30, random_state=42)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        # Metrics
        mse = mean_squared_error(y_test, y_pred) if len(y_test) > 0 else 0
        rmse = math.sqrt(mse) if mse > 0 else 0
        mae = mean_absolute_error(y_test, y_pred) if len(y_test) > 0 else 0
        r2 = r2_score(y_train, y_pred_train) if len(y_train) > 1 else 0

        # Generate recommendation
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
        
        # Create features and train model
        df_lags = make_lag_features(df, lags=2)
        if df_lags.empty:
            return {"error": "Insufficient data for forecasting", "sensor_id": sensor_id, "sensor_type": sensor}
        
        X = df_lags[["lag1", "lag2"]].values
        y = df_lags["value"].values
        
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # Generate forecasts
        forecasts = []
        current_features = X[-1].copy()
        current_date = df_lags["timestamp"].iloc[-1]
        
        for i in range(min(days, 7)):  # Limit to 7 days max
            next_pred = model.predict(current_features.reshape(1, -1))[0]
            current_date += timedelta(days=1)
            
            forecasts.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "forecast": float(next_pred),
                "recommendation": generate_recommendation(sensor, float(next_pred))
            })
            
            # Update features for next prediction
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

# Simple test endpoint for basic functionality
@app.get("/test/{sensor_id}")
def test_endpoint(sensor_id: str, sensor: str = Query("temperature")):
    """Test endpoint to verify basic functionality"""
    return {
        "message": "API is working",
        "sensor_id": sensor_id,
        "sensor_type": sensor,
        "timestamp": datetime.now().isoformat()
    }

# Explain endpoint
@app.get("/explain/{sensor_id}")
def explain(sensor_id: str, sensor: str = Query(...)):
    records = fetch_sensor_history(sensor_id)
    df = preprocess_dataframe(records, sensor)
    if df.empty or len(df) < 10:
        return JSONResponse({"error": "Not enough data"})
    cutoff = datetime.now() - timedelta(days=7)
    df_recent = df[df["timestamp"] >= cutoff]
    if df_recent.empty:
        return JSONResponse({"error": "No recent data"})
    df_recent["hour"] = df_recent["timestamp"].dt.hour
    agg = df_recent.groupby("hour")["value"].mean().reset_index()
    X = agg[["hour"]]
    y = agg["value"]
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return JSONResponse({
        "sensor_id": sensor_id,
        "sensor_type": sensor,
        "shap_values": shap_values.tolist(),
        "features": X.to_dict(orient="records")
    })

# SHAP Hour endpoint
@app.get("/shap_hour/{sensor_id}")
def shap_hour(sensor_id: str, sensor: str = Query(...)):
    records = fetch_sensor_history(sensor_id)
    df = preprocess_dataframe(records, sensor)
    if df.empty:
        return JSONResponse({"error": "No data"})
    df["hour"] = df["timestamp"].dt.hour
    agg = df.groupby("hour")["value"].mean().reset_index()
    return JSONResponse(agg.to_dict(orient="records"))
    
    # Hyperparameter grids for different classifiers
HYPERPARAM_GRIDS = {
    'xgboost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    },
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    },
    'svc': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.1, 1],
        'degree': [2, 3, 4]
    },
    'knn': {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree'],
        'leaf_size': [20, 30, 40]
    },
    'logistic_regression': {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [100, 200, 500]
    }
}

# NEW PERFORMANCE ENDPOINT
@app.get("/performance/{sensor_id}")
def performance_metrics(
    sensor_id: str,
    sensor: str = Query(..., description="Sensor type"),
    test_size: float = Query(0.2, description="Test set size ratio"),
    cv_folds: int = Query(5, description="Cross-validation folds")
):
    """Get performance metrics for different classifiers with hyperparameter tuning"""
    try:
        records = fetch_sensor_history(sensor_id)
        df = preprocess_dataframe(records, sensor)
        
        if df.empty or len(df) < 10:
            return {"error": "Not enough data for performance analysis", "sensor_id": sensor_id, "sensor_type": sensor}
        
        # Create classification dataset
        df_class = create_classification_labels(df, sensor)
        if df_class.empty:
            return {"error": "Failed to create classification labels", "sensor_id": sensor_id, "sensor_type": sensor}
        
        # Create features (using lag features for time series)
        df_lags = make_lag_features(df_class, lags=2)
        if df_lags.empty:
            return {"error": "Insufficient data after feature engineering", "sensor_id": sensor_id, "sensor_type": sensor}
        
        # Prepare features and target
        feature_cols = [col for col in df_lags.columns if col.startswith('lag')]
        X = df_lags[feature_cols]
        y = df_lags['class']
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        classes = le.classes_.tolist()
        
        if len(classes) < 2:
            return {"error": "Need at least 2 classes for classification", "sensor_id": sensor_id, "sensor_type": sensor}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        
        # 1. XGBoost Classifier with Hyperparameter Tuning
        try:
            xgb_clf = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
            grid_search_xgb = GridSearchCV(
                xgb_clf, HYPERPARAM_GRIDS['xgboost'], 
                cv=min(cv_folds, len(X_train)), scoring='f1_weighted', n_jobs=-1
            )
            grid_search_xgb.fit(X_train_scaled, y_train)
            best_xgb = grid_search_xgb.best_estimator_
            y_pred_xgb = best_xgb.predict(X_test_scaled)
            
            results['XGBoost'] = {
                'best_params': grid_search_xgb.best_params_,
                'metrics': compute_classification_metrics(y_test, y_pred_xgb, classes),
                'cv_score': float(grid_search_xgb.best_score_)
            }
        except Exception as e:
            logger.error(f"XGBoost tuning failed: {e}")
            results['XGBoost'] = {'error': str(e)}
        
        # 2. Random Forest Classifier with Hyperparameter Tuning
        try:
            rf_clf = RandomForestClassifier(random_state=42)
            grid_search_rf = GridSearchCV(
                rf_clf, HYPERPARAM_GRIDS['random_forest'],
                cv=min(cv_folds, len(X_train)), scoring='f1_weighted', n_jobs=-1
            )
            grid_search_rf.fit(X_train, y_train)  # RF doesn't always need scaling
            best_rf = grid_search_rf.best_estimator_
            y_pred_rf = best_rf.predict(X_test)
            
            results['RandomForest'] = {
                'best_params': grid_search_rf.best_params_,
                'metrics': compute_classification_metrics(y_test, y_pred_rf, classes),
                'cv_score': float(grid_search_rf.best_score_)
            }
        except Exception as e:
            logger.error(f"Random Forest tuning failed: {e}")
            results['RandomForest'] = {'error': str(e)}
        
        # 3. SVC with Hyperparameter Tuning
        try:
            svc_clf = SVC(random_state=42, probability=True)
            grid_search_svc = GridSearchCV(
                svc_clf, HYPERPARAM_GRIDS['svc'],
                cv=min(cv_folds, len(X_train_scaled)), scoring='f1_weighted', n_jobs=-1
            )
            grid_search_svc.fit(X_train_scaled, y_train)
            best_svc = grid_search_svc.best_estimator_
            y_pred_svc = best_svc.predict(X_test_scaled)
            
            results['SVC'] = {
                'best_params': grid_search_svc.best_params_,
                'metrics': compute_classification_metrics(y_test, y_pred_svc, classes),
                'cv_score': float(grid_search_svc.best_score_)
            }
        except Exception as e:
            logger.error(f"SVC tuning failed: {e}")
            results['SVC'] = {'error': str(e)}
        
        # 4. K-Nearest Neighbors with Hyperparameter Tuning
        try:
            knn_clf = KNeighborsClassifier()
            grid_search_knn = GridSearchCV(
                knn_clf, HYPERPARAM_GRIDS['knn'],
                cv=min(cv_folds, len(X_train_scaled)), scoring='f1_weighted', n_jobs=-1
            )
            grid_search_knn.fit(X_train_scaled, y_train)
            best_knn = grid_search_knn.best_estimator_
            y_pred_knn = best_knn.predict(X_test_scaled)
            
            results['KNN'] = {
                'best_params': grid_search_knn.best_params_,
                'metrics': compute_classification_metrics(y_test, y_pred_knn, classes),
                'cv_score': float(grid_search_knn.best_score_)
            }
        except Exception as e:
            logger.error(f"KNN tuning failed: {e}")
            results['KNN'] = {'error': str(e)}
        
        # 5. Logistic Regression with Hyperparameter Tuning
        try:
            lr_clf = LogisticRegression(random_state=42, max_iter=1000)
            grid_search_lr = GridSearchCV(
                lr_clf, HYPERPARAM_GRIDS['logistic_regression'],
                cv=min(cv_folds, len(X_train_scaled)), scoring='f1_weighted', n_jobs=-1
            )
            grid_search_lr.fit(X_train_scaled, y_train)
            best_lr = grid_search_lr.best_estimator_
            y_pred_lr = best_lr.predict(X_test_scaled)
            
            results['LogisticRegression'] = {
                'best_params': grid_search_lr.best_params_,
                'metrics': compute_classification_metrics(y_test, y_pred_lr, classes),
                'cv_score': float(grid_search_lr.best_score_)
            }
        except Exception as e:
            logger.error(f"Logistic Regression tuning failed: {e}")
            results['LogisticRegression'] = {'error': str(e)}
        
        # Determine best algorithm
        successful_models = {k: v for k, v in results.items() if 'metrics' in v}
        if successful_models:
            best_algorithm = max(successful_models.keys(), 
                               key=lambda x: successful_models[x]['metrics']['f1_score'])
        else:
            best_algorithm = "No successful models"
        
        return {
            "sensor_id": sensor_id,
            "sensor_type": sensor,
            "dataset_info": {
                "total_samples": len(df_lags),
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "feature_count": len(feature_cols),
                "classes": classes,
                "class_distribution": dict(df_lags['class'].value_counts())
            },
            "algorithms": results,
            "best_algorithm": best_algorithm,
            "test_size": test_size,
            "cv_folds": cv_folds
        }
        
    except Exception as e:
        logger.error(f"Performance metrics error: {e}")
        return {"error": str(e), "sensor_id": sensor_id, "sensor_type": sensor}

@app.get("/confusion_matrix/{sensor_id}")
def confusion_matrix_chart(
    sensor_id: str,
    sensor: str = Query(..., description="Sensor type"),
    test_size: float = Query(0.2, description="Test size"),
    cv_folds: int = Query(5, description="CV folds")
):
    """
    Generate confusion matrix chart for best performing algorithm
    """
    try:
        # Reuse performance endpoint logic
        perf = performance_metrics(sensor_id, sensor, test_size, cv_folds)
        if "error" in perf:
            return error_image(perf["error"])

        best_algo = perf.get("best_algorithm")
        if not best_algo or best_algo == "No successful models":
            return error_image("No valid model for confusion matrix")

        metrics = perf["algorithms"][best_algo]["metrics"]
        conf_matrix = np.array(metrics["confusion_matrix"])
        classes = metrics["classes"]

        # Plot confusion matrix
        buf = io.BytesIO()
        plt.figure(figsize=(6, 5))
        plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix - {best_algo}")
        plt.colorbar()

        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        # Normalize
        cm_normalized = conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
        thresh = conf_matrix.max() / 2.
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(
                    j, i,
                    f"{conf_matrix[i, j]} ({cm_normalized[i, j]:.2f})",
                    horizontalalignment="center",
                    color="white" if conf_matrix[i, j] > thresh else "black"
                )

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(buf, format="png", dpi=100)
        plt.close()
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        logger.error(f"Confusion matrix error: {e}")
        return error_image(str(e))

# Run the application
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)