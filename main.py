
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
from xgboost import XGBRegressor
   
from io import BytesIO

# Global model and data (in-memory)
model = None
data = None
features = None
target = None
db = None

# Try to import optional dependencies
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available, falling back to feature importance")

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available, some statistical tests will be skipped")
   
  
  
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
 # --- FastAPI setup ---
app = FastAPI(title="Gas Monitoring XGBoost API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def load_data_from_firebase():
    """Load data from Firebase Realtime Database /history"""
    try:
        # Get data from /history path in Realtime Database
        history_ref = db.reference('/history')
        history_data = history_ref.get()
        
        if not history_data:
            logger.warning("No data found in Firebase /history. Using local CSV as fallback.")
            return pd.read_csv("data.csv")
        
        # Convert the nested dictionary to DataFrame
        data_list = []
        for key, value in history_data.items():
            if isinstance(value, dict):
                value['firebase_key'] = key  # Keep the Firebase key for reference
                data_list.append(value)
        
        df = pd.DataFrame(data_list)
        logger.info(f"Loaded {len(df)} records from Firebase /history")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data from Firebase: {e}")
        logger.info("Using local CSV as fallback")
        return pd.read_csv("data.csv")

def save_prediction_to_firebase(sensor_id, prediction_data):
    """Save prediction results to Firebase Realtime Database"""
    try:
        predictions_ref = db.reference('/predictions')
        
        for pred in prediction_data:
            # Create a unique key using sensor_id and timestamp
            unique_key = f"{sensor_id}_{pred['timestamp']}".replace(':', '_').replace(' ', '_')
            
            # Prepare prediction data
            pred_data = {
                'sensor_id': sensor_id,
                'timestamp': pred['timestamp'],
                'predicted_riskIndex': pred['predicted_riskIndex'],
                'prediction_date': datetime.now().isoformat()
            }
            
            # Save to Firebase under /predictions/{unique_key}
            predictions_ref.child(unique_key).set(pred_data)
        
        logger.info(f"Saved {len(prediction_data)} predictions to Firebase for sensor {sensor_id}")
        
    except Exception as e:
        logger.error(f"Error saving prediction to Firebase: {e}")

def prepare_data(df):
    """Prepare data for training"""
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    X = df.drop(columns=['riskIndex', 'time', 'timestamp', 'apiUserID', 'apiPass', 'sensorID'], errors='ignore')
    y = df['riskIndex']
    X = X.fillna(X.mean())
    return X, y

def train_model(X_train, y_train):
    """Train XGBoost model with GridSearch"""
    xgb_model = XGBRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100],
        'max_depth': [3],
        'learning_rate': [0.1],
        'subsample': [1.0],
        'colsample_bytree': [1.0]
    }
    grid_search = GridSearchCV(xgb_model, param_grid, scoring='neg_root_mean_squared_error', cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

@app.on_event("startup")
async def load_and_train():
    """Startup event to load data and train model"""
    global data, features, target, model

    # Initialize Firebase
    initialize_firebase()
    
    # Load data from Firebase (with CSV fallback)
    data = load_data_from_firebase()

    if data is None or data.empty:
        logger.error("No data available for training")
        return

    # Prepare and train model
    features, target = prepare_data(data)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)
    logger.info("Model trained and ready.")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>Gas Monitoring API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 2em; line-height: 1.6; background: #fafafa; }
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
        <h1>Gas Monitoring XGBoost API</h1>
        <p>Welcome to the <strong>Render-ready</strong> Gas Monitoring Backend. This API performs XGBoost training and visualization on Firebase Realtime Database data.</p>
        <p><strong>CORS Status:</strong> ✅ Enabled for all origins</p>
        <p><strong>Database:</strong> ✅ Firebase Realtime Database (/history)</p>

        <h2>API Documentation</h2>
        <ul>
            <li><a href="/docs">Swagger UI</a> (interactive API docs)</li>
            <li><a href="/redoc">ReDoc</a> (alternative docs)</li>
        </ul>

        <h2>Available Endpoints</h2>
        
        <div class="endpoint">
            <h3>🔍 Data & Predictions</h3>
            <ul>
                <li><code>GET /explain/{sensor_id}</code> - Get predictions for a sensor</li>
                <li><code>GET /predictions/{sensor_id}</code> - Get stored predictions</li>
                <li><code>GET /health</code> - Health check</li>
            </ul>
        </div>

        <div class="endpoint new">
            <h3>🧠 Machine Learning Endpoints</h3>
            <ul>
                <li><code>POST /train_xgboost</code> — Train XGBoost using Firebase /history data</li>
            </ul>
            <p>Request body (JSON):</p>
<pre>{
  "sensor_id": "3221",
  "test_size": 0.2,
  "n_estimators": [100, 300],
  "max_depth": [3, 5, 7],
  "learning_rate": [0.01, 0.1],
  "subsample": [0.8, 1.0],
  "colsample_bytree": [0.8, 1.0],
  "plot_type": "scatter",
  "plot_output": "base64"
}</pre>
        </div>

        <hr/>
        <p style="font-size: 0.9em; color: #666;">Powered by FastAPI, XGBoost & Firebase Realtime Database | Gas Monitoring Project</p>
    </body>
    </html>
    """

@app.get("/explain/{sensor_id}")
def explain(sensor_id: int):
    global data, features, target, model

    if model is None:
        raise HTTPException(status_code=503, detail="Model not trained yet")

    # Filter data for the requested sensor_id
    sensor_data = data[data['sensorID'] == sensor_id]

    if sensor_data.empty:
        raise HTTPException(status_code=404, detail="Sensor ID not found")

    # Prepare features for prediction
    X_sensor = sensor_data.drop(columns=['riskIndex', 'time', 'timestamp', 'apiUserID', 'apiPass', 'sensorID'], errors='ignore')
    X_sensor = X_sensor.fillna(X_sensor.mean())

    # Predict riskIndex for the sensor data
    predictions = model.predict(X_sensor)

    # Build response with timestamps and predictions
    response = []
    for ts, pred in zip(sensor_data['timestamp'], predictions):
        response.append({
            "timestamp": str(ts),
            "predicted_riskIndex": float(pred)
        })

    # Save predictions to Firebase
    save_prediction_to_firebase(sensor_id, response)

    return {
        "sensor_id": sensor_id,
        "predictions": response
    }

@app.get("/predictions/{sensor_id}")
def get_predictions(sensor_id: int, limit: int = 10):
    """Get recent predictions for a sensor from Firebase"""
    try:
        predictions_ref = db.reference('/predictions')
        all_predictions = predictions_ref.get()
        
        if not all_predictions:
            return {"sensor_id": sensor_id, "recent_predictions": []}
        
        # Filter and sort predictions
        sensor_predictions = []
        for key, pred_data in all_predictions.items():
            if pred_data.get('sensor_id') == sensor_id:
                sensor_predictions.append(pred_data)
        
        # Sort by timestamp descending and limit results
        sensor_predictions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        recent_predictions = sensor_predictions[:limit]
        
        return {
            "sensor_id": sensor_id,
            "recent_predictions": recent_predictions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving predictions: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy", 
        "message": "ML Model API with Firebase Realtime Database integration is running",
        "model_ready": model is not None,
        "data_loaded": data is not None and not data.empty,
        "timestamp": datetime.now().isoformat()
    }
    return status
    
#===========
# Run the application
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)