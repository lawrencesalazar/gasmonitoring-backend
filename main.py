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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import io
import base64
import logging
from typing import Dict, Any, Optional

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

  </code></pre>

        <hr />
        <p style="font-size: 0.9em; color: #666;">
            Powered by FastAPI & XGBoost | Gas Monitoring Project
        </p>
    </body>
    </html>
    """

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

        
# Run the application
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)