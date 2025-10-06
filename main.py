
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
   
from io import BytesIO

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

# --- HOME ROUTE (Render Documentation Page) ---
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

        <h2>API Documentation</h2>
        <ul>
            <li><a href="/docs">Swagger UI</a> (interactive API docs)</li>
            <li><a href="/redoc">ReDoc</a> (alternative docs)</li>
        </ul>

        <h2>Available Endpoints</h2>
        <div class="endpoint new">
            <h3>🧠 Machine Learning Endpoint (NEW)</h3>
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
        <p style="font-size: 0.9em; color: #666;">Powered by FastAPI, XGBoost & Seaborn | Gas Monitoring Project</p>
    </body>
    </html>
    """

# --- XGBoost Training Endpoint ---
@app.post("/train_xgboost")
def train_xgboost(params: dict):
    try:
        sensor_id = params.get("sensor_id")
        if not sensor_id:
            raise HTTPException(status_code=400, detail="Missing sensor_id")

        test_size = float(params.get("test_size", 0.2))
        plot_type = params.get("plot_type", "scatter")
        plot_output = params.get("plot_output", "base64")

        # --- Fetch Firebase data ---
        ref = db.reference(f"/history/{sensor_id}")
        data = ref.get()
        if not data:
            raise HTTPException(status_code=404, detail="No data found for sensor_id")

        df = pd.DataFrame.from_dict(data, orient="index")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        # --- Prepare features and target ---
        features = df.drop(columns=["riskIndex", "time", "timestamp", "apiUserID", "apiPass", "sensorID"], errors="ignore")
        target = df["riskIndex"]

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=42)

        # --- Grid Search ---
        xgb = XGBRegressor(random_state=42)
        param_grid = {
            "n_estimators": params.get("n_estimators", [100, 300]),
            "max_depth": params.get("max_depth", [3, 5, 7]),
            "learning_rate": params.get("learning_rate", [0.01, 0.1]),
            "subsample": params.get("subsample", [0.8, 1.0]),
            "colsample_bytree": params.get("colsample_bytree", [0.8, 1.0])
        }

        grid_search = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=3,
            verbose=0,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # --- Predictions & metrics ---
        y_pred = best_model.predict(X_test)
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred))
        }

        # --- Visualization ---
        plt.figure(figsize=(8, 6))
        if plot_type == "scatter":
            sns.scatterplot(x=y_test, y=y_pred)
            plt.xlabel("True Values")
            plt.ylabel("Predicted Values")
            plt.title("Predicted vs True Risk Index")
        elif plot_type == "line":
            plt.plot(y_test.values, label="True")
            plt.plot(y_pred, label="Predicted")
            plt.legend()
            plt.title("True vs Predicted Over Time")
        elif plot_type == "heatmap":
            corr = features.corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm")
            plt.title("Feature Correlation Heatmap")

        # --- Convert or Save plot ---
        image_data = None
        if plot_output == "base64":
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            image_data = base64.b64encode(buf.read()).decode("utf-8")
            plt.close()
        else:
            os.makedirs("static/plots", exist_ok=True)
            filename = f"static/plots/{sensor_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, bbox_inches="tight")
            plt.close()
            image_data = f"/{filename}"

        # --- Response ---
        response = {
            "sensor_id": sensor_id,
            "best_params": grid_search.best_params_,
            "metrics": metrics,
            "plot_type": plot_type,
            "plot_output": plot_output,
            "plot_data": image_data
        }
        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
#===========
# Run the application
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)