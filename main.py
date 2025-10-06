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

# Configure logging FIRST
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global models and data storage (in-memory)
models = {}  # Store models by sensor_id: {sensor_id: model}
data_cache = {}  # Store data by sensor_id: {sensor_id: data}
training_metrics = {}  # Store training metrics by sensor_id
shap_explainers = {}  # Store SHAP explainers by sensor_id
firebase_db = None

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

# Set Seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Initialize FastAPI app
app = FastAPI(
    title="Gas Monitoring API", 
    version="1.0.0",
    description="API for gas sensor monitoring, forecasting, and SHAP explanations"
)

# Advanced CORS Middleware
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
# Visualization Functions
# ---------------------------------------------------
def create_shap_plots(model, X, sensor_id: str):
    """Create SHAP explanation plots"""
    if not SHAP_AVAILABLE:
        return {"error": "SHAP not available"}
    
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Store explainer for later use
        shap_explainers[sensor_id] = explainer
        
        # Create plots
        plots = {}
        
        # 1. SHAP Summary Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plots['summary_plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        # 2. SHAP Bar Plot (mean absolute SHAP values)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plots['bar_plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        # 3. SHAP Waterfall Plot for first observation
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(explainer.expected_value, shap_values[0], X.iloc[0], show=False)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plots['waterfall_plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        # 4. SHAP Force Plot for first observation
        plt.figure(figsize=(12, 4))
        shap.force_plot(explainer.expected_value, shap_values[0], X.iloc[0], matplotlib=True, show=False)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plots['force_plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return plots
        
    except Exception as e:
        logger.error(f"Error creating SHAP plots for sensor {sensor_id}: {e}")
        return {"error": f"SHAP plot creation failed: {str(e)}"}

def create_seaborn_plots(df, sensor_id: str):
    """Create various Seaborn visualization plots"""
    try:
        plots = {}
        
        # 1. Correlation Heatmap
        plt.figure(figsize=(12, 10))
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title(f'Correlation Heatmap - Sensor {sensor_id}', fontsize=16, pad=20)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plots['correlation_heatmap'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        # 2. Feature Distribution Plots
        numeric_features = [col for col in numeric_cols if col != 'riskIndex' and df[col].nunique() > 1]
        if numeric_features:
            n_features = min(6, len(numeric_features))
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.ravel()
            
            for i, feature in enumerate(numeric_features[:n_features]):
                sns.histplot(data=df, x=feature, kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {feature}')
                axes[i].tick_params(axis='x', rotation=45)
            
            # Hide empty subplots
            for i in range(n_features, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plots['feature_distributions'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
        
        # 3. Scatter Plots vs riskIndex
        if 'riskIndex' in df.columns and numeric_features:
            n_scatter = min(4, len(numeric_features))
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.ravel()
            
            for i, feature in enumerate(numeric_features[:n_scatter]):
                sns.scatterplot(data=df, x=feature, y='riskIndex', alpha=0.6, ax=axes[i])
                axes[i].set_title(f'{feature} vs Risk Index')
                axes[i].tick_params(axis='x', rotation=45)
            
            # Hide empty subplots
            for i in range(n_scatter, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plots['scatter_plots'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
        
        # 4. Pairplot (sample if too many features)
        if len(numeric_features) > 0 and 'riskIndex' in df.columns:
            pairplot_features = numeric_features[:4] + ['riskIndex']
            plt.figure(figsize=(12, 10))
            sample_df = df[pairplot_features].sample(n=min(500, len(df)), random_state=42)
            sns.pairplot(sample_df, diag_kind='kde', corner=True)
            plt.suptitle(f'Feature Relationships - Sensor {sensor_id}', y=1.02)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plots['pairplot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
        
        # 5. Time Series Plot (if timestamp exists)
        if 'timestamp' in df.columns and 'riskIndex' in df.columns:
            plt.figure(figsize=(15, 6))
            time_df = df.copy()
            time_df['timestamp'] = pd.to_datetime(time_df['timestamp'])
            time_df = time_df.sort_values('timestamp')
            
            plt.plot(time_df['timestamp'], time_df['riskIndex'], alpha=0.7, linewidth=1)
            plt.title(f'Risk Index Over Time - Sensor {sensor_id}')
            plt.xlabel('Timestamp')
            plt.ylabel('Risk Index')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plots['time_series'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
        
        return plots
        
    except Exception as e:
        logger.error(f"Error creating Seaborn plots for sensor {sensor_id}: {e}")
        return {"error": f"Seaborn plot creation failed: {str(e)}"}

def create_model_performance_plots(y_true, y_pred, sensor_id: str):
    """Create model performance visualization plots"""
    try:
        plots = {}
        
        # 1. Prediction vs Actual Scatter Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Risk Index')
        plt.ylabel('Predicted Risk Index')
        plt.title(f'Actual vs Predicted - Sensor {sensor_id}')
        
        # Add R² to plot
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plots['actual_vs_predicted'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        # 2. Residual Plot
        plt.figure(figsize=(10, 6))
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Risk Index')
        plt.ylabel('Residuals')
        plt.title(f'Residual Plot - Sensor {sensor_id}')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plots['residual_plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        # 3. Error Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title(f'Error Distribution - Sensor {sensor_id}')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plots['error_distribution'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return plots
        
    except Exception as e:
        logger.error(f"Error creating performance plots for sensor {sensor_id}: {e}")
        return {"error": f"Performance plot creation failed: {str(e)}"}

# ---------------------------------------------------
# Simplified Firebase Setup for Realtime Database
# ---------------------------------------------------
def initialize_firebase():
    """Initialize Firebase Realtime Database"""
    global firebase_db
    try:
        # Try to use service account file first
        cred_path = '/content/serviceAccountKey.json'
        if os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            logger.info("Using service account file from /content/serviceAccountKey.json")
        else:
            # Fallback to environment variables
            firebase_config = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
            if firebase_config:
                service_account_info = json.loads(firebase_config)
                cred = credentials.Certificate(service_account_info)
                logger.info("Using Firebase config from environment variable")
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
                cred = credentials.Certificate(service_account_info)
                logger.info("Using Firebase config from individual environment variables")
        
        database_url = os.getenv(
            "FIREBASE_DB_URL",
            "https://gasmonitoring-ec511-default-rtdb.asia-southeast1.firebasedatabase.app"
        )

        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {"databaseURL": database_url})
            firebase_db = db.reference()
            logger.info("Firebase Realtime Database initialized successfully")
        else:
            firebase_db = db.reference()
            logger.info("Firebase already initialized, using existing app")
            
    except Exception as e:
        logger.error(f"Firebase initialization failed: {e}")
        raise

def load_data_from_firebase(sensor_id: str):
    """Load data from Firebase Realtime Database /history/{sensor_id}"""
    try:
        # Get data from specific sensor path
        ref = db.reference(f'/history/{sensor_id}')
        data_dict = ref.get()
        
        if not data_dict:
            logger.warning(f"No data found in Firebase /history/{sensor_id}")
            return None
        
        # Convert the nested dictionary to DataFrame using orient='index'
        df = pd.DataFrame.from_dict(data_dict, orient='index')
        logger.info(f"Loaded {len(df)} records from Firebase /history/{sensor_id}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data from Firebase for sensor {sensor_id}: {e}")
        return None

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

def prepare_data(df, sensor_id: str):
    """Prepare data for training"""
    if df is None or df.empty:
        raise ValueError(f"No data available for sensor {sensor_id}")
    
    # Check if required columns exist
    if 'riskIndex' not in df.columns:
        raise ValueError(f"Required column 'riskIndex' not found in data for sensor {sensor_id}")
    
    # Handle timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Drop columns that are not useful or non-numeric
    columns_to_drop = ['riskIndex']
    optional_columns_to_drop = ['time', 'timestamp', 'apiUserID', 'apiPass', 'sensorID']
    
    for col in optional_columns_to_drop:
        if col in df.columns:
            columns_to_drop.append(col)
    
    X = df.drop(columns=columns_to_drop, errors='ignore')
    y = df['riskIndex']
    
    # Fill NaN values
    X = X.fillna(X.mean())
    
    logger.info(f"Prepared data for sensor {sensor_id}: X shape {X.shape}, y shape {y.shape}")
    logger.info(f"Features used: {X.columns.tolist()}")
    
    return X, y

def train_model_for_sensor(sensor_id: str, use_enhanced_gridsearch: bool = True):
    """Train XGBoost model for a specific sensor"""
    try:
        # Load data
        df = load_data_from_firebase(sensor_id)
        if df is None or df.empty:
            raise ValueError(f"No data available for sensor {sensor_id}")
        
        # Prepare data
        X, y = prepare_data(df, sensor_id)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if use_enhanced_gridsearch:
            # Enhanced training with comprehensive GridSearchCV
            xgb_model = XGBRegressor(random_state=42)
            
            param_grid = {
                'n_estimators': [100, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
            
            grid_search = GridSearchCV(
                estimator=xgb_model, 
                param_grid=param_grid,
                scoring='neg_root_mean_squared_error',
                cv=3, 
                verbose=1, 
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Make predictions and calculate metrics
            y_pred = best_model.predict(X_test)
            
            # Calculate metrics
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_mae = mean_absolute_error(y_test, y_pred)
            test_r2 = r2_score(y_test, y_pred)
            
            # Create visualizations
            shap_plots = create_shap_plots(best_model, X_train, sensor_id)
            seaborn_plots = create_seaborn_plots(df, sensor_id)
            performance_plots = create_model_performance_plots(y_test, y_pred, sensor_id)
            
            # Store metrics
            training_metrics[sensor_id] = {
                'best_params': grid_search.best_params_,
                'cv_rmse': -grid_search.best_score_,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'training_date': datetime.now().isoformat(),
                'sample_predictions': [
                    {'true': float(true_val), 'predicted': float(pred_val)} 
                    for true_val, pred_val in zip(y_test[:10], y_pred[:10])
                ],
                'visualizations_available': True
            }
            
            logger.info(f"Enhanced model trained for sensor {sensor_id}")
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best RMSE (CV): {-grid_search.best_score_:.6f}")
            logger.info(f"Test RMSE: {test_rmse:.6f}")
            logger.info(f"Test MAE: {test_mae:.6f}")
            logger.info(f"Test R² Score: {test_r2:.4f}")
            
        else:
            # Basic training with simple parameters
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
            best_model = grid_search.best_estimator_
            
            training_metrics[sensor_id] = {
                'best_params': grid_search.best_params_,
                'training_date': datetime.now().isoformat(),
                'model_type': 'basic',
                'visualizations_available': False
            }
            
            logger.info(f"Basic model trained for sensor {sensor_id}")
        
        # Store model and data
        models[sensor_id] = best_model
        data_cache[sensor_id] = df
        
        return best_model, training_metrics[sensor_id]
        
    except Exception as e:
        logger.error(f"Error training model for sensor {sensor_id}: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize Firebase on startup"""
    initialize_firebase()
    logger.info("API started - models will be trained on-demand via API endpoints")

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
        <p>Welcome to the <strong>Dynamic Sensor</strong> Gas Monitoring Backend. This API performs XGBoost training and predictions for multiple sensors.</p>
        <p><strong>CORS Status:</strong> ✅ Enabled for all origins</p>
        <p><strong>Database:</strong> ✅ Firebase Realtime Database (/history/{sensor_id})</p>
        <p><strong>Visualizations:</strong> ✅ SHAP, Seaborn, Performance Plots</p>

        <h2>API Documentation</h2>
        <ul>
            <li><a href="/docs">Swagger UI</a> (interactive API docs)</li>
            <li><a href="/redoc">ReDoc</a> (alternative docs)</li>
        </ul>

        <h2>Available Endpoints</h2>
        
        <div class="endpoint">
            <h3>🔧 Model Training</h3>
            <ul>
                <li><code>POST /train/{sensor_id}</code> - Train model for specific sensor</li>
                <li><code>GET /models</code> - List all trained models</li>
                <li><code>GET /models/{sensor_id}</code> - Get model info for sensor</li>
            </ul>
        </div>

        <div class="endpoint">
            <h3>📊 Visualizations</h3>
            <ul>
                <li><code>GET /shap/{sensor_id}</code> - SHAP explanation plots</li>
                <li><code>GET /plots/seaborn/{sensor_id}</code> - Seaborn data visualizations</li>
                <li><code>GET /plots/performance/{sensor_id}</code> - Model performance plots</li>
                <li><code>GET /plots/all/{sensor_id}</code> - All available visualizations</li>
            </ul>
        </div>

        <div class="endpoint">
            <h3>🔍 Predictions & Data</h3>
            <ul>
                <li><code>GET /explain/{sensor_id}</code> - Get predictions for a sensor</li>
                <li><code>GET /predictions/{sensor_id}</code> - Get stored predictions</li>
                <li><code>GET /data/{sensor_id}</code> - Get sensor data structure</li>
            </ul>
        </div>

        <div class="endpoint">
            <h3>📈 System Info</h3>
            <ul>
                <li><code>GET /health</code> - Health check</li>
                <li><code>GET /metrics/{sensor_id}</code> - Get training metrics</li>
            </ul>
        </div>

        <hr/>
        <p style="font-size: 0.9em; color: #666;">Powered by FastAPI, XGBoost, SHAP & Seaborn | Gas Monitoring Project</p>
    </body>
    </html>
    """

@app.post("/train/{sensor_id}")
def train_sensor_model(
    sensor_id: str,
    enhanced: bool = True,
    retrain: bool = False
):
    """Train or retrain model for a specific sensor"""
    try:
        # Check if model already exists
        if sensor_id in models and not retrain:
            return {
                "status": "exists",
                "message": f"Model for sensor {sensor_id} already exists. Use retrain=true to retrain.",
                "sensor_id": sensor_id
            }
        
        # Train the model
        model, metrics = train_model_for_sensor(sensor_id, use_enhanced_gridsearch=enhanced)
        
        return {
            "status": "success",
            "message": f"Model trained successfully for sensor {sensor_id}",
            "sensor_id": sensor_id,
            "model_type": "enhanced" if enhanced else "basic",
            "metrics": metrics,
            "visualizations_available": metrics.get('visualizations_available', False)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model for sensor {sensor_id}: {str(e)}")

# =========== VISUALIZATION ENDPOINTS ===========

@app.get("/shap/{sensor_id}")
def get_shap_plots(sensor_id: str):
    """Get SHAP explanation plots for a trained model"""
    if sensor_id not in models:
        raise HTTPException(status_code=404, detail=f"No trained model found for sensor {sensor_id}")
    
    if sensor_id not in data_cache:
        raise HTTPException(status_code=404, detail=f"No data available for sensor {sensor_id}")
    
    try:
        model = models[sensor_id]
        df = data_cache[sensor_id]
        
        # Prepare features
        X, _ = prepare_data(df, sensor_id)
        
        # Create SHAP plots
        shap_plots = create_shap_plots(model, X, sensor_id)
        
        return {
            "sensor_id": sensor_id,
            "shap_available": SHAP_AVAILABLE,
            "plots": shap_plots
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating SHAP plots: {str(e)}")

@app.get("/plots/seaborn/{sensor_id}")
def get_seaborn_plots(sensor_id: str):
    """Get Seaborn data visualization plots"""
    if sensor_id not in data_cache:
        # Try to load data if not in cache
        df = load_data_from_firebase(sensor_id)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for sensor {sensor_id}")
        data_cache[sensor_id] = df
    
    try:
        df = data_cache[sensor_id]
        seaborn_plots = create_seaborn_plots(df, sensor_id)
        
        return {
            "sensor_id": sensor_id,
            "plots": seaborn_plots
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating Seaborn plots: {str(e)}")

@app.get("/plots/performance/{sensor_id}")
def get_performance_plots(sensor_id: str):
    """Get model performance visualization plots"""
    if sensor_id not in models:
        raise HTTPException(status_code=404, detail=f"No trained model found for sensor {sensor_id}")
    
    try:
        # We need to recreate test predictions for performance plots
        df = data_cache[sensor_id]
        X, y = prepare_data(df, sensor_id)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = models[sensor_id]
        y_pred = model.predict(X_test)
        
        performance_plots = create_model_performance_plots(y_test, y_pred, sensor_id)
        
        return {
            "sensor_id": sensor_id,
            "plots": performance_plots
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating performance plots: {str(e)}")

@app.get("/plots/all/{sensor_id}")
def get_all_plots(sensor_id: str):
    """Get all available visualizations for a sensor"""
    if sensor_id not in models and sensor_id not in data_cache:
        raise HTTPException(status_code=404, detail=f"No model or data found for sensor {sensor_id}")
    
    try:
        plots = {}
        
        # Get Seaborn plots (always available if data exists)
        if sensor_id in data_cache:
            plots['seaborn'] = create_seaborn_plots(data_cache[sensor_id], sensor_id)
        
        # Get SHAP and performance plots (only if model exists)
        if sensor_id in models:
            df = data_cache[sensor_id]
            X, _ = prepare_data(df, sensor_id)
            plots['shap'] = create_shap_plots(models[sensor_id], X, sensor_id)
            
            # Performance plots
            X_train, X_test, y_train, y_test = train_test_split(X, df['riskIndex'], test_size=0.2, random_state=42)
            y_pred = models[sensor_id].predict(X_test)
            plots['performance'] = create_model_performance_plots(y_test, y_pred, sensor_id)
        
        return {
            "sensor_id": sensor_id,
            "visualizations": plots
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating all plots: {str(e)}")

# =========== EXISTING ENDPOINTS ===========

@app.get("/models")
def list_models():
    """List all trained models"""
    return {
        "trained_models": list(models.keys()),
        "total_models": len(models),
        "available_sensors": list(data_cache.keys())
    }

@app.get("/models/{sensor_id}")
def get_model_info(sensor_id: str):
    """Get information about a specific model"""
    if sensor_id not in models:
        raise HTTPException(status_code=404, detail=f"No model found for sensor {sensor_id}")
    
    metrics = training_metrics.get(sensor_id, {})
    
    return {
        "sensor_id": sensor_id,
        "model_available": True,
        "training_metrics": metrics,
        "data_points": len(data_cache[sensor_id]) if sensor_id in data_cache else 0,
        "visualizations_available": metrics.get('visualizations_available', False)
    }

@app.get("/explain/{sensor_id}")
def explain(sensor_id: str):
    """Get predictions for a specific sensor"""
    if sensor_id not in models:
        raise HTTPException(status_code=404, detail=f"No trained model found for sensor {sensor_id}. Train the model first using POST /train/{sensor_id}")
    
    if sensor_id not in data_cache:
        raise HTTPException(status_code=404, detail=f"No data available for sensor {sensor_id}")
    
    model = models[sensor_id]
    sensor_data = data_cache[sensor_id]
    
    # Prepare features for prediction
    columns_to_drop = ['riskIndex']
    optional_columns_to_drop = ['time', 'timestamp', 'apiUserID', 'apiPass', 'sensorID']
    
    for col in optional_columns_to_drop:
        if col in sensor_data.columns:
            columns_to_drop.append(col)
    
    X_sensor = sensor_data.drop(columns=columns_to_drop, errors='ignore')
    X_sensor = X_sensor.fillna(X_sensor.mean())

    # Predict riskIndex for the sensor data
    predictions = model.predict(X_sensor)

    # Build response with timestamps and predictions
    response = []
    