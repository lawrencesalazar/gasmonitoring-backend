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
    """Create SHAP explanation plots with compatible function calls"""
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
        
        # 3. SHAP Beeswarm Plot
        plt.figure(figsize=(12, 8))
        shap.plots.beeswarm(shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=X), show=False)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plots['beeswarm_plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        # 4. Feature Importance Bar Plot
        plt.figure(figsize=(12, 8))
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(shap_values).mean(0)
        }).sort_values('importance', ascending=True)
        
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.xlabel('Mean |SHAP value|')
        plt.title(f'Feature Importance - Sensor {sensor_id}')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plots['feature_importance'] = base64.b64encode(buf.getvalue()).decode('utf-8')
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
# Firebase Setup
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

# ---------------------------------------------------
# Main XGBoost Prediction Endpoint
# ---------------------------------------------------
@app.get("/predict/xgboost/")
def xgboost_regressor_predict(
    sensor_id: str = Query(..., description="Sensor ID to train model for"),
    test_size: float = Query(0.2, description="Test size ratio (0.1 to 0.4)", ge=0.1, le=0.4),
    n_estimators: str = Query("100,300", description="n_estimators values (comma separated)"),
    max_depth: str = Query("3,5,7", description="max_depth values (comma separated)"),
    learning_rate: str = Query("0.01,0.1", description="learning_rate values (comma separated)"),
    subsample: str = Query("0.8,1.0", description="subsample values (comma separated)"),
    colsample_bytree: str = Query("0.8,1.0", description="colsample_bytree values (comma separated)"),
    cv_folds: int = Query(3, description="Number of cross-validation folds", ge=2, le=10),
    random_state: int = Query(42, description="Random state for reproducibility")
):
    """XGBoost Regressor with dynamic GridSearchCV parameters via GET request"""
    try:
        # Parse parameter strings to lists
        n_estimators_list = [int(x.strip()) for x in n_estimators.split(',')]
        max_depth_list = [int(x.strip()) for x in max_depth.split(',')]
        learning_rate_list = [float(x.strip()) for x in learning_rate.split(',')]
        subsample_list = [float(x.strip()) for x in subsample.split(',')]
        colsample_bytree_list = [float(x.strip()) for x in colsample_bytree.split(',')]
        
        # Load data from Firebase
        ref = db.reference(f'/history/{sensor_id}')
        data = ref.get()
        
        if not data:
            raise HTTPException(status_code=404, detail=f"No data found for sensor {sensor_id}")
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(data, orient='index')
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Empty dataset for sensor {sensor_id}")
        
        # Prepare data
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce') 
        
        # Drop columns that are not useful or non-numeric for now
        features = df.drop(columns=['riskIndex', 'time', 'timestamp', 'apiUserID', 'apiPass', 'sensorID'], errors='ignore')
        target = df['riskIndex']
        
        # Check if we have enough data
        if len(features) < 10:
            raise HTTPException(status_code=400, detail=f"Insufficient data for sensor {sensor_id}. Need at least 10 samples.")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=random_state
        )
        
        # Initialize and train XGBoost with GridSearch
        xgb = XGBRegressor(random_state=random_state)
        
        param_grid = {
            'n_estimators': n_estimators_list,
            'max_depth': max_depth_list,
            'learning_rate': learning_rate_list,
            'subsample': subsample_list,
            'colsample_bytree': colsample_bytree_list
        }
        
        grid_search = GridSearchCV(
            estimator=xgb, 
            param_grid=param_grid,
            scoring='neg_root_mean_squared_error',
            cv=cv_folds, 
            verbose=1, 
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model and predictions
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store model and metrics
        models[sensor_id] = best_model
        training_metrics[sensor_id] = {
            'best_params': grid_search.best_params_,
            'cv_rmse': -grid_search.best_score_,
            'test_rmse': rmse,
            'test_mae': mae,
            'test_r2': r2,
            'training_date': datetime.now().isoformat(),
            'model_type': 'xgboost_regressor',
            'parameters_used': {
                'test_size': test_size,
                'cv_folds': cv_folds,
                'random_state': random_state
            }
        }
        
        # Sample predictions for inspection
        sample_predictions = []
        for true_val, pred_val in zip(y_test[:10], y_pred[:10]):
            sample_predictions.append({
                "true": float(true_val),
                "predicted": float(pred_val),
                "error": float(abs(true_val - pred_val))
            })
        
        # Feature importance
        feature_importance = dict(zip(features.columns, best_model.feature_importances_))
        sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])
        
        return {
            "sensor_id": sensor_id,
            "status": "success",
            "data_info": {
                "total_samples": len(df),
                "training_samples": len(X_train),
                "testing_samples": len(X_test),
                "features_used": list(features.columns)
            },
            "grid_search_parameters": param_grid,
            "best_parameters": grid_search.best_params_,
            "performance_metrics": {
                "cross_validation": {
                    "RMSE": round(-grid_search.best_score_, 6)
                },
                "test_set": {
                    "RMSE": round(rmse, 6),
                    "MAE": round(mae, 6),
                    "R2_Score": round(r2, 4)
                }
            },
            "top_features": sorted_importance,
            "sample_predictions": sample_predictions,
            "training_details": {
                "test_size": test_size,
                "cv_folds": cv_folds,
                "random_state": random_state
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in xgboost_regressor_predict for sensor {sensor_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ---------------------------------------------------
# Visualization Endpoints
# ---------------------------------------------------
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
        features = df.drop(columns=['riskIndex', 'time', 'timestamp', 'apiUserID', 'apiPass', 'sensorID'], errors='ignore')
        features = features.fillna(features.mean())
        
        # Create SHAP plots
        shap_plots = create_shap_plots(model, features, sensor_id)
        
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
        features = df.drop(columns=['riskIndex', 'time', 'timestamp', 'apiUserID', 'apiPass', 'sensorID'], errors='ignore')
        features = features.fillna(features.mean())
        target = df['riskIndex']
        
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        
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
            features = df.drop(columns=['riskIndex', 'time', 'timestamp', 'apiUserID', 'apiPass', 'sensorID'], errors='ignore')
            features = features.fillna(features.mean())
            plots['shap'] = create_shap_plots(models[sensor_id], features, sensor_id)
            
            # Performance plots
            target = df['riskIndex']
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
            y_pred = models[sensor_id].predict(X_test)
            plots['performance'] = create_model_performance_plots(y_test, y_pred, sensor_id)
        
        return {
            "sensor_id": sensor_id,
            "visualizations": plots
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating all plots: {str(e)}")

# ---------------------------------------------------
# Data & System Endpoints
# ---------------------------------------------------
@app.get("/data/{sensor_id}")
def get_sensor_data(sensor_id: str, limit: int = 5):
    """Get sensor data structure"""
    try:
        # Load data if not in cache
        if sensor_id not in data_cache:
            df = load_data_from_firebase(sensor_id)
            if df is None or df.empty:
                raise HTTPException(status_code=404, detail=f"No data found for sensor {sensor_id}")
            data_cache[sensor_id] = df
        else:
            df = data_cache[sensor_id]
        
        return {
            "sensor_id": sensor_id,
            "columns": df.columns.tolist(),
            "shape": df.shape,
            "sample_data": df.head(limit).to_dict('records')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data for sensor {sensor_id}: {str(e)}")

@app.get("/models")
def list_models():
    """List all trained models"""
    return {
        "trained_models": list(models.keys()),
        "total_models": len(models),
        "available_sensors": list(data_cache.keys())
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy", 
        "message": "Dynamic XGBoost API with Firebase integration is running",
        "trained_models": len(models),
        "cached_datasets": len(data_cache),
        "timestamp": datetime.now().isoformat()
    }
    return status

@app.get("/metrics/{sensor_id}")
def get_training_metrics(sensor_id: str):
    """Get training metrics for a sensor"""
    if sensor_id not in training_metrics:
        raise HTTPException(status_code=404, detail=f"No training metrics found for sensor {sensor_id}")
    
    return {
        "sensor_id": sensor_id,
        "metrics": training_metrics[sensor_id]
    }

# ---------------------------------------------------
# Home Page
# ---------------------------------------------------
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
            .param { color: #d63384; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Gas Monitoring XGBoost API</h1>
        <p>Welcome to the <strong>Dynamic Sensor</strong> Gas Monitoring Backend with dynamic XGBoost training.</p>
        <p><strong>CORS Status:</strong> ✅ Enabled for all origins</p>
        <p><strong>Database:</strong> ✅ Firebase Realtime Database (/history/{sensor_id})</p>

        <h2>API Documentation</h2>
        <ul>
            <li><a href="/docs">Swagger UI</a> (interactive API docs)</li>
            <li><a href="/redoc">ReDoc</a> (alternative docs)</li>
        </ul>

        <h2>Main Endpoint</h2>
        
        <div class="endpoint new">
            <h3>🎯 Dynamic XGBoost Training & Prediction</h3>
            <code>GET /predict/xgboost/?sensor_id=3221&test_size=0.2&n_estimators=100,300&max_depth=3,5,7&learning_rate=0.01,0.1&subsample=0.8,1.0&colsample_bytree=0.8,1.0&cv_folds=3&random_state=42</code>
            
            <h4>Query Parameters:</h4>
            <ul>
                <li><span class="param">sensor_id</span> (required) - Sensor ID to train model for</li>
                <li><span class="param">test_size</span> (default: 0.2) - Test size ratio (0.1 to 0.4)</li>
                <li><span class="param">n_estimators</span> (default: "100,300") - n_estimators values (comma separated)</li>
                <li><span class="param">max_depth</span> (default: "3,5,7") - max_depth values (comma separated)</li>
                <li><span class="param">learning_rate</span> (default: "0.01,0.1") - learning_rate values (comma separated)</li>
                <li><span class="param">subsample</span> (default: "0.8,1.0") - subsample values (comma separated)</li>
                <li><span class="param">colsample_bytree</span> (default: "0.8,1.0") - colsample_bytree values (comma separated)</li>
                <li><span class="param">cv_folds</span> (default: 3) - Number of cross-validation folds (2-10)</li>
                <li><span class="param">random_state</span> (default: 42) - Random state for reproducibility</li>
            </ul>
            
            <h4>Example Usage:</h4>
            <pre>
GET /predict/xgboost/?sensor_id=3221&test_size=0.2
GET /predict/xgboost/?sensor_id=3221&n_estimators=50,100,200&max_depth=2,4,6
GET /predict/xgboost/?sensor_id=3221&learning_rate=0.05,0.1,0.2&cv_folds=5
            </pre>
        </div>

        <div class="endpoint">
            <h3>📊 Visualization Endpoints</h3>
            <ul>
                <li><code>GET /shap/{sensor_id}</code> - SHAP explanation plots</li>
                <li><code>GET /plots/seaborn/{sensor_id}</code> - Seaborn data visualizations</li>
                <li><code>GET /plots/performance/{sensor_id}</code> - Model performance plots</li>
                <li><code>GET /plots/all/{sensor_id}</code> - All available visualizations</li>
            </ul>
        </div>

        <div class="endpoint">
            <h3>🔍 Data & System</h3>
            <ul>
                <li><code>GET /data/{sensor_id}</code> - Get sensor data structure</li>
                <li><code>GET /models</code> - List all trained models</li>
                <li><code>GET /health</code> - Health check</li>
                <li><code>GET /metrics/{sensor_id}</code> - Get training metrics</li>
            </ul>
        </div>

        <hr/>
        <p style="font-size: 0.9em; color: #666;">Powered by FastAPI & XGBoost | Gas Monitoring Project</p>
    </body>
    </html>
    """

@app.on_event("startup")
async def startup_event():
    """Initialize Firebase on startup"""
    initialize_firebase()
    logger.info("API started - use GET /predict/xgboost/ to train models")

# ===========
# Run the application
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)