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
import shap
from scipy import stats
# Configure logging FIRST
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global models and data storage (in-memory)
models = {}  # Store models by sensor_id: {sensor_id: model}
data_cache = {}  # Store data by sensor_id: {sensor_id: data}
training_metrics = {}  # Store training metrics by sensor_id
shap_explainers = {}  # Store SHAP explainers by sensor_id
firebase_db = None
 
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
 

# ---------------------------------------------------
# Visualization Functions
# ---------------------------------------------------
def fetch_history(
    sensor_ID,
    range: str = Query("1month", description="Date range: 1week, 1month, 3months, 6months, 1year, all")
):
  ref = db.reference(f'/history/{sensor_ID}')
  data = ref.get()
  df = pd.DataFrame.from_dict(data, orient='index') 
  df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')     
    # Filter data based on the range parameter
  if range != "all":
      current_date = datetime.now()
      
      if range == "1week":
          start_date = current_date - timedelta(weeks=1)
      elif range == "1month":
          start_date = current_date - timedelta(days=30)
      elif range == "3months":
          start_date = current_date - timedelta(days=90)
      elif range == "6months":
          start_date = current_date - timedelta(days=180)
      elif range == "1year":
          start_date = current_date - timedelta(days=365)
      else:
          start_date = df['timestamp'].min()
        
        # Filter the dataframe
      df = df[df['timestamp'] >= start_date]


  return df

def xgboost_model(
    sensor_ID,
    range: str = Query("1month", description="Date range: 1week, 1month, 3months, 6months, 1year, all")
):

  records = []
  result_predict_test = []
  ref = db.reference(f'/history/{sensor_ID}')
  data = ref.get()
  df = pd.DataFrame.from_dict(data, orient='index') 
  # print(data)

  df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')     
    # Filter data based on the range parameter
  if range != "all":
      current_date = datetime.now()
      
      if range == "1week":
          start_date = current_date - timedelta(weeks=1)
      elif range == "1month":
          start_date = current_date - timedelta(days=30)
      elif range == "3months":
          start_date = current_date - timedelta(days=90)
      elif range == "6months":
          start_date = current_date - timedelta(days=180)
      elif range == "1year":
          start_date = current_date - timedelta(days=365)
      else:
          start_date = df['timestamp'].min()
        
        # Filter the dataframe
      df = df[df['timestamp'] >= start_date]
  # Drop columns that are not useful or non-numeric for now
  features = df.drop(columns=['riskIndex', 'time', 'timestamp', 'apiUserID','apiPass', 'sensorID'])
  target = df['riskIndex']

    
  X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
  xgb = XGBRegressor(random_state=42)

  param_grid = {
      'n_estimators': [100, 300],
      'max_depth': [3, 5, 7],
      'learning_rate': [0.01, 0.1],
      'subsample': [0.8, 1.0],
      'colsample_bytree': [0.8, 1.0]
  }

  grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid,
                            scoring='neg_root_mean_squared_error',
                            cv=3, verbose=2, n_jobs=-1)
  grid_search.fit(X_train, y_train)

   # Predict on test set
  best_model = grid_search.best_estimator_
  y_pred = best_model.predict(X_test) 

  # Assuming you kept indices aligned
  test_results = X_test.copy()
  test_results['true_riskIndex'] = y_test
  test_results['predicted_riskIndex'] = y_pred

  result_predict_test = { 
     "TestRMSE" : np.sqrt(mean_squared_error(y_test, y_pred)),
     "Test MAE":mean_absolute_error(y_test, y_pred),
     "Test r_score": r2_score(y_test, y_pred)
  }

  result = {
     "BP" : grid_search.best_params_,
     "RMSE":-grid_search.best_score_,
     "test_result": result_predict_test,
     "Predicted":test_results.head(),
     "data_info": {
            "total_records": len(df),
            "training_records": len(X_train),
            "test_records": len(X_test),
            "date_range_used": f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}",
            "selected_range": range
        },
      
  }


  return result
  def plot_correlation_heatmap(df, figsize=(12, 10)):
    """Display only correlation heatmap with proper error handling"""
    try:
        # Ensure figsize is valid
        if figsize is None:
            figsize = (12, 10)
            print("Using default figsize (12, 10)")
        else:
            # Convert to tuple of floats
            try:
                figsize = (float(figsize[0]), float(figsize[1]))
            except (TypeError, ValueError, IndexError):
                print("Warning: Invalid figsize, using default (12, 10)")
                figsize = (12, 10)
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            print("No numeric columns found for correlation heatmap")
            return None
        
        # Check if we have enough data for correlations
        if len(numeric_df.columns) < 2:
            print("Need at least 2 numeric columns for correlation heatmap")
            return None
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create the plot
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, 
                    mask=mask,
                    annot=True, 
                    cmap='RdBu_r', 
                    center=0,
                    square=True,
                    fmt='.2f',
                    linewidths=0.5,
                    cbar_kws={"shrink": .8})
        
        plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return corr_matrix
        
    except Exception as e:
        print(f"Error creating correlation heatmap: {str(e)}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return None
        

def plot_target_correlations(df, target_col='riskIndex', top_n=10):
    """Display target correlations bar chart"""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if target_col not in numeric_df.columns:
        print(f"Target column '{target_col}' not found")
        return None
    
    corr_matrix = numeric_df.corr()
    target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
    
    # Take top N correlations (positive and negative)
    top_positive = target_corr.head(top_n//2)
    top_negative = target_corr.tail(top_n//2)
    top_correlations = pd.concat([top_positive, top_negative])
    
    plt.figure(figsize=(10, 8))
    
    # Create a color mapping based on correlation values
    colors = ['red' if x > 0 else 'blue' for x in top_correlations.values]
    
    # Updated barplot without palette warning
    bars = plt.barh(top_correlations.index, top_correlations.values, color=colors)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.title(f' Feature Correlations with {target_col}', fontsize=14, fontweight='bold')
    plt.xlabel('Correlation Coefficient')
    
    # Add value annotations on bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', 
                ha='left' if width > 0 else 'right', 
                va='center',
                fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return target_corr

def plot_correlation_scatterplots(df, target_col='riskIndex', top_n=6):
    """Simple version - plot each figure separately"""
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        
        if target_col not in numeric_df.columns:
            print(f"Target column '{target_col}' not found")
            return None
        
        corr_matrix = numeric_df.corr()
        target_corr = corr_matrix[target_col].drop(target_col)
        top_features = target_corr.abs().nlargest(top_n).index.tolist()
        
        # Count features without len
        n_plots = 0
        for _ in top_features:
            n_plots += 1
        
        if n_plots == 0:
            print("No features to plot")
            return None
        
        # Plot each feature in its own figure
        figures = []
        for feature in top_features:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=df, x=feature, y=target_col, alpha=0.6, ax=ax)
            correlation_val = target_corr[feature]
            ax.set_title(f'{feature} vs {target_col}\nCorrelation: {correlation_val:.3f}')
            ax.set_xlabel(feature)
            ax.set_ylabel(target_col)
            plt.tight_layout()
            plt.show()
            figures.append(fig)
        
        return figures
        
    except Exception as e:
        print(f"Error creating scatter plots: {str(e)}")
        return None
        
def print_correlation_summary(df, target_col='riskIndex'):
    """Print numerical correlation summary"""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if target_col not in numeric_df.columns:
        print(f"Target column '{target_col}' not found")
        return None
    
    corr_matrix = numeric_df.corr()
    target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
    
    print("="*60)
    print("CORRELATION ANALYSIS SUMMARY")
    print("="*60)
    print(f"Target Variable: {target_col}")
    print(f"Total Features Analyzed: {len(target_corr)}")
    print("\nTOP POSITIVE CORRELATIONS:")
    print("-" * 30)
    for feature, corr in target_corr.head(5).items():
        print(f"  {feature:20} : {corr:+.3f}")
    
    print("\nTOP NEGATIVE CORRELATIONS:")
    print("-" * 30)
    for feature, corr in target_corr.tail(5).items():
        print(f"  {feature:20} : {corr:+.3f}")
    
    print("\nCORRELATION STRENGTH BREAKDOWN:")
    print("-" * 35)
    print(f"  Strong Positive (>0.7)   : {len(target_corr[target_corr > 0.7]):2d} features")
    print(f"  Moderate Positive (0.3-0.7): {len(target_corr[(target_corr > 0.3) & (target_corr <= 0.7)]):2d} features")
    print(f"  Weak (-0.3 to 0.3)       : {len(target_corr[(target_corr >= -0.3) & (target_corr <= 0.3)]):2d} features")
    print(f"  Moderate Negative (-0.7 to -0.3): {len(target_corr[(target_corr >= -0.7) & (target_corr < -0.3)]):2d} features")
    print(f"  Strong Negative (<-0.7)  : {len(target_corr[target_corr < -0.7]):2d} features")
    print("="*60)
    
    return target_corr

# ---------------------------------------------------
# Data & System Endpoints
# --------------------------------------------------- 

@app.get("/api/xgboost/{sensor_id}")
def get_xgboost_model(
    sensor_id: str,
    range: str = Query("1month", description="Date range: 1week, 1month, 3months, 6months, 1year, all")
):
  result = xgboost_model(sensor_id,range)
  return result

@app.get("/api/correlation_summary/{sensor_id}")
def getprint_correlation_summary(
    sensor_id: str,
    range: str = Query("1month", description="Date range: 1week, 1month, 3months, 6months, 1year, all"),
    target_col: str = Query("riskIndex", description="Target column for correlation analysis")
):
  df = fetch_history(sensor_id, range) 
  result =  print_correlation_summary(df, target_col) 
  return result

@app.get("/api/correlation_scatterplots/{sensor_id}")
def getplot_correlation_scatterplots(
    sensor_id: str,
    range: str = Query("1month", description="Date range: 1week, 1month, 3months, 6months, 1year, all"),
    target_col: str = Query("riskIndex", description="Target column for correlation analysis")
):
  df = fetch_history(sensor_id, range) 
  result =  plot_correlation_scatterplots(df, target_col) 
  return result

@app.get("/api/correlations/{sensor_id}")
def getplot_target_correlations(
    sensor_id: str,
    range: str = Query("1month", description="Date range: 1week, 1month, 3months, 6months, 1year, all"),
    target_col: str = Query("riskIndex", description="Target column for correlation analysis")
):
  df = fetch_history(sensor_id, range) 
  result = plot_target_correlations(df, target_col) 
  return result
@app.get("/api/correlations_heatmap/{sensor_id}")
def getplot_correlation_heatmap(
    sensor_id: str,
    range: str = Query("1month", description="Date range: 1week, 1month, 3months, 6months, 1year, all"),
    target_col: str = Query("riskIndex", description="Target column for correlation heatmap")
):
  df = fetch_history(sensor_id, range) 
  result = plot_correlation_heatmap(df) 
  return result

#______________________________________________________
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
             
            <h4>Query Parameters:</h4>
             <div class="card">
                <h2>Dataset Overview</h2>
                <div class="stats">
                    <div class="stat-item">Total Records: {{ dataset_info.total_records }}</div>
                    <div class="stat-item">Total Features: {{ dataset_info.total_features }}</div>
                    <div class="stat-item">Numeric Features: {{ dataset_info.numeric_features }}</div>
                    <div class="stat-item">Categorical Features: {{ dataset_info.categorical_features }}</div>
                    <div class="stat-item">Missing Values: {{ dataset_info.missing_values }}</div>
                    <div class="stat-item">Duplicate Rows: {{ dataset_info.duplicate_rows }}</div>
                </div>
            </div>

            <div class="card">
                <h2>Sample Data</h2>
                <table>
                    <thead>
                        <tr>
                            {% for column in column_names %}
                            <th>{{ column }}</th>
                            {% endfor %}
                        </tr>
                    </thead>
                    <tbody>
                        {% for row in sample_data %}
                        <tr>
                            {% for column in column_names %}
                            <td>{{ row[column] }}</td>
                            {% endfor %}
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
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