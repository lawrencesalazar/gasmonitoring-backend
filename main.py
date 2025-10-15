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
 
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import threading
import pickle
import base64
import joblib

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
# Improved Firebase Setup
# ---------------------------------------------------
def initialize_firebase():
    """Initialize Firebase Realtime Database with better error handling"""
    global firebase_db
    try:
        # Check if Firebase is already initialized
        if firebase_db is not None:
            logger.info("Firebase already initialized")
            return True

        # Method 1: Check for service account JSON string in environment
        firebase_config_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
        if firebase_config_json:
            try:
                service_account_info = json.loads(firebase_config_json)
                cred = credentials.Certificate(service_account_info)
                logger.info("Using Firebase config from FIREBASE_SERVICE_ACCOUNT environment variable")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in FIREBASE_SERVICE_ACCOUNT: {e}")
                return False
        else:
            # Method 2: Individual environment variables
            private_key = os.getenv("FIREBASE_PRIVATE_KEY", "")
            if private_key:
                # Handle newline characters in private key
                private_key = private_key.replace('\\n', '\n')
            
            service_account_info = {
                "type": "service_account",
                "project_id": os.getenv("FIREBASE_PROJECT_ID"),
                "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
                "private_key": private_key,
                "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
                "client_id": os.getenv("FIREBASE_CLIENT_ID"),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_CERT_URL")
            }
            
            # Validate required fields
            required_fields = ["project_id", "private_key", "client_email"]
            for field in required_fields:
                if not service_account_info.get(field):
                    logger.error(f"Missing required Firebase config field: {field}")
                    return False
            
            cred = credentials.Certificate(service_account_info)
            logger.info("Using Firebase config from individual environment variables")

        database_url = os.getenv(
            "FIREBASE_DB_URL",
            "https://gasmonitoring-ec511-default-rtdb.asia-southeast1.firebasedatabase.app"
        )

        # Initialize Firebase app
        if not firebase_admin._apps:
            firebase_app = firebase_admin.initialize_app(cred, {
                "databaseURL": database_url
            })
            logger.info(f"Firebase app initialized: {firebase_app.name}")
        else:
            logger.info("Using existing Firebase app")
        
        # Get database reference
        firebase_db = db.reference()
        logger.info("Firebase Realtime Database initialized successfully")
        logger.info(f"Database URL: {database_url}")
        
        return True
        
    except Exception as e:
        logger.error(f"Firebase initialization failed: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        
        # Log environment variables (without sensitive values)
        env_info = {
            "FIREBASE_PROJECT_ID": bool(os.getenv("FIREBASE_PROJECT_ID")),
            "FIREBASE_CLIENT_EMAIL": bool(os.getenv("FIREBASE_CLIENT_EMAIL")),
            "FIREBASE_PRIVATE_KEY": bool(os.getenv("FIREBASE_PRIVATE_KEY")),
            "FIREBASE_PRIVATE_KEY_ID": bool(os.getenv("FIREBASE_PRIVATE_KEY_ID")),
            "FIREBASE_SERVICE_ACCOUNT": bool(os.getenv("FIREBASE_SERVICE_ACCOUNT")),
        }
        logger.info(f"Firebase environment variables: {env_info}")
        
        firebase_db = None
        return False

# ---------------------------------------------------
# Enhanced Email Config with Firebase Integration
# ---------------------------------------------------
class EmailConfig:
    """Email configuration for alerts with Firebase integration"""
    def __init__(self):
        self._config_cache = None
        self._cache_timestamp = None
        self._cache_timeout = 300  # 5 minutes cache
        self._recipient_cache = {}
        self._recipient_cache_time = {}
        
        # Load initial config from Firebase or environment
        self._load_initial_config()
    
    def _load_initial_config(self):
        """Load configuration from Firebase or fallback to environment variables"""
        try:
            fb_config = self._get_config_from_firebase()
            if fb_config:
                self._apply_config(fb_config)
                logger.info("Email configuration loaded from Firebase")
            else:
                self._apply_config(self._get_config_from_env())
                logger.info("Email configuration loaded from environment variables")
        except Exception as e:
            logger.error(f"Error loading email config: {e}")
            self._apply_config(self._get_config_from_env())
    
    def _get_config_from_env(self):
        """Get configuration from environment variables"""
        return {
            "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
            "smtp_port": int(os.getenv("SMTP_PORT", "587")),
            "sender_email": os.getenv("SENDER_EMAIL", ""),
            "sender_password": os.getenv("SENDER_PASSWORD", ""),
            "enable_ssl": os.getenv("SMTP_SSL", "False").lower() == "true",
            "config_source": "environment"
        }
    
    def _get_config_from_firebase(self):
        """Get configuration from Firebase"""
        try:
            if not firebase_db:
                return None
                
            ref = db.reference('/email_config/default')
            config = ref.get()
            
            if config and isinstance(config, dict):
                config['config_source'] = 'firebase'
                return config
            return None
        except Exception as e:
            logger.error(f"Error fetching email config from Firebase: {e}")
            return None
    
    def _save_config_to_firebase(self, config_data):
        """Save configuration to Firebase"""
        try:
            if not firebase_db:
                logger.error("Firebase not initialized - cannot save email config")
                return False
            
            # Remove sensitive data and source before saving
            save_data = config_data.copy()
            if 'sender_password' in save_data:
                # Only save if password is not masked
                if save_data['sender_password'] != '********':
                    # Encrypt or mask password before saving (basic masking)
                    save_data['sender_password'] = save_data['sender_password']
                else:
                    # Don't update password if it's masked
                    del save_data['sender_password']
            
            if 'config_source' in save_data:
                del save_data['config_source']
            
            ref = db.reference('/email_config/default')
            ref.set(save_data)
            
            # Update cache
            self._config_cache = config_data
            self._cache_timestamp = datetime.now()
            
            logger.info("Email configuration saved to Firebase")
            return True
        except Exception as e:
            logger.error(f"Error saving email config to Firebase: {e}")
            return False
    
    def _apply_config(self, config):
        """Apply configuration to instance variables"""
        self.smtp_server = config.get("smtp_server", "smtp.gmail.com")
        self.smtp_port = int(config.get("smtp_port", 587))
        self.sender_email = config.get("sender_email", "")
        self.sender_password = config.get("sender_password", "")
        self.enable_ssl = config.get("enable_ssl", False)
        self.config_source = config.get("config_source", "unknown")
    
    def get_current_config(self, mask_sensitive=True):
        """Get current configuration"""
        config = {
            "smtp_server": self.smtp_server,
            "smtp_port": self.smtp_port,
            "sender_email": self.sender_email,
            "sender_password": "********" if mask_sensitive and self.sender_password else self.sender_password,
            "enable_ssl": self.enable_ssl,
            "config_source": self.config_source,
            "fully_configured": self.is_configured(),
            "last_updated": self._cache_timestamp.isoformat() if self._cache_timestamp else None
        }
        return config
    
    def update_config(self, new_config, confirm=True):
        """
        Update email configuration with optional confirmation
        
        Args:
            new_config: Dictionary with new configuration values
            confirm: If True, will only update if configuration is valid and confirmed
        """
        try:
            # Validate required fields
            required_fields = ['smtp_server', 'smtp_port', 'sender_email']
            for field in required_fields:
                if field not in new_config or not new_config[field]:
                    return {
                        "success": False,
                        "error": f"Missing required field: {field}",
                        "current_config": self.get_current_config()
                    }
            
            # Validate port
            try:
                port = int(new_config['smtp_port'])
                if not (1 <= port <= 65535):
                    return {
                        "success": False,
                        "error": "SMTP port must be between 1 and 65535",
                        "current_config": self.get_current_config()
                    }
            except ValueError:
                return {
                    "success": False,
                    "error": "SMTP port must be a valid number",
                    "current_config": self.get_current_config()
                }
            
            # Validate email format
            if '@' not in new_config['sender_email']:
                return {
                    "success": False,
                    "error": "Invalid sender email format",
                    "current_config": self.get_current_config()
                }
            
            # If confirm=True, return the proposed changes for confirmation
            if confirm:
                proposed_changes = {}
                current_config = self.get_current_config(mask_sensitive=False)
                
                for key, new_value in new_config.items():
                    if key in current_config and current_config[key] != new_value:
                        if key == 'sender_password':
                            proposed_changes[key] = {
                                "old": "********",
                                "new": "********" if new_value else "[empty]"
                            }
                        else:
                            proposed_changes[key] = {
                                "old": current_config[key],
                                "new": new_value
                            }
                
                if not proposed_changes:
                    return {
                        "success": False,
                        "error": "No changes detected",
                        "current_config": current_config
                    }
                
                return {
                    "success": True,
                    "confirmation_required": True,
                    "proposed_changes": proposed_changes,
                    "current_config": self.get_current_config(),
                    "message": "Please confirm these changes before applying"
                }
            
            # If no confirmation required, apply changes directly
            updated_config = self.get_current_config(mask_sensitive=False)
            updated_config.update(new_config)
            
            # Save to Firebase
            if self._save_config_to_firebase(updated_config):
                self._apply_config(updated_config)
                return {
                    "success": True,
                    "confirmation_required": False,
                    "message": "Email configuration updated successfully",
                    "new_config": self.get_current_config()
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to save configuration to Firebase",
                    "current_config": self.get_current_config()
                }
                
        except Exception as e:
            logger.error(f"Error updating email config: {e}")
            return {
                "success": False,
                "error": f"Configuration update failed: {str(e)}",
                "current_config": self.get_current_config()
            }
    
    def confirm_update(self, confirmation_data):
        """
        Confirm and apply previously proposed changes
        """
        try:
            if 'proposed_changes' not in confirmation_data:
                return {
                    "success": False,
                    "error": "No proposed changes found in confirmation data"
                }
            
            # Reconstruct the full config from proposed changes
            current_config = self.get_current_config(mask_sensitive=False)
            updated_config = current_config.copy()
            
            for key, change in confirmation_data['proposed_changes'].items():
                if key in updated_config:
                    updated_config[key] = change['new']
            
            # Save to Firebase
            if self._save_config_to_firebase(updated_config):
                self._apply_config(updated_config)
                return {
                    "success": True,
                    "message": "Email configuration updated successfully",
                    "new_config": self.get_current_config()
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to save configuration to Firebase"
                }
                
        except Exception as e:
            logger.error(f"Error confirming email config update: {e}")
            return {
                "success": False,
                "error": f"Configuration confirmation failed: {str(e)}"
            }
    
    def is_configured(self):
        """Check if basic email configuration is complete"""
        return bool(self.sender_email and self.sender_password)
    
    def get_config_status(self):
        """Get detailed configuration status"""
        return {
            "smtp_server": self.smtp_server,
            "smtp_port": self.smtp_port,
            "sender_email": self.sender_email,
            "sender_password_set": bool(self.sender_password),
            "sender_password_length": len(self.sender_password),
            "enable_ssl": self.enable_ssl,
            "fully_configured": self.is_configured(),
            "config_source": self.config_source
        }
    
    def get_recipients_for_sensor(self, sensor_id: str) -> List[str]:
        """
        Fetch recipient emails from Firebase for a specific sensor
        Structure in Firebase:
        /alert_recipients/{sensor_id}/emails: ["email1@com", "email2@com"]
        OR
        /alert_recipients/global/emails: ["email1@com", "email2@com"] (fallback)
        """
        try:
            # Check cache first (cache for 5 minutes)
            current_time = datetime.now()
            if (sensor_id in self._recipient_cache and 
                sensor_id in self._recipient_cache_time and
                (current_time - self._recipient_cache_time[sensor_id]).total_seconds() < 300):
                return self._recipient_cache[sensor_id]
            
            if not firebase_db:
                logger.error("Firebase not initialized - cannot fetch recipients")
                return []
            
            recipients = []
            
            # First try: Get sensor-specific recipients
            sensor_ref = db.reference(f'/alert_recipients/{sensor_id}/emails')
            sensor_emails = sensor_ref.get()
            
            if sensor_emails:
                recipients = sensor_emails if isinstance(sensor_emails, list) else [sensor_emails]
                logger.info(f"Found {len(recipients)} sensor-specific recipients for {sensor_id}")
            else:
                # Second try: Get global recipients
                global_ref = db.reference('/alert_recipients/global/emails')
                global_emails = global_ref.get()
                
                if global_emails:
                    recipients = global_emails if isinstance(global_emails, list) else [global_emails]
                    logger.info(f"Using {len(recipients)} global recipients for {sensor_id}")
                else:
                    # Third try: Get from sensorLocations (backward compatibility)
                    locations_ref = db.reference('/sensorLocations')
                    locations_data = locations_ref.get()
                    
                    if locations_data:
                        for key, sensor_data in locations_data.items():
                            if sensor_data.get('sensorID') == sensor_id:
                                # Check for email fields in sensor location data
                                if 'alertEmail' in sensor_data:
                                    recipients = [sensor_data['alertEmail']] if sensor_data['alertEmail'] else []
                                elif 'contactEmail' in sensor_data:
                                    recipients = [sensor_data['contactEmail']] if sensor_data['contactEmail'] else []
                                break
                    
                    if recipients:
                        logger.info(f"Found {len(recipients)} recipients in sensorLocations for {sensor_id}")
            
            # Validate email format
            valid_recipients = []
            for email in recipients:
                if email and isinstance(email, str) and '@' in email:
                    valid_recipients.append(email.strip())
            
            # Update cache
            self._recipient_cache[sensor_id] = valid_recipients
            self._recipient_cache_time[sensor_id] = current_time
            
            logger.info(f"Final recipient list for {sensor_id}: {valid_recipients}")
            return valid_recipients
            
        except Exception as e:
            logger.error(f"Error fetching recipients for sensor {sensor_id}: {e}")
            return []
    
    def clear_cache(self, sensor_id: str = None):
        """Clear configuration and recipient cache"""
        if sensor_id:
            self._recipient_cache.pop(sensor_id, None)
            self._recipient_cache_time.pop(sensor_id, None)
        else:
            self._recipient_cache.clear()
            self._recipient_cache_time.clear()
            self._config_cache = None
            self._cache_timestamp = None

# ---------------------------------------------------
# Email Alert Functions
# ---------------------------------------------------
def send_risk_alert(sensor_id: str, risk_data: Dict[str, Any], email_config: EmailConfig):
    """
    Send email alert for high-risk and critical predictions
    """
    try:
        if not email_config.is_configured():
            logger.warning("Email not configured - skipping alert")
            return False

        # Get recipients for this specific sensor
        recipients = email_config.get_recipients_for_sensor(sensor_id)
        
        if not recipients:
            logger.warning(f"No recipients found for sensor {sensor_id} - skipping alert")
            return False

        # Create message
        subject = f"🚨 GAS MONITORING ALERT - Sensor {sensor_id} - {risk_data['risk_label']}"
        
        # HTML email content
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .alert-box {{ 
                    padding: 20px; 
                    border-radius: 10px; 
                    margin: 10px 0;
                    {'background: #ffebee; color: #c62828; border: 2px solid #f44336;' if risk_data['risk_level'] >= 4 else 'background: #fff3e0; color: #ef6c00; border: 2px solid #ff9800;'}
                }}
                .risk-level {{ font-size: 24px; font-weight: bold; }}
                .sensor-info {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .values {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 15px 0; }}
                .value-item {{ padding: 8px; background: white; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="alert-box">
                <div class="risk-level">Risk Level: {risk_data['risk_label']}</div>
                <p><strong>Sensor ID:</strong> {sensor_id}</p>
                <p><strong>Time:</strong> {risk_data['timestamp']}</p>
            </div>
            
            <div class="sensor-info">
                <h3>📊 Sensor Readings</h3>
                <div class="values">
                    <div class="value-item"><strong>Risk Index:</strong> {risk_data['predicted_risk_index']:.2f}</div>
                    <div class="value-item"><strong>AQI:</strong> {risk_data['aqi']:.1f}</div>
                    <div class="value-item"><strong>AQI Category:</strong> {risk_data['aqi_category']}</div>
                    <div class="value-item"><strong>Methane:</strong> {risk_data['input_data']['methane']}</div>
                    <div class="value-item"><strong>CO2:</strong> {risk_data['input_data']['co2']}</div>
                    <div class="value-item"><strong>Ammonia:</strong> {risk_data['input_data']['ammonia']}</div>
                    <div class="value-item"><strong>Humidity:</strong> {risk_data['input_data']['humidity']}%</div>
                    <div class="value-item"><strong>Temperature:</strong> {risk_data['input_data']['temperature']}°C</div>
                </div>
            </div>
            
            <div style="margin-top: 20px; padding: 15px; background: #e3f2fd; border-radius: 5px;">
                <h3>💡 Recommended Actions</h3>
                <p>{risk_data['recommendation']}</p>
                <p><strong>Calculation Method:</strong> {risk_data.get('calculation_method', 'Unknown')}</p>
            </div>
            
            <hr>
            <p style="color: #666; font-size: 12px;">
                This is an automated alert from the Gas Monitoring System. 
                Please take appropriate safety measures immediately.
            </p>
        </body>
        </html>
        """

        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = email_config.sender_email
        msg['To'] = ", ".join(recipients)
        
        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)

        # Send email
        if email_config.enable_ssl:
            server = smtplib.SMTP_SSL(email_config.smtp_server, email_config.smtp_port)
        else:
            server = smtplib.SMTP(email_config.smtp_server, email_config.smtp_port)
            server.starttls()
        
        server.login(email_config.sender_email, email_config.sender_password)
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Risk alert email sent for sensor {sensor_id} to {len(recipients)} recipients - Risk level: {risk_data['risk_label']}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send email alert: {e}")
        return False
        
def send_alert_async(sensor_id: str, risk_data: Dict[str, Any], email_config: EmailConfig):
    """
    Send email alert asynchronously to avoid blocking the API response
    """
    thread = threading.Thread(
        target=send_risk_alert, 
        args=(sensor_id, risk_data, email_config)
    )
    thread.daemon = True
    thread.start()

# ---------------------------------------------------
# Data Fetching Function with Fallback
# ---------------------------------------------------
def fetch_history(sensor_ID, range="1month"):
    """Fetch sensor history data from Firebase with fallback"""
    try:
        if not firebase_db:
            logger.error("Firebase not initialized - cannot fetch data")
            return pd.DataFrame()
        
        ref = db.reference(f'/history/{sensor_ID}')
        data = ref.get()
        
        if not data:
            logger.warning(f"No data found for sensor {sensor_ID}")
            return pd.DataFrame()
            
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
                start_date = df['timestamp'].min() if not df.empty else current_date
            
            # Filter the dataframe
            if not df.empty:
                df = df[df['timestamp'] >= start_date]

        logger.info(f"Fetched {len(df)} records for sensor {sensor_ID}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching history for {sensor_ID}: {e}")
        return pd.DataFrame()

# ---------------------------------------------------
# Model Management Functions
# ---------------------------------------------------
def save_model_to_firebase(model, sensor_ID, model_type="xgboost"):
    """
    Save trained model to Firebase as base64 encoded string
    """
    try:
        if not firebase_db:
            logger.error("Firebase not initialized - cannot save model")
            return None
        
        # Serialize model to bytes
        model_bytes = pickle.dumps(model)
        model_b64 = base64.b64encode(model_bytes).decode('utf-8')
        
        # Save model metadata and data to Firebase
        ref = db.reference(f'/trained_models/{sensor_ID}')
        model_data = {
            'model_data': model_b64,
            'model_type': model_type,
            'timestamp': datetime.now().isoformat(),
            'sensor_ID': sensor_ID,
            'version': '1.0'
        }
        
        ref.set(model_data)
        logger.info(f"Model saved to Firebase for sensor {sensor_ID}")
        return f"firebase:/trained_models/{sensor_ID}"
        
    except Exception as e:
        logger.error(f"Error saving model to Firebase: {e}")
        return None

def load_model_from_firebase(sensor_ID):
    """
    Load trained model from Firebase
    """
    try:
        if not firebase_db:
            logger.error("Firebase not initialized - cannot load model")
            return None
        
        ref = db.reference(f'/trained_models/{sensor_ID}')
        model_data = ref.get()
        
        if not model_data or 'model_data' not in model_data:
            logger.warning(f"No trained model found for sensor {sensor_ID}")
            return None
        
        # Decode and deserialize model
        model_b64 = model_data['model_data']
        model_bytes = base64.b64decode(model_b64)
        model = pickle.loads(model_bytes)
        
        logger.info(f"Model loaded from Firebase for sensor {sensor_ID}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model from Firebase: {e}")
        return None

def save_model_metadata(sensor_ID, metrics, feature_columns, best_params):
    """
    Save model metadata and performance metrics to Firebase
    """
    try:
        if not firebase_db:
            return None
        
        ref = db.reference(f'/model_metadata/{sensor_ID}')
        metadata = {
            'training_metrics': metrics,
            'feature_columns': feature_columns,
            'best_params': best_params,
            'last_trained': datetime.now().isoformat(),
            'sensor_ID': sensor_ID
        }
        
        ref.set(metadata)
        logger.info(f"Model metadata saved for sensor {sensor_ID}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving model metadata: {e}")
        return False

def load_model_metadata(sensor_ID):
    """
    Load model metadata from Firebase
    """
    try:
        if not firebase_db:
            return None
        
        ref = db.reference(f'/model_metadata/{sensor_ID}')
        return ref.get()
        
    except Exception as e:
        logger.error(f"Error loading model metadata: {e}")
        return None

def model_exists_in_firebase(sensor_ID):
    """
    Check if a trained model exists in Firebase
    """
    try:
        if not firebase_db:
            return False
        
        ref = db.reference(f'/trained_models/{sensor_ID}')
        model_data = ref.get()
        return model_data is not None and 'model_data' in model_data
        
    except Exception as e:
        logger.error(f"Error checking model existence: {e}")
        return False

def delete_model_from_firebase(sensor_ID):
    """
    Delete trained model from Firebase
    """
    try:
        if not firebase_db:
            return False
        
        # Delete model data
        model_ref = db.reference(f'/trained_models/{sensor_ID}')
        model_ref.delete()
        
        # Delete metadata
        metadata_ref = db.reference(f'/model_metadata/{sensor_ID}')
        metadata_ref.delete()
        
        logger.info(f"Model deleted from Firebase for sensor {sensor_ID}")
        return True
        
    except Exception as e:
        logger.error(f"Error deleting model from Firebase: {e}")
        return False

# ---------------------------------------------------
# XGBoost Model Function with Sample Data Fallback
# ---------------------------------------------------
def xgboost_model(
    sensor_ID, 
    range: str = "1month",
    save_model: bool = True,
    retrain: bool = False
):
    """Train XGBoost model and save to Firebase"""
    try:
        # Check if model already exists and we don't want to retrain
        if not retrain and model_exists_in_firebase(sensor_ID):
            logger.info(f"Using existing model for sensor {sensor_ID}")
            existing_model = load_model_from_firebase(sensor_ID)
            metadata = load_model_metadata(sensor_ID)
            
            return {
                "status": "existing_model",
                "message": "Using pre-trained model from Firebase",
                "sensor_ID": sensor_ID,
                "model_available": True,
                "last_trained": metadata.get('last_trained') if metadata else None
            }
        
        df = fetch_history(sensor_ID, range)       
      
        if df.empty:
            return {"error": f"No data available for sensor {sensor_ID} with range {range}"}
        
        # Drop columns that are not useful or non-numeric for now
        columns_to_drop = ['riskIndex', 'time', 'timestamp', 'apiUserID', 'apiPass', 'sensorID']
        available_columns = [col for col in columns_to_drop if col in df.columns]
        
        features = df.drop(columns=available_columns)
        target = df['riskIndex']

        if len(features.columns) == 0:
            return {"error": "No features available for training"}

        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
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
            verbose=2,
            n_jobs=1
        )
        
        grid_search.fit(X_train, y_train)

        # Get the best model
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test) 

        # Calculate metrics
        result_predict_test = { 
            "TestRMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "Test_MAE": mean_absolute_error(y_test, y_pred),
            "Test_r_score": r2_score(y_test, y_pred)
        }

        # SAVE MODEL TO FIREBASE
        model_saved = False
        model_path = None
        if save_model:
            model_path = save_model_to_firebase(best_model, sensor_ID)
            if model_path:
                # Save metadata
                save_model_metadata(
                    sensor_ID, 
                    result_predict_test, 
                    list(features.columns), 
                    grid_search.best_params_
                )
                model_saved = True

        # Create test results
        test_results = X_test.copy()
        test_results['true_riskIndex'] = y_test
        test_results['predicted_riskIndex'] = y_pred

        result = {
            "best_params": grid_search.best_params_,
            "RMSE": -grid_search.best_score_,
            "test_result": result_predict_test,
            "predicted_sample": test_results.head().to_dict('records'),
            "model_saved": model_saved,
            "model_path": model_path,
            "data_info": {
                "total_records": len(df),
                "training_records": len(X_train),
                "test_records": len(X_test),
                "date_range_used": f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}" if not df.empty and 'timestamp' in df.columns else "Unknown",
                "selected_range": range,
                "features_used": list(features.columns), 
            }
        }

        return result
        
    except Exception as e:
        logger.error(f"Error in xgboost_model for {sensor_ID}: {e}")
        return {"error": f"Model training failed: {str(e)}"}

def calculate_regulatory_risk(methane, co2, ammonia, humidity, temperature):
    """
    Risk index based on regulatory exposure limits
    """
    # OSHA and NIOSH exposure limits (adjust based on your region)
    exposure_limits = {
        'methane': {'stel': 1000, 'twa': 1000, 'idlh': 50000},  # Short-term, Time-weighted, Immediately Dangerous
        'co2': {'stel': 30000, 'twa': 5000, 'idlh': 40000},
        'ammonia': {'stel': 35, 'twa': 25, 'idlh': 500}
    }
    
    risks = []
    
    # Calculate compliance risk for each gas
    for gas, concentration in [('methane', methane), ('co2', co2), ('ammonia', ammonia)]:
        limits = exposure_limits[gas]
        
        if concentration <= limits['twa']:
            risk = 1  # Within safe limits
        elif concentration <= limits['stel']:
            risk = 3  # Exceeded TWA but within STEL
        elif concentration <= limits['idlh']:
            risk = 6  # Exceeded STEL but below IDLH
        else:
            risk = 9 # Immediately dangerous
            
        risks.append(risk)
    
    # Consider environmental factors
    env_risk = 0
    if humidity > 85:  # High humidity can affect gas dispersion
        env_risk += 1
    if temperature > 35:  # High temperature increases volatility
        env_risk += 1
        
    # Overall risk (maximum individual risk + environmental factors)
    overall_risk = max(risks) + env_risk
    
    return min(overall_risk, 10)
    
def predict_risk_index(sensor_ID, sensor_data, use_regulatory_calculation=True):
    """
    Predict risk index using either regulatory calculation or pre-trained model
    """
    try:
        if use_regulatory_calculation:
            # Use regulatory risk calculation
            required_params = ['methane', 'co2', 'ammonia', 'humidity', 'temperature']
            
            for param in required_params:
                if param not in sensor_data:
                    return {"error": f"Missing required parameter: {param}"}
            
            methane = sensor_data['methane']
            co2 = sensor_data['co2']
            ammonia = sensor_data['ammonia']
            humidity = sensor_data['humidity']
            temperature = sensor_data['temperature']
            
            calculated_risk = calculate_regulatory_risk(methane, co2, ammonia, humidity, temperature)
            risk_value = calculated_risk
            
        else:
            # Use ML model prediction (original code)
            model = load_model_from_firebase(sensor_ID)
            if not model:
                return {
                    "error": f"No trained model found for sensor {sensor_ID}",
                    "suggestion": "Train a model first using /api/xgboost endpoint"
                }
            
            metadata = load_model_metadata(sensor_ID)
            if not metadata:
                return {"error": "Model metadata not found"}
            
            feature_columns = metadata.get('feature_columns', [])
            input_df = pd.DataFrame([sensor_data])
            
            for col in feature_columns:
                if col not in input_df.columns:
                    return {"error": f"Missing required feature: {col}"}
            
            input_df = input_df[feature_columns]
            risk_value = model.predict(input_df)[0]
        
        # Convert to AQI and get category
        aqi = convert_risk_to_aqi(risk_value)
        risk_level, risk_label, aqi_category, recommendation = calculate_risk_category(risk_value)
        
        result = {
            "sensor_ID": sensor_ID,
            "predicted_risk_index": float(risk_value),
            "aqi": float(aqi),
            "risk_level": risk_level,
            "risk_label": risk_label,
            "aqi_category": aqi_category,
            "recommendation": recommendation,
            "timestamp": datetime.now().isoformat(),
            "calculation_method": "regulatory_calculation" if use_regulatory_calculation else "ml_model"
        }
            
        # 🔥 NEW: Send email alert for high risk and critical levels
        if risk_level >= 4:  # High risk (level 4) and Critical (level 5)
            email_config = EmailConfig()
            if email_config.is_configured():
                recipients = email_config.get_recipients_for_sensor(sensor_ID)
                if recipients:
                    send_alert_async(sensor_ID, result, email_config)
                    result["alert_sent"] = True
                    result["alert_recipients"] = recipients
                    result["recipient_count"] = len(recipients)
                else:
                    result["alert_sent"] = False
                    result["alert_reason"] = "No recipients found for this sensor"
            else:
                result["alert_sent"] = False
                result["alert_reason"] = "Email not configured"
        
        if not use_regulatory_calculation:
            result["features_used"] = feature_columns
            
        result["input_data"] = sensor_data
        
        return result
        
    except Exception as e:
        logger.error(f"Error in risk prediction for sensor {sensor_ID}: {e}")
        return {"error": f"Risk prediction failed: {str(e)}"}
        
def generate_sample_data():
    """Generate sample data for testing when Firebase is unavailable"""
    try:
        # Create sample data with realistic ranges
        np.random.seed(42)
        n_samples = 100
        
        sample_data = {
            'timestamp': [datetime.now() - timedelta(hours=i) for i in range(n_samples)],
            'methane': np.random.uniform(0, 100, n_samples),
            'co2': np.random.uniform(300, 2000, n_samples),
            'ammonia': np.random.uniform(0, 50, n_samples),
            'humidity': np.random.uniform(20, 90, n_samples),
            'temperature': np.random.uniform(15, 40, n_samples),
            'riskIndex': np.random.uniform(1, 10, n_samples)
        }
        
        df = pd.DataFrame(sample_data)
        logger.info("Generated sample data for testing")
        return df
        
    except Exception as e:
        logger.error(f"Error generating sample data: {e}")
        return pd.DataFrame()
        
def convert_risk_to_aqi(risk_index):
    """
    Piecewise Linear Scaling
    Convert risk index to AQI (0-500 scale) 
    """
    
    # AQI categories and their corresponding risk index ranges 
    aqi_thresholds = [
        (0, 50, 0, 3),      # Good: 0-50 AQI, 0-3 risk
        (51, 100, 3.1, 7),  # Moderate: 51-100 AQI, 3.1-7 risk  
        (101, 150, 7.1, 9), # Unhealthy for Sensitive: 101-150 AQI, 7.1-9 risk
        (151, 500, 9.1, 10) # Unhealthy/Very Unhealthy: 151-500 AQI, 9.1-10 risk
    ]
    
    for aqi_min, aqi_max, risk_min, risk_max in aqi_thresholds:
        if risk_min <= risk_index <= risk_max:
            # Linear interpolation within this category
            aqi = aqi_min + (risk_index - risk_min) * (aqi_max - aqi_min) / (risk_max - risk_min)
            return min(max(aqi, 0), 500)
    
    # Fallback: cap at extremes
    if risk_index < 1:
        return 10
    else:
        return 500 

def calculate_risk_category(risk_index):
    """
    Categorize risk and provide recommendations
    """
    if risk_index <= 2:
        return 1, "Low", "Good", "Air quality is good. No action required."
    elif risk_index <= 4:
        return 2, "Moderate", "Moderate", "Air quality is acceptable. Sensitive individuals should consider reducing prolonged outdoor exposure."
    elif risk_index <= 6:
        return 3, "High", "Unhealthy for Sensitive Groups", "Air quality is poor. Reduce outdoor activities. Consider improving ventilation."
    elif risk_index <= 8:
        return 4, "Very High", "Unhealthy", "Air quality is unhealthy. Limit outdoor exposure. Use air purifiers if available."
    else:
        return 5, "Hazardous", "Very Unhealthy", "Air quality is hazardous. Avoid outdoor activities. Evacuate if necessary."

def get_feature_importance(sensor_id):
    """Get feature importance from trained XGBoost model"""
    try:
        # Load the trained model
        model = load_model_from_firebase(sensor_id)
        if not model:
            return {"error": f"No trained model found for sensor {sensor_id}"}
        
        # Load model metadata to get feature names
        metadata = load_model_metadata(sensor_id)
        if not metadata:
            return {"error": "Model metadata not found"}
        
        feature_columns = metadata.get('feature_columns', [])
        
        # Get feature importance
        importance_scores = model.feature_importances_
        
        # Convert numpy types to native Python types
        feature_importance = {}
        for feature, score in zip(feature_columns, importance_scores):
            # Convert numpy.float32 to native Python float
            feature_importance[feature] = float(score)
        
        # Sort by importance
        sorted_importance = dict(sorted(feature_importance.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        # Create feature importance plot
        plt.figure(figsize=(10, 8))
        features = list(sorted_importance.keys())[:10]  # Top 10 features
        scores = [float(score) for score in list(sorted_importance.values())[:10]]  # Convert to float
        
        plt.barh(features, scores)
        plt.xlabel('Feature Importance Score')
        plt.title('Top Feature Importances from XGBoost Model')
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Calculate total importance for percentages
        total_importance = sum(sorted_importance.values())
        
        return {
            "feature_importance": sorted_importance,
            "importance_plot": f"data:image/png;base64,{image_base64}",
            "top_features": list(sorted_importance.keys())[:5],  # Top 5 most important features
            "metadata": {
                "total_features": len(feature_columns),
                "model_type": "XGBoost",
                "total_importance": float(total_importance)  # Convert to float
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        return {"error": f"Failed to get feature importance: {str(e)}"}

# ---------------------------------------------------
# Visualization Functions (Return Base64 Images)
# ---------------------------------------------------
def plot_correlation_heatmap(df, figsize=(12, 10)):
    """Return correlation heatmap as base64 image"""
    try:
        # Enhanced validation
        if df is None or df.empty:
            return {"error": "No data provided for correlation heatmap"}
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {"error": "No numeric columns found for correlation heatmap"}
        
        if len(numeric_df.columns) < 2:
            return {"error": "Need at least 2 numeric columns for correlation heatmap"}
        
        # Check for sufficient data points
        if len(numeric_df) < 3:
            return {"error": "Insufficient data points (need at least 3) for correlation analysis"}
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Check if correlation matrix is valid
        if corr_matrix.isnull().all().all():
            return {"error": "Unable to calculate correlations - data may be constant"}
        
        # Create the plot
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(
            corr_matrix, 
            mask=mask,
            annot=True, 
            cmap='RdBu_r', 
            center=0,
            square=True,
            fmt='.2f',
            linewidths=0.5,
            cbar_kws={"shrink": .8}
        )
        
        plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Return additional metadata
        return {
            "image": f"data:image/png;base64,{image_base64}",
            "correlation_matrix": corr_matrix.to_dict(),
            "metadata": {
                "data_points": len(numeric_df),
                "features_analyzed": len(numeric_df.columns),
                "matrix_shape": corr_matrix.shape
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating correlation heatmap: {e}")
        return {"error": f"Failed to create heatmap: {str(e)}"}
        
def plot_target_correlations(df, target_col='riskIndex', top_n=10):
    """Return target correlations bar chart as base64 image"""
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        
        if target_col not in numeric_df.columns:
            return {"error": f"Target column '{target_col}' not found in numeric data"}
        
        corr_matrix = numeric_df.corr()
        target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
        
        if target_corr.empty:
            return {"error": "No correlations to display"}
        
        # Take top N correlations (positive and negative)
        top_positive = target_corr.head(top_n//2)
        top_negative = target_corr.tail(top_n//2)
        top_correlations = pd.concat([top_positive, top_negative])
        
        plt.figure(figsize=(10, 8))
        
        # Create a color mapping based on correlation values
        colors = ['red' if x > 0 else 'blue' for x in top_correlations.values]
        
        bars = plt.barh(top_correlations.index, top_correlations.values, color=colors)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.title(f'Feature Correlations with {target_col}', fontsize=14, fontweight='bold')
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
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            "image": f"data:image/png;base64,{image_base64}",
            "correlations": top_correlations.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Error creating target correlations: {e}")
        return {"error": f"Failed to create correlations plot: {str(e)}"}

def plot_correlation_scatterplots(df, target_col='riskIndex', top_n=6):
    """Return scatter plots as base64 images"""
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        
        if target_col not in numeric_df.columns:
            return {"error": f"Target column '{target_col}' not found"}
        
        corr_matrix = numeric_df.corr()
        target_corr = corr_matrix[target_col].drop(target_col)
        top_features = target_corr.abs().nlargest(top_n).index.tolist()
        
        if not top_features:
            return {"error": "No features to plot"}
        
        # Create scatter plots for top features
        scatter_images = []
        scatter_data = []
        
        for feature in top_features:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df, x=feature, y=target_col, alpha=0.6)
            correlation_val = target_corr[feature]
            plt.title(f'{feature} vs {target_col}\nCorrelation: {correlation_val:.3f}')
            plt.xlabel(feature)
            plt.ylabel(target_col)
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            scatter_images.append({
                "feature": feature,
                "image": f"data:image/png;base64,{image_base64}",
                "correlation": correlation_val
            })
            
            scatter_data.append({
                "feature": feature,
                "correlation": correlation_val
            })
        
        return {
            "scatter_plots": scatter_images,
            "summary": scatter_data
        }
        
    except Exception as e:
        logger.error(f"Error creating scatter plots: {e}")
        return {"error": f"Failed to create scatter plots: {str(e)}"}
        
def print_correlation_summary(df, target_col='riskIndex'):
    """Return correlation summary as structured data"""
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        
        if target_col not in numeric_df.columns:
            return {"error": f"Target column '{target_col}' not found"}
        
        corr_matrix = numeric_df.corr()
        target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
        
        # FIX: Properly separate positive and negative correlations
        positive_corr = target_corr[target_corr > 0].sort_values(ascending=False)
        negative_corr = target_corr[target_corr < 0].sort_values(ascending=True)  # Most negative first
        
        # Convert to dictionaries for JSON serialization
        top_positive = positive_corr.head(5).to_dict()
        top_negative = negative_corr.head(5).to_dict()  # This will now show actual negative values
        
        return {
            "target_variable": target_col,
            "total_features_analyzed": len(target_corr),
            "top_positive_correlations": top_positive,
            "top_negative_correlations": top_negative,
            "correlation_strength_breakdown": {
                "strong_positive": int(len(target_corr[target_corr > 0.7])),
                "moderate_positive": int(len(target_corr[(target_corr > 0.3) & (target_corr <= 0.7)])),
                "weak": int(len(target_corr[(target_corr >= -0.3) & (target_corr <= 0.3)])),
                "moderate_negative": int(len(target_corr[(target_corr >= -0.7) & (target_corr < -0.3)])),
                "strong_negative": int(len(target_corr[target_corr < -0.7]))
            },
            "all_correlations": target_corr.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Error creating correlation summary: {e}")
        return {"error": f"Failed to create correlation summary: {str(e)}"}

def enhanced_correlation_summary(df, target_col='riskIndex'):
    """Return enhanced correlation summary with statistical significance"""
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        
        if target_col not in numeric_df.columns:
            return {"error": f"Target column '{target_col}' not found"}
        
        corr_matrix = numeric_df.corr()
        target_corr = corr_matrix[target_col].drop(target_col)
        
        # Calculate p-values for correlations
        p_values = {}
        for feature in target_corr.index:
            if feature != target_col:
                corr_coef, p_value = stats.pearsonr(numeric_df[feature], numeric_df[target_col])
                p_values[feature] = p_value
        
        # Separate correlations
        positive_corr = target_corr[target_corr > 0].sort_values(ascending=False)
        negative_corr = target_corr[target_corr < 0].sort_values(ascending=True)
        
        # Get statistically significant correlations (p < 0.05)
        significant_features = {
            feature: {
                "correlation": target_corr[feature],
                "p_value": p_values[feature],
                "significant": p_values[feature] < 0.05
            }
            for feature in target_corr.index
        }
        
        return {
            "target_variable": target_col,
            "total_features_analyzed": len(target_corr),
            "data_points": len(df),
            "top_positive_correlations": positive_corr.head(5).to_dict(),
            "top_negative_correlations": negative_corr.head(5).to_dict(),
            "statistically_significant": {
                feature: data for feature, data in significant_features.items() 
                if data["significant"]
            },
            "correlation_strength_breakdown": {
                "strong_positive": int(len(target_corr[target_corr > 0.7])),
                "moderate_positive": int(len(target_corr[(target_corr > 0.3) & (target_corr <= 0.7)])),
                "weak": int(len(target_corr[(target_corr >= -0.3) & (target_corr <= 0.3)])),
                "moderate_negative": int(len(target_corr[(target_corr >= -0.7) & (target_corr < -0.3)])),
                "strong_negative": int(len(target_corr[target_corr < -0.7]))
            },
            "all_correlations": target_corr.to_dict(),
            "p_values": p_values
        }
        
    except Exception as e:
        logger.error(f"Error creating enhanced correlation summary: {e}")
        return {"error": f"Failed to create correlation summary: {str(e)}"}

# ---------------------------------------------------
# API Endpoints with Error Handling
# ---------------------------------------------------
@app.get("/api/xgboost/{sensor_id}")
def get_xgboost_model(
    sensor_id: str,
    range: str = Query("1month", description="Date range: 1week, 1month, 3months, 6months, 1year, all"),
    save_model: bool = Query(True, description="Save trained model to Firebase"),
    retrain: bool = Query(False, description="Force retrain even if model exists")
):
    """Train XGBoost model for specific sensor and optionally save to Firebase"""
    try:
        result = xgboost_model(sensor_id, range, save_model, retrain)
        return result
    except Exception as e:
        logger.error(f"Error in XGBoost endpoint for {sensor_id}: {e}")
        raise HTTPException(status_code=500, detail=f"XGBoost training failed: {str(e)}")

@app.get("/api/correlation_summary/{sensor_id}")
def get_correlation_summary(
    sensor_id: str,
    range: str = Query("1month", description="Date range: 1week, 1month, 3months, 6months, 1year, all"),
    target_col: str = Query("riskIndex", description="Target column for correlation analysis")
):
    """Get correlation summary for sensor data"""
    try:
        df = fetch_history(sensor_id, range)
        if df.empty:
            df = generate_sample_data()  # Fallback to sample data
        
        result = print_correlation_summary(df, target_col)
        return result
    except Exception as e:
        logger.error(f"Error in correlation summary for {sensor_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Correlation analysis failed: {str(e)}")

@app.get("/api/correlation_scatterplots/{sensor_id}")
def get_correlation_scatterplots(
    sensor_id: str,
    range: str = Query("1month", description="Date range: 1week, 1month, 3months, 6months, 1year, all"),
    target_col: str = Query("riskIndex", description="Target column for correlation analysis")
):
    """Get correlation scatter plots for sensor data"""
    try:
        df = fetch_history(sensor_id, range)
        if df.empty:
            df = generate_sample_data()  # Fallback to sample data
        
        result = plot_correlation_scatterplots(df, target_col)
        return result
    except Exception as e:
        logger.error(f"Error in scatter plots for {sensor_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Scatter plot generation failed: {str(e)}")

@app.get("/api/correlations/{sensor_id}")
def get_target_correlations(
    sensor_id: str,
    range: str = Query("1month", description="Date range: 1week, 1month, 3months, 6months, 1year, all"),
    target_col: str = Query("riskIndex", description="Target column for correlation analysis")
):
    """Get target correlations for sensor data"""
    try:
        df = fetch_history(sensor_id, range)
        if df.empty:
            df = generate_sample_data()  # Fallback to sample data
        
        result = plot_target_correlations(df, target_col)
        return result
    except Exception as e:
        logger.error(f"Error in target correlations for {sensor_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Target correlation analysis failed: {str(e)}")

@app.get("/api/correlations_heatmap/{sensor_id}")
def get_correlation_heatmap(
    sensor_id: str,
    range: str = Query("1month", description="Date range: 1week, 1month, 3months, 6months, 1year, all"),
    target_col: str = Query("riskIndex", description="Target column for correlation heatmap")
):
    """Get correlation heatmap for sensor data"""
    try:
        df = fetch_history(sensor_id, range)
        if df.empty:
            df = generate_sample_data()  # Fallback to sample data
        
        result = plot_correlation_heatmap(df)
        return result
    except Exception as e:
        logger.error(f"Error in correlation heatmap for {sensor_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Heatmap generation failed: {str(e)}")

@app.get("/api/verify_sensor/{sensor_id}")
def check_sensor_credential(
    sensor_id: str,
    apiuser: str,
    apipassword: str,
):
    try:
        if not firebase_db:
            logger.error("Firebase not initialized - cannot fetch data")
            return False       
      
        ref = db.reference(f'/sensorLocations')
        data = ref.get()
        result = False

        if data:
            # Iterate through all sensor entries
            for key, sensor_data in data.items():
                # Check if this sensor matches the sensor_id
                if sensor_data.get('sensorID') == sensor_id:
                    # Check credentials
                    if (sensor_data.get('APIUserID') == apiuser and 
                        sensor_data.get('APIUserPass') == apipassword):
                        result = True
                        break  # Found matching sensor with correct credentials
                    else:
                        result = False  # Sensor exists but wrong credentials
                        break

        return result
        
    except Exception as e:
        logger.error(f"Error fetching sensor locations for {sensor_id}: {e}")
        return False

@app.get("/api/feature_importance/{sensor_id}")
def get_feature_importance_endpoint(sensor_id: str):
    """Get feature importance from trained model"""
    try:
        result = get_feature_importance(sensor_id)
        return result
    except Exception as e:
        logger.error(f"Error in feature importance for {sensor_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Feature importance analysis failed: {str(e)}")
        
@app.get("/api/data_quality/{sensor_id}")
def get_data_quality_report(
    sensor_id: str,
    range: str = Query("1month", description="Date range: 1week, 1month, 3months, 6months, 1year, all")
):
    """Get data quality report for sensor data"""
    try:
        df = fetch_history(sensor_id, range)
        if df.empty:
            return {"error": f"No data found for sensor {sensor_id} with range {range}"}
        
        # Basic data quality metrics
        numeric_df = df.select_dtypes(include=[np.number])
        
        quality_report = {
            "sensor_id": sensor_id,
            "date_range": range,
            "total_records": len(df),
            "numeric_columns": len(numeric_df.columns),
            "date_range_actual": {
                "start": df['timestamp'].min().isoformat() if 'timestamp' in df.columns else "Unknown",
                "end": df['timestamp'].max().isoformat() if 'timestamp' in df.columns else "Unknown"
            },
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "basic_statistics": numeric_df.describe().to_dict() if not numeric_df.empty else {}
        }
        
        return quality_report
        
    except Exception as e:
        logger.error(f"Error in data quality report for {sensor_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Data quality analysis failed: {str(e)}")

@app.get("/api/predict/{sensor_id}")
def predict_risk_endpoint(
    sensor_id: str,
    methane: float = Query(..., description="Methane level"),
    co2: float = Query(..., description="CO2 level"),
    ammonia: float = Query(..., description="Ammonia level"),
    humidity: float = Query(..., description="Humidity percentage"),
    temperature: float = Query(..., description="Temperature in Celsius")
):
    """Predict risk index using pre-trained model"""
    try:
        sensor_data = {
            "methane": methane,
            "co2": co2,
            "ammonia": ammonia,
            "humidity": humidity,
            "temperature": temperature
        }
        
        prediction_result = predict_risk_index(sensor_id, sensor_data)
        
        if "error" in prediction_result:
            raise HTTPException(status_code=404, detail=prediction_result["error"])
        
        # Enhance with AQI and risk category
        risk_index = prediction_result["predicted_risk_index"]
        aqi = convert_risk_to_aqi(risk_index)
        risk_level, risk_label, aqi_category, recommendation = calculate_risk_category(risk_index)
        
        enhanced_result = {
            **prediction_result,
            "aqi": aqi,
            "aqi_category": aqi_category,
            "risk_level": risk_level,
            "risk_label": risk_label,
            "recommendation": recommendation
        }
        
        return enhanced_result
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint for {sensor_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/predict/{sensor_id}")
def predict_risk_bulk(
    sensor_id: str,
    sensor_readings: List[Dict[str, float]]
):
    """Bulk prediction for multiple sensor readings"""
    try:
        results = []
        alert_count = 0
        email_config = EmailConfig()
        
        for reading in sensor_readings:
            result = predict_risk_index(sensor_id, reading)
            if "error" not in result:
                risk_index = result["predicted_risk_index"]
                aqi = convert_risk_to_aqi(risk_index)
                risk_level, risk_label, aqi_category, recommendation = calculate_risk_category(risk_index)
                
                enhanced_result = {
                    **result,
                    "aqi": aqi,
                    "aqi_category": aqi_category,
                    "risk_level": risk_level,
                    "risk_label": risk_label,
                    "recommendation": recommendation
                }
                
                # Check if alert was sent
                if enhanced_result.get("alert_sent"):
                    alert_count += 1
                    
                results.append(enhanced_result)
            else:
                results.append({"error": result["error"], "input_data": reading})
        
        return {
            "sensor_ID": sensor_id,
            "predictions": results,
            "total_predictions": len(results),
            "successful_predictions": len([r for r in results if "error" not in r]),
            "alerts_triggered": alert_count,
            "email_alerts_enabled": email_config.is_configured()
        }
        
    except Exception as e:
        logger.error(f"Error in bulk prediction for {sensor_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Bulk prediction failed: {str(e)}")
        
@app.get("/api/model/status/{sensor_id}")
def get_model_status(sensor_id: str):
    """Check if a trained model exists and get its status"""
    try:
        model_exists = model_exists_in_firebase(sensor_id)
        metadata = load_model_metadata(sensor_id) if model_exists else None
        
        return {
            "sensor_ID": sensor_id,
            "model_exists": model_exists,
            "last_trained": metadata.get('last_trained') if metadata else None,
            "training_metrics": metadata.get('training_metrics') if metadata else None,
            "feature_columns": metadata.get('feature_columns') if metadata else None
        }
        
    except Exception as e:
        logger.error(f"Error checking model status for {sensor_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Model status check failed: {str(e)}")

@app.delete("/api/model/{sensor_id}")
def delete_model_endpoint(sensor_id: str):
    """Delete trained model from Firebase"""
    try:
        success = delete_model_from_firebase(sensor_id)
        
        if success:
            return {
                "status": "success",
                "message": f"Model for sensor {sensor_id} deleted successfully"
            }
        else:
            raise HTTPException(status_code=404, detail=f"Failed to delete model or model not found for sensor {sensor_id}")
            
    except Exception as e:
        logger.error(f"Error deleting model for {sensor_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Model deletion failed: {str(e)}")

@app.post("/api/model/retrain/{sensor_id}")
def retrain_model(
    sensor_id: str,
    range: str = Query("1month", description="Date range for retraining")
):
    """Force retrain model and save to Firebase"""
    try:
        result = xgboost_model(sensor_id, range, save_model=True, retrain=True)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "status": "success",
            "message": f"Model retrained and saved for sensor {sensor_id}",
            "training_result": result
        }
        
    except Exception as e:
        logger.error(f"Error retraining model for {sensor_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Model retraining failed: {str(e)}")

@app.get("/api/sensor_readings/{sensor_id}")
def get_latest_sensor_reading(
    sensor_id: str
):
    """
    Get the most recent sensor reading for a specific sensor
    """
    try:
        if not firebase_db:
            logger.error("Firebase not initialized - cannot fetch data")
            return {"error": "Firebase not initialized", "sensor_id": sensor_id}
        
        ref = db.reference(f'/sensorReadings')
        data = ref.get()
        
        if not data:
            return {"error": "No sensor readings data available", "sensor_id": sensor_id}
        
        latest_reading = None
        latest_timestamp = None
        
        # Find the most recent reading for this sensor
        for key, sensor_data in data.items():
            if sensor_data.get('sensorID') == sensor_id:
                current_timestamp = sensor_data.get('timestamp')
                
                # If this is the first matching reading or it's more recent
                if not latest_reading or (current_timestamp and current_timestamp > latest_timestamp):
                    latest_reading = {
                        'timestamp': current_timestamp,
                        'methane': sensor_data.get('methane'),
                        'co2': sensor_data.get('co2'),
                        'ammonia': sensor_data.get('ammonia'),
                        'humidity': sensor_data.get('humidity'),
                        'temperature': sensor_data.get('temperature'),
                        'sensorID': sensor_data.get('sensorID')
                    }
                    latest_timestamp = current_timestamp
        
        if not latest_reading:
            return {"error": f"No readings found for sensor {sensor_id}", "sensor_id": sensor_id}
        
        return {
            "sensor_id": sensor_id,
            "latest_reading": latest_reading,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error fetching sensor readings for {sensor_id}: {str(e)}")
        return {"error": f"Failed to fetch sensor readings: {str(e)}", "sensor_id": sensor_id}

# ---------------------------------------------------
# Email Configuration Endpoints
# ---------------------------------------------------
@app.get("/api/email-config")
def get_email_config():
    """Get current email configuration"""
    email_config = EmailConfig()
    return email_config.get_current_config()

@app.post("/api/email-config/update")
def update_email_config(
    config_update: Dict[str, Any],
    confirm: bool = Query(True, description="Ask for confirmation before updating")
):
    """Update email configuration with confirmation"""
    email_config = EmailConfig()
    result = email_config.update_config(config_update, confirm=confirm)
    return result

@app.post("/api/email-config/confirm-update")
def confirm_email_config_update(confirmation_data: Dict[str, Any]):
    """Confirm and apply email configuration changes"""
    email_config = EmailConfig()
    result = email_config.confirm_update(confirmation_data)
    return result

@app.post("/api/email-config/reset")
def reset_email_config():
    """Reset email configuration to environment variables"""
    email_config = EmailConfig()
    env_config = email_config._get_config_from_env()
    
    result = email_config.update_config(env_config, confirm=False)
    if result["success"]:
        email_config.clear_cache()
    
    return result

@app.get("/api/email-config/test-connection")
def test_email_connection():
    """Test email configuration with a connection test"""
    email_config = EmailConfig()
    
    if not email_config.is_configured():
        return {
            "success": False,
            "error": "Email not configured - missing sender email or password"
        }
    
    try:
        # Test SMTP connection
        if email_config.enable_ssl:
            server = smtplib.SMTP_SSL(email_config.smtp_server, email_config.smtp_port)
        else:
            server = smtplib.SMTP(email_config.smtp_server, email_config.smtp_port)
            server.starttls()
        
        server.login(email_config.sender_email, email_config.sender_password)
        server.quit()
        
        return {
            "success": True,
            "message": "Email connection test successful",
            "config": email_config.get_current_config()
        }
        
    except Exception as e:
        logger.error(f"Email connection test failed: {e}")
        return {
            "success": False,
            "error": f"Email connection test failed: {str(e)}",
            "config": email_config.get_current_config()
        }

@app.delete("/api/email-config/cache")
def clear_email_cache():
    """Clear email configuration cache"""
    email_config = EmailConfig()
    email_config.clear_cache()
    return {"success": True, "message": "Email cache cleared"}

@app.post("/api/test-email-alert/{sensor_id}")
def test_email_alert(sensor_id: str):
    """Test email alert functionality for a specific sensor"""
    try:
        email_config = EmailConfig()
        
        if not email_config.is_configured():
            return {"error": "Email not configured - check SMTP environment variables"}
        
        recipients = email_config.get_recipients_for_sensor(sensor_id)
        if not recipients:
            return {"error": f"No recipients found for sensor {sensor_id}"}

        # Create test risk data
        test_risk_data = {
            "predicted_risk_index": 8.5,
            "aqi": 280.5,
            "risk_level": 4,
            "risk_label": "Very High",
            "aqi_category": "Unhealthy",
            "recommendation": "Air quality is unhealthy. Limit outdoor exposure. Use air purifiers if available.",
            "timestamp": datetime.now().isoformat(),
            "calculation_method": "test",
            "input_data": {
                "methane": 45.2,
                "co2": 420.5,
                "ammonia": 12.8,
                "humidity": 65.2,
                "temperature": 25.8
            }
        }
        
        success = send_risk_alert(sensor_id, test_risk_data, email_config)
        
        return {
            "status": "success" if success else "failed",
            "message": "Test email sent" if success else "Failed to send test email",
            "sensor_id": sensor_id,
            "recipients": recipients,
            "email_config": {
                "smtp_server": email_config.smtp_server,
                "sender_email": email_config.sender_email,
                "recipient_count": len(recipients),
                "configured": email_config.is_configured()
            }
        }
        
    except Exception as e:
        logger.error(f"Email test failed for {sensor_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Email test failed: {str(e)}")
        
@app.get("/api/alert-recipients/{sensor_id}")
def get_alert_recipients(sensor_id: str):
    """Get current alert recipients for a sensor"""
    try:
        email_config = EmailConfig()
        recipients = email_config.get_recipients_for_sensor(sensor_id)
        
        return {
            "sensor_id": sensor_id,
            "recipients": recipients,
            "recipient_count": len(recipients),
            "cache_status": "cached" if sensor_id in email_config._recipient_cache else "fresh"
        }
    except Exception as e:
        logger.error(f"Error getting recipients for {sensor_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recipients: {str(e)}")

@app.post("/api/alert-recipients/{sensor_id}")
def set_alert_recipients(
    sensor_id: str,
    recipients: List[str],
    recipient_type: str = Query("sensor", description="sensor or global")
):
    """Set alert recipients for a sensor or globally"""
    try:
        if not firebase_db:
            raise HTTPException(status_code=500, detail="Firebase not initialized")
        
        # Validate emails
        valid_recipients = []
        for email in recipients:
            if email and isinstance(email, str) and '@' in email:
                valid_recipients.append(email.strip())
        
        if not valid_recipients:
            raise HTTPException(status_code=400, detail="No valid email addresses provided")
        
        # Save to Firebase
        if recipient_type == "global":
            ref = db.reference('/alert_recipients/global/emails')
        else:
            ref = db.reference(f'/alert_recipients/{sensor_id}/emails')
        
        ref.set(valid_recipients)
        
        # Clear cache
        email_config = EmailConfig()
        if recipient_type == "global":
            email_config.clear_cache()  # Clear all cache
        else:
            email_config.clear_cache(sensor_id)
        
        logger.info(f"Updated {recipient_type} recipients for {sensor_id}: {valid_recipients}")
        
        return {
            "status": "success",
            "message": f"Recipients updated for {recipient_type}",
            "sensor_id": sensor_id,
            "recipients": valid_recipients,
            "recipient_count": len(valid_recipients)
        }
        
    except Exception as e:
        logger.error(f"Error setting recipients for {sensor_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set recipients: {str(e)}")

@app.delete("/api/alert-recipients/{sensor_id}")
def clear_alert_recipients(
    sensor_id: str,
    recipient_type: str = Query("sensor", description="sensor or global")
):
    """Clear alert recipients for a sensor or globally"""
    try:
        if not firebase_db:
            raise HTTPException(status_code=500, detail="Firebase not initialized")
        
        if recipient_type == "global":
            ref = db.reference('/alert_recipients/global')
        else:
            ref = db.reference(f'/alert_recipients/{sensor_id}')
        
        ref.delete()
        
        # Clear cache
        email_config = EmailConfig()
        if recipient_type == "global":
            email_config.clear_cache()
        else:
            email_config.clear_cache(sensor_id)
        
        logger.info(f"Cleared {recipient_type} recipients for {sensor_id}")
        
        return {
            "status": "success",
            "message": f"Recipients cleared for {recipient_type}",
            "sensor_id": sensor_id
        }
        
    except Exception as e:
        logger.error(f"Error clearing recipients for {sensor_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear recipients: {str(e)}")
        
@app.get("/api/debug-email-config")
def debug_email_config():
    """Debug email configuration to see what's actually set"""
    email_config = EmailConfig()
    
    # Get actual values (mask passwords)
    sender_email = os.getenv("SENDER_EMAIL", "NOT_SET")
    sender_password = os.getenv("SENDER_PASSWORD", "NOT_SET")
    
    config_status = {
        "SMTP_SERVER": os.getenv("SMTP_SERVER", "NOT_SET"),
        "SMTP_PORT": os.getenv("SMTP_PORT", "NOT_SET"),
        "SENDER_EMAIL": sender_email,
        "SENDER_PASSWORD_SET": bool(sender_password and sender_password != "NOT_SET"),
        "SENDER_PASSWORD_LENGTH": len(sender_password) if sender_password and sender_password != "NOT_SET" else 0,
        "SMTP_SSL": os.getenv("SMTP_SSL", "NOT_SET"),
        "is_configured": email_config.is_configured(),
        "all_env_vars_available": {
            "SMTP_SERVER": bool(os.getenv("SMTP_SERVER")),
            "SMTP_PORT": bool(os.getenv("SMTP_PORT")),
            "SENDER_EMAIL": bool(os.getenv("SENDER_EMAIL")),
            "SENDER_PASSWORD": bool(os.getenv("SENDER_PASSWORD")),
            "SMTP_SSL": bool(os.getenv("SMTP_SSL")),
        }
    }
    
    return config_status

# ---------------------------------------------------
# Health and Debug Endpoints
# ---------------------------------------------------
@app.get("/health")
def health_check():
    """Health check endpoint"""
    firebase_status = "initialized" if firebase_db else "not_initialized"
    
    status = {
        "status": "healthy", 
        "message": "Dynamic XGBoost API is running",
        "timestamp": datetime.now().isoformat(),
        "firebase": firebase_status,
        "fallback_data_available": True,
        "trained_models": len(models),
        "cached_datasets": len(data_cache),
        "system": {
            "python_version": os.sys.version,
            "platform": os.sys.platform
        }
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
            .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
            .status-ok { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .status-warning { background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
            .status-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        </style>
    </head>
    <body>
        <h1>Gas Monitoring XGBoost API</h1>
        
        <div class="status {'status-ok' if firebase_db else 'status-warning'}">
            <strong>Firebase Status:</strong> {'✅ Initialized' if firebase_db else '⚠️ Not Initialized'}
        </div>
        
        <p>Welcome to the <strong>Dynamic Sensor</strong> Gas Monitoring Backend with dynamic XGBoost training.</p>
        <p><strong>CORS Status:</strong> ✅ Enabled for all origins</p>
        <p><strong>Fallback Data:</strong> ✅ Available when Firebase is unavailable</p>

        <h2>API Documentation</h2>
        <ul>
            <li><a href="/docs">Swagger UI</a> (interactive API docs)</li>
            <li><a href="/redoc">ReDoc</a> (alternative docs)</li>
        </ul>

        <h2>Email Configuration Endpoints</h2>
        <ul>
            <li><code>GET /api/email-config</code> - Get current email configuration</li>
            <li><code>POST /api/email-config/update?confirm=true</code> - Update email config with confirmation</li>
            <li><code>POST /api/email-config/confirm-update</code> - Confirm and apply changes</li>
            <li><code>GET /api/email-config/test-connection</code> - Test email connection</li>
            <li><code>POST /api/test-email-alert/{sensor_id}</code> - Test email alerts</li>
        </ul>

        <h2>Main Endpoints</h2>
        
        <div class="endpoint">
            <h3>🎯 Dynamic XGBoost Training & Prediction</h3>
            <code>GET /api/xgboost/{sensor_id}?range=1month</code>
        </div>

        <div class="endpoint">
            <h3>📊 Correlation Analysis</h3>
            <code>GET /api/correlation_summary/{sensor_id}?range=1month&target_col=riskIndex</code><br>
            <code>GET /api/correlations_heatmap/{sensor_id}?range=1month</code><br>
            <code>GET /api/correlations/{sensor_id}?range=1month&target_col=riskIndex</code><br>
            <code>GET /api/correlation_scatterplots/{sensor_id}?range=1month&target_col=riskIndex</code>
        </div>

        <hr/>
        <p style="font-size: 0.9em; color: #666;">Powered by FastAPI & XGBoost | Gas Monitoring Project</p>
    </body>
    </html>
    """

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        firebase_success = initialize_firebase()
        if firebase_success:
            logger.info("✅ Firebase initialized successfully")
        else:
            logger.warning("⚠️ Firebase initialization failed - using fallback mode")
        
        # Log environment info for debugging
        logger.info(f"Python version: {os.sys.version}")
        logger.info(f"Platform: {os.sys.platform}")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")

# ===========
# Run the application
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)