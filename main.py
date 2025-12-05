import asyncio
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
  
import threading
import pickle
import base64
import joblib

import requests   
  
 
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
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Add specific CORS headers for SSE
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    # Add CORS headers for SSE
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Cache-Control"
    
    # Important for SSE
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    
    return response
    
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
class EmailJSEmailSystem:
    """Email system using EmailJS.com - WORKS ON RENDER"""
    
    def __init__(self, db):
        self.db = db
        self._recipient_cache = {}
        self._cache_timeout = 300
        
        # EmailJS configuration - get these from your EmailJS account
        self.service_id = os.environ.get("EMAILJS_SERVICE_ID")
        self.template_id = os.environ.get("EMAILJS_TEMPLATE_ID") 
        self.user_id = os.environ.get("EMAILJS_USER_ID")
        self.access_token = os.environ.get("EMAILJS_ACCESS_TOKEN", "")  # Optional
        
        # Your sender email (configured in EmailJS dashboard)
        self.sender_email = "gasvanguard@gmail.com"
        
        self._load_config()
    
    def _load_config(self):
        """Load configuration from Firebase"""
        try:
            config_ref = self.db.reference('/email_config/default')
            config = config_ref.get()
            
            if config and config.get('sender_email'):
                self.sender_email = config['sender_email']
                logger.info(f"Sender email from Firebase: {self.sender_email}")
                
        except Exception as e:
            logger.error(f"Error loading email config: {e}")
    
    def is_configured(self):
        """Check if EmailJS is properly configured"""
        return bool(self.service_id and self.template_id and self.user_id)
    
    def _get_recipients_for_sensor(self, sensor_id: str) -> List[str]:
        """Get email recipients from Firebase"""
        try:
            # Check cache first
            current_time = datetime.now()
            cache_data = self._recipient_cache.get(sensor_id, {})
            
            if (cache_data.get('timestamp') and 
                (current_time - cache_data['timestamp']).total_seconds() < self._cache_timeout):
                return cache_data['recipients']
            
            recipients = []
            
            # Try sensor-specific recipients
            sensor_ref = self.db.reference(f'/alert_recipients/{sensor_id}')
            sensor_data = sensor_ref.get()
            
            if sensor_data and 'emails' in sensor_data:
                emails = sensor_data['emails']
                if isinstance(emails, list):
                    recipients = [email for email in emails if self._is_valid_email(email)]
                elif isinstance(emails, str) and self._is_valid_email(emails):
                    recipients = [emails]
            
            # Fallback to global recipients
            if not recipients:
                global_ref = self.db.reference('/alert_recipients/global')
                global_data = global_ref.get()
                
                if global_data and 'emails' in global_data:
                    emails = global_data['emails']
                    if isinstance(emails, list):
                        recipients = [email for email in emails if self._is_valid_email(email)]
                    elif isinstance(emails, str) and self._is_valid_email(emails):
                        recipients = [emails]
            
            # Update cache
            self._recipient_cache[sensor_id] = {
                'recipients': recipients,
                'timestamp': current_time
            }
            
            logger.info(f"Found {len(recipients)} recipients for sensor {sensor_id}")
            return recipients
            
        except Exception as e:
            logger.error(f"Error fetching recipients for {sensor_id}: {e}")
            return []
    
    def _is_valid_email(self, email: str) -> bool:
        """Basic email validation"""
        return isinstance(email, str) and '@' in email and '.' in email
    
    def send_email(self, subject: str, message: str, recipient_emails: List[str]) -> Dict[str, Any]:
        """Send email using EmailJS HTTP API - FIXED RECIPIENT ISSUE"""
        if not self.is_configured():
            return {
                "success": False, 
                "error": "EmailJS not configured. Set EMAILJS_SERVICE_ID, EMAILJS_TEMPLATE_ID, and EMAILJS_USER_ID environment variables."
            }
        
        try:
            if isinstance(recipient_emails, str):
                recipient_emails = [recipient_emails]
            
            valid_recipients = [email for email in recipient_emails if self._is_valid_email(email)]
            if not valid_recipients:
                return {"success": False, "error": "No valid email addresses provided"}
            
            # EmailJS API endpoint
            url = "https://api.emailjs.com/api/v1.0/email/send"
            
            # FIXED: Ensure proper recipient mapping
            template_params = {
                "to_email": valid_recipients[0],  # This goes to lawrence.c.salazar@gmail.com
                "to_name": "Gas Alert Recipient",
                "from_name": "GasVanguard Alerts",
                "from_email": self.sender_email,  # This is gasvanguard@gmail.com (sender)
                "reply_to": self.sender_email,    # Add reply-to for better deliverability
                "subject": subject,
                "message": message,
                "sensor_id": "Gas Monitoring System",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Prepare request data
            email_data = {
                "service_id": self.service_id,
                "template_id": self.template_id,
                "user_id": self.user_id,
                "template_params": template_params
            }
            
            # Add access token if available (for newer accounts)
            if self.access_token:
                email_data["accessToken"] = self.access_token
            
            # Send email via EmailJS HTTP API
            headers = {
                "Content-Type": "application/json",
                "origin": "https://gasmonitoring-backend-evw0.onrender.com"
            }
            
            response = requests.post(url, json=email_data, headers=headers, timeout=30)
            
            logger.info(f"Attempting to send email from {self.sender_email} to {valid_recipients[0]}")
            
            if response.status_code == 200:
                logger.info(f"✅ Email sent via EmailJS from {self.sender_email} to {valid_recipients[0]}")
                
                return {
                    "success": True,
                    "message": f"Email sent from {self.sender_email} to {valid_recipients[0]}",
                    "from_email": self.sender_email,
                    "to_email": valid_recipients[0],
                    "recipient": valid_recipients[0],
                    "service": "EmailJS",
                    "cost": "FREE (200 emails/month)",
                    "works_on_render": True
                }
            else:
                error_msg = f"EmailJS API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "status_code": response.status_code,
                    "service": "EmailJS",
                    "debug_info": {
                        "from_email": self.sender_email,
                        "to_email": valid_recipients[0],
                        "template_id": self.template_id
                    }
                }
            
        except requests.exceptions.Timeout:
            error_msg = "EmailJS API timeout"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "service": "EmailJS"}
        except Exception as e:
            error_msg = f"EmailJS failed: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "service": "EmailJS"}
    
    def send_alert_email(self, sensor_id: str, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send alert email for sensor with risk data"""
        # Get recipients for this sensor
        recipients = self._get_recipients_for_sensor(sensor_id)
        
        if not recipients:
            return {
                "success": False, 
                "error": f"No recipients found for sensor {sensor_id}",
                "sensor_id": sensor_id
            }
        
        # Create alert content
        subject = self._create_alert_subject(sensor_id, risk_data)
        message = self._create_alert_message(sensor_id, risk_data)
        
        # Send to each recipient (EmailJS sends one email per call)
        successful_sends = 0
        errors = []
        sent_to_emails = []
        
        for recipient in recipients:
            result = self.send_email(subject, message, [recipient])
            if result["success"]:
                successful_sends += 1
                sent_to_emails.append(recipient)
                logger.info(f"✅ Alert sent to {recipient} for sensor {sensor_id}")
            else:
                errors.append(f"{recipient}: {result.get('error', 'Unknown error')}")
                logger.error(f"❌ Failed to send alert to {recipient}: {result.get('error')}")
        
        if successful_sends > 0:
            return {
                "success": True,
                "message": f"Alerts sent to {successful_sends} recipients",
                "sensor_id": sensor_id,
                "recipients_count": successful_sends,
                "sent_to": sent_to_emails,
                "from_email": self.sender_email,
                "total_recipients": len(recipients),
                "errors": errors if errors else None,
                "service": "EmailJS",
                "works_on_render": True
            }
        else:
            return {
                "success": False,
                "error": f"Failed to send alerts to any recipients: {', '.join(errors)}",
                "sensor_id": sensor_id,
                "from_email": self.sender_email,
                "intended_recipients": recipients,
                "service": "EmailJS"
            }
    
    def _create_alert_subject(self, sensor_id: str, risk_data: Dict[str, Any]) -> str:
        """Create alert email subject"""
        risk_level = risk_data.get('risk_level', 0)
        risk_label = risk_data.get('risk_label', 'Unknown')
        
        if risk_level >= 4:
            return f"🚨 CRITICAL GAS ALERT - Sensor {sensor_id} - {risk_label}"
        elif risk_level >= 3:
            return f"⚠️ HIGH RISK ALERT - Sensor {sensor_id} - {risk_label}"
        else:
            return f"ℹ️ GAS ALERT - Sensor {sensor_id} - {risk_label}"
    
    def _create_alert_message(self, sensor_id: str, risk_data: Dict[str, Any]) -> str:
        """Create alert email message body"""
        risk_level = risk_data.get('risk_level', 0)
        
        if risk_level >= 4:
            alert_header = "🚨 CRITICAL GAS ALERT 🚨"
        elif risk_level >= 3:
            alert_header = "⚠️ HIGH RISK ALERT ⚠️"
        else:
            alert_header = "ℹ️ GAS ALERT ℹ️"
        
        # Create detailed message
        message = f"""
        {alert_header}

        SENSOR ALERT DETAILS:
        • Sensor ID: {sensor_id}
        • Risk Level: {risk_data.get('risk_label', 'Unknown')}
        • Risk Index: {risk_data.get('predicted_risk_index', 0):.2f}
        • AQI: {risk_data.get('aqi', 0):.1f} ({risk_data.get('aqi_category', 'Unknown')})
        • Time: {risk_data.get('timestamp', 'Unknown')}

        SENSOR READINGS:"""
        
        # Add sensor readings
        input_data = risk_data.get('input_data', {})
        for key, value in input_data.items():
            message += f"\n• {key.title()}: {value}"
        
        message += f"""

            RECOMMENDED ACTIONS:
            {risk_data.get('recommendation', 'Please check the sensor immediately and ensure proper ventilation.')}

            ---
            Gas Monitoring System Alert
            From: {self.sender_email}
          
            """
        
        return message
    
    def clear_cache(self, sensor_id: str = None):
        """Clear recipient cache"""
        if sensor_id:
            self._recipient_cache.pop(sensor_id, None)
        else:
            self._recipient_cache.clear()
    
    def get_config_status(self):
        """Get current configuration status"""
        return {
            "service": "EmailJS",
            "configured": self.is_configured(),
            "sender_email": self.sender_email,
            "service_id_set": bool(self.service_id),
            "template_id_set": bool(self.template_id),
            "user_id_set": bool(self.user_id),
            "access_token_set": bool(self.access_token),
            "works_on_render": True,
            "cost": "FREE (200 emails/month)"
        }
 
# Email Configuration
@app.get("/api/email-config")
def get_email_config():
    """Get current email configuration"""
    try:
        email_system = EmailJSEmailSystem(db)
        return {
            "success": True,
            "service": "EmailJS",
            "sender_email": email_system.sender_email,
            "configured": email_system.is_configured(),
            "cost": "FREE (200 emails/month)",
            "works_on_render": True,
            "service_id_set": bool(email_system.service_id),
            "template_id_set": bool(email_system.template_id),
            "user_id_set": bool(email_system.user_id)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# Test Endpoints
@app.get("/test-email-emailjs")
def test_email_emailjs():
    """Test EmailJS integration - WILL WORK ON RENDER"""
    try:
        email_system = EmailJSEmailSystem(db)
        
        if not email_system.is_configured():
            return {
                "success": False,
                "error": "EmailJS not configured. Set environment variables.",
                "required_vars": ["EMAILJS_SERVICE_ID", "EMAILJS_TEMPLATE_ID", "EMAILJS_USER_ID"],
                "works_on_render": True
            }
        
        result = email_system.send_email(
            subject="EmailJS Test from Render",
            message="""
                🎉 EmailJS Integration Test Successful!

                This email was sent from your Render.com application using EmailJS HTTP API.

                ✅ Works on Render: YES
                ✅ Cost: FREE (200 emails/month)
                ✅ Service: EmailJS HTTP API
                ✅ Delivery: Instant

                Your gas alert system is now ready to use!
            """,
            recipient_emails=["lawrence.c.salazar@gmail.com"]
        )
        
        return result
        
    except Exception as e:
        return {"success": False, "error": str(e), "works_on_render": True}

@app.post("/api/test-email-alert/{sensor_id}")
def test_email_alert(sensor_id: str):
    """Test email alert functionality"""
    try:
        email_system = EmailJSEmailSystem(db)
        
        if not email_system.is_configured():
            return {"success": False, "error": "EmailJS not configured"}
        
        test_risk_data = {
            "risk_level": 4,
            "risk_label": "CRITICAL",
            "predicted_risk_index": 8.5,
            "aqi": 280.5,
            "aqi_category": "Unhealthy",
            "recommendation": "Air quality is unhealthy. Limit outdoor exposure. Use air purifiers if available.",
            "timestamp": datetime.now().isoformat(),
            "input_data": {
                "methane": 45.2,
                "co2": 420.5,
                "ammonia": 12.8,
                "humidity": 65.2,
                "temperature": 25.8
            }
        }
        
        result = email_system.send_alert_email(sensor_id, test_risk_data)
        return result
        
    except Exception as e:
        return {"success": False, "error": str(e)} 
        
# ---------------------------------------------------
# Threshold Management Functions
# ---------------------------------------------------

def fetch_exposure_thresholds():
    """Fetch exposure thresholds from Firebase Realtime Database"""
    try:
        if not firebase_db:
            logger.error("Firebase not initialized - cannot fetch thresholds")
            return get_default_thresholds()
        
        ref = db.reference('/threshold')
        thresholds_data = ref.get()
        
        if not thresholds_data:
            logger.warning("No threshold data found in Firebase, using defaults")
            return get_default_thresholds()
        
        # Transform Firebase data to match expected format
        exposure_limits = {}
        
        for gas_name, limits in thresholds_data.items():
            if isinstance(limits, dict):
                exposure_limits[gas_name.lower()] = {
                    'stel': limits.get('stel', 0),
                    'twa': limits.get('twa', 0),
                    'idlh': limits.get('idlh', 0)
                }
        
        logger.info(f"Loaded thresholds for gases: {list(exposure_limits.keys())}")
        return exposure_limits
        
    except Exception as e:
        logger.error(f"Error fetching exposure thresholds: {e}")
        return get_default_thresholds()

def get_default_thresholds():
    """Return default thresholds as fallback"""
    return {
        'methane': {'stel': 1000, 'twa': 1000, 'idlh': 50000},
        'co2': {'stel': 30000, 'twa': 5000, 'idlh': 40000},
        'ammonia': {'stel': 35, 'twa': 25, 'idlh': 500}
    }

def update_thresholds_in_firebase(gas_name, new_limits):
    """Update thresholds for a specific gas in Firebase"""
    try:
        if not firebase_db:
            logger.error("Firebase not initialized - cannot update thresholds")
            return False
        
        ref = db.reference(f'/threshold/{gas_name}')
        ref.set(new_limits)
        
        logger.info(f"Updated thresholds for {gas_name} in Firebase")
        return True
        
    except Exception as e:
        logger.error(f"Error updating thresholds for {gas_name}: {e}")
        return False

# ---------------------------------------------------
# Threshold Management Endpoints
# ---------------------------------------------------

@app.get("/api/thresholds")
def get_current_thresholds():
    """Get current exposure thresholds from Firebase"""
    try:
        thresholds = fetch_exposure_thresholds()
        return {
            "success": True,
            "thresholds": thresholds,
            "source": "firebase",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching thresholds: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/thresholds/{gas_name}")
def update_gas_thresholds(
    gas_name: str,
    stel: float = Query(..., description="Short Term Exposure Limit"),
    twa: float = Query(..., description="Time Weighted Average"),
    idlh: float = Query(..., description="Immediately Dangerous to Life and Health")
):
    """Update thresholds for a specific gas"""
    try:
        new_limits = {
            'stel': stel,
            'twa': twa,
            'idlh': idlh
        }
        
        success = update_thresholds_in_firebase(gas_name, new_limits)
        
        if success:
            return {
                "success": True,
                "message": f"Thresholds updated for {gas_name}",
                "new_limits": new_limits,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update thresholds in Firebase")
            
    except Exception as e:
        logger.error(f"Error updating thresholds for {gas_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Threshold update failed: {str(e)}")

@app.get("/api/thresholds/verify/{sensor_id}")
def verify_thresholds_with_sensor_data(sensor_id: str):
    """Verify current thresholds against recent sensor data"""
    try:
        # Get current thresholds
        thresholds = fetch_exposure_thresholds()
        
        # Get recent sensor data
        df = fetch_history(sensor_id, "1week")
        
        # Enhanced empty DataFrame check
        if df.empty or len(df) == 0:
            logger.warning(f"No data found for sensor {sensor_id}")
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": f"No recent data found for sensor {sensor_id}",
                    "sensor_id": sensor_id,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Check if required columns exist
        required_columns = ['methane', 'co2', 'ammonia']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns in sensor data: {missing_columns}")
        
        # Get the latest reading with safety checks
        try:
            latest_reading = df.iloc[-1] if not df.empty else None
            
            # Check if latest_reading is valid
            if latest_reading is None or pd.isna(latest_reading).all():
                return JSONResponse(
                    status_code=404,
                    content={
                        "success": False,
                        "error": "No valid sensor readings found",
                        "sensor_id": sensor_id,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
        except (IndexError, KeyError) as e:
            logger.error(f"Error accessing latest reading: {e}")
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": "Cannot access sensor readings",
                    "sensor_id": sensor_id,
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Check each gas against thresholds with enhanced error handling
        compliance_check = {}
        for gas in ['methane', 'co2', 'ammonia']:
            try:
                if gas in thresholds:
                    limits = thresholds[gas]
                    
                    # Safely get concentration value
                    concentration = 0
                    if gas in df.columns and gas in latest_reading:
                        concentration_val = latest_reading[gas]
                        if pd.isna(concentration_val) or concentration_val is None:
                            concentration = 0
                        else:
                            concentration = float(concentration_val)
                    
                    # Calculate compliance status
                    within_twa = concentration <= limits['twa']
                    within_stel = concentration <= limits['stel']
                    within_idlh = concentration <= limits['idlh']
                    
                    # Determine status
                    if within_twa:
                        status = 'SAFE'
                    elif within_stel:
                        status = 'WARNING'
                    elif within_idlh:
                        status = 'DANGER'
                    else:
                        status = 'CRITICAL'
                    
                    compliance_check[gas] = {
                        'current_value': concentration,
                        'twa_limit': limits['twa'],
                        'stel_limit': limits['stel'],
                        'idlh_limit': limits['idlh'],
                        'within_twa': within_twa,
                        'within_stel': within_stel,
                        'within_idlh': within_idlh,
                        'status': status,
                        'data_available': gas in df.columns and not pd.isna(latest_reading.get(gas))
                    }
                else:
                    # Handle missing threshold data
                    compliance_check[gas] = {
                        'current_value': 0,
                        'twa_limit': 0,
                        'stel_limit': 0,
                        'idlh_limit': 0,
                        'within_twa': True,
                        'within_stel': True,
                        'within_idlh': True,
                        'status': 'UNKNOWN',
                        'note': f'No threshold data available for {gas}'
                    }
                    
            except Exception as gas_error:
                logger.error(f"Error processing gas {gas}: {gas_error}")
                compliance_check[gas] = {
                    'current_value': 0,
                    'twa_limit': 0,
                    'stel_limit': 0,
                    'idlh_limit': 0,
                    'within_twa': True,
                    'within_stel': True,
                    'within_idlh': True,
                    'status': 'ERROR',
                    'note': f'Error processing {gas} data: {str(gas_error)}'
                }
        
        # Add data summary
        data_summary = {
            "total_readings": len(df),
            "date_range": {
                "start": df['timestamp'].min().isoformat() if 'timestamp' in df.columns and not df.empty else "Unknown",
                "end": df['timestamp'].max().isoformat() if 'timestamp' in df.columns and not df.empty else "Unknown"
            },
            "data_quality": {
                "has_methane": 'methane' in df.columns,
                "has_co2": 'co2' in df.columns,
                "has_ammonia": 'ammonia' in df.columns
            }
        }
        
        return {
            "success": True,
            "sensor_id": sensor_id,
            "timestamp": datetime.now().isoformat(),
            "compliance_check": compliance_check,
            "thresholds_source": "firebase",
            "data_summary": data_summary
        }
        
    except Exception as e:
        logger.error(f"Error verifying thresholds for {sensor_id}: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error details: {str(e)}")
        
        # Return proper JSON response with CORS-friendly format
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Internal server error: {str(e)}",
                "sensor_id": sensor_id,
                "timestamp": datetime.now().isoformat(),
                "error_type": type(e).__name__
            }
        )      
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
    Risk index based on regulatory exposure limits from Firebase
    """
    # Fetch current thresholds from Firebase
    exposure_limits = fetch_exposure_thresholds()
    
    risks = []
    
    # Calculate compliance risk for each gas
    for gas, concentration in [('methane', methane), ('co2', co2), ('ammonia', ammonia)]:
        if gas not in exposure_limits:
            logger.warning(f"No thresholds found for {gas}, using default risk calculation")
            risks.append(5)  # Medium risk as fallback
            continue
            
        limits = exposure_limits[gas]
        
        if concentration <= limits['twa']:
            risk = 1  # Within safe limits
        elif concentration <= limits['stel']:
            risk = 3  # Exceeded TWA but within STEL
        elif concentration <= limits['idlh']:
            risk = 6  # Exceeded STEL but below IDLH
        else:
            risk = 9  # Immediately dangerous
            
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
        
# def convert_risk_to_aqi(risk_index):
    # """
    # Piecewise Linear Scaling
    # Convert risk index to AQI (0-500 scale) 
    # """
    
   ## AQI categories and their corresponding risk index ranges 
    # aqi_thresholds = [
        # (0, 50, 0, 3),      # Good: 0-50 AQI, 0-3 risk
        # (51, 100, 3.1, 7),  # Moderate: 51-100 AQI, 3.1-7 risk  
        # (101, 150, 7.1, 9), # Unhealthy for Sensitive: 101-150 AQI, 7.1-9 risk
        # (151, 500, 9.1, 10) # Unhealthy/Very Unhealthy: 151-500 AQI, 9.1-10 risk
    # ]
    
    # for aqi_min, aqi_max, risk_min, risk_max in aqi_thresholds:
        # if risk_min <= risk_index <= risk_max:
           # Linear interpolation within this category
            # aqi = aqi_min + (risk_index - risk_min) * (aqi_max - aqi_min) / (risk_max - risk_min)
            # return min(max(aqi, 0), 500)
    
   ## Fallback: cap at extremes
    # if risk_index < 1:
        # return 10
    # else:
        # return 500 
# ---------------------------------------------------
# AQI Threshold Management Functions
# ---------------------------------------------------

def fetch_aqi_thresholds():
    """Fetch AQI thresholds from Firebase Realtime Database"""
    try:
        if not firebase_db:
            logger.error("Firebase not initialized - cannot fetch AQI thresholds")
            return get_default_aqi_thresholds()
        
        ref = db.reference('/aqi_thresholds')
        aqi_thresholds_data = ref.get()
        
        if not aqi_thresholds_data:
            logger.warning("No AQI threshold data found in Firebase, using defaults")
            return get_default_aqi_thresholds()
        
        # Validate and transform the data
        validated_thresholds = []
        for threshold in aqi_thresholds_data:
            if all(key in threshold for key in ['aqi_min', 'aqi_max', 'risk_min', 'risk_max']):
                validated_thresholds.append((
                    float(threshold['aqi_min']),
                    float(threshold['aqi_max']),
                    float(threshold['risk_min']),
                    float(threshold['risk_max'])
                ))
        
        if not validated_thresholds:
            logger.warning("Invalid AQI threshold structure in Firebase, using defaults")
            return get_default_aqi_thresholds()
        
        logger.info(f"Loaded {len(validated_thresholds)} AQI thresholds from Firebase")
        return validated_thresholds
        
    except Exception as e:
        logger.error(f"Error fetching AQI thresholds: {e}")
        return get_default_aqi_thresholds()

def get_default_aqi_thresholds():
    """Return default AQI thresholds as fallback"""
    return [
        (0, 50, 0, 3),      # Good: 0-50 AQI, 0-3 risk
        (51, 100, 3.1, 7),  # Moderate: 51-100 AQI, 3.1-7 risk  
        (101, 150, 7.1, 9), # Unhealthy for Sensitive: 101-150 AQI, 7.1-9 risk
        (151, 500, 9.1, 10) # Unhealthy/Very Unhealthy: 151-500 AQI, 9.1-10 risk
    ]

def update_aqi_thresholds_in_firebase(new_thresholds):
    """Update AQI thresholds in Firebase"""
    try:
        if not firebase_db:
            logger.error("Firebase not initialized - cannot update AQI thresholds")
            return False
        
        # Validate the new thresholds structure
        validated_thresholds = []
        for threshold in new_thresholds:
            if (isinstance(threshold, (list, tuple)) and len(threshold) == 4 and
                all(isinstance(x, (int, float)) for x in threshold)):
                validated_thresholds.append({
                    'aqi_min': float(threshold[0]),
                    'aqi_max': float(threshold[1]),
                    'risk_min': float(threshold[2]),
                    'risk_max': float(threshold[3])
                })
        
        if not validated_thresholds:
            logger.error("Invalid AQI thresholds format provided")
            return False
        
        ref = db.reference('/aqi_threshold')
        ref.set(validated_thresholds)
        
        logger.info(f"Updated AQI thresholds in Firebase with {len(validated_thresholds)} categories")
        return True
        
    except Exception as e:
        logger.error(f"Error updating AQI thresholds in Firebase: {e}")
        return False

def convert_risk_to_aqi(risk_index):
    """
    Piecewise Linear Scaling
    Convert risk index to AQI (0-500 scale) using thresholds from Firebase
    """
    # Fetch thresholds from Firebase (with fallback to defaults)
    aqi_thresholds = fetch_aqi_thresholds()
    
    for aqi_min, aqi_max, risk_min, risk_max in aqi_thresholds:
        if risk_min <= risk_index <= risk_max:
            # Linear interpolation within this category
            aqi = aqi_min + (risk_index - risk_min) * (aqi_max - aqi_min) / (risk_max - risk_min)
            return min(max(aqi, 0), 500)
    
    # Fallback: cap at extremes
    if risk_index < aqi_thresholds[0][2]:  # Below first risk_min
        return max(aqi_thresholds[0][0], 0)
    else:  # Above last risk_max
        return min(aqi_thresholds[-1][1], 500)

# ---------------------------------------------------
# AQI Threshold Management Endpoints
# ---------------------------------------------------

@app.get("/api/aqi-thresholds")
def get_current_aqi_thresholds():
    """Get current AQI thresholds from Firebase"""
    try:
        thresholds = fetch_aqi_thresholds()
        return {
            "success": True,
            "aqi_thresholds": [
                {
                    "aqi_min": t[0],
                    "aqi_max": t[1], 
                    "risk_min": t[2],
                    "risk_max": t[3],
                    "category": get_aqi_category_name(t[0], t[1])
                }
                for t in thresholds
            ],
            "source": "firebase",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching AQI thresholds: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/aqi-thresholds")
def update_aqi_thresholds(new_thresholds: List[Dict[str, float]]):
    """Update AQI thresholds in Firebase"""
    try:
        # Convert the list of dicts to the expected format
        thresholds_list = []
        for threshold in new_thresholds:
            if all(key in threshold for key in ['aqi_min', 'aqi_max', 'risk_min', 'risk_max']):
                thresholds_list.append((
                    threshold['aqi_min'],
                    threshold['aqi_max'], 
                    threshold['risk_min'],
                    threshold['risk_max']
                ))
        
        success = update_aqi_thresholds_in_firebase(thresholds_list)
        
        if success:
            return {
                "success": True,
                "message": f"AQI thresholds updated successfully",
                "new_thresholds": new_thresholds,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update AQI thresholds in Firebase")
            
    except Exception as e:
        logger.error(f"Error updating AQI thresholds: {e}")
        raise HTTPException(status_code=500, detail=f"AQI threshold update failed: {str(e)}")

@app.post("/api/aqi-thresholds/reset")
def reset_aqi_thresholds():
    """Reset AQI thresholds to default values"""
    try:
        default_thresholds = get_default_aqi_thresholds()
        success = update_aqi_thresholds_in_firebase(default_thresholds)
        
        if success:
            return {
                "success": True,
                "message": "AQI thresholds reset to default values",
                "default_thresholds": [
                    {
                        "aqi_min": t[0],
                        "aqi_max": t[1],
                        "risk_min": t[2], 
                        "risk_max": t[3],
                        "category": get_aqi_category_name(t[0], t[1])
                    }
                    for t in default_thresholds
                ],
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to reset AQI thresholds")
            
    except Exception as e:
        logger.error(f"Error resetting AQI thresholds: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

def get_aqi_category_name(aqi_min, aqi_max):
    """Get human-readable AQI category name"""
    categories = {
        (0, 50): "Good",
        (51, 100): "Moderate", 
        (101, 150): "Unhealthy for Sensitive Groups",
        (151, 200): "Unhealthy",
        (201, 300): "Very Unhealthy",
        (301, 500): "Hazardous"
    }
    
    for (min_val, max_val), name in categories.items():
        if aqi_min == min_val and aqi_max == max_val:
            return name
    
    return "Custom Range"


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
import threading
from datetime import datetime

def save_alert_to_firebase(sensor_id: str, risk_data: Dict[str, Any], email_result: Dict[str, Any]):
    """Save alert record to Firebase Realtime Database"""
    try:
        alert_data = {
            "sensor_id": sensor_id,
            "risk_index": risk_data.get('predicted_risk_index', 0),
            "risk_level": risk_data.get('risk_level', 0),
            "risk_label": risk_data.get('risk_label', 'Unknown'),
            "aqi": risk_data.get('aqi', 0),
            "aqi_category": risk_data.get('aqi_category', 'Unknown'),
            "timestamp": datetime.now().isoformat(),
            "email_sent": email_result.get('success', False),
            "email_service": email_result.get('service', 'Unknown'),
            "recipients_count": email_result.get('recipients_count', 0),
            "recipients": email_result.get('sent_to', []),
            "email_error": email_result.get('error'),
            "sensor_readings": risk_data.get('input_data', {}),
            "recommendation": risk_data.get('recommendation', '')
        }
        
        # Generate unique alert ID
        alert_id = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Save to Firebase: /alerts/{sensor_id}/{alert_id}
        alert_ref = db.reference(f'/alerts/{sensor_id}/{alert_id}')
        alert_ref.set(alert_data)
        
        logger.info(f"✅ Alert saved to Firebase: {sensor_id}/{alert_id}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to save alert to Firebase: {e}")
        return False

def send_alert_async(sensor_id: str, risk_data: Dict[str, Any]):
    """Send alert email in background thread and save to Firebase"""
    def _send_alert():
        try:
            email_system = EmailJSEmailSystem(db)
            email_result = {}
            
            if email_system.is_configured():
                email_result = email_system.send_alert_email(sensor_id, risk_data)
                if email_result["success"]:
                    logger.info(f"✅ Async alert sent for sensor {sensor_id} - Risk: {risk_data['predicted_risk_index']:.2f}")
                else:
                    logger.error(f"❌ Async alert failed for {sensor_id}: {email_result.get('error')}")
            else:
                email_result = {
                    "success": False,
                    "error": "Email system not configured",
                    "service": "EmailJS"
                }
                logger.warning(f"Email system not configured - alert not sent for {sensor_id}")
            
            # Save alert record to Firebase regardless of email success
            save_alert_to_firebase(sensor_id, risk_data, email_result)
            
        except Exception as e:
            logger.error(f"Async alert error for {sensor_id}: {e}")
            # Still try to save to Firebase even if email fails
            error_result = {
                "success": False,
                "error": str(e),
                "service": "EmailJS"
            }
            save_alert_to_firebase(sensor_id, risk_data, error_result)
    
    thread = threading.Thread(target=_send_alert, daemon=True)
    thread.start()

# Alert throttling to prevent spam
_last_alert_times = {}
_alert_throttle_minutes = 30  # Only send one alert per sensor every 30 minutes

def should_send_alert(sensor_id: str, risk_index: float) -> bool:
    """Check if we should send alert (prevent spam)"""
    global _last_alert_times
    
    current_time = datetime.now()
    last_alert = _last_alert_times.get(sensor_id)
    
    # Always send critical alerts (risk >= 4)
    if risk_index >= 4:
        return True
    
    # For non-critical alerts, check throttle time
    if not last_alert or (current_time - last_alert).total_seconds() > (_alert_throttle_minutes * 60):
        _last_alert_times[sensor_id] = current_time
        return True
    
    return False
@app.get("/api/alerts/{sensor_id}")
def get_sensor_alerts(
    sensor_id: str,
    limit: int = Query(50, description="Number of alerts to return"),
    days: int = Query(7, description="Number of past days to include")
):
    """Get alert history for a specific sensor"""
    try:
        alerts_ref = db.reference(f'/alerts/{sensor_id}')
        alerts_data = alerts_ref.order_by_child('timestamp').limit_to_last(limit).get()
        
        if not alerts_data:
            return {
                "success": True,
                "sensor_id": sensor_id,
                "alerts": [],
                "count": 0,
                "message": "No alerts found for this sensor"
            }
        
        # Convert to list and sort by timestamp (newest first)
        alerts_list = []
        for alert_id, alert_data in alerts_data.items():
            alerts_list.append({
                "alert_id": alert_id,
                **alert_data
            })
        
        # Sort by timestamp descending (newest first)
        alerts_list.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Filter by days if specified
        if days > 0:
            cutoff_date = datetime.now() - timedelta(days=days)
            alerts_list = [alert for alert in alerts_list 
                         if datetime.fromisoformat(alert['timestamp'].replace('Z', '+00:00')) > cutoff_date]
        
        return {
            "success": True,
            "sensor_id": sensor_id,
            "alerts": alerts_list,
            "count": len(alerts_list),
            "timeframe_days": days
        }
        
    except Exception as e:
        logger.error(f"Error fetching alerts for {sensor_id}: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/alerts")
def get_all_alerts(
    limit: int = Query(100, description="Number of alerts to return"),
    risk_level: int = Query(None, description="Filter by risk level")
):
    """Get all alerts across all sensors"""
    try:
        alerts_ref = db.reference('/alerts')
        all_alerts_data = alerts_ref.get()
        
        if not all_alerts_data:
            return {
                "success": True,
                "alerts": [],
                "count": 0,
                "message": "No alerts found"
            }
        
        # Flatten all alerts from all sensors
        all_alerts = []
        for sensor_id, sensor_alerts in all_alerts_data.items():
            for alert_id, alert_data in sensor_alerts.items():
                all_alerts.append({
                    "sensor_id": sensor_id,
                    "alert_id": alert_id,
                    **alert_data
                })
        
        # Sort by timestamp descending (newest first)
        all_alerts.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Filter by risk level if specified
        if risk_level is not None:
            all_alerts = [alert for alert in all_alerts if alert.get('risk_level') == risk_level]
        
        # Apply limit
        all_alerts = all_alerts[:limit]
        
        return {
            "success": True,
            "alerts": all_alerts,
            "count": len(all_alerts),
            "total_sensors": len(all_alerts_data.keys()),
            "risk_level_filter": risk_level
        }
        
    except Exception as e:
        logger.error(f"Error fetching all alerts: {e}")
        return {"success": False, "error": str(e)}

@app.delete("/api/alerts/{sensor_id}/{alert_id}")
def delete_alert(sensor_id: str, alert_id: str):
    """Delete a specific alert"""
    try:
        alert_ref = db.reference(f'/alerts/{sensor_id}/{alert_id}')
        alert_ref.delete()
        
        logger.info(f"Alert deleted: {sensor_id}/{alert_id}")
        
        return {
            "success": True,
            "message": f"Alert {alert_id} deleted for sensor {sensor_id}"
        }
        
    except Exception as e:
        logger.error(f"Error deleting alert {sensor_id}/{alert_id}: {e}")
        return {"success": False, "error": str(e)}

@app.get("/test-alert-saving/{sensor_id}")
def test_alert_saving(sensor_id: str):
    """Test the alert system with Firebase saving"""
    try:
        # Test with high risk data
        test_risk_data = {
            "predicted_risk_index": 8.2,
            "aqi": 290.5,
            "risk_level": 4,
            "risk_label": "CRITICAL",
            "aqi_category": "Unhealthy",
            "recommendation": "Test recommendation for Firebase saving",
            "input_data": {
                "methane": 48.7,
                "co2": 435.2,
                "ammonia": 15.3,
                "humidity": 68.9,
                "temperature": 26.4
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Send alert (this will save to Firebase)
        send_alert_async(sensor_id, test_risk_data)
        
        return {
            "success": True,
            "sensor_id": sensor_id,
            "risk_index": test_risk_data["predicted_risk_index"],
            "message": "Test alert sent - check Firebase for saved record",
            "firebase_path": f"/alerts/{sensor_id}/alert_*",
            "expected_data": {
                "sensor_id": sensor_id,
                "risk_index": test_risk_data["predicted_risk_index"],
                "risk_level": test_risk_data["risk_level"],
                "email_sent": True,
                "timestamp": "Should be current time"
            }
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}
        
@app.get("/api/predict/{sensor_id}")
def predict_risk_endpoint(
    sensor_id: str,
    methane: float = Query(..., description="Methane level"),
    co2: float = Query(..., description="CO2 level"),
    ammonia: float = Query(..., description="Ammonia level"),
    humidity: float = Query(..., description="Humidity percentage"),
    temperature: float = Query(..., description="Temperature in Celsius")
):
    """Predict risk index using pre-trained model and send alerts if risk > 1"""
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
            "recommendation": recommendation,
            "input_data": sensor_data,  # Include input data for email alerts
            "timestamp": datetime.now().isoformat()
        }
        
        # Send email alert if risk index > 1 and not throttled
        if risk_index > 1 and should_send_alert(sensor_id, risk_index):
            enhanced_result["alert_triggered"] = True
            enhanced_result["alert_message"] = "Alert being sent via EmailJS"
            
            # Send alert asynchronously (don't wait for it to complete)
            send_alert_async(sensor_id, enhanced_result)
            
            logger.info(f"🚨 Alert triggered for sensor {sensor_id} - Risk: {risk_index:.2f}")
        else:
            enhanced_result["alert_triggered"] = False
            if risk_index <= 1:
                enhanced_result["alert_message"] = "No alert needed (risk <= 1)"
            else:
                # Alert throttled
                last_alert = _last_alert_times.get(sensor_id)
                if last_alert:
                    minutes_since_last = (datetime.now() - last_alert).total_seconds() / 60
                    minutes_remaining = _alert_throttle_minutes - minutes_since_last
                    enhanced_result["alert_message"] = f"Alert throttled ({minutes_remaining:.0f} min remaining)"
                else:
                    enhanced_result["alert_message"] = "Alert throttled"
        
        return enhanced_result
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint for {sensor_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# @app.get("/api/predict/{sensor_id}")
# def predict_risk_endpoint(
    # sensor_id: str,
    # methane: float = Query(..., description="Methane level"),
    # co2: float = Query(..., description="CO2 level"),
    # ammonia: float = Query(..., description="Ammonia level"),
    # humidity: float = Query(..., description="Humidity percentage"),
    # temperature: float = Query(..., description="Temperature in Celsius")
# ):
    # """Predict risk index using pre-trained model and send alerts if risk > 1"""
    # try:
        # sensor_data = {
            # "methane": methane,
            # "co2": co2,
            # "ammonia": ammonia,
            # "humidity": humidity,
            # "temperature": temperature
        # }
        
        # prediction_result = predict_risk_index(sensor_id, sensor_data)
        
        # if "error" in prediction_result:
            # raise HTTPException(status_code=404, detail=prediction_result["error"])
        
        # Enhance with AQI and risk category
        # risk_index = prediction_result["predicted_risk_index"]
        # aqi = convert_risk_to_aqi(risk_index)
        # risk_level, risk_label, aqi_category, recommendation = calculate_risk_category(risk_index)
        
        # enhanced_result = {
            # **prediction_result,
            # "aqi": aqi,
            # "aqi_category": aqi_category,
            # "risk_level": risk_level,
            # "risk_label": risk_label,
            # "recommendation": recommendation,
            # "input_data": sensor_data,  # Include input data for email alerts
            # "timestamp": datetime.now().isoformat()
        # }
         
        # return enhanced_result
        
    # except Exception as e:
        # logger.error(f"Error in prediction endpoint for {sensor_id}: {e}")
        # raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
@app.post("/api/predict/{sensor_id}")
def predict_risk_bulk(
    sensor_id: str,
    sensor_readings: List[Dict[str, float]]
):
    """Bulk prediction for multiple sensor readings"""
    try:
        results = [] 
        
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
                results.append(enhanced_result)
            else:
                results.append({"error": result["error"], "input_data": reading})
        
        return {
            "sensor_ID": sensor_id,
            "predictions": results,
            "total_predictions": len(results),
            "successful_predictions": len([r for r in results if "error" not in r]),
     
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
###  
# Email Configuration Endpoints
 
 ### Forecast Live preview
# Add these functions to your FastAPI backend

def generate_forecast(sensor_id: str, steps: int = 24):
    """Generate future forecast using trained XGBoost model - IMPROVED"""
    try:
        # Load the trained model
        model = load_model_from_firebase(sensor_id)
        if not model:
            logger.warning(f"No trained model found for sensor {sensor_id}, using default forecast")
            return generate_default_forecast(sensor_id, steps)
        
        # Load model metadata
        metadata = load_model_metadata(sensor_id)
        if not metadata:
            logger.warning(f"No metadata found for sensor {sensor_id}")
            return generate_default_forecast(sensor_id, steps)
        
        feature_columns = metadata.get('feature_columns', [])
        
        # Get recent historical data for context
        df = fetch_history(sensor_id, "1week")
        if df.empty:
            logger.warning(f"No historical data for {sensor_id}, using sample data")
            df = generate_sample_data()
            if df.empty:
                return {"error": "No data available for forecasting"}
        
        # Prepare the latest data point as starting point
        latest_data = df.iloc[-1] if len(df) > 0 else pd.Series()
        
        # Create future timestamps
        last_timestamp = latest_data.get('timestamp', datetime.now()) if not latest_data.empty else datetime.now()
        if isinstance(last_timestamp, pd.Timestamp):
            last_timestamp = last_timestamp.to_pydatetime()
        
        future_timestamps = [last_timestamp + timedelta(hours=i) for i in range(1, steps + 1)]
        
        forecasts = []
        current_features = {}
        
        # Initialize with latest values or averages
        for feature in feature_columns:
            if feature in latest_data and not pd.isna(latest_data[feature]):
                current_features[feature] = float(latest_data[feature])
            elif feature in df.columns:
                current_features[feature] = float(df[feature].mean())
            else:
                current_features[feature] = 0.0
        
        # Generate forecasts
        for i in range(steps):
            try:
                # Create input DataFrame for prediction
                input_df = pd.DataFrame([current_features])
                
                # Ensure all required features are present
                for feature in feature_columns:
                    if feature not in input_df.columns:
                        input_df[feature] = current_features.get(feature, 0)
                
                input_df = input_df[feature_columns]
                
                # Predict next risk index
                risk_index = float(model.predict(input_df)[0])
                
            except Exception as pred_error:
                logger.warning(f"Prediction failed for step {i}: {pred_error}, using historical average")
                risk_index = float(df['riskIndex'].mean() if 'riskIndex' in df.columns else 5.0)
            
            # Convert to AQI
            aqi = convert_risk_to_aqi(risk_index)
            risk_level, risk_label, aqi_category, recommendation = calculate_risk_category(risk_index)
            
            # Add to forecasts
            forecast_entry = {
                "timestamp": future_timestamps[i].isoformat(),
                "predicted_risk_index": risk_index,
                "aqi": float(aqi),
                "aqi_category": aqi_category,
                "risk_level": risk_level,
                "risk_label": risk_label,
                "step": i + 1,
                "hour_offset": i + 1
            }
            forecasts.append(forecast_entry)
            
            # Update features for next prediction (simplified auto-regressive)
            for feature in feature_columns:
                if feature in ['methane', 'co2', 'ammonia', 'humidity', 'temperature']:
                    # Add small variation based on historical patterns
                    if feature in df.columns and len(df) > 1:
                        mean_val = df[feature].mean()
                        std_val = df[feature].std()
                        if std_val > 0:
                            # Add random walk with mean reversion
                            current_val = current_features[feature]
                            new_val = current_val * 0.9 + mean_val * 0.1 + np.random.normal(0, std_val * 0.1)
                            current_features[feature] = max(new_val, 0)
        
        # Get historical risk indices for comparison
        historical_risks = []
        if 'riskIndex' in df.columns and 'timestamp' in df.columns and len(df) > 0:
            historical_data = df.tail(min(24, len(df)))  # Last 24 hours or less
            for idx, row in historical_data.iterrows():
                try:
                    timestamp = row['timestamp']
                    if hasattr(timestamp, 'isoformat'):
                        timestamp_str = timestamp.isoformat()
                    else:
                        timestamp_str = str(timestamp)
                    
                    historical_risks.append({
                        "timestamp": timestamp_str,
                        "risk_index": float(row['riskIndex']),
                        "aqi": float(convert_risk_to_aqi(row['riskIndex'])),
                        "is_historical": True
                    })
                except Exception as e:
                    continue
        
        return {
            "sensor_id": sensor_id,
            "historical_data": historical_risks,
            "forecast": forecasts,
            "forecast_hours": steps,
            "generated_at": datetime.now().isoformat(),
            "model_info": {
                "model_type": "XGBoost",
                "features_used": feature_columns,
                "last_trained": metadata.get('last_trained'),
                "data_points": len(df)
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating forecast for {sensor_id}: {e}")
        return generate_default_forecast(sensor_id, steps)

def generate_default_forecast(sensor_id: str, steps: int = 24):
    """Generate a default forecast when model/data is unavailable"""
    try:
        current_time = datetime.now()
        
        # Create sample forecast data
        forecasts = []
        for i in range(steps):
            # Create a simple sine wave pattern for demonstration
            base_risk = 5.0 + 3.0 * np.sin(i * np.pi / 6)  # 12-hour cycle
            
            forecast_time = current_time + timedelta(hours=i+1)
            risk_index = max(1.0, min(10.0, base_risk + np.random.normal(0, 0.5)))
            aqi = convert_risk_to_aqi(risk_index)
            risk_level, risk_label, aqi_category, _ = calculate_risk_category(risk_index)
            
            forecasts.append({
                "timestamp": forecast_time.isoformat(),
                "predicted_risk_index": float(risk_index),
                "aqi": float(aqi),
                "aqi_category": aqi_category,
                "risk_level": risk_level,
                "risk_label": risk_label,
                "step": i + 1,
                "hour_offset": i + 1
            })
        
        # Create some historical data
        historical_risks = []
        for i in range(12, 0, -1):  # Last 12 hours
            hist_time = current_time - timedelta(hours=i)
            hist_risk = 4.0 + 2.0 * np.sin(i * np.pi / 12)
            historical_risks.append({
                "timestamp": hist_time.isoformat(),
                "risk_index": float(hist_risk),
                "aqi": float(convert_risk_to_aqi(hist_risk)),
                "is_historical": True
            })
        
        return {
            "sensor_id": sensor_id,
            "historical_data": historical_risks,
            "forecast": forecasts,
            "forecast_hours": steps,
            "generated_at": current_time.isoformat(),
            "model_info": {
                "model_type": "Demo (XGBoost model not available)",
                "features_used": ["methane", "co2", "ammonia", "humidity", "temperature"],
                "note": "Using demo data - train model for accurate predictions"
            },
            "is_demo_data": True
        }
        
    except Exception as e:
        logger.error(f"Error generating default forecast: {e}")
        return {"error": f"Forecast generation failed: {str(e)}"}
def get_live_predictions(sensor_id: str):
    """Get live predictions for the current sensor readings"""
    try:
        # Get latest sensor reading
        latest_reading_ref = db.reference(f'/sensorReadings')
        readings_data = latest_reading_ref.order_by_child('sensorID').equal_to(sensor_id).limit_to_last(1).get()
        
        if not readings_data:
            return {"error": f"No live readings found for sensor {sensor_id}"}
        
        # Get the latest reading
        latest_key = list(readings_data.keys())[0]
        latest_reading = readings_data[latest_key]
        
        # Prepare sensor data for prediction
        sensor_data = {
            "methane": latest_reading.get('methane', 0),
            "co2": latest_reading.get('co2', 0),
            "ammonia": latest_reading.get('ammonia', 0),
            "humidity": latest_reading.get('humidity', 0),
            "temperature": latest_reading.get('temperature', 0)
        }
        
        # Predict risk index
        prediction_result = predict_risk_index(sensor_id, sensor_data, use_regulatory_calculation=False)
        
        if "error" in prediction_result:
            # Fallback to regulatory calculation
            prediction_result = predict_risk_index(sensor_id, sensor_data, use_regulatory_calculation=True)
        
        # Add live data info
        prediction_result["is_live"] = True
        prediction_result["reading_timestamp"] = latest_reading.get('timestamp')
        prediction_result["sensor_reading_id"] = latest_key
        
        return prediction_result
        
    except Exception as e:
        logger.error(f"Error getting live predictions for {sensor_id}: {e}")
        return {"error": f"Live prediction failed: {str(e)}"}

# Add these endpoints to your FastAPI app

@app.get("/api/forecast/{sensor_id}")
def get_forecast(
    sensor_id: str,
    steps: int = Query(24, description="Number of forecast steps (hours)", ge=1, le=168)
):
    """Get predictive forecast for sensor"""
    try:
        result = generate_forecast(sensor_id, steps)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except Exception as e:
        logger.error(f"Error in forecast endpoint for {sensor_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

@app.get("/api/live-predict/{sensor_id}")
def get_live_prediction_endpoint(sensor_id: str):
    """Get live prediction for current sensor readings"""
    try:
        result = get_live_predictions(sensor_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except Exception as e:
        logger.error(f"Error in live prediction for {sensor_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Live prediction failed: {str(e)}")

# @app.get("/api/forecast/stream/{sensor_id}")
# async def stream_forecast_updates(sensor_id: str):
    # """Server-Sent Events stream for live forecast updates - FIXED CORS"""
    # async def event_generator():
        # while True:
            # try:
                ##Get latest forecast (with timeout)
                # forecast_data = generate_forecast(sensor_id, 12)
                # live_data = get_live_predictions(sensor_id)
                
                # if "error" not in forecast_data and "error" not in live_data:
                    # data = {
                        # "forecast": forecast_data,
                        # "live_prediction": live_data,
                        # "timestamp": datetime.now().isoformat(),
                        # "type": "update"
                    # }
                    # yield f"data: {json.dumps(data)}\n\n"
                # else:
                    # error_msg = forecast_data.get('error', live_data.get('error', 'Data unavailable'))
                    # yield f"data: {json.dumps({'error': error_msg, 'timestamp': datetime.now().isoformat()})}\n\n"
                
               ## Wait before next update
                # await asyncio.sleep(30)  # Reduce to 30 seconds for better responsiveness
                
            # except asyncio.CancelledError:
                # logger.info(f"SSE connection cancelled for {sensor_id}")
                # break
            # except Exception as e:
                # logger.error(f"Error in forecast stream: {e}")
                # yield f"data: {json.dumps({'error': str(e), 'timestamp': datetime.now().isoformat()})}\n\n"
                # await asyncio.sleep(5)
    
    # return StreamingResponse(
        # event_generator(),
        # media_type="text/event-stream",
        # headers={
            # "Cache-Control": "no-cache",
            # "Connection": "keep-alive",
            # "X-Accel-Buffering": "no",  # Disable buffering for nginx
            # "Access-Control-Allow-Origin": "*",
            # "Access-Control-Allow-Credentials": "true",
            # "Access-Control-Expose-Headers": "*",
            # "Content-Type": "text/event-stream; charset=utf-8"
        # }
    # )

# #Add OPTIONS endpoint for SSE CORS preflight
# @app.options("/api/forecast/stream/{sensor_id}")
# async def sse_preflight(sensor_id: str):
    # return JSONResponse(
        # content={"status": "ok"},
        # headers={
            # "Access-Control-Allow-Origin": "*",
            # "Access-Control-Allow-Methods": "GET, OPTIONS",
            # "Access-Control-Allow-Headers": "*",
            # "Access-Control-Max-Age": "86400",
            # "Access-Control-Allow-Credentials": "true"
        # }
    # )
 
# Remove or comment out the SSE endpoints and add these polling endpoints

@app.get("/api/forecast/poll/{sensor_id}")
async def poll_forecast_updates(sensor_id: str):
    """Polling endpoint for forecast updates - more reliable than SSE"""
    try:
        # Get latest forecast
        forecast_data = generate_forecast(sensor_id, 12)
        live_data = get_live_predictions(sensor_id)
        
        if "error" in forecast_data or "error" in live_data:
            error_msg = forecast_data.get('error', live_data.get('error', 'Data unavailable'))
            return {
                "success": False,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
        
        return {
            "success": True,
            "forecast": forecast_data,
            "live_prediction": live_data,
            "timestamp": datetime.now().isoformat(),
            "type": "update",
            "next_poll_in": 30  # seconds until next recommended poll
        }
        
    except Exception as e:
        logger.error(f"Error in forecast poll for {sensor_id}: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/forecast/batch/{sensor_id}")
async def get_batch_forecast_data(sensor_id: str):
    """Get all forecast data in one request - optimized for polling"""
    try:
        # Get forecast for 24 hours
        forecast_data = generate_forecast(sensor_id, 24)
        live_data = get_live_predictions(sensor_id)
        
        # Get historical data separately
        df = fetch_history(sensor_id, "1day")
        historical_stats = {}
        if not df.empty:
            for column in ['riskIndex', 'methane', 'co2', 'ammonia', 'humidity', 'temperature']:
                if column in df.columns:
                    historical_stats[column] = {
                        "min": float(df[column].min()),
                        "max": float(df[column].max()),
                        "mean": float(df[column].mean()),
                        "latest": float(df[column].iloc[-1]) if len(df) > 0 else 0
                    }
        
        # Check if model exists
        model_exists = model_exists_in_firebase(sensor_id)
        
        return {
            "success": True,
            "sensor_id": sensor_id,
            "forecast": forecast_data,
            "live_prediction": live_data,
            "historical_stats": historical_stats,
            "model_exists": model_exists,
            "timestamp": datetime.now().isoformat(),
            "cache_control": "max-age=30"  # Cache for 30 seconds
        }
        
    except Exception as e:
        logger.error(f"Error in batch forecast for {sensor_id}: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        } 
    
 #### Reports Generations ##### 

# Define report tier limits
REPORT_TIERS = {
    "standard": {
        "max_days": 90,
        "name": "Standard Report (90 days)",
        "features": ["statistics", "gas_analysis", "recommendations", "executive_summary"]
    },
    "extended": {
        "max_days": 180,
        "name": "Extended Report (180 days)",
        "features": ["statistics", "gas_analysis", "recommendations", "executive_summary", "trend_analysis"]
    },
    "annual": {
        "max_days": 365,
        "name": "Annual Report (1 year)",
        "features": ["statistics", "gas_analysis", "recommendations", "executive_summary", "trend_analysis", "yearly_summary"]
    }
}
@app.get("/api/report/{sensor_id}")
async def generate_report(
    sensor_id: str,
    from_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    to_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    report_type: str = Query("standard", description="Report type: standard (90 days), extended (180 days), annual (365 days)")
):
    """Generate comprehensive report data with tiered limits"""
    try:
        # Validate date format
        try:
            from_dt = datetime.strptime(from_date, "%Y-%m-%d")
            to_dt = datetime.strptime(to_date, "%Y-%m-%d")
        except ValueError:
            return {"success": False, "error": "Invalid date format. Use YYYY-MM-DD"}
        
        if from_dt > to_dt:
            return {"success": False, "error": "from_date must be before to_date"}
        
        # Calculate total days
        total_days = (to_dt - from_dt).days + 1
        
        # Validate report type and check limits
        report_tier = REPORT_TIERS.get(report_type.lower(), REPORT_TIERS["standard"])
        max_allowed_days = report_tier["max_days"]
        
        if total_days > max_allowed_days:
            return {
                "success": False, 
                "error": f"Date range exceeds {max_allowed_days} days limit for {report_tier['name']}. Selected: {total_days} days.",
                "max_allowed_days": max_allowed_days,
                "selected_days": total_days,
                "suggested_tier": get_suggested_tier(total_days)
            }
        
        # Check minimum date range (at least 1 day)
        if total_days < 1:
            return {"success": False, "error": "Date range must be at least 1 day"}
        
        # Warn for very large date ranges
        if total_days > 30:
            logger.info(f"Generating large report for {sensor_id}: {total_days} days ({from_date} to {to_date})")
        
        # Fetch historical data with optimization for large date ranges
        logger.info(f"Fetching data for sensor {sensor_id} from {from_date} to {to_date} ({total_days} days)")
        
        # For large date ranges, fetch all data and then filter
        df = fetch_history(sensor_id, "all")
        
        if df.empty:
            return {
                "success": False, 
                "error": f"No data available for sensor {sensor_id}",
                "sensor_id": sensor_id
            }
        
        # Ensure timestamp column exists
        if 'timestamp' not in df.columns:
            logger.error(f"DataFrame missing 'timestamp' column. Columns: {df.columns.tolist()}")
            return {"success": False, "error": "Data format error: missing timestamp"}
        
        # Convert timestamp and filter
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            # Drop rows with invalid timestamps
            df = df.dropna(subset=['timestamp'])
            
            # Filter by date range
            mask = (df['timestamp'] >= from_dt) & (df['timestamp'] <= (to_dt + timedelta(days=1)))
            filtered_df = df.loc[mask].copy()  # Use copy to avoid SettingWithCopyWarning
            
            # Downsample for very large datasets to improve performance
            if len(filtered_df) > 10000:
                logger.info(f"Large dataset detected ({len(filtered_df)} records). Downsampling...")
                # Keep hourly samples for better performance
                filtered_df = downsample_data(filtered_df)
                logger.info(f"Downsampled to {len(filtered_df)} records")
                
        except Exception as e:
            logger.error(f"Error processing timestamps: {e}")
            return {"success": False, "error": f"Error processing data: {str(e)}"}
        
        if filtered_df.empty:
            return {
                "success": False, 
                "error": f"No data found for sensor {sensor_id} between {from_date} and {to_date}",
                "sensor_id": sensor_id,
                "date_range": {"from": from_date, "to": to_date}
            }
        
        # Calculate statistics with safe access
        statistics = calculate_statistics(filtered_df)
        
        # Generate gas analysis with specific timestamp data
        gas_analysis = analyze_gas_concentrations_with_timestamps(filtered_df)
        
        # Generate recommendations
        recommendations = generate_recommendations(filtered_df, gas_analysis)
        
        # Create executive summary
        summary = create_executive_summary(statistics, gas_analysis)
        
        # Prepare base response data
        response_data = {
            "success": True,
            "sensor_id": sensor_id,
            "report_type": report_tier["name"],
            "date_range": {
                "from": from_date, 
                "to": to_date,
                "days": total_days
            },
            "data_summary": {
                "total_records": len(filtered_df),
                "date_range_records": len(filtered_df),
                "date_range_coverage": f"{filtered_df['timestamp'].min().date()} to {filtered_df['timestamp'].max().date()}",
                "date_range_density": f"{len(filtered_df) / total_days:.1f} records per day"
            },
            "statistics": statistics,
            "gas_analysis": gas_analysis,
            "recommendations": recommendations,
            "summary": summary,
            "generated_at": datetime.now().isoformat(),
            "features_included": report_tier["features"]
        }
        
        # Add critical events timeline
        if len(filtered_df) > 0:
            critical_events = generate_critical_events_timeline(filtered_df)
            response_data["critical_events"] = critical_events
        
        # Add trend analysis for extended and annual reports
        if report_type in ["extended", "annual"] and total_days > 30:
            trend_analysis = analyze_trends(filtered_df)
            response_data["trend_analysis"] = trend_analysis
        
        # Add yearly summary for annual reports
        if report_type == "annual" and total_days >= 180:
            yearly_summary = generate_yearly_summary(filtered_df, from_dt, to_dt)
            response_data["yearly_summary"] = yearly_summary
        
        # Add forecast data (only for recent data, up to 30 days)
        if total_days <= 30:
            try:
                forecast_result = generate_forecast(sensor_id, steps=24)
                if forecast_result and 'error' not in forecast_result:
                    response_data["forecast"] = forecast_result
                else:
                    response_data["forecast"] = {
                        "sensor_id": sensor_id,
                        "forecast": [],
                        "historical_data": [],
                        "forecast_hours": 24,
                        "model_info": {
                            "model_type": "Not Available",
                            "note": "Forecast only available for recent data (≤ 30 days)"
                        },
                        "generated_at": datetime.now().isoformat()
                    }
            except Exception as forecast_error:
                logger.warning(f"Forecast generation failed: {forecast_error}")
                response_data["forecast"] = {
                    "sensor_id": sensor_id,
                    "forecast": [],
                    "note": "Forecast unavailable for this date range"
                }
        
        # Add live predictions for recent reports (last 7 days)
        if total_days <= 7:
            try:
                live_predictions = get_live_predictions(sensor_id)
                if live_predictions and 'error' not in live_predictions:
                    response_data["live_predictions"] = live_predictions
            except Exception as live_error:
                logger.warning(f"Live predictions failed: {live_error}")
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error generating report for {sensor_id}: {str(e)}", exc_info=True)
        return {
            "success": False, 
            "error": f"Internal server error: {str(e)}",
            "sensor_id": sensor_id
        }
# Helper functions
def calculate_statistics(df):
    """Calculate statistical metrics"""
    if df.empty:
        return {}
    
    return {
        "total_readings": len(df),
        "avg_risk": float(df['riskIndex'].mean()) if 'riskIndex' in df.columns else 0,
        "avg_aqi": float(df.get('aqi', df['riskIndex'] * 50).mean()) if 'riskIndex' in df.columns else 0,
        "max_aqi": float(df.get('aqi', df['riskIndex'] * 50).max()) if 'riskIndex' in df.columns else 0,
        "min_aqi": float(df.get('aqi', df['riskIndex'] * 50).min()) if 'riskIndex' in df.columns else 0,
        "alerts_count": int((df['riskIndex'] > 6).sum()) if 'riskIndex' in df.columns else 0
    }

def analyze_gas_concentrations_with_timestamps(df: pd.DataFrame) -> dict:
    """Analyze gas concentration data with specific timestamp information"""
    gases = ['methane', 'co2', 'ammonia']
    analysis = {}
    
    for gas in gases:
        if gas in df.columns:
            try:
                values = df[gas].astype(float)
                threshold = get_gas_threshold(gas)
                safe_threshold = get_safe_threshold(gas)  # Lower threshold for "good" levels
                
                # Find exceedances with timestamps
                exceedances_mask = values > threshold
                safe_mask = values < safe_threshold
                
                exceedance_records = []
                safe_records = []
                
                # CRITICAL FIX: Check if 'timestamp' column exists in df
                if 'timestamp' not in df.columns:
                    logger.error(f"DataFrame missing 'timestamp' column. Available columns: {df.columns.tolist()}")
                    # Create a fallback using index or current time
                    timestamps = pd.Series([datetime.now()] * len(df), index=df.index)
                else:
                    timestamps = df['timestamp']
                
                if exceedances_mask.any():
                    # Use .loc to avoid chained indexing
                    exceedance_indices = df.index[exceedances_mask]
                    for idx in exceedance_indices:
                        try:
                            timestamp_val = timestamps.loc[idx]
                            gas_value = values.loc[idx]
                            
                            # Format timestamp
                            if hasattr(timestamp_val, 'isoformat'):
                                timestamp_str = timestamp_val.isoformat()
                            elif isinstance(timestamp_val, (datetime, pd.Timestamp)):
                                timestamp_str = timestamp_val.isoformat()
                            else:
                                timestamp_str = str(timestamp_val)
                            
                            exceedance_records.append({
                                "timestamp": timestamp_str,
                                "value": float(gas_value),
                                "exceedance_percent": float(((gas_value - threshold) / threshold) * 100) if threshold > 0 else 0
                            })
                        except Exception as e:
                            logger.warning(f"Error processing exceedance record for {gas} at index {idx}: {e}")
                            continue
                
                if safe_mask.any():
                    # Use .loc to avoid chained indexing
                    safe_indices = df.index[safe_mask]
                    for idx in safe_indices:
                        try:
                            timestamp_val = timestamps.loc[idx]
                            gas_value = values.loc[idx]
                            
                            # Format timestamp
                            if hasattr(timestamp_val, 'isoformat'):
                                timestamp_str = timestamp_val.isoformat()
                            elif isinstance(timestamp_val, (datetime, pd.Timestamp)):
                                timestamp_str = timestamp_val.isoformat()
                            else:
                                timestamp_str = str(timestamp_val)
                            
                            safe_records.append({
                                "timestamp": timestamp_str,
                                "value": float(gas_value),
                                "safety_margin_percent": float(((safe_threshold - gas_value) / safe_threshold) * 100) if safe_threshold > 0 else 0
                            })
                        except Exception as e:
                            logger.warning(f"Error processing safe record for {gas} at index {idx}: {e}")
                            continue
                
                # Get top 5 highest and lowest readings with timestamps
                top_exceedances = sorted(exceedance_records, key=lambda x: x["value"], reverse=True)[:5]
                top_safe = sorted(safe_records, key=lambda x: x["safety_margin_percent"] if "safety_margin_percent" in x else x["value"], reverse=True)[:5]
                
                # Find first and last exceedance (by timestamp)
                if exceedance_records:
                    # Sort by timestamp if possible
                    try:
                        exceedance_records_sorted = sorted(exceedance_records, key=lambda x: x["timestamp"])
                        first_exceedance = exceedance_records_sorted[0]
                        last_exceedance = exceedance_records_sorted[-1]
                    except:
                        first_exceedance = exceedance_records[0] if exceedance_records else None
                        last_exceedance = exceedance_records[-1] if exceedance_records else None
                else:
                    first_exceedance = None
                    last_exceedance = None
                
                # Calculate longest continuous exceedance period
                longest_exceedance_period = calculate_longest_exceedance_period(df, gas, threshold)
                
                analysis[gas] = {
                    "average": float(values.mean()),
                    "max": float(values.max()),
                    "min": float(values.min()),
                    "std": float(values.std()),
                    "exceedances": int(exceedances_mask.sum()),
                    "safe_readings": int(safe_mask.sum()),
                    "threshold": threshold,
                    "safe_threshold": safe_threshold,
                    "unit": "ppm",
                    "exceedance_details": {
                        "total_exceedances": len(exceedance_records),
                        "top_exceedances": top_exceedances,
                        "first_exceedance": first_exceedance,
                        "last_exceedance": last_exceedance,
                        "longest_exceedance_period": longest_exceedance_period
                    },
                    "safe_period_details": {
                        "total_safe_readings": len(safe_records),
                        "top_safe_readings": top_safe,
                        "safe_percentage": float((safe_mask.sum() / len(values)) * 100) if len(values) > 0 else 0
                    }
                }
            except Exception as e:
                logger.warning(f"Error analyzing {gas} with timestamps: {e}", exc_info=True)
                analysis[gas] = {
                    "error": f"Failed to analyze {gas}: {str(e)}",
                    "average": 0,
                    "max": 0,
                    "min": 0,
                    "exceedances": 0,
                    "safe_readings": 0
                }
    
    return analysis

def analyze_gas_concentrations(df):
    """Analyze gas concentration data (simpler version without timestamps)"""
    gases = ['methane', 'co2', 'ammonia']
    analysis = {}
    
    for gas in gases:
        if gas in df.columns:
            try:
                values = df[gas].astype(float)
                threshold = get_gas_threshold(gas)
                safe_threshold = get_safe_threshold(gas)
                
                exceedances_mask = values > threshold
                safe_mask = values < safe_threshold
                
                analysis[gas] = {
                    "average": float(values.mean()),
                    "max": float(values.max()),
                    "min": float(values.min()),
                    "std": float(values.std()),
                    "exceedances": int(exceedances_mask.sum()),
                    "safe_readings": int(safe_mask.sum()),
                    "threshold": threshold,
                    "safe_threshold": safe_threshold,
                    "unit": "ppm"
                }
            except Exception as e:
                logger.warning(f"Error analyzing {gas}: {e}")
                analysis[gas] = {
                    "average": 0,
                    "max": 0,
                    "min": 0,
                    "exceedances": 0,
                    "safe_readings": 0,
                    "error": str(e)
                }
    
    return analysis
    
def get_safe_threshold(gas: str) -> float:
    """Get safe threshold values for each gas (lower than warning threshold)"""
    safe_thresholds = {
        'methane': 500,   # 50% of warning threshold
        'co2': 1000,      # 20% of warning threshold
        'ammonia': 10     # 40% of warning threshold
    }
    return safe_thresholds.get(gas, 0)

def calculate_longest_exceedance_period(df: pd.DataFrame, gas: str, threshold: float) -> dict:
    """Calculate the longest continuous period when gas exceeded threshold"""
    try:
        # FIX: Check if 'timestamp' column exists
        if gas not in df.columns or 'timestamp' not in df.columns:
            logger.warning(f"Missing columns: gas={gas in df.columns}, timestamp={'timestamp' in df.columns}")
            return {"duration_hours": 0, "start": None, "end": None}
        
        # Make a copy to avoid modifying the original
        df_sorted = df[['timestamp', gas]].copy()
        
        # Ensure timestamp is datetime
        df_sorted['timestamp'] = pd.to_datetime(df_sorted['timestamp'], errors='coerce')
        df_sorted = df_sorted.dropna(subset=['timestamp', gas])
        
        if df_sorted.empty:
            return {"duration_hours": 0, "start": None, "end": None}
        
        # Sort by timestamp
        df_sorted = df_sorted.sort_values('timestamp')
        
        # Create exceedance mask
        exceedance_mask = df_sorted[gas] > threshold
        
        # Find continuous exceedance periods
        exceedance_periods = []
        current_period = None
        
        for idx, row in df_sorted.iterrows():
            is_exceeding = exceedance_mask.loc[idx] if idx in exceedance_mask else False
            
            if is_exceeding and current_period is None:
                # Start new period
                current_period = {
                    "start": row['timestamp'],
                    "start_value": float(row[gas]),
                    "end": row['timestamp'],
                    "end_value": float(row[gas]),
                    "max_value": float(row[gas])
                }
            elif is_exceeding and current_period is not None:
                # Continue current period
                current_period["end"] = row['timestamp']
                current_period["end_value"] = float(row[gas])
                current_period["max_value"] = max(current_period["max_value"], float(row[gas]))
            elif not is_exceeding and current_period is not None:
                # End current period
                exceedance_periods.append(current_period)
                current_period = None
        
        # Add last period if it exists
        if current_period is not None:
            exceedance_periods.append(current_period)
        
        # Find longest period
        if exceedance_periods:
            longest_period = max(exceedance_periods, 
                                key=lambda p: (p["end"] - p["start"]).total_seconds())
            
            duration_hours = (longest_period["end"] - longest_period["start"]).total_seconds() / 3600
            
            return {
                "duration_hours": float(duration_hours),
                "start": longest_period["start"].isoformat() if hasattr(longest_period["start"], 'isoformat') else str(longest_period["start"]),
                "end": longest_period["end"].isoformat() if hasattr(longest_period["end"], 'isoformat') else str(longest_period["end"]),
                "max_value": longest_period["max_value"],
                "start_value": longest_period["start_value"],
                "end_value": longest_period["end_value"]
            }
        
        return {"duration_hours": 0, "start": None, "end": None}
        
    except Exception as e:
        logger.warning(f"Error calculating longest exceedance period for {gas}: {e}", exc_info=True)
        return {"duration_hours": 0, "start": None, "end": None}
         
def create_executive_summary(statistics: dict, gas_analysis: dict) -> dict:
    """Create executive summary text with specific event mentions"""
    avg_aqi = statistics.get('avg_aqi', 0)
    
    if avg_aqi <= 50:
        air_quality = "Good"
        impact = "minimal health impact"
        color = "green"
    elif avg_aqi <= 100:
        air_quality = "Moderate"
        impact = "acceptable air quality"
        color = "yellow"
    elif avg_aqi <= 150:
        air_quality = "Unhealthy for Sensitive Groups"
        impact = "caution advised for sensitive individuals"
        color = "orange"
    else:
        air_quality = "Unhealthy"
        impact = "significant health concerns"
        color = "red"
    
    exceedances = sum(gas.get('exceedances', 0) for gas in gas_analysis.values() if isinstance(gas, dict))
    safe_readings = sum(gas.get('safe_readings', 0) for gas in gas_analysis.values() if isinstance(gas, dict))
    
    # Add specific gas exceedance details to summary
    gas_details = []
    for gas_name, gas_data in gas_analysis.items():
        if isinstance(gas_data, dict) and gas_data.get('exceedances', 0) > 0:
            gas_details.append(f"{gas_name}: {gas_data['exceedances']} exceedances")
    
    gas_detail_text = f" Gas exceedances: {', '.join(gas_details)}." if gas_details else ""
    
    summary = f"""
    During the reporting period, the average Air Quality Index (AQI) was {avg_aqi:.1f}, 
    categorizing air quality as '{air_quality}'. This indicates {impact}. 
    A total of {statistics.get('total_readings', 0)} readings were analyzed, 
    with {statistics.get('alerts_count', 0)} high-risk alerts triggered.{gas_detail_text}
    Gas concentration thresholds were exceeded {exceedances} times across all monitored gases,
    while {safe_readings} readings were within safe limits.
    """
    
    # Add critical event highlights if available
    critical_events_note = ""
    if any(gas_data.get('exceedance_details', {}).get('longest_exceedance_period', {}).get('duration_hours', 0) > 24 
           for gas_data in gas_analysis.values() if isinstance(gas_data, dict)):
        critical_events_note = " Extended exceedance periods (>24 hours) were detected for some gases."
    
    summary = summary.strip() + critical_events_note
    
    return {
        "executive_summary": summary,
        "air_quality_category": air_quality,
        "air_quality_color": color,
        "overall_assessment": "Safe" if avg_aqi <= 100 else "Requires Attention",
        "exceedance_summary": {
            "total_exceedances": exceedances,
            "gas_details": gas_details,
            "safe_readings": safe_readings
        }
    }

# Update the generate_recommendations function to use timestamp data
def generate_recommendations(df: pd.DataFrame, gas_analysis: dict) -> list:
    """Generate recommendations based on data analysis with specific event references"""
    recommendations = []
    
    # Check for high methane levels with specific events
    if 'methane' in gas_analysis and 'exceedance_details' in gas_analysis['methane']:
        methane_data = gas_analysis['methane']
        if methane_data.get('exceedances', 0) > 0:
            first_exceedance = methane_data['exceedance_details'].get('first_exceedance')
            last_exceedance = methane_data['exceedance_details'].get('last_exceedance')
            longest_period = methane_data['exceedance_details'].get('longest_exceedance_period', {})
            
            event_details = []
            if first_exceedance:
                event_details.append(f"First detected at {format_timestamp(first_exceedance['timestamp'])}")
            if last_exceedance:
                event_details.append(f"Last detected at {format_timestamp(last_exceedance['timestamp'])}")
            if longest_period.get('duration_hours', 0) > 0:
                event_details.append(f"Longest continuous exceedance: {longest_period['duration_hours']:.1f} hours")
            
            recommendations.append({
                "title": "Methane Monitoring",
                "description": f"High methane levels detected ({methane_data['exceedances']} exceedances). {' '.join(event_details)}",
                "priority": "high",
                "action": "Conduct immediate leak detection and increase ventilation",
                "category": "safety",
                "gas": "methane",
                "events_count": methane_data['exceedances'],
                "timeline_data": methane_data['exceedance_details'].get('top_exceedances', [])[:3]
            })
    
    # Check for CO2 buildup with specific events
    if 'co2' in gas_analysis and 'exceedance_details' in gas_analysis['co2']:
        co2_data = gas_analysis['co2']
        if co2_data.get('exceedances', 0) > 0:
            top_exceedances = co2_data['exceedance_details'].get('top_exceedances', [])
            
            recommendation_text = f"Elevated CO2 levels detected ({co2_data['exceedances']} exceedances). "
            if top_exceedances:
                peak_time = format_timestamp(top_exceedances[0]['timestamp'])
                recommendation_text += f"Peak concentration of {top_exceedances[0]['value']:.0f} ppm at {peak_time}. "
            
            recommendations.append({
                "title": "CO2 Ventilation",
                "description": recommendation_text + "Poor ventilation may be affecting air quality.",
                "priority": "medium",
                "action": "Improve air circulation in monitored areas",
                "category": "ventilation",
                "gas": "co2",
                "events_count": co2_data['exceedances']
            })
    
    # Check for ammonia presence with specific events
    if 'ammonia' in gas_analysis and 'exceedance_details' in gas_analysis['ammonia']:
        ammonia_data = gas_analysis['ammonia']
        if ammonia_data.get('exceedances', 0) > 0:
            top_exceedances = ammonia_data['exceedance_details'].get('top_exceedances', [])
            
            recommendation_text = f"Ammonia detected above safe levels ({ammonia_data['exceedances']} exceedances). "
            if top_exceedances:
                peak_value = top_exceedances[0]['value']
                threshold = ammonia_data.get('threshold', 25)
                recommendation_text += f"Peak concentration {peak_value:.1f} ppm ({((peak_value - threshold) / threshold * 100):.0f}% above threshold). "
            
            recommendations.append({
                "title": "Ammonia Safety",
                "description": recommendation_text + "Ensure proper containment and ventilation.",
                "priority": "high",
                "action": "Check containment systems and increase ventilation",
                "category": "safety",
                "gas": "ammonia",
                "events_count": ammonia_data['exceedances']
            })
    
    # Check risk index if available
    if 'riskIndex' in df.columns:
        avg_risk = df['riskIndex'].mean()
        if avg_risk > 5:
            # Find high risk periods
            high_risk_periods = df[df['riskIndex'] > 6][['timestamp', 'riskIndex']]
            if not high_risk_periods.empty:
                worst_period = high_risk_periods.nlargest(1, 'riskIndex').iloc[0]
                worst_time = format_timestamp(worst_period['timestamp'])
                
                recommendations.append({
                    "title": "Continuous Monitoring",
                    "description": f"Persistent high risk levels detected (average: {avg_risk:.1f}). Highest risk of {worst_period['riskIndex']:.1f} at {worst_time}.",
                    "priority": "medium",
                    "action": "Implement enhanced monitoring and alert protocols",
                    "category": "monitoring"
                })
    
    # Add standard recommendations
    recommendations.extend([
        {
            "title": "Regular Maintenance",
            "description": "Ensure sensor accuracy with regular calibration.",
            "priority": "low",
            "action": "Schedule calibration every 3 months",
            "category": "maintenance"
        },
        {
            "title": "Emergency Protocols",
            "description": "Review emergency response procedures.",
            "priority": "medium",
            "action": "Conduct quarterly safety drills",
            "category": "safety"
        }
    ])
    
    # Sort by priority (high > medium > low)
    priority_order = {"high": 0, "medium": 1, "low": 2}
    recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))
    
    return recommendations[:5]  # Return top 5 recommendations

def format_timestamp(timestamp_str: str) -> str:
    """Format timestamp string to readable format"""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return timestamp_str

def get_suggested_tier(selected_days: int) -> dict:
    """Suggest appropriate report tier based on selected days"""
    for tier_name, tier_info in REPORT_TIERS.items():
        if selected_days <= tier_info["max_days"]:
            return {
                "tier": tier_name,
                "name": tier_info["name"],
                "max_days": tier_info["max_days"]
            }
    
    # If exceeds all tiers, suggest annual
    return {
        "tier": "annual",
        "name": REPORT_TIERS["annual"]["name"],
        "max_days": REPORT_TIERS["annual"]["max_days"],
        "note": "Date range exceeds maximum allowed. Consider generating multiple reports."
    }

def downsample_data(df: pd.DataFrame, frequency: str = '1H') -> pd.DataFrame:
    """Downsample data to improve performance for large datasets"""
    try:
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Resample to hourly data (take mean)
        resampled = df.resample(frequency).mean()
        
        # Reset index
        resampled.reset_index(inplace=True)
        
        return resampled
    except Exception as e:
        logger.error(f"Error downsampling data: {e}")
        return df

def analyze_trends(df: pd.DataFrame) -> dict:
    """Analyze trends in the data for extended reports"""
    trends = {
        "daily_patterns": {},
        "weekly_patterns": {},
        "overall_trend": "stable"
    }
    
    try:
        if 'timestamp' not in df.columns or 'riskIndex' not in df.columns:
            return trends
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Analyze daily patterns
        if len(df) > 24:  # Need at least 24 hours of data
            hourly_avg = df.groupby('hour')['riskIndex'].mean()
            trends["daily_patterns"] = {
                "highest_risk_hour": int(hourly_avg.idxmax()),
                "lowest_risk_hour": int(hourly_avg.idxmin()),
                "peak_hours": hourly_avg.nlargest(3).index.tolist()
            }
        
        # Analyze weekly patterns
        if len(df) > 7 * 24:  # Need at least 1 week of hourly data
            weekly_avg = df.groupby('day_of_week')['riskIndex'].mean()
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            trends["weekly_patterns"] = {
                "highest_risk_day": day_names[int(weekly_avg.idxmax())],
                "lowest_risk_day": day_names[int(weekly_avg.idxmin())]
            }
        
        # Analyze overall trend
        if len(df) > 30:  # Need at least 30 data points
            # Simple linear regression for trend
            df['time_index'] = range(len(df))
            slope = np.polyfit(df['time_index'], df['riskIndex'], 1)[0]
            
            if slope > 0.01:
                trends["overall_trend"] = "increasing"
            elif slope < -0.01:
                trends["overall_trend"] = "decreasing"
            else:
                trends["overall_trend"] = "stable"
            
            trends["trend_slope"] = float(slope)
        
        return trends
        
    except Exception as e:
        logger.warning(f"Error analyzing trends: {e}")
        return trends

def generate_yearly_summary(df: pd.DataFrame, from_date: datetime, to_date: datetime) -> dict:
    """Generate yearly summary for annual reports"""
    summary = {
        "monthly_analysis": {},
        "seasonal_patterns": {},
        "key_metrics_by_month": {}
    }
    
    try:
        if 'timestamp' not in df.columns:
            return summary
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
        
        # Monthly analysis
        monthly_stats = df.groupby('month').agg({
            'riskIndex': ['mean', 'max', 'min', 'count'],
            'aqi': ['mean', 'max', 'min'] if 'aqi' in df.columns else pd.NamedAgg(column='riskIndex', aggfunc='mean')
        }).round(2)
        
        summary["monthly_analysis"] = monthly_stats.to_dict()
        
        # Seasonal patterns
        seasons = {
            1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall",
            11: "Fall", 12: "Winter"
        }
        
        df['season'] = df['month'].map(seasons)
        seasonal_stats = df.groupby('season')['riskIndex'].mean().round(2).to_dict()
        summary["seasonal_patterns"] = seasonal_stats
        
        # Key metrics by month
        for month in range(1, 13):
            month_data = df[df['month'] == month]
            if not month_data.empty:
                summary["key_metrics_by_month"][month] = {
                    "avg_risk": float(month_data['riskIndex'].mean()),
                    "max_risk": float(month_data['riskIndex'].max()),
                    "alerts": int((month_data['riskIndex'] > 6).sum()),
                    "records": len(month_data)
                }
        
        return summary
        
    except Exception as e:
        logger.warning(f"Error generating yearly summary: {e}")
        return summary

# Helper function to get report tiers info (for frontend)
@app.get("/api/report-tiers")
async def get_report_tiers():
    """Get information about available report tiers"""
    return {
        "success": True,
        "tiers": REPORT_TIERS,
        "default_tier": "standard"
    }
 #### End Reports Generations #####
 
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