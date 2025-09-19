from fastapi import FastAPI
import firebase_admin
from firebase_admin import credentials, db
import os
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend domain to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://gasmonitoring-ec511.web.app",  # your React frontend
        "http://localhost:3000"  # for local React dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Firebase Admin setup
service_account_info = json.loads(os.environ["FIREBASE_SERVICE_ACCOUNT"])
cred = credentials.Certificate(service_account_info)

firebase_admin.initialize_app(cred, {
    "databaseURL": "https://gasmonitoring-ec511.firebaseio.com/"
})

@app.get("/")
def home():
    return {"message": "Python API is running on Render"}

@app.get("/readings")
def get_readings():
    ref = db.reference("sensorReadings")
    return ref.get()

@app.post("/readings")
def add_reading(data: dict):
    ref = db.reference("sensorReadings")
    ref.push(data)
    return {"status": "success"}
