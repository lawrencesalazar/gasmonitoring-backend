from fastapi import FastAPI
import firebase_admin
from firebase_admin import credentials, db

app = FastAPI()

# Firebase Admin setup
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://gasmonitoring-ec511.firebaseio.com/"
})

@app.get("/")
def home():
    return {"message": "Python API is running on Cloud Run"}

@app.get("/readings")
def get_readings():
    ref = db.reference("sensorReadings")
    return ref.get()

@app.post("/readings")
def add_reading(data: dict):
    ref = db.reference("sensorReadings")
    ref.push(data)
    return {"status": "success"}
