from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
MODEL_PATH = "model.pkl"
model = joblib.load(MODEL_PATH)

# Initialize FastAPI
app = FastAPI(title="Heart Disease Predictor")

# Setup templates and static folder
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Pydantic model for JSON requests
class HeartData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# Home route to render HTML page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# Form submission route (HTML)
@app.post("/predict_form", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    age: int = Form(...),
    sex: int = Form(...),
    cp: int = Form(...),
    trestbps: int = Form(...),
    chol: int = Form(...),
    fbs: int = Form(...),
    restecg: int = Form(...),
    thalach: int = Form(...),
    exang: int = Form(...),
    oldpeak: float = Form(...),
    slope: int = Form(...),
    ca: int = Form(...),
    thal: int = Form(...)
):
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
                          oldpeak, slope, ca, thal]])
    prediction = model.predict(features)[0]
    if prediction == 1:
        result = "Heart Disease Detected 💔. Please visit a doctor, do not panic."
    else:
        result = "No Heart Disease ❤️. Keep healthy, visit a doctor regularly."
    return templates.TemplateResponse("index.html", {"request": request, "result": result})

# JSON API route
@app.post("/predict", response_class=JSONResponse)
async def predict_api(data: HeartData):
    features = np.array([[data.age, data.sex, data.cp, data.trestbps, data.chol, data.fbs,
                          data.restecg, data.thalach, data.exang, data.oldpeak,
                          data.slope, data.ca, data.thal]])
    prediction = model.predict(features)[0]
    if prediction == 1:
        result = "Heart Disease Detected 💔. Please visit a doctor, do not panic."
    else:
        result = "No Heart Disease ❤️. Keep healthy, visit a doctor regularly."
    return {"prediction": int(prediction), "result": result}
