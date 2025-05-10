from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import cv2
import numpy as np
import io
import base64
from typing import List
from pydantic import BaseModel

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageAnalysis(BaseModel):
    blur_score: float
    eyes_open: bool
    smiling: bool
    exposure_quality: str
    total_score: float

def process_image(image_bytes: bytes) -> ImageAnalysis:
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Calculate blur score
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Basic face analysis (simplified for example)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Default values
    eyes_open = len(faces) > 0  # Simplified
    smiling = len(faces) > 0    # Simplified
    
    # Calculate exposure
    mean_brightness = np.mean(gray)
    if mean_brightness < 60:
        exposure_quality = "underexposed"
    elif mean_brightness > 190:
        exposure_quality = "overexposed"
    else:
        exposure_quality = "good"
    
    # Calculate total score (simplified)
    total_score = min(10, blur_score / 1000)
    if eyes_open:
        total_score += 1
    if smiling:
        total_score += 1
    if exposure_quality == "good":
        total_score += 1
    
    return ImageAnalysis(
        blur_score=blur_score,
        eyes_open=eyes_open,
        smiling=smiling,
        exposure_quality=exposure_quality,
        total_score=total_score
    )

@app.post("/cull")
async def cull_image(file: UploadFile = File(...)):
    contents = await file.read()
    analysis = process_image(contents)
    return analysis