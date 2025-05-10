import cv2
import numpy as np
from scipy import stats

def analyze_exposure(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # Normalize histogram
    hist = hist.ravel() / hist.sum()
    
    # Calculate mean and std of pixel values
    mean = np.mean(gray)
    std = np.std(gray)
    
    # Calculate histogram peaks
    peaks = stats.find_peaks(hist)[0]
    
    # Determine exposure quality
    if mean < 60:
        quality = "underexposed"
    elif mean > 190:
        quality = "overexposed"
    else:
        quality = "good"
    
    return {
        "quality": quality,
        "mean": mean,
        "std": std,
        "peaks": len(peaks)
    }

def calculate_image_score(blur_score, face_attributes, exposure_data):
    # Base score from blur detection (0-10)
    base_score = min(10, blur_score / 1000)
    
    # Adjust for face attributes
    if face_attributes["eyes_open"]:
        base_score += 1
    if face_attributes["smiling"]:
        base_score += 1
        
    # Adjust for exposure
    if exposure_data["quality"] == "good":
        base_score += 1
    elif exposure_data["quality"] == "underexposed":
        base_score -= 1
    elif exposure_data["quality"] == "overexposed":
        base_score -= 2
        
    # Normalize final score to 0-10 range
    final_score = max(0, min(10, base_score))
    
    return round(final_score, 1)