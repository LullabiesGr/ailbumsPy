import face_recognition
import numpy as np
from PIL import Image
import imagehash

def get_face_embedding(image):
    rgb_img = image[:, :, ::-1]  # BGR to RGB
    encodings = face_recognition.face_encodings(rgb_img)
    return encodings[0] if encodings else None

def get_image_hash(image):
    pil_img = Image.fromarray(image[:, :, ::-1])  # BGR to RGB
    return imagehash.phash(pil_img)

def are_images_duplicates(hash1, hash2, threshold=5):
    return abs(hash1 - hash2) <= threshold
