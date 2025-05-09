import cv2
import os

def load_images_from_folder(folder):
    images = {}
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images[filename] = img
    return images

