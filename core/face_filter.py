import mediapipe as mp
import cv2
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

def detect_face_attributes(image):
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return {"eyes_open": False, "smiling": False}

        for face_landmarks in results.multi_face_landmarks:
            # Eye landmarks
            left_eye_top = face_landmarks.landmark[159].y
            left_eye_bottom = face_landmarks.landmark[145].y
            right_eye_top = face_landmarks.landmark[386].y
            right_eye_bottom = face_landmarks.landmark[374].y

            left_eye_open = abs(left_eye_top - left_eye_bottom) > 0.01
            right_eye_open = abs(right_eye_top - right_eye_bottom) > 0.01

            # Smile (check distance between top and bottom lips)
            upper_lip = face_landmarks.landmark[13].y
            lower_lip = face_landmarks.landmark[14].y
            mouth_open = abs(upper_lip - lower_lip) > 0.02

            return {
                "eyes_open": left_eye_open and right_eye_open,
                "smiling": mouth_open,
            }

        return {"eyes_open": False, "smiling": False}
