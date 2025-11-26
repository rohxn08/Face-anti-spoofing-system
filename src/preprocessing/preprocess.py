import os
import cv2
import numpy as np
import sys

# Add src to path to import features
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.features.lbp_extractor import LBPExtractor
attack_dir=r"data\MSU-MFSD\pics\attack"
real_dir=r"data\MSU-MFSD\pics\real"

def load_data_from_dirs(real_dir, attack_dir):
    X = []
    y = []
    extractor = LBPExtractor()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # --- Real Images ---
    if os.path.exists(real_dir):
        for filename in os.listdir(real_dir):
            if filename.lower().endswith(('.jpg', '.png')):
                img_path = os.path.join(real_dir, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    # Face Detection & Cropping
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)
                    
                    if len(faces) > 0:
                        (x, y_rect, w, h) = faces[0]
                        face_img = img[y_rect:y_rect+h, x:x+w]
                        face_img = cv2.resize(face_img, (128, 128))
                        
                        X.append(extractor.extract(face_img))
                        y.append(1) # Real
                    else:
                        continue

    # --- Attack Images ---
    if os.path.exists(attack_dir):
        for filename in os.listdir(attack_dir):
            if filename.lower().endswith(('.jpg', '.png')):
                img_path = os.path.join(attack_dir, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    # Face Detection & Cropping
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)
                    
                    if len(faces) > 0:
                        (x, y_rect, w, h) = faces[0]
                        face_img = img[y_rect:y_rect+h, x:x+w]
                        face_img = cv2.resize(face_img, (128, 128))
                        
                        X.append(extractor.extract(face_img))
                        y.append(0) # Attack
                    else:
                        continue

    return np.array(X), np.array(y)