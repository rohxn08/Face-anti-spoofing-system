import cv2
import joblib
import numpy as np
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.lbp_extractor import LBPExtractor
from src.preprocessing.preprocess import preprocess_face


try:
    model = joblib.load('models/svm_face_antispoofing.pkl')
    scaler = joblib.load('models/scaler.pkl')
except FileNotFoundError:
   
    model = joblib.load('../models/svm_face_antispoofing.pkl')
    scaler = joblib.load('../models/scaler.pkl')

extractor = LBPExtractor()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

   
    face = preprocess_face(frame)
    
    label_text = "No Face Detected"
    color = (0, 255, 255) 

    if face is not None:
        try:
            feat = extractor.extract(face)
            feat = feat.reshape(1, -1)
            feat_scaled = scaler.transform(feat)
            
            prediction = model.predict(feat_scaled)
            
            if prediction[0] == 1:
                label_text = "Real"
                color = (0, 255, 0) 
            else:
                label_text = "Spoof"
                color = (0, 0, 255) 
        except Exception as e:
            print(f"Prediction error: {e}")
            label_text = "Error"
            color = (0, 255, 255)

    cv2.putText(frame, label_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow("Real-time Anti-Spoofing", frame)
    if face is not None:
        cv2.imshow("Debug Crop", face)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
