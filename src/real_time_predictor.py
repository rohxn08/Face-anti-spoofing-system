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

from collections import deque
from scipy.stats import mode


history_length = 15
prediction_history = deque(maxlen=history_length)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face, bbox = preprocess_face(frame)
    
    label_text = "No Face Detected"
    color = (0, 255, 255) 

    if face is not None:
        x, y, w, h = bbox
        try:
            feat = extractor.extract(face)
            feat = feat.reshape(1, -1)
            feat_scaled = scaler.transform(feat)
            
           
            current_pred = model.predict(feat_scaled)[0]
            
          
            prediction_history.append(current_pred)
            
           
          
            real_votes = sum(prediction_history)
            
          
            if real_votes > (len(prediction_history) / 2):
                final_prediction = 1 # Real
                label_text = "Real"
                color = (0, 255, 0)
            else:
                final_prediction = 0 # Spoof
                label_text = "Spoof"
                color = (0, 0, 255)
                
          

        except Exception as e:
            print(f"Prediction error: {e}")
            label_text = "Error"
            color = (0, 255, 255)

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    else:
        # Clear history if face is lost, so it doesn't carry over old votes
        prediction_history.clear()

    # Show frame
    cv2.imshow("Real-time Anti-Spoofing", frame)
    if face is not None:
        cv2.imshow("Debug Crop", face)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
