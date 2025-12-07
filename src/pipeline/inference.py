import os
import cv2
import joblib
import numpy as np
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.features.lbp_extractor import LBPExtractor
from src.preprocessing.preprocess import preprocess_face

class FaceAntiSpoofingSystem:
    def __init__(self, model_path='models/svm_face_antispoofing.pkl', scaler_path='models/scaler.pkl'):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.extractor = LBPExtractor()
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                print("Model and scaler loaded successfully.")
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"Model files not found at {self.model_path} or {self.scaler_path}. Please train the model first.")

    def predict(self, image_path_or_array):
        if self.model is None or self.scaler is None:
            return {"error": "Model not loaded"}

        # Preprocess the face (Detect -> Crop -> Resize)
        processed_face, _ = preprocess_face(image_path_or_array)
        
        if processed_face is None:
            return {"error": "No face detected"}

        # Extract features
        features = self.extractor.extract(processed_face)
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        confidence = self.model.decision_function(features_scaled)[0]
        
        label = "REAL" if prediction == 1 else "SPOOF"
        
        return {
            "label": label,
            "confidence": float(confidence),
            "prediction": int(prediction)
        }

if __name__ == "__main__":
    # Example usage
    system = FaceAntiSpoofingSystem()
    
    # Test with an image if provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = system.predict(image_path)
        print(result)
