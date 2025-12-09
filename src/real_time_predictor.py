import cv2
import joblib
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.lbp_extractor import LBPExtractor
from src.preprocessing.preprocess import preprocess_face

class RealTimePredictor:
    def __init__(self, model_type='cnn'):
        """
        Args:
            model_type (str): 'cnn' or 'svm'
        """
        self.model_type = model_type.lower()
        self.model = None
        self.scaler = None
        self.extractor = None
        self.base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))

        self._load_model()
        
    def _load_model(self):
        if self.model_type == 'svm':
            try:
                svm_path = os.path.join(self.base_path, 'svm_face_antispoofing.pkl')
                scaler_path = os.path.join(self.base_path, 'scaler.pkl')
                
                print(f"Loading SVM from {svm_path}")
                self.model = joblib.load(svm_path)
                self.scaler = joblib.load(scaler_path)
                self.extractor = LBPExtractor()
                print("✅ SVM Model Loaded")
            except Exception as e:
                print(f"❌ Error loading SVM: {e}")
                
        elif self.model_type == 'cnn':
            try:
                cnn_path = os.path.join(self.base_path, 'face_antispoofing_model.h5')
                print(f"Loading CNN from {cnn_path}")
                self.model = load_model(cnn_path)
                print("✅ CNN Model Loaded")
            except Exception as e:
                print(f"❌ Error loading CNN: {e}")
        else:
            raise ValueError("Invalid model_type. Choose 'svm' or 'cnn'.")

    def predict(self, frame):
        """
        Predicts if a face in the frame is Real or Spoof.
        Returns: 
            (label_text, color, bbox)
            label_text: "Real (0.99)" or "Spoof (0.99)"
            color: (G, B, R) tuple for drawing
            bbox: (x, y, w, h) or None if no face
        """
        # Detect and Crop Face
        face, bbox = preprocess_face(frame, target_size=(128, 128))
        
        if face is None:
            return "No Face", (0, 255, 255), None

        score = 0
        is_real = False
        
        try:
            if self.model_type == 'svm':
                # --- SVM LOGIC ---
                feat = self.extractor.extract(face)
                feat = feat.reshape(1, -1)
                feat_scaled = self.scaler.transform(feat)
                prediction = self.model.predict(feat_scaled)[0] # 1=Real, 0=Spoof (Usually)
                
                # SVM predict usually returns class label directly
                is_real = (prediction == 1)
                score = 1.0 # SVM doesn't give probability easily with predict(), use predict_proba if needed
                
                # Try getting probability if available
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(feat_scaled)[0]
                    score = probs[1] if is_real else probs[0]

            elif self.model_type == 'cnn':
                # --- CNN LOGIC ---
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                img = img_to_array(face_rgb)
                img = np.expand_dims(img, axis=0)
                
                # Predict gives probability of Class 1 (Imposter)
                raw_score = self.model.predict(img, verbose=0)[0][0]
                
                # Logic: < 0.5 is Real (Client), > 0.5 is Spoof (Imposter)
                if raw_score < 0.5:
                    is_real = True
                    score = 1.0 - raw_score # Confidence in being Real
                else:
                    is_real = False
                    score = raw_score # Confidence in being Spoof

            # --- FORMAT OUTPUT ---
            if is_real:
                label = f"REAL ({score:.2f})"
                color = (0, 255, 0) # Green
            else:
                label = f"SPOOF ({score:.2f})"
                color = (0, 0, 255) # Red
                
            return label, color, bbox

        except Exception as e:
            print(f"Prediction Error: {e}")
            return "Error", (0, 255, 255), bbox

# --- STANDALONE TEST SUPPORT ---
if __name__ == "__main__":
    # If run directly, default to CNN and webcam loop
    mode = sys.argv[1] if len(sys.argv) > 1 else 'cnn'
    predictor = RealTimePredictor(model_type=mode)
    
    cap = cv2.VideoCapture(0)
    print(f"Starting {mode.upper()} Predictor. Press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        label, color, bbox = predictor.predict(frame)
        
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
        cv2.imshow(f"Anti-Spoofing [{mode.upper()}]", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
