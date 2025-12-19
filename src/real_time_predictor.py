import cv2
import joblib
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Input, Lambda, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from collections import deque

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
        self.base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'saved_models'))
        
        # --- VOTING HISTORY ---
        self.history = deque(maxlen=10) # Store last 10 predictions

        self._load_model()
        
    def _load_model(self):
        print(f"DEBUG: Initializing {self.model_type.upper()} model...")
        
        if self.model_type == 'svm':
            try:
                svm_path = os.path.join(self.base_path, 'face_antispoof_svm.pkl')
                scaler_path = os.path.join(self.base_path, 'scaler.pkl')
                
                if not os.path.exists(svm_path):
                    raise FileNotFoundError(f"SVM model not found at {svm_path}")
                
                print(f"Loading SVM from {svm_path}")
                self.model = joblib.load(svm_path)
                self.scaler = joblib.load(scaler_path)
                self.extractor = LBPExtractor()
                print("✅ SVM Model Loaded Successfully")
            except Exception as e:
                print(f"❌ Error loading SVM: {e}")
                self.model = None
                
        elif self.model_type == 'cnn':
            cnn_path = os.path.join(self.base_path, 'face_antispoofing_v3_224.keras')
            if not os.path.exists(cnn_path):
                raise FileNotFoundError(f"CRITICAL: Model file missing at {cnn_path}")

            try:
                print("DEBUG: Reconstructing model architecture and loading weights...")
                
                from tensorflow.keras.models import Model
                from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense, Lambda
                from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
                
                # 1. Re-define the EXACT architecture (functional API)
                inputs = Input(shape=(224, 224, 3))
                x = Lambda(preprocess_input)(inputs)
                
                # Load MobileNetV2 without weights (we load them later)
                base_model = MobileNetV2(include_top=False, weights=None, input_shape=(224, 224, 3))
                x = base_model(x)
                
                x = GlobalAveragePooling2D()(x)
                x = Dropout(0.5)(x)
                x = Dense(64, activation="relu")(x)
                x = Dropout(0.5)(x)
                outputs = Dense(1, activation='sigmoid')(x)
                
                self.model = Model(inputs, outputs)
                
                # 2. Load weights
                self.model.load_weights(cnn_path)
                print("✅ CNN Model Loaded Successfully (Weights Only Mode)")
            except Exception as e:
                print(f"❌ Critical Error loading CNN: {e}")
                import traceback
                traceback.print_exc()
                self.model = None

        if self.model is None:
            raise RuntimeError(f"Failed to initialize {self.model_type} model. See logs above.")

    def reset_history(self):
        """Reset the voting history when switching users or contexts."""
        self.history.clear()

    def predict(self, frame):
        """
        Predicts if a face in the frame is Real or Spoof using VOTING.
        Returns: 
            (label_text, color, bbox)
            label_text: "Real (0.99)" or "Spoof (0.99)"
            color: (G, B, R) tuple for drawing
            bbox: (x, y, w, h) or None if no face
        """
        # Detect and Crop Face
        if self.model_type=='cnn':
            target_size=(224,224)
        else:
            target_size=(128,128)
        face, bbox = preprocess_face(frame, target_size=target_size)
       
        
        if face is None:
            self.history.clear() # Reset if face is lost
            return "No Face", (0, 255, 255), None

        raw_score = 0
        is_real_frame = False # Prediction for THIS specific frame
        
        try:
            if self.model_type == 'svm':
                # --- SVM LOGIC ---
                feat = self.extractor.extract(face)
                feat = feat.reshape(1, -1)
                feat_scaled = self.scaler.transform(feat)
                prediction = self.model.predict(feat_scaled)[0] # 1=Real, 0=Spoof
                
                is_real_frame = (prediction == 0)
                
                # Confidence workaround for SVM
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(feat_scaled)[0]
                    raw_score = probs[0] if is_real_frame else probs[1]
                else:
                    raw_score = 1.0

            elif self.model_type == 'cnn':
                # --- CNN LOGIC ---
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                img = img_to_array(face_rgb)
                img = np.expand_dims(img, axis=0)
                
                # Predict gives probability of Class 1 (Imposter)
                cnn_score = self.model.predict(img, verbose=0)[0][0]
                
                # Logic: < 0.5 is Real (Client), > 0.5 is Spoof (Imposter)
                if cnn_score < 0.5:
                    is_real_frame = True
                    raw_score = 1.0 - cnn_score
                else:
                    is_real_frame = False
                    raw_score = cnn_score

            # --- VOTING LOGIC ---
            # Append 1 for Real, 0 for Spoof
            self.history.append(1 if is_real_frame else 0)
            
            # Calculate average vote
            avg_vote = sum(self.history) / len(self.history)
            
            # Threshold > 0.5 means Majority say Real
            is_real_final = (avg_vote > 0.5)
            
            # --- FORMAT OUTPUT ---
            if is_real_final:
                label = f"REAL ({raw_score:.2f})"
                color = (0, 255, 0) # Green
            else:
                label = f"SPOOF ({raw_score:.2f})"
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
