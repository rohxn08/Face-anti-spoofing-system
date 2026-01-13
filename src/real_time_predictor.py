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
        self.last_bbox=None

        self._load_model()
        
    def _load_model(self):
        print(f"DEBUG: Initializing {self.model_type.upper()} model...")
        
        if self.model_type == 'svm':
            try:
                # User provided Dictionary Artifacts
                svm_path = os.path.join(self.base_path, 'svm_texture_pipeline.pkl')
                
                if not os.path.exists(svm_path):
                    raise FileNotFoundError(f"SVM pipeline not found at {svm_path}")
                
                print(f"Loading SVM Artifacts from {svm_path}...")
                artifacts = joblib.load(svm_path)
                
                # Unpack the dictionary
                self.scaler = artifacts['scaler']
                self.pca = artifacts['pca']
                self.model = artifacts['svm_model']
                lbp_params = artifacts.get('lbp_params', {'num_points': 24, 'radius': 3})
                
                # Initialize Extractor with saved params
                self.extractor = LBPExtractor(
                    num_points=lbp_params['num_points'], 
                    radius=lbp_params['radius'], 
                    grid_x=4, grid_y=4
                )
                print("✅ SVM Texture Pipeline Loaded Successfully")
            except Exception as e:
                print(f"❌ Error loading SVM Pipeline: {e}")
                import traceback
                traceback.print_exc()
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
    def _is_scene_change(self,current_bbox):
        if self.last_bbox is None:
            return True
        
        x1,y1,w1,h1=current_bbox
        x2,y2,w2,h2=self.last_bbox
        c1=(x1+w1/2,y1+h1/2)
        c2=(x2+w2/2,y2+h2/2)
        dist=np.sqrt((c1[0]-c2[0])**2+(c1[1]-c2[1])**2)
        if dist>100:
            return True
        return False
        
        

    def predict(self, frame,is_live=False):
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
            self.history.clear() 
            self.last_bbox = None 
            return "No Face", (0, 255, 255), None

        if is_live:
            if self._is_scene_change(bbox):
                self.history.clear()
            
            self.last_bbox = bbox
            
        raw_score = 0
        is_real_frame = False # Prediction for THIS specific frame
        
        try:
            if self.model_type == 'svm':
                # --- SVM LOGIC ---
                # 1. Pipeline Alignment: Training used a "Center 50% Crop" to focus on skin texture
                # We emulate this here to match the feature distribution.
                h, w = face.shape[:2]
                
                # Take center 50% (25% margin)
                # Note: 'face' is already 128x128 from preprocess_face
                y_start, y_end = int(h*0.25), int(h*0.75)
                x_start, x_end = int(w*0.25), int(w*0.75)
                
                face_cropped = face[y_start:y_end, x_start:x_end]
                
                # Resize back to 128x128 as done in training
                face_final = cv2.resize(face_cropped, (128, 128))
                
                # 2. Extract
                feat = self.extractor.extract(face_final)
                feat = feat.reshape(1, -1)
                
                # 3. Scale
                feat_scaled = self.scaler.transform(feat)
                
                # 4. PCA
                feat_pca = self.pca.transform(feat_scaled)
                
                # 5. Predict
                prediction = self.model.predict(feat_pca)[0] 
                
                # Mapping: 0=Real, 1=Spoof (Confirmed in train_svm.py)
                is_real_frame = (prediction == 0)
                
                # Confidence
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(feat_pca)[0]
                    raw_score = probs[0] if is_real_frame else probs[1]
                elif hasattr(self.model, "decision_function"):
                    score = self.model.decision_function(feat_pca)[0]
                    raw_score = 1 / (1 + np.exp(-score)) 
                    if not is_real_frame: raw_score = 1 - raw_score
                else:
                    raw_score = 1.0
                
                print(f"DEBUG [SVM]: Pred={prediction} (Real={is_real_frame}) Conf={raw_score:.2f}")

            elif self.model_type == 'cnn':
                # --- CNN LOGIC ---
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                img = img_to_array(face_rgb)
                img = np.expand_dims(img, axis=0)
                
                # Predict gives probability of Class 1 (Imposter)
                cnn_score = self.model.predict(img, verbose=0)[0][0]
                
                # Logic branching based on deployment mode
                if is_live:
                    # LIVE MODE: Inverted Logic (Based on observed behavior)
                    # Score > 0.5 means REAL
                    if cnn_score > 0.5:
                        is_real_frame = True
                        raw_score = cnn_score
                    else:
                        is_real_frame = False
                        raw_score = 1.0 - cnn_score
                else:
                    # STATIC MODE: Standard Logic
                    # Score < 0.5 means REAL (Class 0)
                    if cnn_score < 0.5:
                        is_real_frame = True
                        raw_score = 1.0 - cnn_score
                    else:
                        is_real_frame = False
                        raw_score = cnn_score

            # --- VOTING LOGIC ---
            if is_live:
                self.history.append(1 if is_real_frame else 0)
                
                # Calculate average vote
                avg_vote = sum(self.history) / len(self.history)
                
                # Threshold > 0.5 means Majority say Real
                is_real_final = (avg_vote > 0.5)
            else:
                is_real_final = is_real_frame
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

    # --- GRAD-CAM EXPLAINABILITY ---
    def predict_with_heatmap(self, frame, is_live=False):
        """
        Returns standard prediction PLUS a verified Grad-CAM overlay image.
        Only works for CNN.
        """
        from src.gradcam import GradCAM # Lazy import

        # 1. Standard Predict first to get Label/BBox
        label, color, bbox = self.predict(frame, is_live=is_live)
        
        if self.model_type != 'cnn' or bbox is None:
            return label, color, bbox, None

        try:
            # Initialize Explainer Lazy
            if not hasattr(self, 'explainer'):
                self.explainer = GradCAM(self.model)

            # 2. Re-Prepare image for Grad-CAM
            face, _ = preprocess_face(frame, target_size=(224,224))
            if face is None: return label, color, bbox, None
            
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            img_array = img_to_array(face_rgb)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array) 

            # 3. Generate Heatmap via External Module
            heatmap = self.explainer.compute_heatmap(img_array)
            
            if heatmap is None:
                return label, color, bbox, None

            # 4. Colorize Heatmap
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.resize(heatmap, (224, 224)) # Ensure 224x224
            
            # Application of ColorMap
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Create Superimposed Image (Overlay)
            # face is BGR, heatmap_colored is BGR. Both 224x224.
            superimposed_img = cv2.addWeighted(face, 0.6, heatmap_colored, 0.4, 0)
            
            return label, color, bbox, superimposed_img
            
        except Exception as e:
            print(f"Grad-CAM Failed: {e}")
            return label, color, bbox, None

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
