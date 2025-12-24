from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import numpy as np
import os
import cv2
from sklearn.preprocessing import StandardScaler
import sys
import joblib

# Add src to path to import modules if running from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.features.lbp_extractor import LBPExtractor

# Define dataset paths
attack_dir = r"data\Detectedface\ImposterFace"
real_dir = r"data\Detectedface\ClientFace"
IMG_SIZE = (128, 128)

# Initialize Extractor with new settings
extractor = LBPExtractor(num_points=24, radius=3)

def load_data_robust(real_dir, attack_dir):
    x = []
    y = []
    
    def process_dir(directory, label):
        count = 0
        if not os.path.exists(directory):
            print(f"Warning: Directory not found: {directory}")
            return 0
            
        print(f"Processing {directory}...")
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                    img_path = os.path.join(root, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        try:
                            # INTERNAL CROP: Ignore hair/ears/background
                            h, w = img.shape[:2]
                            # Take center 50% of the image (25% margin on all sides)
                            img_cropped = img[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
                            img_resized = cv2.resize(img_cropped, IMG_SIZE)
                            
                            feat = extractor.extract(img_resized)
                            x.append(feat)
                            y.append(label)
                            count += 1
                        except Exception as e:
                            print(f"Error processing {filename}: {e}")
        return count

    n_real = process_dir(real_dir, 0)
    n_imposter = process_dir(attack_dir, 1)
    
    print(f"Loaded {n_real} real images and {n_imposter} imposter images.")
    return np.array(x), np.array(y)

def train_model(x, y):
    # Split
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    xtr = scaler.fit_transform(xtr)
    xte = scaler.transform(xte)
    
    # PCA STARVATION: Dropping to 70% variance kills the high-frequency identity/noise
    pca = PCA(n_components=0.70, random_state=42)
    xtr = pca.fit_transform(xtr)
    xte = pca.transform(xte)
    
    print(f"PCA Components retained: {pca.n_components_}")
    
    # Strong L1 Regularization (Feature Selection)
    print("Training Calibrated LinearSVC...")
    base_svm = LinearSVC(penalty='l1', C=0.005, dual=False, class_weight='balanced', max_iter=5000)
    model = CalibratedClassifierCV(base_svm, cv=5)
    model.fit(xtr, ytr)
    
    return model, scaler, pca, xte, yte, xtr, ytr

def predict(model, xte):
    y_preds = model.predict(xte)
    return y_preds

if __name__ == "__main__":
    print("Loading Chrominance-Texture data...")
    # Using the local load function since logic is custom (cropping)
    x, y = load_data_robust(real_dir, attack_dir)

    if len(x) == 0:
        print("No data found. Please check the dataset paths.")
    else:
        model, scaler, pca, xte, yte, xtr, ytr = train_model(x, y)
        
        # --- DIAGNOSTICS ---
        print("\n" + "="*40)
        print("     MODEL DIAGNOSTICS")
        print("="*40)
        
        train_preds = model.predict(xtr)
        print(f"Train Accuracy: {accuracy_score(ytr, train_preds):.4f}")
        
        print("\nTest Set Evaluation:")
        y_preds = predict(model, xte)
        print(classification_report(yte, y_preds, target_names=['Real', 'Spoof']))
        
        # Save artifacts to 'saved_models' to match the predictor's expectations
        os.makedirs('saved_models', exist_ok=True)
        joblib.dump(model, 'saved_models/face_antispoof_svm.pkl')
        joblib.dump(scaler, 'saved_models/scaler.pkl')
        joblib.dump(pca, 'saved_models/pca.pkl')
        print("Model, scaler, and PCA saved to 'saved_models/' directory.")