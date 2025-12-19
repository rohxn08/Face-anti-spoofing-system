from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import sys
import joblib

# Add src to path to import modules if running from root
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.features.lbp_extractor import LBPExtractor
from src.preprocessing.preprocess import load_data_from_dirs

attack_dir = r"data\Detectedface\ImposterFace"
real_dir = r"data\Detectedface\ClientFace"
extractor = LBPExtractor()

def train_model(x, y):
    # Split first to avoid data leakage
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Scale after splitting
    scaler = StandardScaler()
    xtr = scaler.fit_transform(xtr)
    xte = scaler.transform(xte)
    
    param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale'], 'kernel': ['rbf']}
    
    # 3. SVM GridSearch
    
    grid = GridSearchCV(estimator=SVC(class_weight="balanced", probability=True), param_grid=param_grid, scoring="roc_auc", cv=3, n_jobs=-1, verbose=1)
    grid.fit(xtr, ytr)
    model = grid.best_estimator_
    
    return model, scaler, xte, yte

def predict(model, xte, yte):
    y_preds = model.predict(xte)
    return y_preds

if __name__ == "__main__":
    print("Loading data...")
    x, y = load_data_from_dirs(real_dir, attack_dir)
    print(f"Data loaded: {len(x)} samples")

    if len(x) == 0:
        print("No data found. Please check the dataset paths.")
    else:
        print("Training model...")
        model, scaler, xte, yte = train_model(x, y)
        
        print("Predicting...")
        y_preds = predict(model, xte, yte)
        
        print("-" * 30)
        print("Accuracy:", accuracy_score(yte, y_preds))
        print("Precision:", precision_score(yte, y_preds))
        print("Recall:", recall_score(yte, y_preds))
        print("F1 Score:", f1_score(yte, y_preds))
        print("-" * 30)

        # Save model and scaler
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/svm_face_antispoofing.pkl')
        joblib.dump(scaler, 'models/scaler.pkl')
        print("Model and scaler saved to 'models/' directory.")