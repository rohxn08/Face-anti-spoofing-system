import os
import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.features.lbp_extractor import LBPExtractor
from src.models.train_svm import load_data

def evaluate():
    # Get paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(current_dir))
    
    data_dir = os.path.join(base_dir, 'data', 'MSU-MFSD')
    test_list = os.path.join(data_dir, 'test_sub_list.txt')
    model_path = os.path.join(current_dir, 'svm_model.pkl')
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    if not os.path.exists(test_list):
        print(f"Error: Test list not found at {test_list}")
        return

    print("Loading test data...")
    # Reuse load_data from train_svm
    X_test, y_test = load_data(data_dir, test_list)
    
    if len(X_test) == 0:
        print("Error: No test data found.")
        return

    print(f"Test data shape: {X_test.shape}")
    print(f"Labels shape: {y_test.shape}")
    
    print("Loading model...")
    clf = joblib.load(model_path)
    
    print("Evaluating...")
    y_pred = clf.predict(X_test)
    
    print("\nAccuracy Score:")
    print(accuracy_score(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Spoof', 'Real']))

if __name__ == '__main__':
    evaluate()
