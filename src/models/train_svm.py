import os
import cv2
import numpy as np
from sklearn.svm import SVC
import joblib
import sys

# Add src to path to import features
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.features.lbp_extractor import LBPExtractor

def load_data(data_dir, sub_list_file):
    with open(sub_list_file, 'r') as f:
        subjects = [line.strip() for line in f.readlines()]
    
    # Convert subjects to integers for easier comparison
    subject_ids = set()
    for s in subjects:
        try:
            subject_ids.add(int(s))
        except ValueError:
            continue
            
    print(f"Loading data for subjects: {subject_ids}")
    
    X = []
    y = []
    
    extractor = LBPExtractor()
    
    # Load Real images
    real_dir = os.path.join(data_dir, 'pics', 'real')
    if os.path.exists(real_dir):
        for filename in os.listdir(real_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                # Format: real_client001_...
                parts = filename.split('_')
                if len(parts) > 1 and parts[1].startswith('client'):
                    try:
                        client_id_str = parts[1][6:] # remove 'client'
                        client_id = int(client_id_str)
                        if client_id in subject_ids:
                            img_path = os.path.join(real_dir, filename)
                            img = cv2.imread(img_path)
                            if img is not None:
                                hist = extractor.extract(img)
                                X.append(hist)
                                y.append(1) # 1 for Real
                    except ValueError:
                        continue
    
    # Load Attack images
    attack_dir = os.path.join(data_dir, 'pics', 'attack')
    if os.path.exists(attack_dir):
        for filename in os.listdir(attack_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                # Format: attack_client001_...
                parts = filename.split('_')
                if len(parts) > 1 and parts[1].startswith('client'):
                    try:
                        client_id_str = parts[1][6:]
                        client_id = int(client_id_str)
                        if client_id in subject_ids:
                            img_path = os.path.join(attack_dir, filename)
                            img = cv2.imread(img_path)
                            if img is not None:
                                hist = extractor.extract(img)
                                X.append(hist)
                                y.append(0) # 0 for Attack
                    except ValueError:
                        continue
                        
    return np.array(X), np.array(y)

def train():
    # Get the absolute path of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to get to the project root
    base_dir = os.path.dirname(os.path.dirname(current_dir))
    
    data_dir = os.path.join(base_dir, 'data', 'MSU-MFSD')
    train_list = os.path.join(data_dir, 'train_sub_list.txt')
    
    if not os.path.exists(train_list):
        print(f"Error: Training list not found at {train_list}")
        return

    print("Loading training data...")
    X_train, y_train = load_data(data_dir, train_list)
    
    if len(X_train) == 0:
        print("Error: No training data found.")
        return

    print(f"Training data shape: {X_train.shape}")
    print(f"Labels shape: {y_train.shape}")
    
    print("Training SVM...")
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)
    
    model_path = os.path.join(current_dir, 'svm_model.pkl')
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train()
