import os
import cv2
import numpy as np
import sys

# Add src to path to import features
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.features.lbp_extractor import LBPExtractor

def load_data(data_dir, sub_list_file):
    """
    Load data from the dataset directory based on the subject list file.
    
    Args:
        data_dir (str): Path to the data directory (e.g., .../MSU-MFSD)
        sub_list_file (str): Path to the subject list file (e.g., train_sub_list.txt)
        
    Returns:
        tuple: (X, y) where X is the array of features/images and y is the array of labels
    """
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

if __name__ == '__main__':
    # Example usage
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(current_dir))
    data_dir = os.path.join(base_dir, 'data', 'MSU-MFSD')
    train_list = os.path.join(data_dir, 'train_sub_list.txt')
    
    if os.path.exists(train_list):
        X, y = load_data(data_dir, train_list)
        print(f"Loaded data shape: {X.shape}, Labels shape: {y.shape}")
    else:
        print("Train list not found.")
