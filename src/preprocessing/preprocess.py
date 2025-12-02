import os
import cv2
import numpy as np
import os
import cv2
import numpy as np
import sys

# Add src to path to import features
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.features.lbp_extractor import LBPExtractor
def load_data_from_dirs(real_dir, attack_dir):
    X = []
    y = []
    extractor = LBPExtractor()
    
    # Helper to process a directory recursively
    def process_directory(directory, label):
        if not os.path.exists(directory):
            print(f"Warning: Directory not found: {directory}")
            return

        print(f"Processing {directory}...")
        count = 0
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                    img_path = os.path.join(root, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Resize to expected size (128x128)
                        # Assuming images are already cropped faces
                        try:
                            img_resized = cv2.resize(img, (128, 128))
                            feat = extractor.extract(img_resized)
                            X.append(feat)
                            y.append(label)
                            count += 1
                        except Exception as e:
                            print(f"Error processing {filename}: {e}")
        print(f"Loaded {count} images from {directory}")

    # Process Real Images (Label 1)
    process_directory(real_dir, 1)

    # Process Attack Images (Label 0)
    process_directory(attack_dir, 0)

    return np.array(X), np.array(y)