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
def load_data_from_dirs(real_dir, attack_dir, target_size=(128, 128)):
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

    # Process Real Images (Label 0)
    process_directory(real_dir, 0)

    # Process Attack Images (Label 1)
    process_directory(attack_dir, 1)

    return np.array(X), np.array(y)

# Face detection model loaded once to improve performance
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(image_or_path, target_size=(128, 128)):
    
    # Load image if path is given
    if isinstance(image_or_path, str):
        img = cv2.imread(image_or_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_or_path}")
    else:
        img = image_or_path

    if img is None:
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use the global face_cascade
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        # No face found
        return None, None
    
    # Find largest face
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face
    
    # Add a small margin (optional but recommended)
    margin = 0
    x_start = max(0, x - margin)
    y_start = max(0, y - margin)
    x_end = min(img.shape[1], x + w + margin)
    y_end = min(img.shape[0], y + h + margin)
    
    face_img = img[y_start:y_end, x_start:x_end]
    
    # Resize
    try:
        face_resized = cv2.resize(face_img, target_size)
        return face_resized, (x, y, w, h)
    except Exception as e:
        print(f"Error resizing face: {e}")
        return None, None