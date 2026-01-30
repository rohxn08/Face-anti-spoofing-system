import os
import sys
import glob
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Setup project root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src.features.lbp_extractor import LBPExtractor
except ImportError:
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path: sys.path.append(src_path)
    from features.lbp_extractor import LBPExtractor

# Configuration
IMG_SIZE_SVM = (128, 128)
IMG_SIZE_CNN = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10 

def load_data(root_dir, limit=None):
    """
    Loads data from structured subject folders.
    Returns:
        images: List of image paths
        labels: List of labels (0=Real, 1=Spoof)
    """
    paths = []
    labels = []
    
    print(f"Scanning directory: {root_dir}")
    subject_dirs = glob.glob(os.path.join(root_dir, "*"))
    
    for subj_dir in tqdm(subject_dirs, desc="Scanning Subjects"):
        if not os.path.isdir(subj_dir): continue
        
        # Live
        live_dir = os.path.join(subj_dir, "live")
        if os.path.exists(live_dir):
            imgs = glob.glob(os.path.join(live_dir, "*.png")) + glob.glob(os.path.join(live_dir, "*.jpg"))
            paths.extend(imgs)
            labels.extend([0] * len(imgs))
        
        # Spoof
        spoof_dir = os.path.join(subj_dir, "spoof")
        if os.path.exists(spoof_dir):
            imgs = glob.glob(os.path.join(spoof_dir, "*.png")) + glob.glob(os.path.join(spoof_dir, "*.jpg"))
            paths.extend(imgs)
            labels.extend([1] * len(imgs))
            
    # VALIDATION STEP: Filter out corrupted images
    print("Validating images with TensorFlow...")
    valid_paths = []
    valid_labels = []
    
    for p, l in tqdm(zip(paths, labels), total=len(paths), desc="Checking"):
        try:
            # Full Decode Check to mimic pipeline
            img_raw = tf.io.read_file(p)
            _ = tf.io.decode_image(img_raw, channels=3, expand_animations=False)
            
            valid_paths.append(p)
            valid_labels.append(l)
        except Exception as e:
            # print(f"Corrupted: {p}")
            continue
            
    print(f"Removed {len(paths) - len(valid_paths)} corrupted file(s).")
    paths = valid_paths
    labels = valid_labels

    if limit:
        # Simple shuffle and limit
        combined = list(zip(paths, labels))
        np.random.shuffle(combined)
        combined = combined[:limit]
        paths, labels = zip(*combined)
        paths = list(paths)
        labels = list(labels)
        
    print(f"Loaded {len(paths)} valid images.")
    return paths, labels

def train_svm(paths, labels, save_path):
    print("\n--- Training SVM ---")
    
    extractor = LBPExtractor(num_points=24, radius=3)
    X = []
    y = []
    
    print("Extracting LBP features...")
    # Loading images for SVM
    for i, path in enumerate(tqdm(paths)):
        try:
            img = cv2.imread(path)
            if img is None: continue
            
            # Preprocessing (Crop Center)
            h, w = img.shape[:2]
            y1, y2 = int(h*0.25), int(h*0.75)
            x1, x2 = int(w*0.25), int(w*0.75)
            img_cropped = img[y1:y2, x1:x2]
            if img_cropped.size == 0: img_cropped = img
            
            img_resized = cv2.resize(img_cropped, IMG_SIZE_SVM)
            feat = extractor.extract(img_resized)
            X.append(feat)
            y.append(labels[i])
        except Exception as e:
            continue
            
    X = np.array(X)
    y = np.array(y)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Pipeline components
    scaler = StandardScaler()
    pca = PCA(n_components=0.95, whiten=True) # Keep 95% variance
    svm = SVC(kernel='rbf', probability=True, class_weight='balanced')
    
    # Fit
    print("Fitting Scaler and PCA...")
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    print("Training SVM Classifier...")
    svm.fit(X_train_pca, y_train)
    
    # Eval
    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_scaled)
    preds = svm.predict(X_test_pca)
    
    print("\nSVM Classification Report:")
    print(classification_report(y_test, preds, target_names=['Real', 'Spoof']))
    
    # Save
    pipeline = {
        'scaler': scaler,
        'pca': pca,
        'svm_model': svm,
        'lbp_params': {'num_points': 24, 'radius': 3}
    }
    joblib.dump(pipeline, save_path)
    print(f"SVM Pipeline saved to {save_path}")

def train_cnn(paths, labels, save_path):
    print("\n--- Training CNN ---")
    
    # Split paths
    X_train_paths, X_val_paths, y_train, y_val = train_test_split(paths, labels, test_size=0.2, random_state=42, stratify=labels)
    
    from sklearn.utils import class_weight

    # Calculate Class Weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"Class Weights: {class_weights_dict}")

    # TF Data Generators with Augmentation
    def load_img(path, label):
        img_raw = tf.io.read_file(path)
        # expand_animations=False fixes shape issues for some formats
        img = tf.io.decode_image(img_raw, channels=3, expand_animations=False) 
        img.set_shape([None, None, 3]) # Explicitly set shape rank
        img = tf.image.resize(img, IMG_SIZE_CNN)
        img = preprocess_input(img)
        return img, label

    def augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        return img, label

    def create_dataset(paths, labels, is_train=False):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
        if is_train:
            ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return ds
        
    train_ds = create_dataset(X_train_paths, y_train, is_train=True)
    val_ds = create_dataset(X_val_paths, y_val, is_train=False)
    
    # Model Architecture
    inputs = Input(shape=(224, 224, 3))
    # Preprocessing is done in data pipeline, but can be explicit here too if needed
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_tensor=inputs)
    
    # Fine-tine: Freeze base first
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Callbacks
    checkpoint = ModelCheckpoint(save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    print("Training CNN with Class Weights...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stop],
        class_weight=class_weights_dict
    )
    
    # Optional: Fine-tuning phase (unfreeze top layers) could be added here
    print(f"CNN Model saved to {save_path}")

if __name__ == "__main__":
    # Settings
    DATA_PATH = r"C:\ALL PROJECTS\Face anti spoofing system\data\CelebA_Spoof\CelebA_Spoof\CelebA_Spoof\Data\test"
    OUTPUT_DIR = r"C:\ALL PROJECTS\Face anti spoofing system\saved_models"
    SVM_SAVE_PATH = os.path.join(OUTPUT_DIR, "face_antispoof_svm.pkl") # Overwrite existing to be used by app
    CNN_SAVE_PATH = os.path.join(OUTPUT_DIR, "face_antispoofing_v3_224.keras") # Overwrite existing to be used by app
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load Data
    paths, labels = load_data(DATA_PATH)
    
    if len(paths) == 0:
        print("No data found!")
    else:
        # Train SVM
        train_svm(paths, labels, SVM_SAVE_PATH)
        
        # Train CNN
        train_cnn(paths, labels, CNN_SAVE_PATH)
        
    print("\nTraining Complete.")
