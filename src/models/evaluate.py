import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model():
    # --- CONFIG ---
    BATCH_SIZE = 32
    IMG_SIZE = (128, 128)
    
    # Paths
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'Detectedface')
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'face_antispoofing_model.h5')
    PLOTS_DIR = os.path.join(BASE_DIR, 'models')

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)

    # Use ImageDataGenerator to load data (Validation set as Test set)
    # We use shuffle=False to ensure predictions match labels order for metrics
    test_datagen = ImageDataGenerator(validation_split=0.2)
    
    print("Loading Validation/Test Data...")
    test_generator = test_datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False  # IMPORTANT: Must be False for Confusion Matrix
    )

    # --- PREDICTION ---
    print("Running predictions (this may take a moment)...")
    # Predict returns probabilities [0.1, 0.9, ...]
    y_pred_prob = model.predict(test_generator, verbose=1)
    
    # Convert probabilities to binary class (0 or 1)
    y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)
    
    # Get True Labels
    y_true = test_generator.classes
    
    # Class Names
    class_indices = test_generator.class_indices
    # Invert dictionary: {0: 'ClientFace', 1: 'ImposterFace'}
    classes = {v: k for k, v in class_indices.items()}
    class_names = [classes[0], classes[1]]

    # --- METRICS ---
    print("\n" + "="*40)
    print("CLASSIFICATION REPORT")
    print("="*40)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # --- PLOTS ---
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Saved Confusion Matrix to {cm_path}")

    # 2. ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    roc_path = os.path.join(PLOTS_DIR, 'roc_curve.png')
    plt.savefig(roc_path)
    print(f"Saved ROC Curve to {roc_path}")

if __name__ == "__main__":
    evaluate_model()
