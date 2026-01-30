import os
import sys
import glob
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense, Lambda
import tensorflow as tf

# Setup project root for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Verify import
try:
    from src.features.lbp_extractor import LBPExtractor
except ImportError:
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path: sys.path.append(src_path)
    from features.lbp_extractor import LBPExtractor

def load_svm_pipeline(svm_path):
    if not os.path.exists(svm_path):
        print(f"Error: SVM model not found at {svm_path}")
        return None, None, None, None
    
    try:
        svm_artifacts = joblib.load(svm_path)
        svm_scaler = svm_artifacts['scaler']
        svm_pca = svm_artifacts['pca']
        svm_model = svm_artifacts['svm_model']
        lbp_params = svm_artifacts.get('lbp_params', {'num_points': 24, 'radius': 3})
        extractor = LBPExtractor(num_points=lbp_params['num_points'], radius=lbp_params['radius'])
        print("SVM Pipeline loaded successfully.")
        return svm_model, svm_scaler, svm_pca, extractor
    except Exception as e:
        print(f"Failed to load SVM pipeline: {e}")
        return None, None, None, None

def load_cnn_model(cnn_path):
    if not os.path.exists(cnn_path):
        print(f"Error: CNN model not found at {cnn_path}")
        return None
    
    try:
        custom_objects = {'preprocess_input': preprocess_input, 'tf': tf}
        try:
            model = load_model(cnn_path, custom_objects=custom_objects, compile=False, safe_mode=False)
        except Exception as e1:
             print(f"Standard load failed: {e1}. Trying architecture reconstruction...")
             try:
                inputs = Input(shape=(224, 224, 3))
                x = Lambda(preprocess_input)(inputs)
                base_model = tf.keras.applications.MobileNetV2(include_top=False, weights=None, input_shape=(224, 224, 3))
                x = base_model(x)
                x = GlobalAveragePooling2D()(x)
                x = Dropout(0.5)(x)
                x = Dense(64, activation="relu")(x)
                x = Dropout(0.5)(x)
                outputs = Dense(1, activation='sigmoid')(x)
                model = Model(inputs, outputs)
                model.load_weights(cnn_path)
                print("CNN Model reconstructed successfully.")
                return model
             except Exception as e2:
                 print(f"Reconstruction failed: {e2}")
                 return None
             
        print("CNN Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Failed to load CNN model: {e}")
        return None

def process_image_svm(img, extractor, scaler, pca, model):
    try:
        h, w = img.shape[:2]
        # Crop center 50%
        y1, y2 = int(h*0.25), int(h*0.75)
        x1, x2 = int(w*0.25), int(w*0.75)
        face_cropped = img[y1:y2, x1:x2]
        
        if face_cropped.size == 0: 
            face_cropped = cv2.resize(img, (128, 128))
        else: 
            face_cropped = cv2.resize(face_cropped, (128, 128))
            
        feat = extractor.extract(face_cropped).reshape(1, -1)
        feat_scaled = scaler.transform(feat)
        feat_pca = pca.transform(feat_scaled)
        
        pred = model.predict(feat_pca)[0]
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(feat_pca)[0][1] # Probability of Class 1 (Spoof)
        else:
            prob = float(pred)
            
        return pred, prob
    except Exception as e:
        return 1, 1.0 

def process_image_cnn(img, model):
    try:
        img_resized = cv2.resize(img, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_array = img_to_array(img_rgb)
        img_preprocessed = preprocess_input(img_array)
        img_batch = np.expand_dims(img_preprocessed, axis=0)
        
        prob = model.predict(img_batch, verbose=0)[0][0]
        pred = 1 if prob > 0.5 else 0
        return pred, prob
    except Exception as e:
        return 1, 1.0

def load_celeba_data(root_dir):
    data = [] 
    
    # Structure: root_dir/SubjectID/live/*.png and root_dir/SubjectID/spoof/*.png
    # Use glob to find all subject folders
    print(f"Scanning directory: {root_dir}")
    
    subject_dirs = glob.glob(os.path.join(root_dir, "*"))
    
    for subj_dir in tqdm(subject_dirs, desc="Scanning Subjects"):
        if not os.path.isdir(subj_dir): continue
        
        # Check for live folder
        live_dir = os.path.join(subj_dir, "live")
        if os.path.exists(live_dir):
            images = glob.glob(os.path.join(live_dir, "*.png")) + glob.glob(os.path.join(live_dir, "*.jpg"))
            for img_path in images:
                data.append({'path': img_path, 'label': 0}) # 0 = Real
        
        # Check for spoof folder
        spoof_dir = os.path.join(subj_dir, "spoof")
        if os.path.exists(spoof_dir):
            images = glob.glob(os.path.join(spoof_dir, "*.png")) + glob.glob(os.path.join(spoof_dir, "*.jpg"))
            for img_path in images:
                data.append({'path': img_path, 'label': 1}) # 1 = Spoof

    print(f"Found {len(data)} total images.")
    # NO RANDOM SAMPLING as requested
    return data

def evaluate_celeba_dataset(data_path, output_dir):
    dataset = load_celeba_data(data_path)
    
    if len(dataset) == 0:
        print("No data found.")
        return
        
    # Load Models
    svm_path = os.path.join(project_root, "saved_models", "face_antispoof_svm.pkl")
    cnn_path = os.path.join(project_root, "saved_models", "face_antispoofing_v3_224.keras")
    
    svm_model, svm_scaler, svm_pca, extractor = load_svm_pipeline(svm_path)
    cnn_model = load_cnn_model(cnn_path)
    
    if not svm_model and not cnn_model:
        print("No models loaded to test.")
        return

    # Results Containers
    results = {
        "svm": {"true": [], "pred": [], "score": []},
        "cnn": {"true": [], "pred": [], "score": []}
    }
    
    print("\nProcessing images...")
    processed_count = 0
    
    for item in tqdm(dataset, desc="Evaluating"):
        full_path = item['path']
        label = item['label']
            
        img = cv2.imread(full_path)
        if img is None:
            continue
            
        processed_count += 1
        
        # SVM Eval
        if svm_model:
            s_pred, s_prob = process_image_svm(img, extractor, svm_scaler, svm_pca, svm_model)
            results["svm"]["true"].append(label)
            results["svm"]["pred"].append(s_pred)
            results["svm"]["score"].append(s_prob)
        
        # CNN Eval
        if cnn_model:
            c_pred, c_prob = process_image_cnn(img, cnn_model)
            results["cnn"]["true"].append(label)
            results["cnn"]["pred"].append(c_pred)
            results["cnn"]["score"].append(c_prob)

    print(f"\nProcessed {processed_count} valid images.")
    
    # --- METRICS & PLOTS ---
    target_names = ["Real", "Spoof"] # 0, 1
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        if svm_model:
            if len(results["svm"]["true"]) > 0:
                report_s = classification_report(results["svm"]["true"], results["svm"]["pred"], target_names=target_names, labels=[0, 1], zero_division=0)
                print("\n--- SVM CLASSIFICATION REPORT ---")
                print(report_s)
                f.write("--- SVM CLASSIFICATION REPORT ---\n")
                f.write(report_s + "\n\n")
            
        if cnn_model:
            if len(results["cnn"]["true"]) > 0:
                report_c = classification_report(results["cnn"]["true"], results["cnn"]["pred"], target_names=target_names, labels=[0, 1], zero_division=0)
                print("\n--- CNN CLASSIFICATION REPORT ---")
                print(report_c)
                f.write("--- CNN CLASSIFICATION REPORT ---\n")
                f.write(report_c + "\n\n")

    # PLOTS
    # 1. Confusion Matrices
    n_plots = 0
    if svm_model: n_plots += 1
    if cnn_model: n_plots += 1
    
    if n_plots > 0:
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        if n_plots == 1: axes = [axes]
        
        ax_idx = 0
        if svm_model and len(results["svm"]["true"]) > 0:
            cm_svm = confusion_matrix(results["svm"]["true"], results["svm"]["pred"], labels=[0, 1])
            sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=axes[ax_idx])
            axes[ax_idx].set_title("SVM Confusion Matrix")
            ax_idx += 1
            
        if cnn_model and len(results["cnn"]["true"]) > 0:
            cm_cnn = confusion_matrix(results["cnn"]["true"], results["cnn"]["pred"], labels=[0, 1])
            sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Reds', xticklabels=target_names, yticklabels=target_names, ax=axes[ax_idx])
            axes[ax_idx].set_title("CNN Confusion Matrix")
        
        plt.tight_layout()
        cm_plot_path = os.path.join(output_dir, "confusion_matrices.png")
        plt.savefig(cm_plot_path)
        print(f"Confusion Matrices saved to {cm_plot_path}")
    
    # 2. ROC Curve
    if n_plots > 0:
        plt.figure(figsize=(8, 6))
        if svm_model and len(set(results["svm"]["true"])) > 1:
            fpr_s, tpr_s, _ = roc_curve(results["svm"]["true"], results["svm"]["score"])
            auc_s = auc(fpr_s, tpr_s)
            plt.plot(fpr_s, tpr_s, label=f'SVM (AUC = {auc_s:.2f})')
            
        if cnn_model and len(set(results["cnn"]["true"])) > 1:
            fpr_c, tpr_c, _ = roc_curve(results["cnn"]["true"], results["cnn"]["score"])
            auc_c = auc(fpr_c, tpr_c)
            plt.plot(fpr_c, tpr_c, label=f'CNN (AUC = {auc_c:.2f})')
            
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        
        roc_plot_path = os.path.join(output_dir, "roc_curve.png")
        plt.savefig(roc_plot_path)
        print(f"ROC Curve saved to {roc_plot_path}")

if __name__ == "__main__":
    # Path to the 'test' folder inside CelebA_Spoof
    celeba_path = r"C:\ALL PROJECTS\Face anti spoofing system\data\CelebA_Spoof\CelebA_Spoof\CelebA_Spoof\Data\test"
    output_results_path = r"C:\ALL PROJECTS\Face anti spoofing system\notebooks\validation_results_celeba"
    
    print("Starting Validation on CelebA_Spoof Dataset (Full Test Set)...")
    evaluate_celeba_dataset(celeba_path, output_results_path)
    print("Done.")
