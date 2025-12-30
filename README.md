# Face Anti-Spoofing System

A machine learning system designed to distinguish between real faces and spoof attacks (e.g., photos, videos) using dual approaches:
1.  **Traditional ML**: A robust pipeline using **YCbCr + Spatial LBP** (Material & Structure analysis), dimensionality reduction via **PCA**, and classification using **LinearSVC**.
2.  **Deep Learning**: Custom CNN (Convolutional Neural Network) trained on raw pixel data.

## Dataset

The model is trained on the NUAA Imposter Database. You can download the dataset from the link below:

- **NUAA Imposter Database**: [Download Link](https://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/NUAAImposterDB_download.html)

**Note:** After downloading, extract the dataset into the `data/Detectedface` directory. The structure should look like:
```
data/
  Detectedface/
    ClientFace/
    ImposterFace/
```

## Project Structure

- `src/features`: Feature extraction logic (LBP).
- `src/models`: Training scripts for SVM and other models.
- `src/preprocessing`: Data loading and face detection/cropping utilities.
- `src/pipeline`: Inference pipeline for making predictions.
- `src/real_time_predictor.py`: Real-time webcam prediction script.
- `notebooks/`: Experimental notebooks (`svm_model_experiments.ipynb`, `cnn_model_experiments.ipynb`).
- `models/`: Stores trained models (`.pkl` and `.h5` files).

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. SVM Model (LBP + SVM)
**Train:**
```bash
python src/models/train_svm.py
```
or run `notebooks/svm_model_experiments.ipynb`.

**Real-time Prediction:**
Ensure `models/svm_face_antispoofing.pkl` and `models/scaler.pkl` exist.
```bash
python src/real_time_predictor.py
```

### 3. CNN Model (Deep Learning)
**Train:**
Open and run `notebooks/cnn_model_experiments.ipynb`. This notebook handles data generation, model architecture, training, and model saving (e.g., `models/cnn_model.h5`).

**Real-time Prediction:**
(Future Update: Integration into `real_time_predictor.py` coming soon. Currently supported within the notebook environment).

## Features
- **Real-time Detection**: Uses OpenCV for face detection and preprocessing.
- **Dual Support**: Switch between lightweight SVM (CPU-friendly) and robust CNN analysis.
- **Robust Preprocessing**: Includes face cropping and scaling to `128x128`.

## Performance
- **SVM Model Accuracy**: 96% (YCbCr + Spatial LBP + PCA + LinearSVC)
