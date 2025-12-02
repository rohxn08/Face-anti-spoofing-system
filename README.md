# Face Anti-Spoofing System

A machine learning system designed to distinguish between real faces and spoof attacks (e.g., photos, videos) using LBP (Local Binary Patterns) features and an SVM classifier.

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
- `models/`: Stores trained models (`.pkl` files).

## Usage

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Train the Model**:
    ```bash
    python src/models/train_svm.py
    ```

3.  **Run Inference**:
    You can use the `FaceAntiSpoofingSystem` class in `src/pipeline/inference.py` to make predictions on new images.
