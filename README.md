# Face Anti-Spoofing System

## Contents
1. [Introduction](#1-introduction)
2. [Demo](#2-demo)
3. [Model Summary](#3-model-summary)
4. [Features](#4-features)
5. [Tech Stack](#6-tech-stack)
6. [Project Structure](#7-project-structure)
7. [How to Run the App](#8-how-to-run-the-app)
8. [Difficulties Faced](#9-difficulties-faced)
9. [Future Improvements](#10-future-improvements)

## 1. Introduction

The **Face Anti-Spoofing System** is a robust biometric security application designed to distinguish between genuine faces ("Real") and presentation attacks ("Spoof"). Whether it's a printed photo, a video replay, or a digital screen mask, this system leverages a hybrid AI approach—combining classical **Texture Analysis (LBP + SVM)** and modern **Deep Learning (MobileNetV2 CNN)**—to ensure high-accuracy verification.

The application features a polished **Streamlit** interface that supports both high-resolution static image analysis and a low-latency, real-time webcam feed with smart temporal smoothing.

## 2. Demo

*(Add your demo video or GIF here)*

## 3. Model Summary

The system employs a dual-model architecture to handle different security scenarios:

### A. Deep Learning: MobileNetV2 CNN
The primary engine for real-time analysis is a **MobileNetV2-based CNN**, fine-tuned for the binary classification task (Real vs. Spoof).

*   **Input Resolution**: 224x224 RGB
*   **Architecture**: MobileNetV2 Encoder (Top 20 layers fine-tuned) + Custom Dense Head.
*   **Key Behavior**: Optimized for identifying general spoofing artifacts like moiré patterns, screen glare, and depth inconsistencies.

### B. Classical Machine Learning: LBP + SVM
For purely texture-based analysis on static images, we utilize a classical pipeline.

*   **Feature Extractor**: Local Binary Patterns (LBP) to capture micro-texture details.
*   **Dimensionality Reduction**: PCA (Principal Component Analysis) to retain 95% variance.
*   **Classifier**: Support Vector Machine (SVM) with RBF kernel.

## 4. Features

### Dual-Mode Interface
1.  **Static Image Verification**:
    *   **High-Fidelity Analysis**: Uses the full resolution of uploaded images.
    *   **Instant Feedback**: Bypasses temporal smoothing for immediate, raw model predictions.
    *   **Visual Debugging**: Displays bounding boxes and precise confidence scores.

2.  **Live Webcam Protection**:
    *   **Real-Time Voting**: Implements a **10-frame voting window** to stabilize predictions and eliminate flickering.
    *   **Smart Scene Detection**: Automatically detects when a subject enters/leaves or if the scene changes, instantly resetting the voting history for a snappy response.
    *   **Visual Feedback**: Color-coded frames (Green=Real, Red=Spoof) with smooth, stable labels.

### Advanced Logic
*   **Smart Continuity Check**: The `RealTimePredictor` tracks the face's position and size across frames. If a face moves unnaturally (teleports) or disappears, the system clears its memory to prevent "lag" when attackers swap a real face for a spoof.
*   **Calibration**: The system logic is fine-tuned to handle specific camera color profiles (BGR/RGB) properly for accurate inference.

## 5. Tech Stack

### Frontend & Application
*   **Streamlit**: For the interactive web interface, sidebar controls, and real-time video rendering.
*   **OpenCV**: For face detection, image processing operations (BGR/RGB conversion, resizing), and drawing bounding boxes.

### AI & Backend
*   **TensorFlow/Keras**: Framework for the MobileNetV2 CNN model.
*   **Scikit-Learn**: For the classic SVM + PCA pipeline.
*   **Joblib**: For efficient model serialization (loading the SVM pipeline).
*   **Python 3.10+**: Core logic.

### Infrastructure
*   **Numpy**: High-performance array manipulation.
*   **Pillow (PIL)**: Image file handling.

## 6. Project Structure

```bash
Face anti spoofing system/
├── app/
│   └── streamlit_app.py        # Main Streamlit Application (Frontend)
├── src/
│   ├── real_time_predictor.py  # Core logic engine (Voting, Scene Detection, Prediction)
│   ├── preprocessing/          # Image cleaning and face cropping utilities
│   ├── features/               # LBP Feature Extractor
│   └── models/                 # Training scripts for CNN and SVM
├── saved_models/
│   ├── face_antispoofing_v3_224.keras  # Trained CNN Model
│   └── svm_texture_pipeline.pkl        # Trained SVM+PCA Pipeline
├── data/                       # Dataset directory
├── requirements.txt            # Project dependencies
└── README.md                   # Documentation
```

## 7. How to Run the App

### Prerequisites
*   Python 3.10 or higher installed.
*   A webcam (for real-time testing).

### Step 1: Install Dependencies
Navigate to the project root and install the required packages:
```bash
pip install -r requirements.txt
```

### Step 2: Launch the App
Run the Streamlit server:
```bash
streamlit run app/streamlit_app.py
```

### Step 3: Access the Interface
*   The app will automatically open in your default browser at `http://localhost:8501`.
*   Select **"CNN"** from the sidebar for the best real-time performance.
*   Toggle **"Live Prediction (Webcam)"** and check **"Turn on Webcam"** to test.

## 8. Difficulties Faced

**1. The "Inertia" Lag in Video:**
Initially, using a continuous voting window meant the system was slow to react when a user swapped a real face for a spoof photo. The "Real" votes from history drowned out the "Spoof" votes.
*   *Solution:* We implemented a **Scene Change Detector** that tracks face centroid movement. Any unnatural jump (>100px) or disappearance immediately wipes the history, making the system feel "snappy."

**2. Model Calibration vs. Deployment:**
The CNN model was trained on specifically normalized RGB images, but the webcam feed (via OpenCV) provides BGR frames with different color profiles. This caused the model to seemingly "invert" its predictions in real-time.
*   *Solution:* We implemented a dedicated calibration logic branch in `real_time_predictor.py` that handles Live Video streams differently from Static Uploads, ensuring accurate thresholds for both.

**3. Balancing Architecture:**
Integrating two completely different model types (SVM vs. Deep Learning) into a single unified `Predictor` class required careful design to ensure the API remained clean (`predict(image)`) regardless of the underlying engine.

## 9. Future Improvements

*   **FastAPI Backend Migration**: Move the heavy AI inference to a dedicated FastAPI server and use **WebSockets** for the video stream. This will decouple the UI from the logic and allow for remote client support.
*   **Liveness Detection**: Integrate simple liveness checks (like blink detection or head pose estimation) to defeat high-quality static photo attacks.
*   **Mobile App Integration**: Build a React Native frontend to bring this security feature to mobile devices.
