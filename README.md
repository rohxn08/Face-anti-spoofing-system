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

The system employs a dual-model architecture trained and validated on diverse, high-benchmark datasets to ensure robustness against various attack types.

### Dataset Evolution
*   **Initial Training (NUAA / DetectedFace)**: The models were originally trained on the **NUAA Imposter Database**, a classic benchmark containing real and printed photo attacks.
*   **Robustness Upgrade (CelebA-Spoof)**: To handle modern "in-the-wild" scenarios, the system was further trained and rigorously validated on the **[CelebA-Spoof Dataset](https://github.com/ZhangYuanhan-AI/CelebA-Spoof)** (Large-scale Face Anti-Spoofing Dataset). This dataset includes diverse lighting, sensors, and attack types including print, replay, and 3D masks.

### A. Deep Learning: MobileNetV2 CNN
The primary engine for real-time analysis is a **MobileNetV2-based CNN**, fine-tuned for the binary classification task (Real vs. Spoof).

*   **Input Resolution**: 224x224 RGB
*   **Architecture**: MobileNetV2 Encoder (Top 20 layers fine-tuned) + Custom Dense Head.
*   **Performance (CelebA-Spoof Test Set)**:
    *   **Accuracy**: **~96%**
    *   **Precision (Spoof)**: **1.00** (Perfect Score)
    *   **Recall (Real)**: **0.99**
*   **Key Behavior**: Optimized for identifying general spoofing artifacts like moiré patterns, screen glare, and depth inconsistencies.

### B. Classical Machine Learning: LBP + SVM
For purely texture-based analysis on static images, we utilize a classical pipeline trained on the same challenging dataset.

*   **Feature Extractor**: Local Binary Patterns (LBP) to capture micro-texture details.
*   **Dimensionality Reduction**: PCA (Principal Component Analysis) to retain 95% variance.
*   **Classifier**: Support Vector Machine (SVM) with RBF kernel.
*   **Performance**: Achieves **~94% Accuracy**, specializing in detecting texture anomalies (e.g., printed dot patterns).

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

### 3. Explainable AI (XAI)
*   **Grad-CAM++ Integration**: We implemented the advanced **Grad-CAM++** algorithm to visualize the model's decision-making process.
*   **Overlay Visualization**: Users can see a "Heatmap Overlay" on uploaded images, showing exactly which regions (eyes, torso reflection, background) contributed to the "Real" or "Spoof" verdict.
*   **Robust Implementation**: Uses a specialized "Split Execution" strategy to handle the complex nested architecture of MobileNetV2 without breaking the TensorFlow graph.

### 4. High-Performance Architecture
*   **FastAPI Backend**: By decoupling the AI logic from the UI, we achieved a **massive performance boost**.
    *   **Startup Time**: Reduced from **15-20 seconds** (monolithic loading) to **< 2 seconds**.
    *   **Responsive**: The API handles requests asynchronously, keeping the UI fluid even during heavy processing.

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
│   └── streamlit_app.py        # Streamlit Frontend (UI Only)
├── api/
│   └── main.py                 # FastAPI Backend (AI Logic)
├── src/
│   ├── real_time_predictor.py  # Core logic engine
│   ├── ...
├── saved_models/               # Model artifacts
├── start_app.bat               # Windows launcher script
└── requirements.txt            # Dependencies
```

## 7. How to Run the App

The system is now split into two parts: a **Backend API** (FastAPI) and a **Frontend UI** (Streamlit). You must run both.

### Quick Start (Windows)
Double-click the `start_app.bat` file in the project folder. This will automatically launch both the backend and frontend in separate windows.

### Manual Start
If you prefer running them manually, open two terminals:

**Terminal 1 (Backend):**
```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```
*Wait for "Application startup complete"*

**Terminal 2 (Frontend):**
```bash
python -m streamlit run app/streamlit_app.py
```

### Accessing the App
*   **Web Interface**: [http://localhost:8501](http://localhost:8501)
*   **API Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## 8. Difficulties Faced

**1. Architecture Decoupling:**
Transitioning from a monolithic Streamlit app to a client-server model required careful handling of state (voting history). We had to move the `deque` history logic into the FastAPI state management to ensure session consistency across HTTP requests.

**2. The "Inertia" Lag in Video:**
Initially, using a continuous voting window meant the system was slow to react when a user swapped a real face for a spoof photo. The "Real" votes from history drowned out the "Spoof" votes.
*   *Solution:* We implemented a **Scene Change Detector** that tracks face centroid movement. Any unnatural jump (>100px) or disappearance immediately wipes the history, making the system feel "snappy."

**3. Model Calibration vs. Deployment:**
The CNN model was trained on specifically normalized RGB images, but the webcam feed (via OpenCV) provides BGR frames with different color profiles. We implemented dedicated calibration logic to handle Live Video streams differently from Static Uploads.

## 9. Future Improvements

*   **WebSocket Streaming**: Currently, the Streamlit app sends individual HTTP POST requests for every frame. Migrating to WebSockets would significantly reduce latency and network overhead.
*   **Dockerization**: Containerize the API and Frontend for easy cloud deployment.
*   **Mobile App Integration**: Build a React Native frontend to consume the API.
