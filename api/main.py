from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import sys
import os
from contextlib import asynccontextmanager
from collections import deque

# Add project root to path to find src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.real_time_predictor import RealTimePredictor

predictors = {}
# Dictionary to store voting history for each model
# Format: { 'cnn': deque([0, 1, ...], maxlen=10), 'svm': deque(...) }
voting_histories = {}
last_bboxes = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    global predictors, voting_histories, last_bboxes
    print("Loading models...")
    try:
        try:
            predictors["cnn"] = RealTimePredictor(model_type="cnn")
            voting_histories["cnn"] = deque(maxlen=10)
            last_bboxes["cnn"] = None
            print("CNN model loaded.")
        except Exception as e:
            print(f"Failed to load CNN: {e}")

        try:
            predictors["svm"] = RealTimePredictor(model_type="svm")
            voting_histories["svm"] = deque(maxlen=10)
            last_bboxes["cnn"] = None
            print("SVM model loaded.")
        except Exception as e:
            print(f"Failed to load SVM: {e}")
            
    except Exception as e:
        print(f"Error initializing models: {e}")
    yield
    # Clean up
    predictors.clear()
    voting_histories.clear()
    last_bboxes.clear()

app = FastAPI(title="Face Anti-Spoofing API", lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message": "Face Anti-Spoofing System API is Ready"}

# --- HELPER: Scene Change Detection (Logic Extracted from Class) ---
def check_scene_change(model_key, current_bbox):
    global last_bboxes, voting_histories
    
    last_bbox = last_bboxes.get(model_key)
    
    if last_bbox is None:
        return True # First frame is always a "new scene"
    
    x1, y1, w1, h1 = current_bbox
    x2, y2, w2, h2 = last_bbox
    
    # Calculate Centroids
    c1 = (x1 + w1/2, y1 + h1/2)
    c2 = (x2 + w2/2, y2 + h2/2)
    
    # Euclidean distance
    dist = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
    
    # Threshold (in pixels)
    if dist > 100:
        return True
        
    return False

# --- SEPARATE HANDLERS ---

def handle_realtime_voting(model_key, is_real_frame, bbox):
    """
    Manages the Voting System completely within the API layer.
    """
    global voting_histories, last_bboxes
    
    history = voting_histories[model_key]
    
    # 1. Scene Change Detection
    if bbox:
        if check_scene_change(model_key, bbox):
            history.clear()
        # Update last known position
        last_bboxes[model_key] = bbox
    else:
        # No face? Reset history immediately to prevent stale votes
        history.clear()
        last_bboxes[model_key] = None
        return "No Face", (0, 255, 255) # Yellow
        
    # 2. Add Vote
    # 1 = Real, 0 = Spoof
    vote = 1 if is_real_frame else 0
    history.append(vote)
    
    # 3. Calculate Average
    if len(history) == 0:
        avg_vote = 0
    else:
        avg_vote = sum(history) / len(history)
        
    # 4. Final Decision
    is_real_final = (avg_vote > 0.5)
    
    # 5. Format Output
    if is_real_final:
        # We can calculate a 'stability' score based on avg_vote confidence
        score = avg_vote if is_real_final else (1.0 - avg_vote)
        label = f"REAL ({score:.2f})"
        color = (0, 255, 0) # Green
    else:
        score = 1.0 - avg_vote
        label = f"SPOOF ({score:.2f})"
        color = (0, 0, 255) # Red
        
    return label, color


def handle_cnn_prediction(predictor, image, is_live: bool):
    """
    Specific handler for CNN predictions.
    """
    # Disable internal history in predictor class, we handle it here
    params = {'is_live': False} 
    
    try:
        # Get RAW single-frame prediction from class
        # (It ignores is_live=False so it returns raw frame result)
        label_raw, color_raw, bbox = predictor.predict(image, **params)
        
        if is_live:
            # Parse the RAW label to get boolean Class
            # Standard Logic: "REAL" in label means Real
            is_real_frame = "REAL" in label_raw
            
            # Use our API-level Voting Mechanism
            label, color = handle_realtime_voting("cnn", is_real_frame, bbox)
            return label, color, bbox
        else:
            # Static Mode: Return Raw
            return label_raw, color_raw, bbox
            
    except Exception as e:
        print(f"CNN Error: {e}")
        raise e

def handle_svm_prediction(predictor, image, is_live: bool):
    """
    Specific handler for SVM predictions.
    """
    params = {'is_live': False}
    
    try:
        label_raw, color_raw, bbox = predictor.predict(image, **params)
        
        if is_live:
            is_real_frame = "REAL" in label_raw
            label, color = handle_realtime_voting("svm", is_real_frame, bbox)
            return label, color, bbox
        else:
            return label_raw, color_raw, bbox
            
    except Exception as e:
        print(f"SVM Error: {e}")
        raise e

# --- ENDPOINT ---

@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    model_name: str = Form("cnn"),
    is_live: bool = Form(False)
):
    """
    Predicts if the face in the image is Real or Spoof.
    Dispatches to specific handlers based on model_name.
    """
    model_key = model_name.lower()
    if model_key not in predictors:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not available or failed to load.")
    
    predictor = predictors[model_key]
    
    # Read image
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
        
    # ROUTING LOGIC
    try:
        if model_key == 'cnn':
            label, color, bbox = handle_cnn_prediction(predictor, image, is_live)
        elif model_key == 'svm':
            label, color, bbox = handle_svm_prediction(predictor, image, is_live)
        else:
            # Fallback
            label, color, bbox = predictor.predict(image, is_live=is_live)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    # Ensure native types for JSON serialization
    if bbox:
        bbox = tuple(map(int, bbox))
        
    if color:
        color = tuple(map(int, color))
    
    return {
        "label": label,
        "color": color, # Tuple (G, B, R)
        "bbox": bbox    # Tuple (x, y, w, h) or None
    }

@app.post("/reset_history")
def reset_history(model_name: str = Form("cnn")):
    model_key = model_name.lower()
    if model_key in voting_histories:
        voting_histories[model_key].clear()
        last_bboxes[model_key] = None
        return {"status": "history_reset", "model": model_key}
    else:
        # Fallback if somehow key is missing
        if model_key in predictors:
             predictors[model_key].reset_history()
             return {"status": "internal_history_reset", "model": model_key}
        raise HTTPException(status_code=400, detail="Model not found")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
