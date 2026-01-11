import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
import io
import time

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8000"

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Face Anti-Spoofing System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main {
        background-color: #f4f6f9;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        font-size: 3rem;
        color: #1E3A8A; /* Dark Blue */
        text-align: center;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        transform: scale(1.05);
    }
    .prediction-card {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .real-card {
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
    }
    .spoof-card {
        background: linear-gradient(135deg, #EF4444, #DC2626);
        color: white;
    }
    .label-text {
        font-size: 2rem;
        font-weight: bold;
    }
    .sidebar-text {
        font-size: 1.1rem;
        font-weight: 500;
    }
    .api-status {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        text-align: center;
        font-weight: bold;
    }
    .status-ok { background-color: #d1fae5; color: #065f46; }
    .status-err { background-color: #fee2e2; color: #991b1b; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
st.sidebar.markdown("## ⚙️ Configuration")

# Check API Status
api_status_placeholder = st.sidebar.empty()

@st.cache_resource(ttl=5) 
def check_api():
    try:
        r = requests.get(f"{API_URL}/", timeout=2)
        if r.status_code == 200:
            return True, "Online"
    except:
        pass
    return False, "Offline"

is_online, status_msg = check_api()
if is_online:
    api_status_placeholder.markdown(f'<div class="api-status status-ok">Backend API: {status_msg} 🟢</div>', unsafe_allow_html=True)
else:
    api_status_placeholder.markdown(f'<div class="api-status status-err">Backend API: {status_msg} 🔴</div>', unsafe_allow_html=True)
    st.sidebar.error("Make sure `api/main.py` is running!")

# 1. Main Selection: Model Type
st.sidebar.markdown('<p class="sidebar-text">Select AI Model</p>', unsafe_allow_html=True)
model_choice = st.sidebar.selectbox(
    "Choose the underlying architecture:",
    ("CNN (Deep Learning)", "SVM (Classical ML)"),
    index=0,
    help="CNN uses MobileNetV2 for robust feature learning. SVM uses LBP features for texture analysis."
)

# Map choice to internally used keys
start_model_key = 'cnn' if "CNN" in model_choice else 'svm'

# 2. Sub-Selection: Input Mode
st.sidebar.markdown("---")
st.sidebar.markdown('<p class="sidebar-text">Select Input Mode</p>', unsafe_allow_html=True)

if start_model_key == 'svm':
    # SVM Mode: Only Static Images allowed
    input_mode = "Upload Image"
    st.sidebar.info("📷 **Note:** Real-time mode is disabled for SVM (Texture Focus) to ensure highest accuracy on static analysis.")
else:
    # CNN Mode: Both allowed
    input_mode = st.sidebar.radio(
        "Choose how to provide data:",
        ("Upload Image", "Live Prediction (Webcam)")
    )

st.sidebar.markdown("---")
st.sidebar.info(f"Currently using **{start_model_key.upper()}** model.")

# --- API HELPERS ---
def predict_frame(image_bgr, model_name, is_live):
    # Encode frame to PNG (Lossless) to prevent compression artifacts affecting CNN
    success, encoded_image = cv2.imencode('.png', image_bgr)
    if not success:
        return "Error Encoding", (0, 0, 0), None
    
    files = {'file': ('frame.png', encoded_image.tobytes(), 'image/png')}
    data = {'model_name': model_name, 'is_live': str(is_live)}
    
    try:
        response = requests.post(f"{API_URL}/predict", files=files, data=data, timeout=5)
        if response.status_code == 200:
            res = response.json()
            # Parse color and bbox back to native types
            color = tuple(res['color']) # [G, B, R]
            bbox = tuple(res['bbox']) if res['bbox'] else None
            return res['label'], color, bbox
        else:
            return f"API Error: {response.status_code}", (0,0,255), None
    except Exception as e:
        return f"Conn Error", (0,0,255), None

def reset_backend_history(model_name):
    try:
        requests.post(f"{API_URL}/reset_history", data={'model_name': model_name}, timeout=2)
    except:
        pass

# --- MAIN CONTENT ---

st.markdown('<div class="main-header">🛡️ Face Anti-Spoofing System</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Powered by {model_choice} (FastAPI Backend) | Secure Biometric Verification</div>', unsafe_allow_html=True)

# --- MODE 1: UPLOAD IMAGE ---
if input_mode == "Upload Image":
    st.write("### 📤 Static Image Verification")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload a face image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load and display
        image_pil = Image.open(uploaded_file)
        
        with col1:
            st.image(image_pil, caption="Uploaded Image", use_container_width=True)
            
        with col2:
            st.write("#### Analysis Result")
            if st.button("🔍 Analyze Authenticity"):
                if not is_online:
                    st.error("Backend is offline. Please start the API.")
                else:
                    with st.spinner("Processing via FastAPI..."):
                        # Convert to OpenCV format (BGR)
                        image_np = np.array(image_pil)
                        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                        
                        # Predict via API
                        label, color, bbox = predict_frame(image_bgr, start_model_key, is_live=False)
                        
                        if bbox is None:
                            if "Error" in label:
                                st.error(f"Analysis failed: {label}")
                            else:
                                st.warning("⚠️ No face detected in the image.")
                        else:
                            # Draw box on image for visualization
                            x, y, w, h = bbox
                            image_vis = image_np.copy()
                            # Color in predictor is BGR, need RGB for PIL/Streamlit
                            color_rgb = (color[2], color[1], color[0]) 
                            cv2.rectangle(image_vis, (x, y), (x+w, y+h), color_rgb, 4)
                            
                            st.image(image_vis, caption="Detected Face", use_container_width=True)
                            
                            # Display Card
                            is_real = "REAL" in label
                            card_class = "real-card" if is_real else "spoof-card"
                            icon = "✅" if is_real else "🚨"
                            
                            st.markdown(f"""
                            <div class="prediction-card {card_class}">
                                <div class="label-text">{icon} {label}</div>
                                <div>Confidence Level Analysis Complete</div>
                            </div>
                            """, unsafe_allow_html=True)

# --- MODE 2: LIVE PREDICTION ---
elif input_mode == "Live Prediction (Webcam)":
    st.write("### 🎥 Live Biometric Security Feed")
    st.write("Click **Start** to open a real-time video feed. Frames are processed by the backend API.")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        run_live = st.checkbox("Turn on Webcam", help="Starts the real-time processing loop.")
        
    with col1:
        st_frame = st.empty()
        
    if run_live:
        if not is_online:
            st.error("Backend is offline. Please start the API.")
        else:
            # Reset voting history for a new session
            reset_backend_history(start_model_key)
            
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Error: Could not open webcam.")
            else:
                while run_live:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture video frame.")
                        break
                    
                    # Predict via API
                    # Note: Sending every frame via HTTP might have latency. 
                    # Ideally, client reduces FPS or resolution if needed.
                    label, color, bbox = predict_frame(frame, start_model_key, is_live=True)
                    
                    # Draw Visuals
                    if bbox:
                        x, y, w, h = bbox
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        
                        # Background for text
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(frame, (x, y - 30), (x + label_size[0], y), color, -1)
                        cv2.putText(frame, label, (x, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Convert BGR to RGB for Streamlit
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st_frame.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    # Slight delay to prevent flooding if local
                    # time.sleep(0.01) 
                
                cap.release()
    else:
        st.info("Webcam is currently off. Check the box to start.")
