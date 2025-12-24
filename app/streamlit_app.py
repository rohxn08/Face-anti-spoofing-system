import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.real_time_predictor import RealTimePredictor

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
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
st.sidebar.markdown("## ⚙️ Configuration")

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

# --- FUNCTIONS ---

@st.cache_resource(show_spinner=True)
def load_predictor(model_type):
    """
    Loads and caches the RealTimePredictor to avoid reloading on every interaction.
    """
    try:
        return RealTimePredictor(model_type=model_type)
    except Exception as e:
        st.error(f"Failed to load {model_type.upper()} model. Error: {e}")
        return None

# Load the selected model
predictor = load_predictor(start_model_key)

# --- MAIN CONTENT ---

st.markdown('<div class="main-header">🛡️ Face Anti-Spoofing System</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Powered by {model_choice} | Secure Biometric Verification</div>', unsafe_allow_html=True)

if predictor is None:
    st.error("Critical Error: Model could not be loaded. Please check logs.")
    st.stop()

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
                with st.spinner("Processing..."):
                    # Convert to OpenCV format (BGR)
                    image_np = np.array(image_pil)
                    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    
                    # Predict
                    label, color, bbox = predictor.predict(image_bgr)
                    
                    if bbox is None:
                        st.warning("⚠️ No face detected in the image.")
                    else:
                        # Draw box on image for visualization
                        x, y, w, h = bbox
                        # Draw rectangle on original RGB image
                        image_vis = image_np.copy()
                        # Color in predictor is BGR, need RGB for PIL/Streaamlit
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
    st.write("Click **Start** to open a real-time video feed. The system will analyze frames frame-by-frame.")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        run_live = st.checkbox("Turn on Webcam", help="Starts the real-time processing loop.")
        
    with col1:
        st_frame = st.empty()
        
    if run_live:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Error: Could not open webcam.")
        else:
            while run_live:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video frame.")
                    break
                
                # Predict
                label, color, bbox = predictor.predict(frame)
                
                # Draw Visuals
                if bbox:
                    x, y, w, h = bbox
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Background for text for better readability
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(frame, (x, y - 30), (x + label_size[0], y), color, -1)
                    cv2.putText(frame, label, (x, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Convert BGR to RGB for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st_frame.image(frame_rgb, channels="RGB", use_container_width=True)
            
            cap.release()
    else:
        st.info("Webcam is currently off. Check the box to start.")
