import streamlit as st
import cv2
import base64
import numpy as np
from PIL import Image
import requests
import io
import time

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8000"

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="VeriFace",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM THEME & CSS ---
st.markdown("""
<style>
    /* IMPORT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

    /* ROOT VARIABLES */
    :root {
        --bg-color: #050505;
        --card-bg: #1A1A1A;
        --border-color: #333;
        --text-primary: #FFFFFF;
        --text-secondary: #B3B3B3;
        --pill-bg: #222;
        --pill-border: #444;
    }

    /* GLOBAL RESET */
    .stApp {
        background-color: var(--bg-color);
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Syne', sans-serif !important;
        color: var(--text-primary) !important;
    }

    /* HIDE DEFAULT ELEMENTS */
    header, footer, .css-1rs6os, .css-17z63m9, div[data-testid="stDecoration"] {
        visibility: hidden;
        height: 0px;
    }

    /* HEADER */
    .header-container {
        margin-bottom: 2rem;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .brand-title {
        font-family: 'Syne', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        line-height: 1;
        color: white;
    }
    .brand-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        color: var(--text-secondary);
        letter-spacing: 1px;
        margin-top: 5px;
    }

    /* PILL BUTTONS (Top Right - "More Like This", "Save", "Design") style from image */
    .pill-container {
        display: flex;
        gap: 10px;
    }
    .pill-btn {
        background-color: transparent;
        border: 1px solid var(--pill-border);
        color: var(--text-primary);
        padding: 8px 16px;
        border-radius: 50px;
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        font-weight: 500;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 6px;
        transition: all 0.3s ease;
    }
    .pill-btn:hover {
        background-color: var(--text-primary);
        color: black;
        border-color: white;
    }
    
    /* CONTROL PANEL */
    .control-panel {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 20px;
        height: 100%;
    }

    /* LOADING BAR */
    @keyframes loadAnim { 0% { width: 0%; opacity: 0.5; } 50% { width: 70%; opacity: 1; } 100% { width: 100%; opacity: 0; } }
    .loading-container { width: 100%; height: 6px; background-color: #222; border-radius: 3px; margin-bottom: 25px; overflow: hidden; }
    .loading-bar { height: 100%; background: linear-gradient(90deg, #666, #fff, #666); width: 0%; }
    .anim-active { animation: loadAnim 2s infinite ease-in-out; }

    /* CUSTOM RUN BUTTON (Wide, Styled like Pill/Badge) */
    .stButton > button {
        background-color: transparent;
        border: 1px solid #666;
        color: white;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 1rem;
        border-radius: 50px; /* Fully rounded pill shape */
        padding: 12px 30px;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: white;
        color: black;
        border-color: white;
        box-shadow: 0 0 15px rgba(255,255,255,0.3);
    }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 20px; background-color: transparent; border-bottom: 1px solid #333; margin-bottom: 20px; }
    .stTabs [data-baseweb="tab"] { height: 50px; font-family: 'Syne', sans-serif; font-weight: 600; color: #888; border: none; background-color: transparent; }
    .stTabs [aria-selected="true"] { color: white !important; border-bottom: 2px solid white !important; }

    /* CARDS */
    .section-box { background: #111; border: 1px dashed #333; border-radius: 12px; padding: 10px; min-height: 200px; display: flex; align-items: center; justify-content: center; }
    .img-caption { text-align: center; font-family: 'Syne'; margin-top: 10px; margin-bottom: 5px; font-size: 0.8rem; color: #888; letter-spacing: 1px; }

</style>
""", unsafe_allow_html=True)

# --- STATE ---
if 'is_loading' not in st.session_state:
    st.session_state['is_loading'] = False

# --- API HELPERS ---
@st.cache_resource(ttl=5) 
def check_api():
    try:
        r = requests.get(f"{API_URL}/", timeout=1)
        return r.status_code == 200
    except:
        return False

def predict_frame(image_bgr, model_name, is_live, explain=False):
    success, encoded_image = cv2.imencode('.png', image_bgr)
    if not success: return "Error", (0, 0, 0), None, None
    files = {'file': ('frame.png', encoded_image.tobytes(), 'image/png')}
    data = {'model_name': model_name, 'is_live': str(is_live), 'explain': str(explain)}
    try:
        response = requests.post(f"{API_URL}/predict", files=files, data=data, timeout=5)
        if response.status_code == 200:
            res = response.json()
            color = tuple(res['color']) 
            bbox = tuple(res['bbox']) if res['bbox'] else None
            return res['label'], color, bbox, res.get('gradcam_image')
        return f"API:{response.status_code}", (0,0,255), None, None
    except: return "Conn Error", (0,0,255), None, None

def reset_backend_history(model_name):
    try: requests.post(f"{API_URL}/reset_history", data={'model_name': model_name}, timeout=1)
    except: pass


# --- HEADER ROW (Title Left, Pill Buttons Right) ---
# We use columns to place title on far left and buttons on far right
h_col1, h_col2 = st.columns([3, 1.5])

with h_col1:
    st.markdown("""
    <div>
        <div class="brand-title">VeriFace</div>
        <div class="brand-subtitle">Your one stop verifier</div>
    </div>
    """, unsafe_allow_html=True)

with h_col2:
    # Mimicking the top-right button group from the reference (UI&UX, NEOP COMPANY, 2024 styled pills)
    st.markdown("""
    <div style="display:flex; justify-content:flex-end; gap:10px; margin-top:10px;">
        <div class="pill-btn">UI&UX</div>
        <div class="pill-btn">SECURE</div>
        <div class="pill-btn">2026</div>
    </div>
    """, unsafe_allow_html=True)

# Divider
st.markdown("<div style='height:1px; background-color:#333; margin: 20px 0;'></div>", unsafe_allow_html=True)


# --- MAIN LAYOUT ---
app_col, ctrl_col = st.columns([3, 1])

# --- CONTROL PANEL (RIGHT) ---
with ctrl_col:
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    anim_class = "anim-active" if st.session_state['is_loading'] else ""
    st.markdown(f"""<div class="loading-container"><div class="loading-bar {anim_class}"></div></div>""", unsafe_allow_html=True)
    
    st.markdown('<div style="font-family:Syne; font-weight:700; font-size:1.2rem; margin-bottom:15px; border-bottom:1px solid #333; padding-bottom:10px;">SYSTEM CONTROL</div>', unsafe_allow_html=True)
    
    is_online = check_api()
    status_col = "#4CAF50" if is_online else "#FF5252"
    st.markdown(f"<div style='color:{status_col}; font-size:0.8rem; font-weight:bold; margin-bottom:20px;'>● {'SYSTEM ONLINE' if is_online else 'SYSTEM OFFLINE'}</div>", unsafe_allow_html=True)

    st.markdown("<div style='font-family:Syne; font-weight:600; margin-bottom:5px;'>ARCHITECTURE</div>", unsafe_allow_html=True)
    model_mode = st.radio("Arch", ("CNN", "SVM"), label_visibility="collapsed", key="model_select")
    model_key = 'cnn' if model_mode == "CNN" else 'svm'
    
    st.markdown("<br>", unsafe_allow_html=True)

    enable_gradcam = False
    if model_key == 'cnn':
        st.markdown("<div style='font-family:Syne; font-weight:600; margin-bottom:5px;'>EXPLAINABILITY</div>", unsafe_allow_html=True)
        enable_gradcam = st.checkbox("Enable GradCAM")
    
    st.markdown('</div>', unsafe_allow_html=True)


# --- APP (LEFT) ---
with app_col:
    tab_static, tab_dynamic = st.tabs(["STATIC ANALYSIS", "DYNAMIC SURVEILLANCE"])
    
    with tab_static:
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        
        # Big Pill styled "Browse Files" or similar area is default Streamlit, 
        # but we style our own ACTION BUTTON below it.
        
        analyze_btn = False
        if uploaded_file:
            st.markdown("<br>", unsafe_allow_html=True)
            # This button will pick up the "pill shape" CSS defined above
            analyze_btn = st.button("RUN VERIFICATION", use_container_width=True)
        else:
             st.markdown("""
            <div style="border: 1px dashed #444; padding: 40px; text-align: center; border-radius: 12px; color: #666; margin-top:20px;">
                <span style="font-family:Inter;">Drag and drop file here or Browse</span>
                <br><span style="font-size:0.8rem;">Limit 200MB per file • JPG, PNG, JPEG</span>
            </div>
            """, unsafe_allow_html=True)


        # LOGIC & RESULTS
        if analyze_btn:
            st.session_state['is_loading'] = True
            st.rerun()
        
        if st.session_state['is_loading'] and uploaded_file:
             if is_online:
                time.sleep(0.8) # Anim delay
                img_pil = Image.open(uploaded_file)
                img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                label, color, bbox, gc_b64 = predict_frame(img_bgr, model_key, False, enable_gradcam)
                st.session_state['last_result'] = {'label': label, 'color': color, 'bbox': bbox, 'gc_b64': gc_b64, 'img_pil': img_pil, 'img_np': np.array(img_pil)}
             st.session_state['is_loading'] = False
             st.rerun()

        if 'last_result' in st.session_state and uploaded_file:
            res = st.session_state['last_result']
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.markdown("<div class='img-caption'>PREDICTION</div>", unsafe_allow_html=True)
                if res['bbox']:
                    vis = res['img_np'].copy()
                    x, y, w, h = res['bbox']
                    c_rgb = (res['color'][2], res['color'][1], res['color'][0])
                    cv2.rectangle(vis, (x, y), (x+w, y+h), c_rgb, 4)
                    st.image(vis, use_container_width=True)
                    tc = "#4CAF50" if "REAL" in res['label'] else "#FF5252"
                    st.markdown(f"<h3 style='text-align:center; color:{tc}; font-family:Syne;'>{res['label']}</h3>", unsafe_allow_html=True)
                else: st.warning("No Face")
            
            with c2:
                st.markdown("<div class='img-caption'>SOURCE</div>", unsafe_allow_html=True)
                st.image(res['img_pil'], use_container_width=True)
                
            with c3:
                st.markdown("<div class='img-caption'>ATTENTION MAP</div>", unsafe_allow_html=True)
                if enable_gradcam and res['gc_b64']:
                    st.image(base64.b64decode(res['gc_b64']), use_container_width=True)
                else:
                    st.markdown("<div class='section-box'><span style='color:#555; font-size:0.7rem;'>GRADCAM DISABLED</span></div>", unsafe_allow_html=True)

    with tab_dynamic:
        st.markdown("<br>", unsafe_allow_html=True)
        if model_key == 'svm':
             st.error("Dynamic Mode Unavailable for SVM")
        else:
            run_live = st.checkbox("Active Surveillance", value=False)
            st_frame = st.empty()
            if run_live and is_online:
                st.session_state['is_loading'] = True
                reset_backend_history(model_key)
                cap = cv2.VideoCapture(0)
                st.session_state['is_loading'] = False
                while run_live:
                    ret, frame = cap.read()
                    if not ret: break
                    label, color, bbox, _ = predict_frame(frame, model_key, True, False)
                    if bbox:
                        x, y, w, h = bbox
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                        cv2.rectangle(frame, (x, y - 30), (x + label_size[0], y), color, -1)
                        cv2.putText(frame, label, (x, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                cap.release()
