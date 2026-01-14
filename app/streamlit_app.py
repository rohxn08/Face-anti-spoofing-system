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
    .brand-title { font-family: 'Syne'; font-size: 3rem; font-weight: 800; line-height: 1; color: white; }
    .brand-subtitle { font-family: 'Inter'; font-size: 0.9rem; color: #888; letter-spacing: 1px; margin-top: 5px; }

    /* LANDING PAGE STYLES */
    .landing-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 80vh;
        text-align: center;
        background: radial-gradient(circle at center, #1a1a1a 0%, #050505 70%);
    }
    
    .landing-header {
        width: 100%;
        display: flex;
        justify-content: space-between;
        padding: 20px 50px;
        position: absolute;
        top: 0;
        left: 0;
        z-index: 99999; /* High Z-index ensures link is clickable above other elements */
    }
    
    /* PILL BADGE & LINKS */
    .pill-badge {
        border: 1px solid #555;
        padding: 8px 20px;
        border-radius: 50px;
        font-family: 'Syne';
        font-weight: 600;
        font-size: 0.8rem;
        letter-spacing: 1px;
        color: white !important; /* Force white color even for links */
        text-transform: uppercase;
        text-decoration: none !important; /* No underline */
        display: inline-block;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .pill-badge:hover {
        background-color: white;
        color: black !important;
        border-color: white;
    }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 6rem;
        font-weight: 800;
        line-height: 1.1;
        text-transform: uppercase;
        background: linear-gradient(180deg, #FFFFFF 0%, #666666 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
        letter-spacing: -2px;
    }
    
    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        color: #AAA;
        max-width: 700px;
        margin-bottom: 50px;
        line-height: 1.6;
    }
    
    /* CONTROL PANEL */
    .control-panel {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 20px;
        height: 100%;
        transition: all 0.5s ease;
    }
    
    .blacked-out {
        opacity: 0.3;
        pointer-events: none;
        filter: grayscale(100%);
    }

    /* LOADING BAR */
    @keyframes loadAnim { 0% { width: 0%; opacity: 0.5; } 50% { width: 70%; opacity: 1; } 100% { width: 100%; opacity: 0; } }
    .loading-container { width: 100%; height: 6px; background-color: #222; border-radius: 3px; margin-bottom: 25px; overflow: hidden; }
    .loading-bar { height: 100%; background: linear-gradient(90deg, #666, #fff, #666); width: 0%; }
    .anim-active { animation: loadAnim 2s infinite ease-in-out; }

    /* BUTTONS */
    .stButton > button {
        background-color: transparent;
        border: 1px solid #666;
        color: white;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 1rem;
        border-radius: 50px;
        padding: 12px 30px;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .stButton > button::before { content: ''; position: absolute; top: 0; left: -100%; width: 100%; height: 100%; background: white; transition: all 0.4s ease; z-index: -1; }
    .stButton > button:hover::before { left: 0; }
    .stButton > button:hover { color: black; border-color: white; box-shadow: 0 0 15px rgba(255,255,255,0.3); }

    /* TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 20px; background-color: transparent; border-bottom: 1px solid #333; margin-bottom: 20px; }
    .stTabs [data-baseweb="tab"] { height: 50px; font-family: 'Syne', sans-serif; font-weight: 600; color: #888; border: none; background-color: transparent; }
    .stTabs [aria-selected="true"] { color: white !important; border-bottom: 2px solid white !important; }

    /* CARDS */
    .section-box { background: #111; border: 1px dashed #333; border-radius: 12px; padding: 10px; min-height: 200px; display: flex; align-items: center; justify-content: center; }
    .img-caption { text-align: center; font-family: 'Syne'; margin-top: 10px; margin-bottom: 5px; font-size: 0.8rem; color: #888; letter-spacing: 1px; }

</style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'page' not in st.session_state:
    st.session_state['page'] = 'landing'
if 'is_loading' not in st.session_state:
    st.session_state['is_loading'] = False
if 'model_arch' not in st.session_state:
    st.session_state['model_arch'] = 'cnn'

def enter_system():
    st.session_state['page'] = 'app'
    st.rerun()

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



# ==========================
# PAGE 1: WELCOME / LANDING
# ==========================
if st.session_state['page'] == 'landing':
    # Top Bar with Z-Index fix
    st.markdown("""
        <div class="landing-header">
            <div style="display:flex; gap:10px;">
                <div class="pill-badge">VERIFACE</div>
                <a href="https://github.com/rohxn08" target="_blank" class="pill-badge">ROHXN08</a>
            </div>
            <div class="pill-badge">2026</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Space for header
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    
    # Hero Section
    c_spacer, c_main, c_spacer2 = st.columns([1, 4, 1])
    with c_main:
        st.markdown("""
            <div style="text-align:center;">
                <div class="hero-title">
                    FACE<br>ANTI SPOOFING<br>SYSTEM
                </div>
                <div class="hero-subtitle" style="margin: 0 auto 40px auto;">
                    Defining the next generation of identity verification. 
                    VeriFace combines state-of-the-art CNN architecture with classical texture analysis 
                    to defend against sophisticated spoofing attacks in real-time.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Enter Button
        if st.button("INITIALIZE SYSTEM", use_container_width=True):
            enter_system()


# ==========================
# PAGE 2: MAIN DASHBOARD
# ==========================
else: # page == app
    # --- HEADER ---
    st.markdown("""
    <div class="header-container">
        <div>
            <div class="brand-title">VeriFace</div>
            <div class="brand-subtitle">Your one stop verifier</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- MAIN LAYOUT ---
    app_col, ctrl_col = st.columns([3, 1])

    # --- CONTROL PANEL (RIGHT) ---
    # --- CONTROL PANEL (RIGHT) ---
    with ctrl_col:
        # Determine if we should blackout controls (Concept: If Live Surveillance is ACTIVE)
        # Note: We can't easily detect which tab is open without extra components, but we can stick to 
        # "If Run Live Checkbox is True" -> Blackout.
        # Since 'run_live' is defined inside the tab, we need to initialize it or put it in session state.
        
        # We'll use a session state var for 'live_active' to control this globally.
        is_live_active = st.session_state.get('live_active', False)
        
        panel_class = "control-panel blacked-out" if is_live_active else "control-panel"
        
        st.markdown(f'<div class="{panel_class}">', unsafe_allow_html=True)
        anim_class = "anim-active" if st.session_state['is_loading'] else ""
        st.markdown(f"""<div class="loading-container"><div class="loading-bar {anim_class}"></div></div>""", unsafe_allow_html=True)
        
        st.markdown('<div style="font-family:Syne; font-weight:700; font-size:1.2rem; margin-bottom:15px; border-bottom:1px solid #333; padding-bottom:10px;">SYSTEM CONTROL</div>', unsafe_allow_html=True)
        
        is_online = check_api()
        status_col = "#4CAF50" if is_online else "#FF5252"
        st.markdown(f"<div style='color:{status_col}; font-size:0.8rem; font-weight:bold; margin-bottom:20px;'>● {'SYSTEM ONLINE' if is_online else 'SYSTEM OFFLINE'}</div>", unsafe_allow_html=True)

        st.markdown("<div style='font-family:Syne; font-weight:600; margin-bottom:10px;'>ARCHITECTURE</div>", unsafe_allow_html=True)
        
        arc_c1, arc_c2 = st.columns(2)
        with arc_c1:
            if st.button("CNN", key="btn_cnn", use_container_width=True, disabled=is_live_active):
                st.session_state['model_arch'] = 'cnn'
        with arc_c2:
            if st.button("SVM", key="btn_svm", use_container_width=True, disabled=is_live_active):
                st.session_state['model_arch'] = 'svm'
                
        current_arch = st.session_state['model_arch']
        st.markdown(f"<div style='text-align:center; font-size:0.8rem; color:#888; margin-top:5px;'>Active: <b style='color:#fff;'>{current_arch.upper()}</b></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        enable_gradcam = False
        if current_arch == 'cnn':
            st.markdown("<div style='font-family:Syne; font-weight:600; margin-bottom:5px;'>EXPLAINABILITY</div>", unsafe_allow_html=True)
            enable_gradcam = st.checkbox("Enable GradCAM", disabled=is_live_active)
        
        st.markdown('</div>', unsafe_allow_html=True)


    # --- APP (LEFT) ---
    with app_col:
        tab_static, tab_dynamic = st.tabs(["STATIC ANALYSIS", "DYNAMIC SURVEILLANCE"])
        
        with tab_static:
            uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
            
            analyze_btn = False
            if uploaded_file:
                st.markdown("<br>", unsafe_allow_html=True)
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
                    label, color, bbox, gc_b64 = predict_frame(img_bgr, current_arch, False, enable_gradcam)
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
            if current_arch == 'svm':
                 st.error("Dynamic Mode Unavailable for SVM")
            else:
                # We use a callback or change logic to update global state
                def update_live_state():
                    st.session_state['live_active'] = st.session_state.get('live_checkbox', False)
                
                run_live = st.checkbox("Active Surveillance", value=False, key='live_checkbox', on_change=update_live_state)
                
                st_frame = st.empty()
                if run_live and is_online:
                    st.session_state['is_loading'] = True
                    reset_backend_history(current_arch)
                    cap = cv2.VideoCapture(0)
                    while run_live:
                        ret, frame = cap.read()
                        if not ret: break
                        label, color, bbox, _ = predict_frame(frame, current_arch, True, False)
                        if bbox:
                            x, y, w, h = bbox
                            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                            cv2.rectangle(frame, (x, y - 30), (x + label_size[0], y), color, -1)
                            cv2.putText(frame, label, (x, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                    cap.release()
                    st.session_state['is_loading'] = False
                    
                # If checkbox unchecked, ensure state is false
                if not run_live:
                    st.session_state['live_active'] = False
