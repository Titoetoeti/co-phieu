import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.api import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from scipy.optimize import minimize 
import warnings
import time
import base64
import os
import gc

# ==============================================================================
# 1. CẤU HÌNH & HÀM HỖ TRỢ
# ==============================================================================
warnings.filterwarnings("ignore")
st.set_page_config(page_title="PIXEL TRADER (TURBO)", layout="wide", page_icon="⚡")
plt.style.use('dark_background') 

# --- HÀM INTRO VIDEO (ĐÃ TỐI ƯU) ---
def show_intro_video(video_file, duration=8):
    if 'intro_done' not in st.session_state:
        st.session_state['intro_done'] = False
    if st.session_state['intro_done']: return
    if not os.path.exists(video_file): st.session_state['intro_done'] = True; return

    try:
        with open(video_file, "rb") as f:
            video_bytes = f.read()
        video_str = base64.b64encode(video_bytes).decode()
        
        intro_placeholder = st.empty()
        intro_placeholder.markdown(
            f"""
            <style>
                .stApp {{ overflow: hidden; }}
                #intro-overlay {{
                    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
                    background-color: #000000; z-index: 999999;
                    display: flex; justify-content: center; align-items: center;
                    flex-direction: column;
                }}
                #intro-video {{ width: 100%; height: 100%; object-fit: cover; }}
            </style>
            <div id="intro-overlay">
                <video id="intro-video" autoplay muted playsinline>
                    <source src="data:video/mp4;base64,{video_str}" type="video/mp4">
                </video>
            </div>
            """, unsafe_allow_html=True)
        time.sleep(duration)
        intro_placeholder.empty()
        st.session_state['intro_done'] = True
        del video_bytes, video_str
        gc.collect() 
        st.rerun()
    except: st.session_state['intro_done'] = True

show_intro_video("intro1.mp4", duration=6)

# --- CSS ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');
        header[data-testid="stHeader"] { visibility: hidden; }
        .stApp { background-color: #0d0d0d; color: #00ff41; font-family: 'VT323', monospace; font-size: 20px; }
        input { color: #ffffff !important; font-family: 'VT323', monospace !important; font-size: 22px !important; }
        div[data-baseweb="select"] > div { background-color: #000 !important; color: #ffffff !important; border-color: #00ff41 !important; }
        label p { font-size: 18px !important; font-family: 'Press Start 2P', cursive !important; color: #00ff41 !important; }
        h1 { font-family: 'Press Start 2P'; text-align: center; color: #00ff41; font-size: 50px; margin-bottom: 0px;}
        div.stButton > button { width: 100%; background-color: #000; color: #00ff41; border: 2px solid #00ff41; font-family: 'Press Start 2P'; padding: 15px; }
        div.stButton > button:hover { background-color: #00ff41; color: #000; box-shadow: 0 0 15px #00ff41; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. HÀM TẢI DỮ LIỆU & TỐI ƯU (CÓ CACHING @st.cache_data)
# ==============================================================================

# [CACHE 1] Tải dữ liệu: Chỉ tải lại sau 1 giờ hoặc khi đổi mã
@st.cache_data(ttl=3600, show_spinner=False)
def load_market_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower().strip() for c in df.columns]
    col = next((c for c in ['adj close', 'close', 'price'] if c in df.columns), df.columns[0])
    return df[col]

def optimize_params(data, model_type, seasonal_periods=None):
    def loss_func(params):
        try:
            if model_type == 'SES':
                model = SimpleExpSmoothing(data).fit(smoothing_level=params[0], optimized=False)
            elif model_type == 'Holt':
                model = ExponentialSmoothing(data, trend='add', seasonal=None, damped_trend=False).fit(
                    smoothing_level=params[0], smoothing_trend=params[1], optimized=False)
            elif model_type == 'HW':
                model = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=seasonal_periods).fit(
                    smoothing_level=params[0], smoothing_trend=params[1], smoothing_seasonal=params[2], optimized=False)
            return np.sum((data - model.fittedvalues)**2)
        except: return 1e10 
    
    bounds = [(0.01, 0.99)]
    if model_type == 'Holt': bounds = [(0.01, 0.99), (0.01, 0.99)]
    if model_type == 'HW': bounds = [(0.01, 0.99), (0.01, 0.99), (0.01, 0.99)]
    
    # Giảm maxiter xuống 50 để chạy nhanh hơn mà vẫn đủ chính xác
    res = minimize(loss_func, [0.5]*len(bounds), bounds=bounds, method='L-BFGS-B', options={'maxiter': 50})
    return res.x

# [CACHE 2] Tính toán AI: Lưu kết quả tính toán, nếu tham số không đổi thì không tính lại
@st.cache_data(show_spinner=False)
def get_forecast(data, model_type, test_size, window_size, future_steps, freq_str):
    if len(data) <= test_size: raise ValueError("Not enough data.")
    
    train = data.iloc[:-test_size]
    test = data.iloc[-test_size:]
    preds = pd.Series(index=test.index, dtype='float64')
    future_series = pd.Series(dtype='float64')
    info = ""
    warning_msg = None
    sp = 1
    
    if freq_str == "DAILY": sp = 5
    elif freq_str == "MONTHLY": sp = 12
    elif freq_str == "QUARTERLY": sp = 4

    # --- AUTO MODE ---
    if model_type == "AUTO (Best Fit)":
        best_rmse = float('inf')
        best_model_name = "Naive"
        candidates = ["Naive", "SES", "Holt"]
        if len(train) > 2 * sp: candidates.append("Holt-Winters")
        
        for m in candidates:
            try:
                # Gọi đệ quy nhưng không cache đoạn này để tránh loop
                _, _, p_try, _, _, _ = get_forecast(data, m, test_size, window_size, 0, freq_str)
                mask = ~np.isnan(p_try) & ~np.isnan(test)
                if mask.sum() > 0:
                    rmse = np.sqrt(mean_squared_error(test[mask], p_try[mask]))
                    if rmse < best_rmse: best_rmse = rmse; best_model_name = m
            except: pass
        model_type = best_model_name
        info = f"AUTO: {best_model_name}"

    try:
        if model_type == "Naive":
            preds[:] = train
