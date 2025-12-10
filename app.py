import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.api import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from scipy.optimize import minimize
import warnings
import time
import base64
import os

# ==============================================================================
# 1. CẤU HÌNH & HÀM HỖ TRỢ (V3.5: UPDATE VIDEO NAME -> intro1.mp4)
# ==============================================================================
warnings.filterwarnings("ignore")
st.set_page_config(page_title="PIXEL TRADER PRO", layout="wide", page_icon="📈")
plt.style.use('dark_background')

# --- HÀM 1: INTRO VIDEO ---
def show_intro_video(video_file, duration=8):
    if 'intro_done' not in st.session_state:
        st.session_state['intro_done'] = False

    if st.session_state['intro_done']:
        return

    # Kiểm tra file tồn tại
    if not os.path.exists(video_file):
        st.warning(f"⚠️ KHÔNG TÌM THẤY FILE: '{video_file}'. Bỏ qua intro...")
        time.sleep(1)
        st.session_state['intro_done'] = True
        st.rerun()
        return

    try:
        with open(video_file, "rb") as f:
            video_bytes = f.read()
        
        if len(video_bytes) > 20 * 1024 * 1024:
            st.warning("⚠️ Video > 20MB, có thể gây chậm.")

        video_str = base64.b64encode(video_bytes).decode()
        
        intro_html = f"""
        <style>
            .stApp {{ overflow: hidden; }}
            #intro-overlay {{
                position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
                background-color: #000000; z-index: 999999;
                display: flex; justify-content: center; align-items: center;
                flex-direction: column;
            }}
            #intro-video {{ width: 100%; height: 100%; object-fit: cover; }}
            #skip-btn {{
                position: absolute; bottom: 30px; right: 30px;
                color: #00ff41; font-family: monospace; font-size: 16px;
                z-index: 1000000; border: 1px solid #00ff41; padding: 10px;
                background: black; opacity: 0.8;
            }}
        </style>
        <div id="intro-overlay">
            <video id="intro-video" autoplay muted playsinline>
                <source src="data:video/mp4;base64,{video_str}" type="video/mp4">
            </video>
            <div id="skip-btn">SYSTEM INITIALIZING...</div>
        </div>
        """
        placeholder = st.empty()
        placeholder.markdown(intro_html, unsafe_allow_html=True)
        time.sleep(duration)
        placeholder.empty()
        st.session_state['intro_done'] = True
        st.rerun()

    except Exception as e:
        st.error(f"Lỗi Intro: {e}")
        st.session_state['intro_done'] = True

# --- GỌI INTRO (ĐÃ SỬA THÀNH intro1.mp4) ---
show_intro_video("intro1.mp4", duration=7)


# ==============================================================================
# 2. CSS GIAO DIỆN (PIXEL STYLE - GIỮ NGUYÊN)
# ==============================================================================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');

        header[data-testid="stHeader"] { visibility: hidden; }
        .block-container { padding-top: 2rem; }

        .stApp {
            background-color: #0d0d0d; 
            color: #00ff41;
            font-family: 'VT323', monospace;
            font-size: 20px;
        }
        
        /* INPUTS CHỮ TRẮNG */
        input { color: #ffffff !important; font-family: 'VT323', monospace !important; font-size: 22px !important; }
        div[data-baseweb="select"] > div { background-color: #000 !important; color: #ffffff !important; border-color: #00ff41 !important; }
        div[data-baseweb="input"] > div { background-color: #000 !important; border: 2px solid #00ff41 !important; border-radius: 0px; }
        div[data-baseweb="select"] svg { fill: #00ff41 !important; }

        /* TEXT STYLES */
        label p { font-size: 18px !important; font-family: 'Press Start 2P', cursive !important; color: #00ff41 !important; }
        
        /* TIÊU ĐỀ KHỔNG LỒ */
        h1 {
            font-family: 'Press Start 2P', cursive !important;
            text-align: center; color: #00ff41;
            text-shadow: 6px 6px 0px #003300;
            font-size: 70px !important; line-height: 1.2 !important;
            margin-bottom: 10px !important; margin-top: 0px !important;
        }
        .sub-title { text-align: center; font-family: 'VT323'; font-size: 24px; color: #555; letter-spacing: 4px; margin-bottom: 30px; }

        /* NÚT BẤM ĐEN XANH */
        div.stButton > button {
            width: 100%;
            background-color: #000000 !important;
            color: #00ff41 !important;
            border: 2px solid #00ff41 !important;
            font-family: 'Press Start 2P', cursive !important;
            padding: 15px; margin-top: 15px;
            border-radius: 0px !important;
            transition: all 0.2s ease-in-out;
            box-shadow: none !important;
        }
        div.stButton > button:hover {
            background-color: #00ff41 !important; color: #000000 !important;
            box-shadow: 0 0 15px #00ff41 !important;
        }
        div.stButton > button:active { transform: scale(0.98); }

    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. LOGIC TÍNH TOÁN (GIỮ NGUYÊN)
# ==============================================================================
def find_optimal_params(train_data, model_type, seasonal_periods=None):
    bounds_limit = (0.01, 0.99)
    def loss_function(params):
        try:
            if model_type == 'SES':
                model = SimpleExpSmoothing(train_data).fit(smoothing_level=params[0], optimized=False)
            elif model_type == 'Holt':
                model = ExponentialSmoothing(train_data, trend='add').fit(
                    smoothing_level=params[0], smoothing_trend=params[1], optimized=False)
            elif model_type == 'HW':
                model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=seasonal_periods).fit(
                    smoothing_level=params[0], smoothing_trend=params[1], smoothing_seasonal=params[2], optimized=False)
            return np.sqrt(mean_squared_error(train_data, model.fittedvalues))
        except: return 1e10

    if model_type == 'SES': init, bnds = [0.5], [bounds_limit]
    elif model_type == 'Holt': init, bnds = [0.5, 0.1], [bounds_limit]*2
    elif model_type == 'HW': init, bnds = [0.5, 0.1, 0.1], [bounds_limit]*3
    else: return []

    res = minimize(loss_function, init, bounds=bnds, method='L-BFGS-B')
    return res.x

def get_forecast(data, model_type, test_size, window_size, seasonal_p):
    train, test = data.iloc[:-test_size], data.iloc[-test_size:]
    preds = pd.Series(index=test.index, dtype='float64')
    info = ""
    try:
        if model_type == "Naive": preds[:] = np.array([train.iloc[-1]] * len(test)); info = "NAIVE"
        elif model_type == "Moving Average": 
            preds = data.rolling(window_size).mean().shift(1).loc[test.index]; info = f"MA({window_size})"
        elif model_type == "SES":
            p = find_optimal_params(train, 'SES')
            preds = SimpleExpSmoothing(train).fit(smoothing_level=p[0], optimized=False).forecast(len(test)); info = f"α:{p[0]:.2f}"
        elif model_type == "Holt":
            p = find_optimal_params(train, 'Holt')
            preds = ExponentialSmoothing(train, trend='add').fit(smoothing_level=p[0], smoothing_trend=p[1], optimized=False).forecast(len(test)); info = f"α:{p[0]:.2f}"
        elif model_type == "Holt-Winters":
            p = find_optimal_params(train, 'HW', seasonal_p)
            preds = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=seasonal_p).fit(smoothing_level=p[0], smoothing_trend=p[1], smoothing_seasonal=p[2], optimized=False).forecast(len(test)); info = f"HW"
    except: preds[:] = np.nan
    return test, preds, info

def clean_yfinance_data(df):
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower().strip() for c in df.columns]
    col = next((c for c in ['adj close', 'close', 'price'] if c in df.columns), df.columns[0])
    return df[col]

# ==============================================================================
# 4. GIAO DIỆN CHÍNH
# ==============================================================================

if 'vs_mode' not in st.session_state: st.session_state.vs_mode = False

st.markdown("<h1>STOCKS FORECAST</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>MADE BY NHÓM 4 CÔNG/KHÁNH/HGB [v3.5]</div>", unsafe_allow_html=True)

with st.container():
    c1, c2, c3 = st.columns([1, 3, 1]) 
    with c2:
        ticker = st.text_input("CODE", value="META", placeholder="EX: AAPL").upper()
        col_inp1, col_inp2 = st.columns(2)
        with col_inp1: freq_display = st.selectbox("TIMEFRAME", ("DAILY", "MONTHLY", "QUARTERLY"))
        with col_inp2: model_display = st.selectbox("MODEL", ("Naive", "Moving Average", "SES", "Holt", "Holt-Winters"))
        with st.expander("⚙️ ADVANCED SETTINGS"):
            window_size = 3
            if model_display == "Moving Average": window_size = st.slider("WINDOW SIZE", 2, 50, 3)
            test_size = st.slider("BACKTEST SIZE", 4, 60, 12)
        
        st.write("") 
        btn_run = st.button(">> START PREDICTION <<")

st.markdown("---")

# ==============================================================================
# 5. XỬ LÝ & HIỂN THỊ
# ==============================================================================
freq_map = {"DAILY": "D", "MONTHLY": "M", "QUARTERLY": "Q"}
freq_val = freq_map[freq_display]

if btn_run: st.session_state.vs_mode = False

if btn_run or st.session_state.get('run_success', False):
    st.session_state.run_success = True
    
    try:
        with st.spinner(f"LOADING DATA: {ticker}..."):
            df = yf.download(ticker, period="5y", progress=False)
            data = clean_yfinance_data(df)
            if data is None: st.error("❌ DATA NOT FOUND."); st.stop()
            data = data.astype(float)
            if data.index.tz is not None: data.index = data.index.tz_localize(None)
            if freq_val == "M": data = data.resample('M').last(); seasonal_p = 12
            elif freq_val == "Q": data = data.resample('Q').last(); seasonal_p = 4
            else: data = data.asfreq('B').fillna(method='ffill'); seasonal_p = 5
            data = data.dropna()
            if len(data) < test_size + 10: st.error("⚠️ DATA TOO SHORT."); st.stop()

            test, preds, info = get_forecast(data, model_display, test_size, window_size, seasonal_p)
            train = data.iloc[:-test_size]

            mask = ~np.isnan(preds) & ~np.isnan(test)
            rmse = np.sqrt(mean_squared_error(test[mask], preds[mask])) if mask.sum() > 0 else 0
            mape = mean_absolute_percentage_error(test[mask], preds[mask]) * 100 if mask.sum() > 0 else 0

            # METRICS DISPLAY (STYLING AS REQUESTED)
            st.markdown(f"<div style='text-align:center; font-family:\"Press Start 2P\"; color:#00ff41; margin-bottom:10px'>TARGET: {ticker}</div>", unsafe_allow_html=True)
            c_m1, c_m2, c_m3 = st.columns(3)
            
            box_style = "border:2px solid #00ff41; padding:10px; text-align:center; height:100%; display:flex; flex-direction:column; justify-content:center;"
            label_font = "font-family: 'Press Start 2P', cursive; font-size: 14px; margin-bottom: 5px; color: #00ff41;"
            value_font = "font-family: 'VT323', monospace; font-size: 40px; margin: 0; line-height: 1; color: #ffffff;"

            c_m1.markdown(f"<div style='{box_style}'><div style='{label_font}'>RMSE</div><div style='{value_font}'>{rmse:.2f}</div></div>", unsafe_allow_html=True)
            c_m2.markdown(f"<div style='{box_style}'><div style='{label_font}'>MAPE</div><div style='{value_font}'>{mape:.2f}%</div></div>", unsafe_allow_html=True)
            c_m3.markdown(f"<div style='border:2px solid #00ffff; padding:10px; text-align:center; height:100%; display:flex; flex-direction:column; justify-content:center;'><div style='font-family: \"Press Start 2P\", cursive; font-size: 14px; margin-bottom: 5px; color: #00ffff;'>PARAMS</div><div style='font-family: \"VT323\", monospace; font-size: 35px; margin: 0; line-height: 1; color: #ffffff;'>{info}</div></div>", unsafe_allow_html=True)

            st.write("")
            fig, ax = plt.subplots(figsize=(14, 6), facecolor='black')
            ax.set_facecolor('black')
            
            ax.plot(train.index[-60:], train.iloc[-60:], color='#777', label='TRAIN')
            ax.plot(test.index, test, color='#00ff41', linewidth=2.5, label='ACTUAL')
            ax.plot(test.index, preds, color='#ff00ff', linestyle='--', linewidth=2, marker='o', markersize=5, label='PREDICT')
            ax.legend(facecolor='black', edgecolor='#333', labelcolor='#fff')
            ax.grid(color='#333', linestyle=':')
            for s in ax.spines.values(): s.set_edgecolor('#333')
            st.pyplot(fig)

            # --- VS MODE ---
            st.markdown("---")
            st.markdown("<h3 style='text-align:center; color:#ffcc00; font-family:\"Press Start 2P\"'>COMPARISON MODE</h3>", unsafe_allow_html=True)
            
            v1, v2, v3 = st.columns([1, 2, 1])
            with v2:
                rivals_input = st.text_input("ENTER RIVALS CODE", value="AAPL, MSFT, GOOG", placeholder="EX: TSLA, AMZN")
                st.write("")
                btn_fight = st.button(">> START COMPARISON <<")

            if btn_fight:
                rivals = [r.strip().upper() for r in rivals_input.split(",") if r.strip()]
                all_tickers = [ticker] + rivals[:4] 
                results_map = {}
                progress_bar = st.progress(0)
                
                for i, t in enumerate(all_tickers):
                    try:
                        d_t = yf.download(t, period="2y", progress=False)
                        val = clean_yfinance_data(d_t)
                        if val is not None and not val.empty:
                            val = val.astype(float)
                            if val.index.tz is not None: val.index = val.index.tz_localize(None)
                            if freq_val == "M": val = val.resample('M').last()
                            elif freq_val == "Q": val = val.resample('Q').last()
                            else: val = val.asfreq('B').fillna(method='ffill')
                            val = val.dropna()
                            if len(val) > test_size + window_size:
                                _, pred_t, _ = get_forecast(val, model_display, test_size, window_size, seasonal_p)
                                if not pred_t.isna().all(): results_map[t] = pred_t
                    except Exception: pass
                    progress_bar.progress((i + 1) / len(all_tickers))
                progress_bar.empty()

                if len(results_map) > 0:
                    st.markdown("<h4 style='text-align:center; font-family:VT323; margin-top:20px'>PREDICTED GROWTH (%) COMPARISON</h4>", unsafe_allow_html=True)
                    fig2, ax2 = plt.subplots(figsize=(14, 7), facecolor='black')
                    ax2.set_facecolor('black')
                    colors = ['#00ff41', '#ff00ff', '#00ffff', '#ffcc00', '#ff3333']
                    for idx, (t_name, pred_series) in enumerate(results_map.items()):
                        if len(pred_series) > 0:
                            start_val = pred_series.iloc[0]
                            if not np.isnan(start_val) and start_val != 0:
                                pct_change = ((pred_series - start_val) / start_val) * 100
                                lw = 3 if t_name == ticker else 2
                                ls = '-' if t_name == ticker else '--'
                                color = colors[idx % len(colors)]
                                ax2.plot(pred_series.index, pct_change, label=f"{t_name}", color=color, linewidth=lw, linestyle=ls)
                    ax2.set_ylabel("GROWTH %")
                    ax2.legend(facecolor='black', edgecolor='#333', labelcolor='#fff')
                    ax2.grid(color='#222', linestyle=':')
                    ax2.axhline(0, color='#555', linewidth=1)
                    for s in ax2.spines.values(): s.set_edgecolor('#333')
                    st.pyplot(fig2)
                else: st.warning("No valid data found for comparison.")

    except Exception as e:
        st.error(f"SYSTEM ERROR: {e}")

else:
    # Màn hình chờ
    st.markdown("""
        <div style='text-align: center; margin-top: 50px; font-family: "Press Start 2P"; color: #00ff41; animation: blinker 1s step-end infinite;'>
            SYSTEM READY...<br>[ WAITING FOR INPUT ]
        </div>
        <style>@keyframes blinker { 50% { opacity: 0; } }</style>
    """, unsafe_allow_html=True)


