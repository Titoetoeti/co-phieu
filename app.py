import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.api import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.optimize import minimize
import warnings

# ==============================================================================
# 1. CẤU HÌNH & CSS (PIXEL STYLE - FIX LỖI LAYOUT)
# ==============================================================================
warnings.filterwarnings("ignore")
st.set_page_config(page_title="PIXEL TRADER PRO", layout="wide", page_icon="📈")
plt.style.use('dark_background')

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');

        /* 1. ẨN THANH MENU TRÊN CÙNG ĐỂ KHÔNG BỊ VƯỚNG */
        header[data-testid="stHeader"] { visibility: hidden; }
        .block-container { padding-top: 2rem; } /* Đẩy nội dung xuống xíu */

        /* 2. NỀN CHÍNH */
        .stApp {
            background-color: #0d0d0d;
            color: #00ff41;
            font-family: 'VT323', monospace;
            font-size: 20px;
        }

        /* 3. TÙY CHỈNH WIDGETS (Ô NHẬP LIỆU) */
        /* Tạo viền xanh neon cho input */
        div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
            background-color: #000 !important;
            color: #00ff41 !important;
            border: 2px solid #00ff41 !important; 
            border-radius: 0px;
            font-family: 'VT323', monospace !important;
            font-size: 22px;
        }
        
        /* Chữ của nhãn (Label) */
        label p {
            font-size: 20px !important;
            font-family: 'Press Start 2P', cursive !important; /* Font Game cho nhãn */
            color: #00ff41 !important;
        }

        /* 4. NÚT BẤM (BUTTON) */
        div.stButton > button {
            width: 100%;
            background-color: #000;
            color: #00ff41;
            border: 4px double #00ff41; /* Viền đôi cho ngầu */
            font-family: 'Press Start 2P', cursive;
            font-size: 16px;
            padding: 15px;
            margin-top: 10px;
            transition: 0.2s;
        }
        div.stButton > button:hover {
            background-color: #00ff41;
            color: #000;
            box-shadow: 0 0 20px #00ff41;
        }

        /* 5. HIỆU ỨNG NHẤP NHÁY */
        @keyframes blinker { 50% { opacity: 0; } }
        .blinking-text {
            animation: blinker 1s step-end infinite;
            color: #00ff41;
            font-family: 'Press Start 2P', cursive;
            font-size: 18px;
            text-align: center;
            margin-top: 30px;
            text-shadow: 0 0 10px #00ff41;
        }

        /* 6. TIÊU ĐỀ */
        h1 {
            font-family: 'Press Start 2P', cursive;
            text-align: center;
            color: #00ff41;
            text-shadow: 4px 4px 0px #003300;
            margin-bottom: 0px;
        }
        .sub-title {
            text-align: center;
            font-family: 'VT323', monospace;
            font-size: 24px;
            color: #555;
            letter-spacing: 4px;
            margin-bottom: 30px;
        }
        
        /* 7. KHOẢNG CÁCH CÁC PHẦN */
        hr { border-color: #333; margin: 30px 0; }

    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. LOGIC TÍNH TOÁN
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

# ==============================================================================
# 3. GIAO DIỆN CHÍNH (LAYOUT AN TOÀN)
# ==============================================================================

# Header
st.markdown("<h1>⚡ PIXEL TRADER ⚡</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>PRO EDITION [v2.0]</div>", unsafe_allow_html=True)

# --- KHU VỰC NHẬP LIỆU (CONTROL PANEL) ---
# Dùng container của Streamlit để không bị vỡ layout
with st.container():
    # Chia cột: [Khoảng trống] - [Nội dung chính] - [Khoảng trống]
    # Tăng độ rộng cột giữa lên (2 -> 3) để input không bị chèn ép
    c1, c2, c3 = st.columns([1, 3, 1]) 
    
    with c2:
        # Nhập mã CK
        ticker = st.text_input("PLAYER TARGET (MÃ CK)", value="META", placeholder="EX: AAPL").upper()
        
        col_inp1, col_inp2 = st.columns(2)
        with col_inp1:
            freq_display = st.selectbox("TIMEFRAME", ("DAILY", "MONTHLY", "QUARTERLY"))
        with col_inp2:
            model_display = st.selectbox("WEAPON (MODEL)", ("Naive", "Moving Average", "SES", "Holt", "Holt-Winters"))
        
        # Expander chuẩn của Streamlit
        with st.expander("⚙️ ADVANCED SETTINGS"):
            window_size = 3
            if model_display == "Moving Average":
                window_size = st.slider("WINDOW SIZE", 2, 50, 3)
            test_size = st.slider("BACKTEST SIZE", 4, 60, 12)
        
        st.write("") # Tạo khoảng trống nhỏ
        btn_run = st.button(">> START PREDICTION <<")

st.markdown("---") # Đường kẻ phân cách rõ ràng để không bị chồng chéo

# ==============================================================================
# 4. XỬ LÝ & KẾT QUẢ
# ==============================================================================
freq_map = {"DAILY": "D", "MONTHLY": "M", "QUARTERLY": "Q"}
freq_val = freq_map[freq_display]

if btn_run:
    st.markdown(f"<h3 style='text-align: center; font-family:VT323'>LOADING DATA FOR: {ticker}...</h3>", unsafe_allow_html=True)
    
    try:
        # Tải dữ liệu
        df = yf.download(ticker, period="5y", progress=False)
        if df.empty:
            st.error("❌ ERROR: DATA NOT FOUND.")
            st.stop()
        
        # Fix lỗi cấu trúc
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower().strip() for c in df.columns]
        
        col = None
        for c in ['adj close', 'close', 'price']:
            if c in df.columns:
                col = c
                break
        if col is None: col = df.columns[0]
            
        data = df[col].astype(float)
        if data.index.tz is not None: data.index = data.index.tz_localize(None)
        
        if freq_val == "M": data = data.resample('M').last(); seasonal_p = 12
        elif freq_val == "Q": data = data.resample('Q').last(); seasonal_p = 4
        else: data = data.asfreq('B').fillna(method='ffill'); seasonal_p = 5

        data = data.dropna()
        if len(data) < test_size + 10: st.error("⚠️ DATA INSUFFICIENT."); st.stop()

        train, test = data.iloc[:-test_size], data.iloc[-test_size:]
        preds = pd.Series(index=test.index, dtype='float64')
        info = ""
        
        # Tính toán mô hình
        if model_display == "Naive": preds[:] = np.array([train.iloc[-1]] * len(test))
        elif model_display == "Moving Average": preds = data.rolling(window_size).mean().shift(1).loc[test.index]
        elif model_display == "SES":
            p = find_optimal_params(train, 'SES')
            preds = SimpleExpSmoothing(train).fit(smoothing_level=p[0], optimized=False).forecast(len(test)); info = f"α={p[0]:.2f}"
        elif model_display == "Holt":
            p = find_optimal_params(train, 'Holt')
            preds = ExponentialSmoothing(train, trend='add').fit(smoothing_level=p[0], smoothing_trend=p[1], optimized=False).forecast(len(test)); info = f"α={p[0]:.2f}, β={p[1]:.2f}"
        elif model_display == "Holt-Winters":
            p = find_optimal_params(train, 'HW', seasonal_p)
            preds = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=seasonal_p).fit(smoothing_level=p[0], smoothing_trend=p[1], smoothing_seasonal=p[2], optimized=False).forecast(len(test)); info = f"α={p[0]:.2f}, β={p[1]:.2f}, γ={p[2]:.2f}"

        # Metrics
        mask = ~np.isnan(preds) & ~np.isnan(test)
        if mask.sum() > 0:
            rmse = np.sqrt(mean_squared_error(test[mask], preds[mask]))
            mape = mean_absolute_percentage_error(test[mask], preds[mask]) * 100
            
            # Hiển thị kết quả (Dùng Column chuẩn)
            m1, m2, m3 = st.columns(3)
            m1.markdown(f"<div style='border:2px solid #00ff41; padding:10px; text-align:center'><h3 style='margin:0'>RMSE</h3><h1 style='font-family:VT323; font-size:40px; margin:0'>{rmse:.2f}</h1></div>", unsafe_allow_html=True)
            m2.markdown(f"<div style='border:2px solid #00ff41; padding:10px; text-align:center'><h3 style='margin:0'>MAPE</h3><h1 style='font-family:VT323; font-size:40px; margin:0'>{mape:.2f}%</h1></div>", unsafe_allow_html=True)
            m3.info(f"PARAMS: {info}")

        # VẼ BIỂU ĐỒ (Cách ly hoàn toàn với phần trên)
        st.write("") # Khoảng trống
        st.write("") 
        
        fig, ax = plt.subplots(figsize=(14, 7), facecolor='black')
        ax.set_facecolor('#050505')
        
        # Vẽ Train
        ax.plot(train.index[-100:], train.iloc[-100:], color='#555', label='TRAIN', linewidth=1)
        # Vẽ Actual (Nét liền, đậm)
        ax.plot(test.index, test, color='#00ff41', linewidth=2.5, label='ACTUAL')
        # Vẽ Predict (Nét đứt, đậm vừa)
        ax.plot(test.index, preds, color='#ff00ff', linestyle='--', linewidth=2, marker='o', markersize=5, label='PREDICT')
        
        ax.grid(color='#222', linestyle=':', linewidth=0.5)
        ax.legend(facecolor='black', edgecolor='#333', labelcolor='#fff', loc='upper left')
        ax.tick_params(colors='#fff', labelsize=12)
        for s in ax.spines.values(): s.set_edgecolor('#333')
        
        st.pyplot(fig)
        
        # Bảng dữ liệu
        with st.expander(">> VIEW RAW DATA"):
            res = pd.DataFrame({'ACTUAL': test, 'PREDICT': preds, 'DIFF': test-preds})
            st.dataframe(res)

    except Exception as e:
        st.error(f"SYSTEM FAILURE: {e}")

else:
    # Màn hình chờ
    st.markdown("""
        <div class="blinking-text">
            SYSTEM READY...<br>
            [ WAITING FOR USER INPUT ]
        </div>
    """, unsafe_allow_html=True)
