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
# 1. CẤU HÌNH & CSS (GIAO DIỆN PIXEL - BIỂU ĐỒ CHUẨN)
# ==============================================================================
warnings.filterwarnings("ignore")
st.set_page_config(page_title="PIXEL TRADER PRO", layout="wide", page_icon="📈")

# Thiết lập style cho biểu đồ
plt.style.use('dark_background')

st.markdown("""
    <style>
        /* Import Font: Press Start 2P (Tiêu đề) và VT323 (Số liệu) */
        @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');

        /* 1. ẨN TOP BAR */
        header[data-testid="stHeader"] { visibility: hidden; }

        /* 2. NỀN TỔNG THỂ */
        .stApp {
            background-color: #0d0d0d; /* Đen dịu hơn chút để đỡ mỏi mắt */
            color: #00ff41;
            font-family: 'VT323', monospace;
            font-size: 22px;
        }

        /* 3. INPUTS & SELECTBOX */
        div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
            background-color: #000 !important;
            color: #00ff41 !important;
            border: 2px solid #333; /* Viền mảnh lại cho tinh tế */
            border-radius: 0px;
            font-family: 'VT323', monospace !important;
            font-size: 22px;
        }
        
        label, .stMarkdown, p, span {
            color: #00ff41 !important;
            font-family: 'VT323', monospace !important;
            font-size: 22px !important;
        }

        /* 4. TIÊU ĐỀ & NÚT BẤM */
        h1, h2, h3 {
            font-family: 'Press Start 2P', cursive !important;
            color: #00ff41 !important;
            line-height: 1.5 !important;
            text-transform: uppercase;
        }

        div.stButton > button {
            width: 100%;
            background-color: #000;
            color: #00ff41;
            border: 2px solid #00ff41;
            font-family: 'Press Start 2P', cursive;
            font-size: 14px;
            padding: 15px;
            transition: 0.3s;
            margin-top: 20px;
        }
        div.stButton > button:hover {
            background-color: #00ff41;
            color: #000;
            box-shadow: 0 0 15px #00ff41;
        }

        /* 5. HIỆU ỨNG NHẤP NHÁY */
        @keyframes blinker { 50% { opacity: 0; } }
        .blinking-text {
            animation: blinker 1s step-end infinite;
            color: #00ff41;
            font-family: 'Press Start 2P', cursive;
            font-size: 18px;
            text-align: center;
            margin-top: 50px;
            text-shadow: 0 0 10px #00ff41;
            line-height: 2;
        }
        
        /* 6. CONTROL PANEL */
        .control-panel {
            border: 1px solid #333;
            padding: 20px;
            background-color: #0a0a0a;
            margin-bottom: 30px;
            box-shadow: 0 0 10px rgba(0, 255, 65, 0.1);
        }
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
# 3. GIAO DIỆN
# ==============================================================================
st.markdown("<h1 style='text-align: center; font-size: 40px; text-shadow: 0 0 10px #00ff41;'>⚡ PIXEL TRADER ⚡</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; letter-spacing: 2px; color: #555 !important;'>ADVANCED PREDICTION SYSTEM [PRO EDITION]</p>", unsafe_allow_html=True)
st.markdown("---")

c1, c2, c3 = st.columns([1, 2, 1]) 

with c2:
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    st.markdown("### >> INPUT PARAMETERS")
    
    ticker = st.text_input("ASSET SYMBOL", value="META", placeholder="EX: AAPL").upper()
    
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        freq_display = st.selectbox("TIMEFRAME", ("DAILY", "MONTHLY", "QUARTERLY"))
    with row1_col2:
        model_display = st.selectbox("ALGORITHM", ("Naive", "Moving Average", "SES", "Holt", "Holt-Winters"))

    with st.expander("⚙️ CONFIGURATION"):
        window_size = 3
        if model_display == "Moving Average":
            window_size = st.slider("WINDOW SIZE", 2, 50, 3)
        test_size = st.slider("BACKTEST SIZE", 4, 60, 12)

    btn_run = st.button(">> INITIALIZE SYSTEM <<")
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# 4. XỬ LÝ & HIỂN THỊ
# ==============================================================================
freq_map = {"DAILY": "D", "MONTHLY": "M", "QUARTERLY": "Q"}
freq_val = freq_map[freq_display]

if btn_run:
    st.markdown(f"<h3 style='text-align: center;'>LOADING DATA: {ticker}...</h3>", unsafe_allow_html=True)
    
    # Thanh loading Text
    progress_text = st.empty()
    for i in range(101):
        bar = "█" * (i // 5) + "-" * ((100 - i) // 5)
        progress_text.text(f"LOADING: [{bar}] {i}%")
    progress_text.empty()
    
    try:
        # Tải và xử lý dữ liệu (Fix lỗi kỹ càng)
        df = yf.download(ticker, period="5y", progress=False)
        if df.empty:
            st.error("❌ ERROR: DATA NOT FOUND.")
            st.stop()
        
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower().strip() for c in df.columns]
        
        # Logic tìm cột giá thông minh
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

        mask = ~np.isnan(preds) & ~np.isnan(test)
        if mask.sum() > 0:
            rmse = np.sqrt(mean_squared_error(test[mask], preds[mask]))
            mape = mean_absolute_percentage_error(test[mask], preds[mask]) * 100
            
            m1, m2, m3 = st.columns(3)
            m1.markdown(f"<div style='border:1px solid #00ff41; padding:10px; text-align:center'><h3>RMSE</h3><h1 style='font-family:VT323; font-size:40px'>{rmse:.2f}</h1></div>", unsafe_allow_html=True)
            m2.markdown(f"<div style='border:1px solid #00ff41; padding:10px; text-align:center'><h3>MAPE</h3><h1 style='font-family:VT323; font-size:40px'>{mape:.2f}%</h1></div>", unsafe_allow_html=True)
            m3.info(f"PARAMS: {info}")

        # --- VẼ BIỂU ĐỒ (ĐƯỜNG THẲNG - CHUẨN) ---
        fig, ax = plt.subplots(figsize=(14, 6), facecolor='black')
        ax.set_facecolor('black')
        
        # Vẽ Train (Màu xám, đường liền)
        ax.plot(train.index[-100:], train.iloc[-100:], color='#555555', label='TRAIN DATA', linewidth=1.5)
        
        # Vẽ Actual (Màu xanh neon, đường liền, nét đậm)
        ax.plot(test.index, test, color='#00ff41', linewidth=2.5, label='ACTUAL PRICE')
        
        # Vẽ Predict (Màu tím, nét đứt, marker chấm tròn nhỏ)
        ax.plot(test.index, preds, color='#ff00ff', linestyle='--', linewidth=2, marker='o', markersize=5, label='PREDICTION')
        
        # Trang trí lưới và trục
        ax.grid(color='#333', linestyle=':', linewidth=0.5)
        ax.legend(facecolor='black', edgecolor='#333', labelcolor='#fff')
        ax.tick_params(colors='#fff', labelsize=10) # Dùng font mặc định của mpl cho dễ đọc
        for s in ax.spines.values(): s.set_edgecolor('#333')
        
        st.pyplot(fig)
        
        with st.expander(">> VIEW DATA MATRIX"):
            res = pd.DataFrame({'ACTUAL': test, 'PREDICT': preds, 'DIFF': test-preds})
            st.dataframe(res)

    except Exception as e:
        st.error(f"SYSTEM FAILURE: {e}")

else:
    st.markdown("""
        <div class="blinking-text">
            SYSTEM READY...<br>
            ENTER PARAMETERS ABOVE TO INITIATE
        </div>
    """, unsafe_allow_html=True)
