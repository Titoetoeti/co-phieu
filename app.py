import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from statsmodels.tsa.api import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.optimize import minimize
import warnings

# ==============================================================================
# 1. CẤU HÌNH & CSS (GIAO DIỆN GAME SCI-FI)
# ==============================================================================
warnings.filterwarnings("ignore")
st.set_page_config(page_title="PRO TRADING SYSTEM", layout="wide", page_icon="⚡")

# Thiết lập style cho biểu đồ Matplotlib sang Dark Mode
plt.style.use('dark_background')

# CSS TÙY CHỈNH MẠNH
st.markdown("""
    <style>
        /* Import Font công nghệ */
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

        /* 1. XỬ LÝ THANH TOP BAR KHÓ CHỊU */
        header[data-testid="stHeader"] {
            background-color: #000000; /* Đổi màu nền top bar thành đen */
            visibility: hidden; /* Hoặc ẩn luôn nếu bạn muốn full màn hình */
        }
        
        /* Nếu muốn hiện menu hamburger nhưng nền đen thì dùng đoạn dưới, bỏ visibility: hidden ở trên đi */
        /*
        header[data-testid="stHeader"] {
            background-color: #000000;
        }
        */

        /* 2. NỀN TỔNG THỂ */
        .stApp {
            background-color: #050505;
            color: #00ff41;
            font-family: 'Share Tech Mono', monospace;
        }

        /* 3. INPUTS & SELECTBOX (Ở GIỮA) */
        div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
            background-color: #0f0f0f !important;
            color: #00ff41 !important;
            border: 1px solid #333;
            border-radius: 0px; /* Vuông vức */
        }
        
        label, .stMarkdown, h1, h2, h3, p {
            color: #00ff41 !important;
            font-family: 'Share Tech Mono', monospace !important;
        }

        /* 4. NÚT BẤM (BUTTON) */
        div.stButton > button {
            width: 100%;
            background-color: #000;
            color: #00ff41;
            border: 1px solid #00ff41;
            font-family: 'Share Tech Mono', monospace;
            font-size: 20px; /* Chữ to hơn */
            font-weight: bold;
            text-transform: uppercase;
            padding: 15px;
            transition: 0.3s;
            margin-top: 20px;
        }
        div.stButton > button:hover {
            background-color: #00ff41;
            color: #000;
            box-shadow: 0 0 20px #00ff41; /* Phát sáng */
        }

        /* 5. HIỆU ỨNG NHẤP NHÁY (BLINK ANIMATION) */
        @keyframes blinker {
            50% { opacity: 0; }
        }
        .blinking-text {
            animation: blinker 1s linear infinite;
            color: #00ff41;
            font-size: 24px;
            text-align: center;
            margin-top: 50px;
            font-weight: bold;
            text-shadow: 0 0 10px #00ff41;
        }
        
        /* 6. KHUNG CHỨA CONTROL PANEL */
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
# 2. LOGIC TÍNH TOÁN (CORE ENGINE)
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
# 3. GIAO DIỆN TRUNG TÂM (CENTER CONTROL)
# ==============================================================================

# Tiêu đề chính
st.markdown("<h1 style='text-align: center; font-size: 60px; text-shadow: 0 0 10px #00ff41;'>⚡ QUANTUM TRADER ⚡</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; letter-spacing: 3px; color: #555 !important;'>ADVANCED MARKET PREDICTION SYSTEM</p>", unsafe_allow_html=True)
st.markdown("---")

# TẠO 3 CỘT ĐỂ CĂN GIỮA PANEL ĐIỀU KHIỂN
# Cột 1 và 3 là khoảng trống, Cột 2 là nơi chứa Input
c1, c2, c3 = st.columns([1, 2, 1]) 

with c2:
    # Bắt đầu khung Control Panel
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    st.markdown("### >> INPUT PARAMETERS")
    
    # Input Mã chứng khoán
    ticker = st.text_input("ASSET SYMBOL (MÃ CỔ PHIẾU)", value="META", placeholder="EX: AAPL, BTC-USD").upper()
    
    # Hai cột nhỏ bên trong cột giữa
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        freq_display = st.selectbox("TIMEFRAME (KHUNG THỜI GIAN)", ("DAILY", "MONTHLY", "QUARTERLY"))
    with row1_col2:
        model_display = st.selectbox("ALGORITHM (THUẬT TOÁN)", ("Naive", "Moving Average", "SES", "Holt", "Holt-Winters"))

    # Cấu hình nâng cao (ẩn trong expander cho gọn)
    with st.expander("⚙️ SYSTEM CONFIGURATION (CẤU HÌNH)"):
        window_size = 3
        if model_display == "Moving Average":
            window_size = st.slider("WINDOW SIZE", 2, 50, 3)
        test_size = st.slider("BACKTEST SIZE (SỐ KỲ TEST)", 4, 60, 12)

    # Nút bấm CHẠY to đùng ở giữa
    btn_run = st.button(">> INITIALIZE SYSTEM <<")
    st.markdown('</div>', unsafe_allow_html=True) # Kết thúc khung div

# ==============================================================================
# 4. XỬ LÝ & HIỂN THỊ KẾT QUẢ
# ==============================================================================

# Mapping lại giá trị
freq_map = {"DAILY": "D", "MONTHLY": "M", "QUARTERLY": "Q"}
freq_val = freq_map[freq_display]

if btn_run:
    # Khi bấm nút -> Hiện kết quả
    st.markdown(f"<h3 style='text-align: center;'>PROCESSING DATA TARGET: {ticker}...</h3>", unsafe_allow_html=True)
    
    # Thanh tiến trình giả lập cho ngầu
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        progress_bar.progress(percent_complete + 1)
    
    try:
        # Tải dữ liệu
        df = yf.download(ticker, period="5y", progress=False)
        if df.empty:
            st.error("❌ CRITICAL ERROR: DATA NOT FOUND.")
            st.stop()
        
        # Fix lỗi Yahoo Finance
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower().strip() for c in df.columns]
        col = 'adj close' if 'adj close' in df.columns else ('close' if 'close' in df.columns else df.columns[0])
        data = df[col].astype(float)
        if data.index.tz is not None: data.index = data.index.tz_localize(None)
        
        # Resample
        if freq_val == "M": data = data.resample('M').last(); seasonal_p = 12
        elif freq_val == "Q": data = data.resample('Q').last(); seasonal_p = 4
        else: data = data.asfreq('B').fillna(method='ffill'); seasonal_p = 5

        data = data.dropna()
        if len(data) < test_size + 10: st.error("⚠️ DATA INSUFFICIENT."); st.stop()

        # Split & Model
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

        # Metrics
        mask = ~np.isnan(preds) & ~np.isnan(test)
        if mask.sum() > 0:
            rmse = np.sqrt(mean_squared_error(test[mask], preds[mask]))
            mape = mean_absolute_percentage_error(test[mask], preds[mask]) * 100
            
            # Hiển thị kết quả dạng Cards
            m1, m2, m3 = st.columns(3)
            m1.markdown(f"<div style='border:1px solid #00ff41; padding:10px; text-align:center'><h3>RMSE ERROR</h3><h1>{rmse:.2f}</h1></div>", unsafe_allow_html=True)
            m2.markdown(f"<div style='border:1px solid #00ff41; padding:10px; text-align:center'><h3>ACCURACY (MAPE)</h3><h1>{mape:.2f}%</h1></div>", unsafe_allow_html=True)
            m3.info(f"OPTIMAL PARAMS: {info}")

        # Chart
        fig, ax = plt.subplots(figsize=(14, 6), facecolor='black')
        ax.set_facecolor('black')
        ax.plot(train.index[-100:], train.iloc[-100:], color='#333', label='TRAIN')
        ax.plot(test.index, test, color='#00ff41', linewidth=2, label='ACTUAL')
        ax.plot(test.index, preds, color='#ff00ff', linestyle='--', linewidth=2, marker='o', label='PREDICT')
        ax.grid(color='#222', linestyle=':')
        ax.legend(facecolor='black', edgecolor='#333', labelcolor='#00ff41')
        ax.tick_params(colors='#00ff41')
        for s in ax.spines.values(): s.set_edgecolor('#333')
        
        st.pyplot(fig)
        
        with st.expander(">> VIEW DATA MATRIX"):
            res = pd.DataFrame({'ACTUAL': test, 'PREDICT': preds, 'DIFF': test-preds})
            st.dataframe(res)

    except Exception as e:
        st.error(f"SYSTEM FAILURE: {e}")

else:
    # --- MÀN HÌNH CHỜ (NHẤP NHÁY) ---
    # Chỉ hiện khi chưa bấm nút
    st.markdown("""
        <div class="blinking-text">
            SYSTEM READY...<br>
            ENTER PARAMETERS ABOVE TO INITIATE
        </div>
    """, unsafe_allow_html=True)
