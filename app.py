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
# 1. CẤU HÌNH & CSS (GIAO DIỆN HACKER)
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

        /* 1. NỀN TỔNG THỂ */
        .stApp {
            background-color: #050505;
            color: #00ff41;
            font-family: 'Share Tech Mono', monospace;
        }

        /* 2. SIDEBAR */
        [data-testid="stSidebar"] {
            background-color: #000000;
            border-right: 1px solid #333;
        }

        /* 3. INPUTS & SELECTBOX */
        div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
            background-color: #0f0f0f !important;
            color: #00ff41 !important;
            border: 1px solid #333;
            border-radius: 0px; /* Vuông vức */
        }
        
        label, .stMarkdown, h1, h2, h3 {
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
            font-size: 18px;
            text-transform: uppercase;
            padding: 15px;
            transition: 0.3s;
        }
        div.stButton > button:hover {
            background-color: #00ff41;
            color: #000;
            box-shadow: 0 0 20px #00ff41;
        }

        /* 5. METRIC CARDS (HỘP KẾT QUẢ) */
        div[data-testid="metric-container"] {
            background-color: #111;
            border: 1px solid #333;
            padding: 10px;
            border-left: 5px solid #00ff41;
        }
        
        /* 6. BẢNG DỮ LIỆU */
        div[data-testid="stDataFrame"] {
            border: 1px solid #333;
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
# 3. SIDEBAR (GIAO DIỆN ĐIỀU KHIỂN)
# ==============================================================================
st.sidebar.markdown('<h1>⚡ SYSTEM CONTROL</h1>', unsafe_allow_html=True)

ticker = st.sidebar.text_input("ASSET SYMBOL", value="META").upper()

col1, col2 = st.sidebar.columns(2)
with col1:
    freq_display = st.selectbox("TIMEFRAME", ("DAILY", "MONTHLY", "QUARTERLY"))
with col2:
    model_display = st.selectbox("ALGORITHM", ("Naive", "Moving Average", "SES", "Holt", "Holt-Winters"))

# Mapping lại giá trị cho logic
freq_map = {"DAILY": "D", "MONTHLY": "M", "QUARTERLY": "Q"}
freq_val = freq_map[freq_display]

# Cấu hình nâng cao
with st.sidebar.expander("⚙️ ADVANCED SETTINGS", expanded=True):
    window_size = 3
    if model_display == "Moving Average":
        window_size = st.slider("WINDOW SIZE", 2, 50, 3)
    test_size = st.slider("TEST SIZE", 4, 60, 12)

btn_run = st.sidebar.button("INITIALIZE PREDICTION")

# ==============================================================================
# 4. MÀN HÌNH CHÍNH (MAIN SCREEN)
# ==============================================================================
if btn_run:
    st.markdown(f"<h2>>> ANALYZING TARGET: {ticker}</h2>", unsafe_allow_html=True)
    
    with st.spinner('ACCESSING DATA FEED...'):
        try:
            # Tải dữ liệu
            df = yf.download(ticker, period="5y", progress=False)
            if df.empty:
                st.error("❌ ERROR: DATA NOT FOUND.")
                st.stop()
            
            # Fix lỗi cấu trúc Yahoo Finance
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df.columns = [str(c).lower().strip() for c in df.columns]
            
            # Chọn cột giá
            col = 'adj close' if 'adj close' in df.columns else ('close' if 'close' in df.columns else df.columns[0])
            data = df[col].astype(float)
            if data.index.tz is not None: data.index = data.index.tz_localize(None)
            
            # Xử lý thời gian
            if freq_val == "M": 
                data = data.resample('M').last(); seasonal_p = 12
            elif freq_val == "Q": 
                data = data.resample('Q').last(); seasonal_p = 4
            else: 
                data = data.asfreq('B').fillna(method='ffill'); seasonal_p = 5

            data = data.dropna()
            
            if len(data) < test_size + 10:
                st.error("⚠️ DATA INSUFFICIENT FOR ANALYSIS.")
                st.stop()

            # Train/Test Split
            train, test = data.iloc[:-test_size], data.iloc[-test_size:]
            
            # Chạy mô hình
            preds = pd.Series(index=test.index, dtype='float64')
            info = ""
            
            if model_display == "Naive":
                preds[:] = np.array([train.iloc[-1]] * len(test))
            elif model_display == "Moving Average":
                preds = data.rolling(window_size).mean().shift(1).loc[test.index]
            elif model_display == "SES":
                p = find_optimal_params(train, 'SES')
                preds = SimpleExpSmoothing(train).fit(smoothing_level=p[0], optimized=False).forecast(len(test))
                info = f"α={p[0]:.2f}"
            elif model_display == "Holt":
                p = find_optimal_params(train, 'Holt')
                preds = ExponentialSmoothing(train, trend='add').fit(smoothing_level=p[0], smoothing_trend=p[1], optimized=False).forecast(len(test))
                info = f"α={p[0]:.2f}, β={p[1]:.2f}"
            elif model_display == "Holt-Winters":
                p = find_optimal_params(train, 'HW', seasonal_p)
                preds = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=seasonal_p).fit(
                    smoothing_level=p[0], smoothing_trend=p[1], smoothing_seasonal=p[2], optimized=False).forecast(len(test))
                info = f"α={p[0]:.2f}, β={p[1]:.2f}, γ={p[2]:.2f}"

            # Hiển thị Metrics
            mask = ~np.isnan(preds) & ~np.isnan(test)
            if mask.sum() > 0:
                rmse = np.sqrt(mean_squared_error(test[mask], preds[mask]))
                mape = mean_absolute_percentage_error(test[mask], preds[mask]) * 100
                
                m1, m2, m3 = st.columns(3)
                m1.metric("RMSE ERROR", f"{rmse:.2f}")
                m2.metric("ACCURACY GAP (MAPE)", f"{mape:.2f}%")
                m3.info(f"PARAMS: {info}")

            # VẼ BIỂU ĐỒ CYBERPUNK
            fig, ax = plt.subplots(figsize=(14, 6), facecolor='black')
            ax.set_facecolor('black')
            
            # Vẽ dữ liệu
            ax.plot(train.index[-100:], train.iloc[-100:], color='#333333', label='TRAINING DATA')
            ax.plot(test.index, test, color='#00ff41', linewidth=2, label='ACTUAL SIGNAL')
            ax.plot(test.index, preds, color='#ff00ff', linestyle='--', linewidth=2, marker='o', label=f'PREDICTION ({model_display})')
            
            # Trang trí biểu đồ
            ax.grid(color='#222222', linestyle=':', linewidth=0.5)
            ax.tick_params(colors='#00ff41')
            for spine in ax.spines.values(): spine.set_edgecolor('#333333')
            
            ax.legend(facecolor='black', edgecolor='#333333', labelcolor='#00ff41')
            ax.set_title(f"SIGNAL ANALYSIS: {ticker}", color='#00ff41', fontsize=14, fontfamily='monospace')
            
            st.pyplot(fig)
            
            # Bảng dữ liệu
            with st.expander(">> VIEW RAW DATA MATRIX"):
                res_df = pd.DataFrame({'ACTUAL': test, 'PREDICT': preds})
                res_df['DIFF'] = res_df['ACTUAL'] - res_df['PREDICT']
                st.dataframe(res_df.style.highlight_max(axis=0))

        except Exception as e:
            st.error(f"SYSTEM FAILURE: {e}")
else:
    # Màn hình chờ
    st.markdown("""
        <div style='text-align: center; margin-top: 100px; color: #333;'>
            <h3>SYSTEM READY...</h3>
            <p>ENTER PARAMETERS ON THE LEFT TO INITIATE</p>
        </div>
    """, unsafe_allow_html=True)
