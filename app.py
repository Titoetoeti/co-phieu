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
# 1. CẤU HÌNH & CSS (PIXEL ART STYLE)
# ==============================================================================
warnings.filterwarnings("ignore")
st.set_page_config(page_title="RETRO TRADER 8-BIT", layout="wide", page_icon="👾")

# Thiết lập style cho biểu đồ Matplotlib sang Dark Mode
plt.style.use('dark_background')

# CSS TÙY CHỈNH MẠNH (PIXEL GAME THEME)
st.markdown("""
    <style>
        /* Import Font Pixel: Press Start 2P (Tiêu đề) và VT323 (Nội dung) */
        @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');

        /* 1. ẨN TOP BAR */
        header[data-testid="stHeader"] {
            visibility: hidden;
        }

        /* 2. NỀN TỔNG THỂ */
        .stApp {
            background-color: #202028; /* Màu xám xanh đậm kiểu Game Boy Advance */
            color: #d0d058; /* Màu vàng nhạt retro */
            font-family: 'VT323', monospace;
            font-size: 20px;
        }

        /* 3. INPUTS & SELECTBOX (KHUNG VIỀN PIXEL) */
        div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
            background-color: #000000 !important;
            color: #00ff00 !important; /* Chữ xanh lá cây cổ điển */
            border: 4px solid #ffffff; /* Viền dày kiểu pixel */
            border-radius: 0px;
            font-family: 'VT323', monospace !important;
            font-size: 22px;
        }
        
        /* Chỉnh Font cho các nhãn */
        label, .stMarkdown p, .stDataFrame {
            color: #ffffff !important;
            font-family: 'VT323', monospace !important;
            font-size: 22px !important;
        }

        /* 4. TIÊU ĐỀ (HEADER) - DÙNG FONT GAME 8-BIT */
        h1, h2, h3 {
            font-family: 'Press Start 2P', cursive !important;
            color: #ffcc00 !important; /* Vàng cam */
            text-shadow: 4px 4px #000000; /* Đổ bóng cứng */
            line-height: 1.5 !important;
        }

        /* 5. NÚT BẤM (BUTTON) - STYLE GAME START */
        div.stButton > button {
            width: 100%;
            background-color: #cc0000; /* Đỏ Nintendo */
            color: #ffffff;
            border: 4px solid #ffffff; /* Viền trắng dày */
            box-shadow: 6px 6px 0px #000000; /* Đổ bóng khối pixel */
            font-family: 'Press Start 2P', cursive;
            font-size: 14px; /* Font này rất to nên để size nhỏ */
            padding: 15px;
            margin-top: 20px;
            transition: 0.1s;
        }
        div.stButton > button:hover {
            background-color: #ff3333;
            transform: translate(2px, 2px); /* Hiệu ứng nhấn nút */
            box-shadow: 4px 4px 0px #000000;
        }
        div.stButton > button:active {
            transform: translate(6px, 6px);
            box-shadow: 0px 0px 0px #000000;
        }

        /* 6. HIỆU ỨNG NHẤP NHÁY (BLINK ANIMATION) */
        @keyframes blinker {
            50% { opacity: 0; }
        }
        .blinking-text {
            animation: blinker 0.8s step-end infinite; /* step-end giúp nháy dứt khoát kiểu game */
            color: #00ff00;
            font-family: 'Press Start 2P', cursive;
            font-size: 16px;
            text-align: center;
            margin-top: 60px;
            line-height: 2;
        }
        
        /* 7. KHUNG CHỨA CONTROL PANEL */
        .control-panel {
            border: 4px solid #ffffff;
            padding: 20px;
            background-color: #000000;
            box-shadow: 8px 8px 0px rgba(0,0,0,0.5);
            margin-bottom: 30px;
        }

        /* 8. BẢNG KẾT QUẢ (METRICS) */
        div[data-testid="column"] > div {
            border: 2px dashed #ffffff;
            background-color: #333;
            padding: 10px;
            text-align: center;
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
# 3. GIAO DIỆN TRUNG TÂM (PIXEL UI)
# ==============================================================================

# Tiêu đề chính
st.markdown("<h1 style='text-align: center; font-size: 35px; color: #ffcc00;'>👾 PIXEL TRADER 👾</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #fff;'>INSERT COIN TO START PREDICTION</p>", unsafe_allow_html=True)
st.markdown("---")

c1, c2, c3 = st.columns([1, 2, 1]) 

with c2:
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    st.markdown("<div style='font-family: \"Press Start 2P\"; font-size: 12px; margin-bottom: 10px; color: #00ff00;'>PLAYER 1 CONFIG:</div>", unsafe_allow_html=True)
    
    ticker = st.text_input("ENTER ASSET (MÃ CK)", value="META", placeholder="EX: AAPL").upper()
    
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        freq_display = st.selectbox("TIMEFRAME", ("DAILY", "MONTHLY", "QUARTERLY"))
    with row1_col2:
        model_display = st.selectbox("SKILL (MODEL)", ("Naive", "Moving Average", "SES", "Holt", "Holt-Winters"))

    with st.expander("🛠️ CHEAT CODES (SETTINGS)"):
        window_size = 3
        if model_display == "Moving Average":
            window_size = st.slider("WINDOW LVL", 2, 50, 3)
        test_size = st.slider("BACKTEST LVL", 4, 60, 12)

    btn_run = st.button("PRESS START")
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# 4. XỬ LÝ & HIỂN THỊ
# ==============================================================================
freq_map = {"DAILY": "D", "MONTHLY": "M", "QUARTERLY": "Q"}
freq_val = freq_map[freq_display]

if btn_run:
    st.markdown(f"<div style='text-align: center; font-family: \"Press Start 2P\"; font-size: 14px;'>LOADING LEVEL: {ticker}...</div>", unsafe_allow_html=True)
    
    # Thanh loading kiểu text
    progress_text = st.empty()
    for i in range(101):
        # Vẽ thanh loading bằng ký tự █
        bar = "█" * (i // 5) + "░" * ((100 - i) // 5)
        progress_text.text(f"LOADING: [{bar}] {i}%")
    progress_text.empty()
    
    try:
        df = yf.download(ticker, period="5y", progress=False)
        if df.empty:
            st.error("GAME OVER: DATA NOT FOUND.")
            st.stop()
        
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower().strip() for c in df.columns]
        col = 'adj close' if 'adj close' in df.columns else ('close' if 'close' in df.columns else df.columns[0])
        data = df[col].astype(float)
        if data.index.tz is not None: data.index = data.index.tz_localize(None)
        
        if freq_val == "M": data = data.resample('M').last(); seasonal_p = 12
        elif freq_val == "Q": data = data.resample('Q').last(); seasonal_p = 4
        else: data = data.asfreq('B').fillna(method='ffill'); seasonal_p = 5

        data = data.dropna()
        if len(data) < test_size + 10: st.error("NOT ENOUGH XP (DATA)."); st.stop()

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
            
            # High Score Display
            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.markdown(f"<div style='color:#ff5555; text-align:center; font-family:\"Press Start 2P\"; font-size:10px;'>RMSE (ERROR)<br><span style='font-size:20px'>{rmse:.2f}</span></div>", unsafe_allow_html=True)
            m2.markdown(f"<div style='color:#55ff55; text-align:center; font-family:\"Press Start 2P\"; font-size:10px;'>ACCURACY<br><span style='font-size:20px'>{100-mape:.1f}%</span></div>", unsafe_allow_html=True)
            m3.markdown(f"<div style='color:#55ffff; text-align:center; font-family:\"Press Start 2P\"; font-size:10px;'>STATS<br><span style='font-size:12px; font-family:\"VT323\"'>{info}</span></div>", unsafe_allow_html=True)

        # Chart Pixel Style
        fig, ax = plt.subplots(figsize=(14, 6), facecolor='black')
        ax.set_facecolor('#111') # Nền biểu đồ hơi xám nhẹ
        
        # Vẽ nét đứt đậm kiểu pixel
        ax.plot(train.index[-100:], train.iloc[-100:], color='#777', label='TRAIN', linewidth=2, linestyle=':')
        ax.plot(test.index, test, color='#0f0', linewidth=3, label='ACTUAL', drawstyle='steps-mid') # Steps-mid tạo cảm giác bậc thang pixel
        ax.plot(test.index, preds, color='#f0f', linestyle='--', linewidth=3, marker='s', markersize=6, label='PREDICT') # Marker vuông (s)
        
        ax.grid(color='#333', linestyle='-', linewidth=1)
        ax.legend(facecolor='black', edgecolor='#fff', labelcolor='#fff', prop={'family':'monospace'})
        ax.tick_params(colors='#fff', labelsize=12)
        for s in ax.spines.values(): s.set_edgecolor('#fff'); s.set_linewidth(2)
        
        st.pyplot(fig)
        
        with st.expander(">> OPEN INVENTORY (DATA)"):
            st.dataframe(pd.DataFrame({'ACTUAL': test, 'PREDICT': preds, 'DIFF': test-preds}))

    except Exception as e:
        st.error(f"SYSTEM CRASHED: {e}")

else:
    # Màn hình chờ Pixel
    st.markdown("""
        <div class="blinking-text">
            WAITING FOR PLAYER 1...<br><br>
            [ ENTER PARAMETERS ABOVE ]
        </div>
    """, unsafe_allow_html=True)
