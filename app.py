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
# 1. CẤU HÌNH & CSS (PIXEL TRADER - VS MODE UPDATE)
# ==============================================================================
warnings.filterwarnings("ignore")
st.set_page_config(page_title="PIXEL TRADER PRO", layout="wide", page_icon="📈")
plt.style.use('dark_background')

# Khởi tạo Session State cho chế độ so sánh
if 'vs_mode' not in st.session_state:
    st.session_state.vs_mode = False

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');

        /* 1. ẨN THANH MENU */
        header[data-testid="stHeader"] { visibility: hidden; }
        .block-container { padding-top: 2rem; }

        /* 2. NỀN CHÍNH */
        .stApp {
            background-color: #0d0d0d;
            color: #00ff41;
            font-family: 'VT323', monospace;
            font-size: 20px;
        }

        /* 3. WIDGETS */
        div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
            background-color: #000 !important;
            color: #00ff41 !important;
            border: 2px solid #00ff41 !important; 
            border-radius: 0px;
            font-family: 'VT323', monospace !important;
            font-size: 22px;
        }
        
        label p {
            font-size: 20px !important;
            font-family: 'Press Start 2P', cursive !important;
            color: #00ff41 !important;
        }

        /* 4. NÚT BẤM CHÍNH */
        div.stButton > button {
            width: 100%;
            background-color: #000;
            color: #00ff41;
            border: 4px double #00ff41;
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

        /* 5. NÚT SO SÁNH (VS MODE) - MÀU CAM */
        .vs-btn > button {
            border-color: #ffcc00 !important;
            color: #ffcc00 !important;
        }
        .vs-btn > button:hover {
            background-color: #ffcc00 !important;
            color: #000 !important;
            box-shadow: 0 0 20px #ffcc00 !important;
        }

        /* 6. TEXT ANIMATION */
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
    except:
        preds[:] = np.nan
        
    return test, preds, info

# ==============================================================================
# 3. GIAO DIỆN CHÍNH
# ==============================================================================

st.markdown("<h1>⚡ PIXEL TRADER ⚡</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>PRO EDITION [v2.1 VS MODE]</div>", unsafe_allow_html=True)

# CONTROL PANEL
with st.container():
    c1, c2, c3 = st.columns([1, 3, 1]) 
    with c2:
        # Input chính
        ticker = st.text_input("PLAYER 1 (MAIN TARGET)", value="META", placeholder="EX: AAPL").upper()
        
        col_inp1, col_inp2 = st.columns(2)
        with col_inp1:
            freq_display = st.selectbox("TIMEFRAME", ("DAILY", "MONTHLY", "QUARTERLY"))
        with col_inp2:
            model_display = st.selectbox("WEAPON (MODEL)", ("Naive", "Moving Average", "SES", "Holt", "Holt-Winters"))
        
        with st.expander("⚙️ ADVANCED SETTINGS"):
            window_size = 3
            if model_display == "Moving Average":
                window_size = st.slider("WINDOW SIZE", 2, 50, 3)
            test_size = st.slider("BACKTEST SIZE", 4, 60, 12)
        
        st.write("") 
        btn_run = st.button(">> START PREDICTION <<")

st.markdown("---")

# ==============================================================================
# 4. XỬ LÝ & KẾT QUẢ
# ==============================================================================
freq_map = {"DAILY": "D", "MONTHLY": "M", "QUARTERLY": "Q"}
freq_val = freq_map[freq_display]

if btn_run:
    st.session_state.vs_mode = False # Reset VS mode khi chạy mới

if btn_run or st.session_state.get('run_success', False):
    st.session_state.run_success = True
    
    # --- PHẦN 1: CHẠY MÃ CHÍNH ---
    try:
        with st.spinner(f"LOADING PLAYER 1: {ticker}..."):
            df = yf.download(ticker, period="5y", progress=False)
            if df.empty: st.error("❌ DATA NOT FOUND."); st.stop()
            
            # Xử lý dữ liệu
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df.columns = [str(c).lower().strip() for c in df.columns]
            col = next((c for c in ['adj close', 'close', 'price'] if c in df.columns), df.columns[0])
            data = df[col].astype(float)
            if data.index.tz is not None: data.index = data.index.tz_localize(None)
            
            # Resample
            if freq_val == "M": data = data.resample('M').last(); seasonal_p = 12
            elif freq_val == "Q": data = data.resample('Q').last(); seasonal_p = 4
            else: data = data.asfreq('B').fillna(method='ffill'); seasonal_p = 5
            data = data.dropna()
            
            if len(data) < test_size + 10: st.error("⚠️ DATA TOO SHORT."); st.stop()

            # Lấy kết quả forecast (Dùng hàm tách riêng cho gọn)
            test, preds, info = get_forecast(data, model_display, test_size, window_size, seasonal_p)
            train = data.iloc[:-test_size]

            # Tính Metrics
            mask = ~np.isnan(preds) & ~np.isnan(test)
            rmse = np.sqrt(mean_squared_error(test[mask], preds[mask])) if mask.sum() > 0 else 0
            mape = mean_absolute_percentage_error(test[mask], preds[mask]) * 100 if mask.sum() > 0 else 0

            # --- HIỂN THỊ KẾT QUẢ CHÍNH ---
            st.markdown(f"<div style='text-align:center; font-family:\"Press Start 2P\"; color:#00ff41; margin-bottom:10px'>TARGET: {ticker}</div>", unsafe_allow_html=True)
            
            c_m1, c_m2, c_m3 = st.columns(3)
            box_style = "border:2px solid #00ff41; padding:5px; text-align:center; height:100%; display:flex; flex-direction:column; justify-content:center;"
            
            c_m1.markdown(f"<div style='{box_style}'><div style='font-size:12px; color:#00ff41'>RMSE</div><div style='font-size:28px; color:#fff'>{rmse:.2f}</div></div>", unsafe_allow_html=True)
            c_m2.markdown(f"<div style='{box_style}'><div style='font-size:12px; color:#00ff41'>MAPE</div><div style='font-size:28px; color:#fff'>{mape:.2f}%</div></div>", unsafe_allow_html=True)
            c_m3.markdown(f"<div style='border:2px solid #00ffff; padding:5px; text-align:center;'><div style='font-size:12px; color:#00ffff'>PARAMS</div><div style='font-size:24px; color:#fff'>{info}</div></div>", unsafe_allow_html=True)

            # Biểu đồ chính
            st.write("")
            fig, ax = plt.subplots(figsize=(14, 6), facecolor='black')
            ax.set_facecolor('#050505')
            ax.plot(train.index[-60:], train.iloc[-60:], color='#555', label='TRAIN')
            ax.plot(test.index, test, color='#00ff41', linewidth=2.5, label='ACTUAL')
            ax.plot(test.index, preds, color='#ff00ff', linestyle='--', linewidth=2, marker='o', markersize=5, label='PREDICT')
            ax.legend(facecolor='black', edgecolor='#333', labelcolor='#fff')
            ax.grid(color='#222', linestyle=':')
            for s in ax.spines.values(): s.set_edgecolor('#333')
            st.pyplot(fig)

            # --- NÚT BẬT CHẾ ĐỘ SO SÁNH (VS MODE) ---
            st.write("---")
            c_vs1, c_vs2, c_vs3 = st.columns([1, 2, 1])
            with c_vs2:
                # Dùng class css 'vs-btn' để đổi màu nút thành màu cam
                st.markdown('<div class="vs-btn">', unsafe_allow_html=True)
                if st.button("⚔️ ACTIVATE VS MODE (COMPARE)"):
                    st.session_state.vs_mode = True
                st.markdown('</div>', unsafe_allow_html=True)

            # --- LOGIC SO SÁNH ---
            if st.session_state.vs_mode:
                st.markdown("<h2 style='text-align:center; color:#ffcc00; margin-top:30px'>⚔️ BATTLE ARENA ⚔️</h2>", unsafe_allow_html=True)
                
                with st.container():
                    c_inp1, c_inp2 = st.columns([3, 1])
                    with c_inp1:
                        # Nhập danh sách đối thủ
                        rivals_input = st.text_input("ENTER RIVALS (COMMA SEPARATED)", value="AAPL, GOOGL, AMZN", placeholder="EX: MSFT, TSLA")
                    with c_inp2:
                        st.write("")
                        st.write("")
                        btn_fight = st.button(">> FIGHT! <<")

                if btn_fight:
                    rivals = [r.strip().upper() for r in rivals_input.split(",") if r.strip()]
                    # Giới hạn tối đa 5 mã (tính cả mã chính)
                    all_tickers = [ticker] + rivals[:4] 
                    
                    st.write(f"ANALYZING: {', '.join(all_tickers)}...")
                    
                    # Dictionary lưu kết quả
                    results_map = {}
                    
                    # Tiến hành chạy vòng lặp
                    progress_bar = st.progress(0)
                    
                    for i, t in enumerate(all_tickers):
                        try:
                            d_t = yf.download(t, period="2y", progress=False) # Lấy 2 năm để so sánh cho nhanh
                            if d_t.empty: continue
                            
                            if isinstance(d_t.columns, pd.MultiIndex): d_t.columns = d_t.columns.get_level_values(0)
                            d_t.columns = [str(c).lower().strip() for c in d_t.columns]
                            col_t = next((c for c in ['adj close', 'close'] if c in d_t.columns), d_t.columns[0])
                            
                            val = d_t[col_t].astype(float)
                            if val.index.tz is not None: val.index = val.index.tz_localize(None)
                            
                            # Resample giống mã chính
                            if freq_val == "M": val = val.resample('M').last()
                            elif freq_val == "Q": val = val.resample('Q').last()
                            else: val = val.asfreq('B').fillna(method='ffill')
                            val = val.dropna()
                            
                            # Lấy Forecast
                            _, pred_t, _ = get_forecast(val, model_display, test_size, window_size, seasonal_p)
                            
                            # Lưu kết quả dự báo
                            results_map[t] = pred_t
                            
                        except: pass
                        progress_bar.progress((i + 1) / len(all_tickers))
                    
                    progress_bar.empty()

                    # --- VẼ BIỂU ĐỒ SO SÁNH ---
                    if results_map:
                        st.markdown("<h3 style='text-align:center; font-family:VT323'>PERFORMANCE COMPARISON (PREDICTED GROWTH %)</h3>", unsafe_allow_html=True)
                        
                        fig2, ax2 = plt.subplots(figsize=(14, 7), facecolor='black')
                        ax2.set_facecolor('#050505')
                        
                        # Màu sắc cho các đường (Màu game)
                        colors = ['#00ff41', '#ff00ff', '#00ffff', '#ffcc00', '#ff3333']
                        
                        for idx, (t_name, pred_series) in enumerate(results_map.items()):
                            # Tính % tăng trưởng so với điểm đầu tiên của dự báo (Normalized)
                            # Để so sánh công bằng giữa mã giá 100$ và mã giá 10$
                            if len(pred_series) > 0:
                                start_val = pred_series.iloc[0]
                                pct_change = ((pred_series - start_val) / start_val) * 100
                                
                                # Mã chính vẽ nét đậm, đối thủ nét mảnh hơn
                                lw = 3 if t_name == ticker else 2
                                ls = '-' if t_name == ticker else '--'
                                color = colors[idx % len(colors)]
                                
                                ax2.plot(pred_series.index, pct_change, label=f"{t_name}", color=color, linewidth=lw, linestyle=ls)

                        ax2.set_ylabel("GROWTH % (FROM TEST START)")
                        ax2.legend(facecolor='black', edgecolor='#333', labelcolor='#fff')
                        ax2.grid(color='#333', linestyle=':')
                        ax2.axhline(0, color='#555', linewidth=1) # Đường tham chiếu 0%
                        for s in ax2.spines.values(): s.set_edgecolor('#333')
                        
                        st.pyplot(fig2)
                        
                        # Bảng so sánh cuối cùng
                        st.info(f"🏆 WINNER CHECK: Based on {model_display} prediction trend.")

    except Exception as e:
        st.error(f"SYSTEM ERROR: {e}")

else:
    st.markdown("""
        <div class="blinking-text">
            SYSTEM READY...<br>
            [ WAITING FOR USER INPUT ]
        </div>
    """, unsafe_allow_html=True)
