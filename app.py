import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.api import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from scipy.optimize import minimize  # <--- VŨ KHÍ BÍ MẬT CỦA COLAB
import warnings
import time
import base64
import os

# ==============================================================================
# 1. CẤU HÌNH & HÀM HỖ TRỢ
# ==============================================================================
warnings.filterwarnings("ignore")
st.set_page_config(page_title="PIXEL TRADER (COLAB CORE)", layout="wide", page_icon="🧪")
plt.style.use('dark_background') 

# --- HÀM TỐI ƯU HÓA THAM SỐ (Trích xuất từ Colab) ---
# Hàm này dùng thuật toán L-BFGS-B để tìm tham số 'ngon' nhất thay vì để máy tự chọn
def optimize_params(data, model_type, seasonal_periods=None):
    def loss_func(params):
        try:
            if model_type == 'SES':
                # params[0] = alpha
                model = SimpleExpSmoothing(data).fit(smoothing_level=params[0], optimized=False)
            elif model_type == 'Holt':
                # params[0]=alpha, params[1]=beta
                model = ExponentialSmoothing(data, trend='add', seasonal=None, damped_trend=True).fit(
                    smoothing_level=params[0], smoothing_trend=params[1], optimized=False)
            elif model_type == 'HW': # Holt-Winters
                # params[0]=alpha, params[1]=beta, params[2]=gamma
                model = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=seasonal_periods).fit(
                    smoothing_level=params[0], smoothing_trend=params[1], smoothing_seasonal=params[2], optimized=False)
            
            # Trả về tổng bình phương sai số (càng nhỏ càng tốt)
            return np.sum((data - model.fittedvalues)**2)
        except:
            return 1e10 # Phạt nặng nếu lỗi

    # Ràng buộc tham số trong khoảng 0.01 đến 0.99 (Giống Colab)
    bounds = [(0.01, 0.99)]
    if model_type == 'Holt': bounds = [(0.01, 0.99), (0.01, 0.99)]
    if model_type == 'HW': bounds = [(0.01, 0.99), (0.01, 0.99), (0.01, 0.99)]

    # Giá trị khởi tạo
    x0 = [0.5] * len(bounds)
    
    # Chạy tối ưu hóa
    res = minimize(loss_func, x0, bounds=bounds, method='L-BFGS-B')
    return res.x

# --- INTRO VIDEO ---
def show_intro_video(video_file, duration=8):
    if 'intro_done' not in st.session_state: st.session_state['intro_done'] = False
    if st.session_state['intro_done']: return
    if not os.path.exists(video_file): st.session_state['intro_done'] = True; return
    try:
        with open(video_file, "rb") as f: v = base64.b64encode(f.read()).decode()
        st.markdown(f"""<style>.stApp {{overflow:hidden}} #intro {{position:fixed;top:0;left:0;width:100%;height:100%;background:#000;z-index:999}}</style><div id="intro"><video style="width:100%;height:100%;object-fit:cover" autoplay muted playsinline><source src="data:video/mp4;base64,{v}" type="video/mp4"></video></div>""", unsafe_allow_html=True)
        time.sleep(duration); st.empty(); st.session_state['intro_done'] = True; st.rerun()
    except: st.session_state['intro_done'] = True

show_intro_video("intro1.mp4", duration=6)

# ==============================================================================
# 2. CSS GIAO DIỆN
# ==============================================================================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');
        header[data-testid="stHeader"] { visibility: hidden; }
        .stApp { background-color: #0d0d0d; color: #00ff41; font-family: 'VT323', monospace; font-size: 20px; }
        input { color: #ffffff !important; font-family: 'VT323', monospace !important; font-size: 22px !important; }
        div[data-baseweb="select"] > div { background-color: #000 !important; color: #ffffff !important; border-color: #00ff41 !important; }
        label p { font-size: 18px !important; font-family: 'Press Start 2P', cursive !important; color: #00ff41 !important; }
        h1 { font-family: 'Press Start 2P'; text-align: center; color: #00ff41; font-size: 60px; }
        div.stButton > button { width: 100%; background-color: #000; color: #00ff41; border: 2px solid #00ff41; font-family: 'Press Start 2P'; padding: 15px; }
        div.stButton > button:hover { background-color: #00ff41; color: #000; box-shadow: 0 0 15px #00ff41; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. LOGIC TÍNH TOÁN (CORE COLAB)
# ==============================================================================

def clean_yfinance_data(df):
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower().strip() for c in df.columns]
    col = next((c for c in ['adj close', 'close', 'price'] if c in df.columns), df.columns[0])
    return df[col]

def get_forecast(data, model_type, test_size, window_size, future_steps, freq_str):
    # Data đã được resample ở bên ngoài
    if len(data) <= test_size: raise ValueError("Not enough data.")
        
    train = data.iloc[:-test_size]
    test = data.iloc[-test_size:]
    
    preds = pd.Series(index=test.index, dtype='float64')
    future_series = pd.Series(dtype='float64')
    info = ""
    warning_msg = None

    # Xác định chu kỳ mùa vụ (Seasonal Period)
    sp = 1
    if freq_str == "DAILY": sp = 5
    elif freq_str == "MONTHLY": sp = 12
    elif freq_str == "QUARTERLY": sp = 4

    try:
        # === NAIVE ===
        if model_type == "Naive":
            preds[:] = train.iloc[-1]
            if future_steps > 0:
                dates = pd.date_range(start=data.index[-1], periods=future_steps+1, freq=data.index.freq)[1:]
                future_series = pd.Series([data.iloc[-1]]*len(dates), index=dates)
            info = "Naive"

        # === MOVING AVERAGE ===
        elif model_type == "Moving Average":
            history = list(train.values)
            predictions = []
            for t in range(len(test)):
                yhat = np.mean(history[-window_size:])
                predictions.append(yhat); history.append(test.iloc[t])
            preds[:] = predictions
            
            if future_steps > 0:
                dates = pd.date_range(start=data.index[-1], periods=future_steps+1, freq=data.index.freq)[1:]
                last_ma = data.rolling(window=window_size).mean().iloc[-1]
                future_series = pd.Series([last_ma]*len(dates), index=dates)
            info = f"MA({window_size})"

        # === SES (Simple Exponential Smoothing) - Có dùng Optimize ===
        elif model_type == "SES":
            # 1. Tìm tham số tốt nhất trên tập train
            best_alpha = optimize_params(train, 'SES')[0]
            
            # 2. Fit lại model để lấy dự báo test
            model = SimpleExpSmoothing(train).fit(smoothing_level=best_alpha, optimized=False)
            preds[:] = model.forecast(len(test)).values
            
            # 3. Future: Tối ưu lại trên TOÀN BỘ DATA (Giống Colab)
            if future_steps > 0:
                best_alpha_full = optimize_params(data, 'SES')[0] # Tối ưu lại trên full data
                model_full = SimpleExpSmoothing(data).fit(smoothing_level=best_alpha_full, optimized=False)
                
                dates = pd.date_range(start=data.index[-1], periods=future_steps+1, freq=data.index.freq)[1:]
                future_series = pd.Series(model_full.forecast(future_steps).values, index=dates)
            info = f"SES (α={best_alpha:.2f})"

        # === HOLT (Double Exp) - Có dùng Optimize ===
        elif model_type == "Holt":
            # 1. Tối ưu Alpha, Beta
            p = optimize_params(train, 'Holt')
            model = ExponentialSmoothing(train, trend='add', seasonal=None, damped_trend=True).fit(
                smoothing_level=p[0], smoothing_trend=p[1], optimized=False)
            preds[:] = model.forecast(len(test)).values
            
            if future_steps > 0:
                p_full = optimize_params(data, 'Holt')
                model_full = ExponentialSmoothing(data, trend='add', seasonal=None, damped_trend=True).fit(
                    smoothing_level=p_full[0], smoothing_trend=p_full[1], optimized=False)
                
                dates = pd.date_range(start=data.index[-1], periods=future_steps+1, freq=data.index.freq)[1:]
                future_series = pd.Series(model_full.forecast(future_steps).values, index=dates)
            info = f"Holt (α={p[0]:.2f}, β={p[1]:.2f})"

        # === HOLT-WINTERS - Có dùng Optimize ===
        elif model_type == "Holt-Winters":
            try:
                # 1. Tối ưu Alpha, Beta, Gamma
                p = optimize_params(train, 'HW', seasonal_periods=sp)
                model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=sp).fit(
                    smoothing_level=p[0], smoothing_trend=p[1], smoothing_seasonal=p[2], optimized=False)
                preds[:] = model.forecast(len(test)).values
                
                if future_steps > 0:
                    p_full = optimize_params(data, 'HW', seasonal_periods=sp)
                    model_full = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=sp).fit(
                        smoothing_level=p_full[0], smoothing_trend=p_full[1], smoothing_seasonal=p_full[2], optimized=False)
                    
                    dates = pd.date_range(start=data.index[-1], periods=future_steps+1, freq=data.index.freq)[1:]
                    future_series = pd.Series(model_full.forecast(future_steps).values, index=dates)
                info = f"HW (sp={sp})"
            except:
                # Fallback về Holt nếu dữ liệu quá ít chu kỳ
                p = optimize_params(train, 'Holt')
                model = ExponentialSmoothing(train, trend='add', seasonal=None, damped_trend=True).fit(
                    smoothing_level=p[0], smoothing_trend=p[1], optimized=False)
                preds[:] = model.forecast(len(test)).values
                info = "Holt (Fallback)"
                warning_msg = "Not enough data for HW -> Switched to Holt"

    except Exception as e:
        info = "ERROR"
        warning_msg = str(e)
        preds[:] = np.nan
        
    return train, test, preds, future_series, info, warning_msg

# ==============================================================================
# 4. GIAO DIỆN CHÍNH
# ==============================================================================
st.markdown("<h1>PIXEL TRADER</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#555; letter-spacing:4px; margin-bottom:30px; font-family:VT323'>COLAB ENGINE EDITION</div>", unsafe_allow_html=True)

with st.container():
    c1, c2, c3 = st.columns([1, 3, 1]) 
    with c2:
        ticker = st.text_input("ENTER TICKER", value="AAPL").upper()
        c_in1, c_in2 = st.columns(2)
        with c_in1: freq_display = st.selectbox("TIMEFRAME", ("DAILY", "MONTHLY", "QUARTERLY"))
        with c_in2: model_display = st.selectbox("MODEL", ("Naive", "Moving Average", "SES", "Holt", "Holt-Winters"))
            
        with st.expander("⚙️ ADVANCED SETTINGS"):
            if model_display == "Moving Average": window_size = st.slider("WINDOW SIZE", 2, 50, 20)
            else: window_size = 20
            test_size = st.slider("BACKTEST STEPS", 5, 60, 12)
            future_steps = st.slider(f"FORECAST STEPS", 4, 60, 6)
        
        st.write("") 
        btn_run = st.button(">> RUN PREDICTION <<")

st.markdown("---")

# ==============================================================================
# 5. XỬ LÝ & HIỂN THỊ
# ==============================================================================
if btn_run:
    try:
        with st.spinner(f"PROCESSING {ticker} (Using Colab Algorithms)..."):
            # 1. Load Data
            df = yf.download(ticker, start="2020-11-23", end="2025-11-21", progress=False)
            data = clean_yfinance_data(df)
            
            if data is None or data.empty: st.error("❌ DATA NOT FOUND."); st.stop()
            
            # 2. Resampling (Cực kỳ quan trọng để khớp Colab)
            if freq_display == "MONTHLY":
                data = data.resample('ME').last().dropna() # Dùng 'ME' cho pandas mới
            elif freq_display == "QUARTERLY":
                data = data.resample('QE').last().dropna() # Dùng 'QE' cho pandas mới
            else:
                data = data.asfreq('B').fillna(method='ffill') # Daily Business Days

            # 3. Chạy dự báo
            train, test, preds, fut, info, warn = get_forecast(data, model_display, test_size, window_size, future_steps, freq_display)

            # Tính lỗi
            mask = ~np.isnan(preds) & ~np.isnan(test)
            rmse = np.sqrt(mean_squared_error(test[mask], preds[mask])) if mask.sum()>0 else 0
            mape = mean_absolute_percentage_error(test[mask], preds[mask])*100 if mask.sum()>0 else 0

            if warn: st.warning(f"⚠️ {warn}")

            # Stats
            cur_price = test.iloc[-1]
            fut_price = fut.iloc[-1] if not fut.empty else preds.iloc[-1]
            trend_pct = ((fut_price - cur_price)/cur_price)*100
            color = "#00ff41" if trend_pct>=0 else "#ff3333"

            s1, s2, s3 = st.columns(3)
            box = "border:1px solid #fff; padding:10px; text-align:center; margin-bottom:20px; background:rgba(255,255,255,0.05)"
            s1.markdown(f"<div style='{box}; border-color:#aaa'><div style='font-size:12px; color:#aaa'>CURRENT</div><div style='font-size:30px'>${cur_price:,.2f}</div></div>", unsafe_allow_html=True)
            s2.markdown(f"<div style='{box}; border-color:#f0f'><div style='font-size:12px; color:#f0f'>TARGET</div><div style='font-size:30px'>${fut_price:,.2f}</div></div>", unsafe_allow_html=True)
            s3.markdown(f"<div style='{box}; border-color:{color}'><div style='font-size:12px; color:{color}'>GROWTH</div><div style='font-size:30px'>{trend_pct:+.2f}%</div></div>", unsafe_allow_html=True)

            # Biểu đồ
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data.values, name='HISTORY', line=dict(color='#555')))
            fig.add_trace(go.Scatter(x=test.index, y=test.values, name='ACTUAL', line=dict(color='#00ff41', width=2)))
            fig.add_trace(go.Scatter(x=preds.index, y=preds.values, name='BACKTEST', line=dict(color='#f0f', width=2, dash='dot')))
            
            if not fut.empty:
                fig.add_trace(go.Scatter(x=fut.index, y=fut.values, name='FUTURE', mode='lines+markers', 
                                         line=dict(color='#ff0', width=3), marker=dict(symbol='star', size=6)))

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Courier New', color='#fff'),
                xaxis=dict(gridcolor='#333'), yaxis=dict(gridcolor='#333'),
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.info(f"RMSE: {rmse:.2f}"); m2.info(f"MAPE: {mape:.2f}%"); m3.success(f"MODEL: {info}")

            if not fut.empty:
                with st.expander("📋 VIEW FUTURE DATA"):
                    st.dataframe(fut.to_frame("Forecast").T)
                    
    except Exception as e:
        st.error(f"SYSTEM ERROR: {e}")
