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
# 1. CẤU HÌNH GIAO DIỆN
# ==============================================================================
warnings.filterwarnings("ignore")
st.set_page_config(page_title="PIXEL TRADER (COLAB CORE)", layout="wide", page_icon="🧪")
plt.style.use('dark_background') 

# --- HÀM INTRO VIDEO (ĐÃ TỐI ƯU RAM) ---
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

# --- CSS PIXEL STYLE ---
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
# 2. COLAB ENGINE: HÀM TỐI ƯU HÓA THAM SỐ (L-BFGS-B)
# ==============================================================================
def optimize_params(data, model_type, seasonal_periods=None):
    """Tìm tham số alpha, beta, gamma tốt nhất để giảm thiểu sai số"""
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
            return np.sum((data - model.fittedvalues)**2) # MSE
        except:
            return 1e10 

    # Giới hạn tham số từ 0.01 đến 0.99 (Tránh biên 0 và 1 gây lỗi)
    bounds = [(0.01, 0.99)]
    if model_type == 'Holt': bounds = [(0.01, 0.99), (0.01, 0.99)]
    if model_type == 'HW': bounds = [(0.01, 0.99), (0.01, 0.99), (0.01, 0.99)]

    x0 = [0.5] * len(bounds)
    res = minimize(loss_func, x0, bounds=bounds, method='L-BFGS-B')
    return res.x

def clean_yfinance_data(df):
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower().strip() for c in df.columns]
    col = next((c for c in ['adj close', 'close', 'price'] if c in df.columns), df.columns[0])
    return df[col]

# ==============================================================================
# 3. LOGIC DỰ BÁO TRUNG TÂM
# ==============================================================================
def get_forecast(data, model_type, test_size, window_size, future_steps, freq_str):
    if len(data) <= test_size: raise ValueError("Not enough data.")
        
    train = data.iloc[:-test_size]
    test = data.iloc[-test_size:]
    
    preds = pd.Series(index=test.index, dtype='float64')
    future_series = pd.Series(dtype='float64')
    info = ""
    warning_msg = None

    # Xác định chu kỳ mùa vụ
    sp = 1
    if freq_str == "DAILY": sp = 5
    elif freq_str == "MONTHLY": sp = 12
    elif freq_str == "QUARTERLY": sp = 4

    # --- CHẾ ĐỘ TỰ ĐỘNG CHỌN MODEL TỐT NHẤT (AUTO) ---
    if model_type == "AUTO (Best Fit)":
        best_rmse = float('inf')
        best_model_name = "Naive"
        
        # Danh sách các ứng cử viên
        candidates = ["Naive", "SES", "Holt"]
        if len(train) > 2 * sp: candidates.append("Holt-Winters") # Chỉ chạy HW nếu đủ dữ liệu

        # Chạy thử vòng lặp để tìm winner
        for m in candidates:
            try:
                _, _, p_try, _, _, _ = get_forecast(data, m, test_size, window_size, 0, freq_str)
                # Tính RMSE
                mask = ~np.isnan(p_try) & ~np.isnan(test)
                if mask.sum() > 0:
                    rmse = np.sqrt(mean_squared_error(test[mask], p_try[mask]))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_model_name = m
            except: pass
        
        # Gán lại model_type thành người chiến thắng để chạy code dưới
        model_type = best_model_name
        info = f"AUTO SELECTED: {best_model_name}"

    try:
        # === NAIVE ===
        if model_type == "Naive":
            preds[:] = train.iloc[-1]
            if future_steps > 0:
                dates = pd.date_range(start=data.index[-1], periods=future_steps+1, freq=data.index.freq)[1:]
                future_series = pd.Series([data.iloc[-1]]*len(dates), index=dates)
            if info == "": info = "Naive"

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
            if info == "": info = f"MA({window_size})"

        # === SES (Simple Exponential Smoothing) ===
        elif model_type == "SES":
            best_alpha = optimize_params(train, 'SES')[0]
            model = SimpleExpSmoothing(train).fit(smoothing_level=best_alpha, optimized=False)
            preds[:] = model.forecast(len(test)).values
            
            if future_steps > 0:
                # Retrain on Full Data with optimized params
                best_alpha_full = optimize_params(data, 'SES')[0]
                model_full = SimpleExpSmoothing(data).fit(smoothing_level=best_alpha_full, optimized=False)
                dates = pd.date_range(start=data.index[-1], periods=future_steps+1, freq=data.index.freq)[1:]
                future_series = pd.Series(model_full.forecast(future_steps).values, index=dates)
            if "AUTO" not in info: info = f"SES (α={best_alpha:.2f})"

        # === HOLT (Standard Linear) ===
        elif model_type == "Holt":
            p = optimize_params(train, 'Holt')
            model = ExponentialSmoothing(train, trend='add', seasonal=None, damped_trend=False).fit(
                smoothing_level=p[0], smoothing_trend=p[1], optimized=False)
            preds[:] = model.forecast(len(test)).values
            
            if future_steps > 0:
                p_full = optimize_params(data, 'Holt')
                model_full = ExponentialSmoothing(data, trend='add', seasonal=None, damped_trend=False).fit(
                    smoothing_level=p_full[0], smoothing_trend=p_full[1], optimized=False)
                dates = pd.date_range(start=data.index[-1], periods=future_steps+1, freq=data.index.freq)[1:]
                future_series = pd.Series(model_full.forecast(future_steps).values, index=dates)
            if "AUTO" not in info: info = f"Holt (α={p[0]:.2f}, β={p[1]:.2f})"

        # === HOLT-WINTERS (Add seasonal) ===
        elif model_type == "Holt-Winters":
            try:
                p = optimize_params(train, 'HW', seasonal_periods=sp)
                model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=sp).fit(
                    smoothing_level=p[0], smoothing_trend=p[1], smoothing_seasonal=p[2], optimized=False)
                
                if future_steps > 0:
                    p_full = optimize_params(data, 'HW', seasonal_periods=sp)
                    model_full = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=sp).fit(
                        smoothing_level=p_full[0], smoothing_trend=p_full[1], smoothing_seasonal=p_full[2], optimized=False)
                if "AUTO" not in info: info = f"HW (sp={sp})"
            except:
                # Fallback to Holt if HW fails
                p = optimize_params(train, 'Holt')
                model = ExponentialSmoothing(train, trend='add', seasonal=None, damped_trend=False).fit(
                    smoothing_level=p[0], smoothing_trend=p[1], optimized=False)
                if future_steps > 0:
                    model_full = ExponentialSmoothing(data, trend='add', seasonal=None, damped_trend=False).fit(optimized=True)
                warning_msg = "HW Failed -> Holt"
                if "AUTO" not in info: info = "Holt (Fallback)"

            preds[:] = model.forecast(len(test)).values
            if future_steps > 0:
                dates = pd.date_range(start=data.index[-1], periods=future_steps+1, freq=data.index.freq)[1:]
                future_series = pd.Series(model_full.forecast(future_steps).values, index=dates)

    except Exception as e:
        info = "ERROR"
        warning_msg = str(e)
        preds[:] = np.nan
        
    return train, test, preds, future_series, info, warning_msg

# ==============================================================================
# 4. GIAO DIỆN CHÍNH
# ==============================================================================
st.markdown("<h1>PIXEL TRADER</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#555; letter-spacing:4px; margin-bottom:30px; font-family:VT323'>COLAB OPTIMIZED ENGINE</div>", unsafe_allow_html=True)

with st.container():
    c1, c2, c3 = st.columns([1, 3, 1]) 
    with c2:
        ticker = st.text_input("ENTER TICKER", value="AAPL").upper()
        c_in1, c_in2 = st.columns(2)
        with c_in1: 
            freq_display = st.selectbox("TIMEFRAME", ("DAILY", "MONTHLY", "QUARTERLY"))
        with c_in2: 
            # [NÂNG CẤP] Thêm chế độ AUTO (Best Fit)
            model_display = st.selectbox("MODEL", ("AUTO (Best Fit)", "Naive", "Moving Average", "SES", "Holt", "Holt-Winters"))
            
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
        with st.spinner(f"CALCULATING OPTIMAL PARAMS FOR {ticker}..."):
            # 1. Load Data
            df = yf.download(ticker, start="2020-11-23", end="2025-11-21", progress=False)
            data = clean_yfinance_data(df)
            
            if data is None or data.empty: st.error("❌ DATA NOT FOUND."); st.stop()
            
            # 2. Resampling & Fillna (Chuẩn Colab)
            if freq_display == "MONTHLY":
                data = data.resample('M').last().ffill().dropna()
            elif freq_display == "QUARTERLY":
                data = data.resample('Q').last().ffill().dropna()
            else:
                data = data.asfreq('B').ffill().dropna() 

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
