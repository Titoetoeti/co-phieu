import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.api import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
import time
import base64
import os

# ==============================================================================
# 1. CẤU HÌNH & HÀM HỖ TRỢ
# ==============================================================================
warnings.filterwarnings("ignore")
st.set_page_config(page_title="PIXEL TRADER (STATISTICS)", layout="wide", page_icon="📈")
plt.style.use('dark_background') 

# --- HÀM INTRO VIDEO ---
def show_intro_video(video_file, duration=8):
    if 'intro_done' not in st.session_state:
        st.session_state['intro_done'] = False

    if st.session_state['intro_done']:
        return

    if not os.path.exists(video_file):
        st.session_state['intro_done'] = True
        return

    try:
        with open(video_file, "rb") as f:
            video_bytes = f.read()
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
            <div id="skip-btn">LOADING STATISTICAL MODELS...</div>
        </div>
        """
        placeholder = st.empty()
        placeholder.markdown(intro_html, unsafe_allow_html=True)
        time.sleep(duration)
        placeholder.empty()
        st.session_state['intro_done'] = True
        st.rerun()

    except Exception:
        st.session_state['intro_done'] = True

show_intro_video("intro1.mp4", duration=6)

# ==============================================================================
# 2. CSS GIAO DIỆN (PIXEL STYLE)
# ==============================================================================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');
        header[data-testid="stHeader"] { visibility: hidden; }
        .block-container { padding-top: 2rem; }
        .stApp { background-color: #0d0d0d; color: #00ff41; font-family: 'VT323', monospace; font-size: 20px; }
        input { color: #ffffff !important; font-family: 'VT323', monospace !important; font-size: 22px !important; }
        div[data-baseweb="select"] > div { background-color: #000 !important; color: #ffffff !important; border-color: #00ff41 !important; }
        div[data-baseweb="input"] > div { background-color: #000 !important; border: 2px solid #00ff41 !important; border-radius: 0px; }
        div[data-baseweb="select"] svg { fill: #00ff41 !important; }
        label p { font-size: 18px !important; font-family: 'Press Start 2P', cursive !important; color: #00ff41 !important; }
        h1 { font-family: 'Press Start 2P', cursive !important; text-align: center; color: #00ff41; text-shadow: 6px 6px 0px #003300; font-size: 60px !important; line-height: 1.2 !important; margin-bottom: 10px !important; margin-top: 0px !important; }
        .sub-title { text-align: center; font-family: 'VT323'; font-size: 24px; color: #555; letter-spacing: 4px; margin-bottom: 30px; }
        div.stButton > button { width: 100%; background-color: #000000 !important; color: #00ff41 !important; border: 2px solid #00ff41 !important; font-family: 'Press Start 2P', cursive !important; padding: 15px; margin-top: 15px; border-radius: 0px !important; transition: all 0.2s ease-in-out; box-shadow: none !important; }
        div.stButton > button:hover { background-color: #00ff41 !important; color: #000000 !important; box-shadow: 0 0 15px #00ff41 !important; }
        div.stButton > button:active { transform: scale(0.98); }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. LOGIC TÍNH TOÁN (CODE 2: STATISTICAL MODELS)
# ==============================================================================

def clean_yfinance_data(df):
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower().strip() for c in df.columns]
    col = next((c for c in ['adj close', 'close', 'price'] if c in df.columns), df.columns[0])
    return df[col]

def get_forecast(data, model_type, test_size, window_size, future_days=0):
    # Chia Train/Test
    train = data.iloc[:-test_size]
    test = data.iloc[-test_size:]
    
    preds = pd.Series(index=test.index, dtype='float64')
    future_series = pd.Series(dtype='float64')
    info = ""
    warning_msg = None

    try:
        # === MÔ HÌNH 1: NAIVE (Ngây thơ) ===
        if model_type == "Naive":
            # Dự báo bằng giá trị cuối cùng của tập train
            last_val = train.iloc[-1]
            preds[:] = last_val
            
            if future_days > 0:
                future_dates = pd.bdate_range(start=data.index[-1], periods=future_days + 1)[1:]
                future_series = pd.Series([data.iloc[-1]] * len(future_dates), index=future_dates)
            info = "Naive Method"

        # === MÔ HÌNH 2: MOVING AVERAGE (Trung bình trượt) ===
        elif model_type == "Moving Average":
            # Dự báo bằng trung bình của N ngày gần nhất
            history = list(train.values)
            predictions = []
            for t in range(len(test)):
                yhat = np.mean(history[-window_size:])
                predictions.append(yhat)
                history.append(test.iloc[t]) # Walk-forward
            preds[:] = predictions
            
            if future_days > 0:
                future_dates = pd.bdate_range(start=data.index[-1], periods=future_days + 1)[1:]
                last_ma = data.rolling(window=window_size).mean().iloc[-1]
                future_series = pd.Series([last_ma] * len(future_dates), index=future_dates)
            info = f"MA({window_size})"

        # === MÔ HÌNH 3: SES (Simple Exponential Smoothing) ===
        elif model_type == "SES":
            model = SimpleExpSmoothing(train).fit(optimized=True)
            preds[:] = model.forecast(len(test)).values
            
            if future_days > 0:
                model_full = SimpleExpSmoothing(data).fit(optimized=True)
                future_vals = model_full.forecast(future_days).values
                future_dates = pd.bdate_range(start=data.index[-1], periods=future_days + 1)[1:]
                future_series = pd.Series(future_vals, index=future_dates)
            info = f"SES (alpha={model.params['smoothing_level']:.2f})"

        # === MÔ HÌNH 4: HOLT (Double Exp Smoothing - Trend) ===
        elif model_type == "Holt":
            model = ExponentialSmoothing(train, trend='add', seasonal=None).fit(optimized=True)
            preds[:] = model.forecast(len(test)).values
            
            if future_days > 0:
                model_full = ExponentialSmoothing(data, trend='add', seasonal=None).fit(optimized=True)
                future_vals = model_full.forecast(future_days).values
                future_dates = pd.bdate_range(start=data.index[-1], periods=future_days + 1)[1:]
                future_series = pd.Series(future_vals, index=future_dates)
            info = "Holt's Linear"

        # === MÔ HÌNH 5: HOLT-WINTERS (Triple Exp Smoothing - Seasonal) ===
        elif model_type == "Holt-Winters":
            # Tự động chọn chu kỳ (seasonal_periods)
            sp = 5 # Mặc định tuần làm việc 5 ngày
            try:
                model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=sp).fit(optimized=True)
                preds[:] = model.forecast(len(test)).values
                
                if future_days > 0:
                    model_full = ExponentialSmoothing(data, trend='add', seasonal='add', seasonal_periods=sp).fit(optimized=True)
                    future_vals = model_full.forecast(future_days).values
                    future_dates = pd.bdate_range(start=data.index[-1], periods=future_days + 1)[1:]
                    future_series = pd.Series(future_vals, index=future_dates)
                info = f"Holt-Winters (sp={sp})"
            except:
                # Fallback về Holt nếu lỗi
                model = ExponentialSmoothing(train, trend='add', seasonal=None).fit(optimized=True)
                preds[:] = model.forecast(len(test)).values
                info = "Holt (Fallback)"
                warning_msg = "HW Failed -> Holt"

    except Exception as e:
        info = "ERROR"
        warning_msg = str(e)
        preds[:] = np.nan
        
    return train, test, preds, future_series, info, warning_msg

# ==============================================================================
# 4. GIAO DIỆN CHÍNH
# ==============================================================================

if 'vs_mode' not in st.session_state: st.session_state.vs_mode = False

st.markdown("<h1>PIXEL TRADER</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>STATISTICAL EDITION</div>", unsafe_allow_html=True)

with st.container():
    c1, c2, c3 = st.columns([1, 3, 1]) 
    with c2:
        ticker = st.text_input("ENTER TICKER (e.g., AAPL)", value="AAPL").upper()
        col_inp1, col_inp2 = st.columns(2)
        with col_inp1: 
            freq_display = st.selectbox("TIMEFRAME", ("DAILY",))
        with col_inp2: 
            # Danh sách mô hình của Code 2
            model_display = st.selectbox("MODEL", ("Naive", "Moving Average", "SES", "Holt", "Holt-Winters"))
            
        with st.expander("⚙️ ADVANCED SETTINGS"):
            window_size = st.slider("WINDOW SIZE (MA)", 2, 50, 20)
            test_size = st.slider("BACKTEST SIZE", 5, 60, 20)
            future_days = st.slider("FUTURE FORECAST (DAYS)", 7, 90, 30)
        
        st.write("") 
        btn_run = st.button(">> START PREDICTION <<")

st.markdown("---")

# ==============================================================================
# 5. XỬ LÝ & HIỂN THỊ
# ==============================================================================

if btn_run: st.session_state.vs_mode = False

if btn_run or st.session_state.get('run_success', False):
    st.session_state.run_success = True
    
    try:
        with st.spinner(f"LOADING DATA: {ticker}..."):
            # [FIX] Cố định thời gian như yêu cầu
            df = yf.download(ticker, start="2020-11-23", end="2025-11-21", progress=False)
            data = clean_yfinance_data(df)
            
            if data is None or data.empty: st.error("❌ DATA NOT FOUND."); st.stop()
            data = data.dropna()

            # GỌI HÀM DỰ BÁO
            train, test, preds, future_series, info, warning_msg = get_forecast(data, model_display, test_size, window_size, future_days)

            # Tính toán lỗi
            mask = ~np.isnan(preds) & ~np.isnan(test)
            rmse = np.sqrt(mean_squared_error(test[mask], preds[mask])) if mask.sum() > 0 else 0
            mape = mean_absolute_percentage_error(test[mask], preds[mask]) * 100 if mask.sum() > 0 else 0

            if warning_msg: st.warning(f"⚠️ SYSTEM WARNING: {warning_msg}")

            # --- MARKET STATS ---
            st.markdown(f"<div style='text-align:center; font-family:\"Press Start 2P\"; color:#00ff41; margin-bottom:10px'>TARGET: {ticker}</div>", unsafe_allow_html=True)
            
            current_price = test.iloc[-1]
            if not future_series.empty:
                final_predicted_price = future_series.iloc[-1]
            else:
                final_predicted_price = preds.iloc[-1]

            if not np.isnan(final_predicted_price):
                trend_pct = ((final_predicted_price - current_price) / current_price) * 100
            else: trend_pct = 0.0
            
            trend_color = "#00ff41" if trend_pct >= 0 else "#ff3333"
            trend_arrow = "▲" if trend_pct >= 0 else "▼"

            stat1, stat2, stat3 = st.columns(3)
            stat_box_style = "border:2px solid #fff; padding:10px; text-align:center; background: rgba(255,255,255,0.05); margin-bottom: 20px;"
            stat_label = "font-family: 'Press Start 2P'; font-size: 12px; color: #aaa; margin-bottom: 8px;"
            stat_val = "font-family: 'VT323'; font-size: 36px; line-height: 1; color: #fff;"

            stat1.markdown(f"<div style='{stat_box_style} border-color: #aaa;'><div style='{stat_label}'>CURRENT PRICE</div><div style='{stat_val}'>${current_price:,.2f}</div></div>", unsafe_allow_html=True)
            stat2.markdown(f"<div style='{stat_box_style} border-color: #ff00ff;'><div style='{stat_label} color:#ff00ff;'>TARGET PRICE</div><div style='{stat_val} color:#ff00ff;'>${final_predicted_price:,.2f}</div></div>", unsafe_allow_html=True)
            stat3.markdown(f"<div style='{stat_box_style} border-color: {trend_color};'><div style='{stat_label} color:{trend_color};'>FORECAST</div><div style='{stat_val} color:{trend_color};'>{trend_arrow} {abs(trend_pct):.2f}%</div></div>", unsafe_allow_html=True)

            # --- METRICS ---
            c_m1, c_m2, c_m3 = st.columns(3)
            box_style = "border:2px solid #00ff41; padding:10px; text-align:center; height:100%; display:flex; flex-direction:column; justify-content:center;"
            
            c_m1.markdown(f"<div style='{box_style}'><div style='font-family: \"Press Start 2P\"; font-size: 14px; margin-bottom: 5px; color: #00ff41;'>RMSE</div><div style='font-family: \"VT323\"; font-size: 40px; color: #ffffff;'>{rmse:.2f}</div></div>", unsafe_allow_html=True)
            c_m2.markdown(f"<div style='{box_style}'><div style='font-family: \"Press Start 2P\"; font-size: 14px; margin-bottom: 5px; color: #00ff41;'>MAPE</div><div style='font-family: \"VT323\"; font-size: 40px; color: #ffffff;'>{mape:.2f}%</div></div>", unsafe_allow_html=True)
            c_m3.markdown(f"<div style='border:2px solid #00ffff; padding:10px; text-align:center; height:100%; display:flex; flex-direction:column; justify-content:center;'><div style='font-family: \"Press Start 2P\"; font-size: 14px; margin-bottom: 5px; color: #00ffff;'>MODEL</div><div style='font-family: \"VT323\"; font-size: 35px; color: #ffffff;'>{info}</div></div>", unsafe_allow_html=True)

            st.write("")
            
            # ==================================================================
            # BIỂU ĐỒ (Đã bao gồm Legend Fix + Future Forecast)
            # ==================================================================
            fig = go.Figure()

            # 1. History
            fig.add_trace(go.Scatter(
                x=data.index, y=data.values,
                mode='lines', name='HISTORY',
                line=dict(color='#555555', width=1.5)
            ))

            # 2. Actual Test
            fig.add_trace(go.Scatter(
                x=test.index, y=test.values,
                mode='lines', name='ACTUAL',
                line=dict(color='#00ff41', width=2)
            ))

            # 3. Forecast Backtest
            fig.add_trace(go.Scatter(
                x=preds.index, y=preds.values,
                mode='lines', name='BACKTEST',
                line=dict(color='#ff00ff', width=2, dash='dot')
            ))

            # 4. Future
            if not future_series.empty:
                fig.add_trace(go.Scatter(
                    x=future_series.index, y=future_series.values,
                    mode='lines+markers', name=f'FUTURE ({future_days}D)',
                    line=dict(color='#ffff00', width=3),
                    marker=dict(size=4, symbol='star')
                ))

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Courier New, monospace', color='#ffffff'),
                xaxis=dict(showgrid=True, gridcolor='#333333', tickfont=dict(color='#00ff41')),
                yaxis=dict(showgrid=True, gridcolor='#333333', tickfont=dict(color='#ffffff')),
                hovermode="x unified",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(color="#ffffff", size=12), bgcolor="rgba(0,0,0,0.5)"
                ),
                margin=dict(l=0, r=0, t=30, b=0)
            )

            st.plotly_chart(fig, use_container_width=True)
            
            if not future_series.empty:
                with st.expander("📋 VIEW FUTURE PRICES"):
                    st.dataframe(future_series.to_frame(name="Predicted Price").T)

            # --- VS MODE ---
            st.markdown("---")
            st.markdown("<h3 style='text-align:center; color:#ffcc00; font-family:\"Press Start 2P\"'>VS MODE</h3>", unsafe_allow_html=True)
            
            c_vs1, c_vs2, c_vs3 = st.columns([1, 2, 1])
            with c_vs2:
                rivals_input = st.text_input("ENTER RIVALS", value="AAPL, MSFT, GOOG")
                st.write("")
                btn_fight = st.button(">> START COMPARISON <<")

            if btn_fight:
                rivals = [r.strip().upper() for r in rivals_input.split(",") if r.strip()]
                all_tickers = [ticker] + rivals[:3] 
                results_map = {}
                progress_bar = st.progress(0)
                
                for i, t in enumerate(all_tickers):
                    try:
                        # [FIX] Dùng cùng khung thời gian 2020-2025
                        d_t = yf.download(t, start="2020-11-23", end="2025-11-21", progress=False)
                        val = clean_yfinance_data(d_t)
                        if val is not None and not val.empty:
                            val = val.dropna()
                            _, _, pred_t, _, _, _ = get_forecast(val, model_display, test_size, window_size, future_days=0)
                            if not pred_t.isna().all(): results_map[t] = pred_t
                    except Exception: pass
                    progress_bar.progress((i + 1) / len(all_tickers))
                progress_bar.empty()

                if len(results_map) > 0:
                    fig2 = go.Figure()
                    colors = ['#00ff41', '#ff00ff', '#00ffff', '#ffcc00', '#ff3333']
                    
                    for idx, (t_name, pred_series) in enumerate(results_map.items()):
                        if len(pred_series) > 0:
                            start_val = pred_series.iloc[0]
                            if not np.isnan(start_val) and start_val != 0:
                                pct_change = ((pred_series - start_val) / start_val) * 100
                                width_line = 4 if t_name == ticker else 2
                                dash_style = 'solid' if t_name == ticker else 'dot'
                                line_color = colors[idx % len(colors)]
                                
                                fig2.add_trace(go.Scatter(
                                    x=pred_series.index,
                                    y=pct_change,
                                    mode='lines',
                                    name=f"{t_name}",
                                    line=dict(color=line_color, width=width_line, dash=dash_style)
                                ))

                    fig2.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Courier New, monospace', color='#ffffff'),
                        xaxis=dict(showgrid=True, gridcolor='#333333', tickfont=dict(color='#ffffff')),
                        yaxis=dict(showgrid=True, gridcolor='#333333', title="Growth %", tickfont=dict(color='#ffffff')),
                        hovermode="x unified",
                        legend=dict(
                            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                            font=dict(color="#ffffff", size=12), bgcolor="rgba(0,0,0,0.5)"
                        )
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else: st.warning("No valid data found for comparison.")

    except Exception as e:
        st.error(f"SYSTEM ERROR: {e}")

else:
    st.markdown("""
        <div style='text-align: center; margin-top: 50px; font-family: "Press Start 2P"; color: #00ff41; animation: blinker 1s step-end infinite;'>
            STATISTICAL MODELS READY...<br>[ WAITING FOR INPUT ]
        </div>
        <style>@keyframes blinker { 50% { opacity: 0; } }</style>
    """, unsafe_allow_html=True)
