import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go # Thư viện biểu đồ tương tác
import yfinance as yf
from statsmodels.tsa.api import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from scipy.optimize import minimize
import warnings
import time
import base64
import os

# ==============================================================================
# 1. CẤU HÌNH & HÀM HỖ TRỢ (GIỮ NGUYÊN)
# ==============================================================================
warnings.filterwarnings("ignore")
st.set_page_config(page_title="PIXEL TRADER PRO", layout="wide", page_icon="📈")
# Vẫn giữ cấu hình này cho Matplotlib dù ta dùng Plotly (để an toàn)
plt.style.use('dark_background') 

# --- HÀM INTRO VIDEO ---
def show_intro_video(video_file, duration=8):
    if 'intro_done' not in st.session_state:
        st.session_state['intro_done'] = False

    if st.session_state['intro_done']:
        return

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

# Gọi Intro
show_intro_video("intro1.mp4", duration=7)


# ==============================================================================
# 2. CSS GIAO DIỆN (PIXEL STYLE FINAL - GIỮ NGUYÊN)
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
        
        input { color: #ffffff !important; font-family: 'VT323', monospace !important; font-size: 22px !important; }
        div[data-baseweb="select"] > div { background-color: #000 !important; color: #ffffff !important; border-color: #00ff41 !important; }
        div[data-baseweb="input"] > div { background-color: #000 !important; border: 2px solid #00ff41 !important; border-radius: 0px; }
        div[data-baseweb="select"] svg { fill: #00ff41 !important; }

        label p { font-size: 18px !important; font-family: 'Press Start 2P', cursive !important; color: #00ff41 !important; }
        
        h1 {
            font-family: 'Press Start 2P', cursive !important;
            text-align: center; color: #00ff41;
            text-shadow: 6px 6px 0px #003300;
            font-size: 70px !important; line-height: 1.2 !important;
            margin-bottom: 10px !important; margin-top: 0px !important;
        }
        .sub-title { text-align: center; font-family: 'VT323'; font-size: 24px; color: #555; letter-spacing: 4px; margin-bottom: 30px; }

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
# 3. LOGIC TÍNH TOÁN (V4.0)
# ==============================================================================

def clean_yfinance_data(df):
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower().strip() for c in df.columns]
    col = next((c for c in ['adj close', 'close', 'price'] if c in df.columns), df.columns[0])
    return df[col]

def get_forecast(full_data, model_type, test_size, window_size, seasonal_p, freq_val):
    train = full_data.iloc[:-test_size]
    test = full_data.iloc[-test_size:]
    
    preds = pd.Series(index=test.index, dtype='float64')
    info = ""
    warning = None

    try:
        if model_type == "Naive": 
            preds[:] = np.array([train.iloc[-1]] * len(test))
            info = "NAIVE"

        elif model_type == "Moving Average": 
            rolling_series = full_data.rolling(window=window_size).mean().shift(1)
            preds = rolling_series.loc[test.index]
            info = f"MA({window_size})"

        elif model_type == "SES":
            def ses_loss(params):
                mdl = SimpleExpSmoothing(train).fit(smoothing_level=params[0], optimized=False)
                return mean_squared_error(train, mdl.fittedvalues)
            res = minimize(ses_loss, [0.5], bounds=[(0.01, 0.99)], method='L-BFGS-B')
            alpha_opt = res.x[0]
            model = SimpleExpSmoothing(train).fit(smoothing_level=alpha_opt, optimized=False)
            preds = model.forecast(len(test))
            info = f"α:{alpha_opt:.2f}"

        elif model_type == "Holt":
            def holt_loss(params):
                mdl = ExponentialSmoothing(train, trend='add').fit(
                    smoothing_level=params[0], smoothing_trend=params[1], optimized=False)
                return mean_squared_error(train, mdl.fittedvalues)
            res = minimize(holt_loss, [0.5, 0.1], bounds=[(0.01, 0.99), (0.01, 0.99)], method='L-BFGS-B')
            alpha_opt, beta_opt = res.x
            model = ExponentialSmoothing(train, trend='add').fit(
                smoothing_level=alpha_opt, smoothing_trend=beta_opt, optimized=False)
            preds = model.forecast(len(test))
            info = f"α:{alpha_opt:.2f} β:{beta_opt:.2f}"

        elif model_type == "Holt-Winters":
            if freq_val == 'D' or seasonal_p > 12: 
                warning = "Mô hình có thể không phù hợp để dự báo (Daily Data)"
            
            def hw_loss(params):
                try:
                    mdl = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=seasonal_p).fit(
                        smoothing_level=params[0], smoothing_trend=params[1], smoothing_seasonal=params[2], optimized=False)
                    return mean_squared_error(train, mdl.fittedvalues)
                except: return 1e10

            initial_guess = [0.3, 0.1, 0.1]
            bounds = [(0.01, 0.99), (0.01, 0.99), (0.01, 0.99)]
            res = minimize(hw_loss, initial_guess, bounds=bounds, method='L-BFGS-B')
            alpha_opt, beta_opt, gamma_opt = res.x
            model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=seasonal_p).fit(
                smoothing_level=alpha_opt, smoothing_trend=beta_opt, smoothing_seasonal=gamma_opt, optimized=False)
            preds = model.forecast(len(test))
            info = f"α:{alpha_opt:.2f} β:{beta_opt:.2f} γ:{gamma_opt:.2f}"

    except Exception as e:
        preds[:] = np.nan
        info = "ERROR"
        
    return train, test, preds, info, warning

# ==============================================================================
# 4. GIAO DIỆN CHÍNH
# ==============================================================================

if 'vs_mode' not in st.session_state: st.session_state.vs_mode = False

st.markdown("<h1>PIXEL TRADER</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>ULTIMATE EDITION [v4.3 - INTERACTIVE]</div>", unsafe_allow_html=True)

with st.container():
    c1, c2, c3 = st.columns([1, 3, 1]) 
    with c2:
        ticker = st.text_input("PLAYER 1 (MÃ CHÍNH)", value="META", placeholder="EX: AAPL").upper()
        col_inp1, col_inp2 = st.columns(2)
        with col_inp1: freq_display = st.selectbox("TIMEFRAME", ("DAILY", "MONTHLY", "QUARTERLY"))
        with col_inp2: model_display = st.selectbox("WEAPON (MODEL)", ("Naive", "Moving Average", "SES", "Holt", "Holt-Winters"))
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
            
            seasonal_p = 5 
            if freq_val == "M": data = data.resample('M').last(); seasonal_p = 12
            elif freq_val == "Q": data = data.resample('Q').last(); seasonal_p = 4
            else: data = data.asfreq('B').fillna(method='ffill'); seasonal_p = 5
            
            data = data.dropna()
            if len(data) < test_size + 10: st.error("⚠️ DATA TOO SHORT."); st.stop()

            # GỌI HÀM DỰ BÁO
            train, test, preds, info, warning_msg = get_forecast(data, model_display, test_size, window_size, seasonal_p, freq_val)

            mask = ~np.isnan(preds) & ~np.isnan(test)
            rmse = np.sqrt(mean_squared_error(test[mask], preds[mask])) if mask.sum() > 0 else 0
            mape = mean_absolute_percentage_error(test[mask], preds[mask]) * 100 if mask.sum() > 0 else 0

            if warning_msg: st.warning(f"⚠️ {warning_msg}")

            # --- MARKET STATS ---
            st.markdown(f"<div style='text-align:center; font-family:\"Press Start 2P\"; color:#00ff41; margin-bottom:10px'>TARGET: {ticker}</div>", unsafe_allow_html=True)
            
            current_price = test.iloc[-1]
            predicted_price = preds.iloc[-1]
            if not np.isnan(predicted_price) and not np.isnan(preds.iloc[0]):
                trend_pct = ((predicted_price - preds.iloc[0]) / preds.iloc[0]) * 100
            else: trend_pct = 0.0
            trend_color = "#00ff41" if trend_pct >= 0 else "#ff3333"
            trend_arrow = "▲" if trend_pct >= 0 else "▼"

            stat1, stat2, stat3 = st.columns(3)
            stat_box_style = "border:2px solid #fff; padding:10px; text-align:center; background: rgba(255,255,255,0.05); margin-bottom: 20px;"
            stat_label = "font-family: 'Press Start 2P'; font-size: 12px; color: #aaa; margin-bottom: 8px;"
            stat_val = "font-family: 'VT323'; font-size: 36px; line-height: 1; color: #fff;"

            stat1.markdown(f"<div style='{stat_box_style} border-color: #aaa;'><div style='{stat_label}'>CURRENT PRICE</div><div style='{stat_val}'>${current_price:,.2f}</div></div>", unsafe_allow_html=True)
            stat2.markdown(f"<div style='{stat_box_style} border-color: #ff00ff;'><div style='{stat_label} color:#ff00ff;'>PREDICTED (END)</div><div style='{stat_val} color:#ff00ff;'>${predicted_price:,.2f}</div></div>", unsafe_allow_html=True)
            stat3.markdown(f"<div style='{stat_box_style} border-color: {trend_color};'><div style='{stat_label} color:{trend_color};'>TREND FORECAST</div><div style='{stat_val} color:{trend_color};'>{trend_arrow} {abs(trend_pct):.2f}%</div></div>", unsafe_allow_html=True)

            # --- METRICS ---
            c_m1, c_m2, c_m3 = st.columns(3)
            box_style = "border:2px solid #00ff41; padding:10px; text-align:center; height:100%; display:flex; flex-direction:column; justify-content:center;"
            label_font = "font-family: 'Press Start 2P', cursive; font-size: 14px; margin-bottom: 5px; color: #00ff41;"
            value_font = "font-family: 'VT323', monospace; font-size: 40px; margin: 0; line-height: 1; color: #ffffff;"

            c_m1.markdown(f"<div style='{box_style}'><div style='{label_font}'>RMSE</div><div style='{value_font}'>{rmse:.2f}</div></div>", unsafe_allow_html=True)
            c_m2.markdown(f"<div style='{box_style}'><div style='{label_font}'>MAPE</div><div style='{value_font}'>{mape:.2f}%</div></div>", unsafe_allow_html=True)
            c_m3.markdown(f"<div style='border:2px solid #00ffff; padding:10px; text-align:center; height:100%; display:flex; flex-direction:column; justify-content:center;'><div style='font-family: \"Press Start 2P\", cursive; font-size: 14px; margin-bottom: 5px; color: #00ffff;'>PARAMS</div><div style='font-family: \"VT323\", monospace; font-size: 35px; margin: 0; line-height: 1; color: #ffffff;'>{info}</div></div>", unsafe_allow_html=True)

            st.write("")
            
            # ==================================================================
            # [THAY ĐỔI] SỬ DỤNG PLOTLY ĐỂ VẼ BIỂU ĐỒ TƯƠNG TÁC
            # ==================================================================
            
            fig = go.Figure()

            # 1. Vẽ dữ liệu Train (Màu xám)
            fig.add_trace(go.Scatter(
                x=train.index[-60:], # Lấy 60 điểm cuối để đỡ rối
                y=train.iloc[-60:],
                mode='lines',
                name='TRAIN',
                line=dict(color='#555555', width=1.5)
            ))

            # 2. Vẽ dữ liệu Thực tế (Màu Xanh Neon)
            fig.add_trace(go.Scatter(
                x=test.index,
                y=test,
                mode='lines+markers',
                name='ACTUAL',
                line=dict(color='#00ff41', width=3),
                marker=dict(size=4)
            ))

            # 3. Vẽ dữ liệu Dự báo (Màu Tím Neon)
            fig.add_trace(go.Scatter(
                x=preds.index,
                y=preds,
                mode='lines+markers',
                name='PREDICT',
                line=dict(color='#ff00ff', width=3, dash='dash'),
                marker=dict(size=6, symbol='circle')
            ))

            # Cấu hình giao diện biểu đồ (Dark Mode)
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', # Nền trong suốt
                plot_bgcolor='rgba(0,0,0,0)',  # Nền trong suốt
                font=dict(family='Courier New, monospace', color='#ffffff'), # Font chữ kiểu code
                xaxis=dict(
                    showgrid=True, gridcolor='#333333', 
                    tickfont=dict(color='#00ff41')
                ),
                yaxis=dict(
                    showgrid=True, gridcolor='#333333', 
                    tickfont=dict(color='#ffffff')
                ),
                hovermode="x unified", # Rê chuột hiện tất cả thông số cùng lúc
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                margin=dict(l=0, r=0, t=30, b=0)
            )

            # Hiển thị biểu đồ Plotly
            st.plotly_chart(fig, use_container_width=True)


            # --- VS MODE ---
            st.markdown("---")
            st.markdown("<h3 style='text-align:center; color:#ffcc00; font-family:\"Press Start 2P\"'>VS MODE ACTIVATED</h3>", unsafe_allow_html=True)
            
            v1, v2, v3 = st.columns([1, 2, 1])
            with v2:
                rivals_input = st.text_input("ENTER RIVALS (MÃ ĐỐI THỦ)", value="AAPL, MSFT, GOOG", placeholder="EX: TSLA, AMZN")
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
                                _, _, pred_t, _, _ = get_forecast(val, model_display, test_size, window_size, seasonal_p, freq_val)
                                if not pred_t.isna().all(): results_map[t] = pred_t
                    except Exception: pass
                    progress_bar.progress((i + 1) / len(all_tickers))
                progress_bar.empty()

                if len(results_map) > 0:
                    st.markdown("<h4 style='text-align:center; font-family:VT323; margin-top:20px'>PREDICTED GROWTH (%) COMPARISON</h4>", unsafe_allow_html=True)
                    
                    # [THAY ĐỔI] BIỂU ĐỒ VS MODE TƯƠNG TÁC
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
                        xaxis=dict(showgrid=True, gridcolor='#333333'),
                        yaxis=dict(showgrid=True, gridcolor='#333333', title="Growth %"),
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
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
