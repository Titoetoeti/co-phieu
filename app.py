import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
import time
import base64
import os

# --- CÁC THƯ VIỆN TÍNH TOÁN MỚI (TỪ FILE JUPYTER) ---
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from pmdarima import auto_arima
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
from tensorflow.keras.optimizers import Adam

# ==============================================================================
# 1. CẤU HÌNH & HÀM HỖ TRỢ GIAO DIỆN
# ==============================================================================
warnings.filterwarnings("ignore")
st.set_page_config(page_title="PIXEL TRADER PRO (AI CORE)", layout="wide", page_icon="📈")
plt.style.use('dark_background') 

# --- HÀM INTRO VIDEO (GIỮ NGUYÊN) ---
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
            <div id="skip-btn">LOADING AI MODELS...</div>
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

show_intro_video("intro1.mp4", duration=7)

# ==============================================================================
# 2. CSS GIAO DIỆN (PIXEL STYLE - GIỮ NGUYÊN)
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
        h1 { font-family: 'Press Start 2P', cursive !important; text-align: center; color: #00ff41; text-shadow: 6px 6px 0px #003300; font-size: 70px !important; line-height: 1.2 !important; margin-bottom: 10px !important; margin-top: 0px !important; }
        .sub-title { text-align: center; font-family: 'VT323'; font-size: 24px; color: #555; letter-spacing: 4px; margin-bottom: 30px; }
        div.stButton > button { width: 100%; background-color: #000000 !important; color: #00ff41 !important; border: 2px solid #00ff41 !important; font-family: 'Press Start 2P', cursive !important; padding: 15px; margin-top: 15px; border-radius: 0px !important; transition: all 0.2s ease-in-out; box-shadow: none !important; }
        div.stButton > button:hover { background-color: #00ff41 !important; color: #000000 !important; box-shadow: 0 0 15px #00ff41 !important; }
        div.stButton > button:active { transform: scale(0.98); }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. LOGIC TÍNH TOÁN MỚI (AI/DEEP LEARNING TỪ DATA_ANALYSIS_4)
# ==============================================================================

def clean_yfinance_data(df):
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower().strip() for c in df.columns]
    col = next((c for c in ['adj close', 'close', 'price'] if c in df.columns), df.columns[0])
    return df[col]

# --- HÀM TẠO SEQUENCE CHO LSTM/GRU ---
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# --- HÀM BUILD MODEL LSTM/GRU ---
def build_dl_model(model_type, input_shape):
    model = Sequential()
    if model_type == 'LSTM':
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
    elif model_type == 'GRU':
        model.add(GRU(50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(GRU(50, return_sequences=False))
    
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1)) # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- HÀM DỰ BÁO CHÍNH (ĐÃ THAY ĐỔI TOÀN BỘ LOGIC) ---
def get_forecast(full_data, model_type, test_size, window_size, future_days=0):
    # Chia Train/Test
    train_data = full_data.iloc[:-test_size]
    test_data = full_data.iloc[-test_size:]
    
    preds = pd.Series(index=test_data.index, dtype='float64')
    future_series = pd.Series(dtype='float64') # Cho dự báo tương lai
    info = ""
    warning = None

    try:
        # === LOGIC 1: ARIMA (AUTO) ===
        if model_type == "ARIMA":
            # Auto ARIMA tìm tham số tốt nhất
            model = auto_arima(train_data, start_p=1, start_q=1,
                               max_p=3, max_q=3, m=1,
                               start_P=0, seasonal=False,
                               d=None, D=0, trace=False,
                               error_action='ignore',  
                               suppress_warnings=True, 
                               stepwise=True)
            
            # Dự báo trên tập Test
            forecast_test = model.predict(n_periods=len(test_data))
            preds[:] = forecast_test.values
            
            # Dự báo tương lai (nếu cần)
            if future_days > 0:
                # Fit lại trên toàn bộ data để dự báo tương lai chính xác hơn
                model_full = auto_arima(full_data, start_p=1, start_q=1, max_p=3, max_q=3, m=1, seasonal=False, stepwise=True)
                future_vals = model_full.predict(n_periods=future_days)
                future_dates = pd.bdate_range(start=full_data.index[-1], periods=future_days + 1)[1:]
                future_series = pd.Series(future_vals.values, index=future_dates)

            order = model.order
            info = f"ARIMA{order}"

        # === LOGIC 2: XGBOOST ===
        elif model_type == "XGBoost":
            # Tạo feature lags (cửa sổ trượt)
            def create_lag_features(series, lag=3):
                df_lag = pd.DataFrame(series)
                for i in range(1, lag + 1):
                    df_lag[f'lag_{i}'] = df_lag.iloc[:, 0].shift(i)
                return df_lag.dropna()

            df_lags = create_lag_features(full_data, lag=window_size)
            
            # Tách lại train/test từ dữ liệu đã lag
            X = df_lags.drop(columns=[df_lags.columns[0]])
            y = df_lags.iloc[:, 0]
            
            # Cắt đuôi test_size
            X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
            y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
            
            model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
            model.fit(X_train, y_train)
            
            # Predict Test
            pred_vals = model.predict(X_test)
            preds[:] = pred_vals
            
            # Predict Future (Recursive)
            if future_days > 0:
                curr_feat = X_test.iloc[-1].values.reshape(1, -1) # Lấy bộ feature cuối cùng
                fut_vals = []
                # Dùng mô hình đã học để predict tiếp (Lưu ý: XGBoost không giỏi extrapolate trend)
                temp_model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
                temp_model.fit(X, y) # Fit full data
                
                curr_seq = list(X.iloc[-1].values) # [lag_1, lag_2, ...]
                
                for _ in range(future_days):
                    input_feat = np.array(curr_seq).reshape(1, -1)
                    pred = temp_model.predict(input_feat)[0]
                    fut_vals.append(pred)
                    curr_seq.pop(-1) # Bỏ cái cũ nhất
                    curr_seq.insert(0, pred) # Thêm cái mới dự báo vào đầu (lag_1)
                
                future_dates = pd.bdate_range(start=full_data.index[-1], periods=future_days + 1)[1:]
                future_series = pd.Series(fut_vals, index=future_dates)

            info = f"XGB (Lags:{window_size})"

        # === LOGIC 3 & 4: DEEP LEARNING (LSTM / GRU) ===
        elif model_type in ["LSTM", "GRU"]:
            # 1. Scale dữ liệu
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(full_data.values.reshape(-1, 1))
            
            # 2. Tạo sequences
            x_total, y_total = create_sequences(scaled_data, window_size)
            
            # Chia Train/Test theo sequences
            # Test size thực tế cho model cần trừ đi window
            split_idx = len(x_total) - test_size
            if split_idx <= 0: raise ValueError("Window size quá lớn so với Data")

            x_train, y_train = x_total[:split_idx], y_total[:split_idx]
            x_test, y_test = x_total[split_idx:], y_total[split_idx:]
            
            # Reshape cho Keras [samples, time steps, features]
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
            
            # 3. Build & Train Model
            model = build_dl_model(model_type, (x_train.shape[1], 1))
            # Epochs thấp để chạy demo nhanh trên Streamlit (thực tế nên để 50-100)
            model.fit(x_train, y_train, batch_size=16, epochs=15, verbose=0) 
            
            # 4. Predict Test
            pred_scaled = model.predict(x_test)
            pred_inverse = scaler.inverse_transform(pred_scaled)
            
            # Gán vào preds (đảm bảo index khớp)
            # Do cơ chế sequence, length có thể lệch 1 xíu, ta lấy đoạn cuối khớp với test_data
            if len(pred_inverse) == len(preds):
                preds[:] = pred_inverse.flatten()
            else:
                # Fallback cắt gọt
                min_len = min(len(pred_inverse), len(preds))
                preds.iloc[-min_len:] = pred_inverse.flatten()[-min_len:]

            # 5. Predict Future
            if future_days > 0:
                # Lấy sequence cuối cùng của toàn bộ dữ liệu
                curr_seq = scaled_data[-window_size:].reshape(1, window_size, 1)
                fut_vals_scaled = []
                
                for _ in range(future_days):
                    pred_step = model.predict(curr_seq, verbose=0)
                    fut_vals_scaled.append(pred_step[0, 0])
                    # Update sequence: bỏ đầu, thêm đuôi dự báo
                    pred_step_reshaped = pred_step.reshape(1, 1, 1)
                    curr_seq = np.append(curr_seq[:, 1:, :], pred_step_reshaped, axis=1)
                
                fut_vals = scaler.inverse_transform(np.array(fut_vals_scaled).reshape(-1, 1))
                future_dates = pd.bdate_range(start=full_data.index[-1], periods=future_days + 1)[1:]
                future_series = pd.Series(fut_vals.flatten(), index=future_dates)

            info = f"{model_type} (Win:{window_size})"

    except Exception as e:
        info = "ERROR"
        warning = str(e)
        preds[:] = np.nan
        
    return train_data, test_data, preds, future_series, info, warning

# ==============================================================================
# 4. GIAO DIỆN CHÍNH (ĐÃ CẬP NHẬT SELECTBOX)
# ==============================================================================

if 'vs_mode' not in st.session_state: st.session_state.vs_mode = False

st.markdown("<h1>PIXEL TRADER AI</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>DEEP LEARNING EDITION</div>", unsafe_allow_html=True)

with st.container():
    c1, c2, c3 = st.columns([1, 3, 1]) 
    with c2:
        ticker = st.text_input("PLAYER 1 (MÃ CHÍNH)", value="META", placeholder="EX: AAPL").upper()
        col_inp1, col_inp2 = st.columns(2)
        with col_inp1: 
            # Giữ nguyên Timeframe nhưng logic lấy data sẽ clean
            freq_display = st.selectbox("TIMEFRAME", ("DAILY", "MONTHLY (Coming Soon)"))
        with col_inp2: 
            # [SỬA] Đổi danh sách mô hình sang AI
            model_display = st.selectbox("AI MODEL", ("ARIMA", "XGBoost", "LSTM", "GRU"))
            
        with st.expander("⚙️ HYPER PARAMETERS"):
            window_size = st.slider("LOOKBACK WINDOW (AI Memory)", 10, 60, 30)
            test_size = st.slider("TEST DATA SIZE", 10, 90, 30)
            future_days = st.slider("FUTURE FORECAST (DAYS)", 7, 60, 30)
        
        st.write("") 
        btn_run = st.button(">> ACTIVATE NEURAL NETWORK <<")

st.markdown("---")

# ==============================================================================
# 5. XỬ LÝ & HIỂN THỊ
# ==============================================================================

if btn_run: st.session_state.vs_mode = False

if btn_run or st.session_state.get('run_success', False):
    st.session_state.run_success = True
    
    try:
        with st.spinner(f"FETCHING DATA & TRAINING AI: {ticker}..."):
            # [EDIT] Lấy dữ liệu cố định theo yêu cầu cũ
            df = yf.download(ticker, start="2020-11-23", end="2025-11-21", progress=False)
            data = clean_yfinance_data(df)
            
            if data is None: st.error("❌ DATA NOT FOUND."); st.stop()
            data = data.astype(float)
            if data.index.tz is not None: data.index = data.index.tz_localize(None)
            data = data.dropna()
            
            # [CHECK] Đảm bảo đủ dữ liệu cho Window Size của AI
            min_req = window_size + test_size + 10
            if len(data) < min_req:
                st.error(f"⚠️ DATA TOO SHORT FOR AI. NEED > {min_req} ROWS."); st.stop()

            # GỌI HÀM DỰ BÁO MỚI
            train, test, preds, future_series, info, warning_msg = get_forecast(data, model_display, test_size, window_size, future_days)

            # Tính toán lỗi
            mask = ~np.isnan(preds) & ~np.isnan(test)
            rmse = np.sqrt(mean_squared_error(test[mask], preds[mask])) if mask.sum() > 0 else 0
            mape = mean_absolute_percentage_error(test[mask], preds[mask]) * 100 if mask.sum() > 0 else 0

            if warning_msg: st.warning(f"⚠️ MODEL WARNING: {warning_msg}")

            # --- MARKET STATS ---
            st.markdown(f"<div style='text-align:center; font-family:\"Press Start 2P\"; color:#00ff41; margin-bottom:10px'>TARGET: {ticker}</div>", unsafe_allow_html=True)
            
            current_price = test.iloc[-1]
            # Lấy giá dự báo cuối cùng (nếu có future thì lấy future, ko thì lấy test)
            if not future_series.empty:
                final_predicted_price = future_series.iloc[-1]
                compare_price = future_series.iloc[0]
            else:
                final_predicted_price = preds.iloc[-1]
                compare_price = preds.iloc[0]

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
            stat2.markdown(f"<div style='{stat_box_style} border-color: #ff00ff;'><div style='{stat_label} color:#ff00ff;'>AI TARGET (Future)</div><div style='{stat_val} color:#ff00ff;'>${final_predicted_price:,.2f}</div></div>", unsafe_allow_html=True)
            stat3.markdown(f"<div style='{stat_box_style} border-color: {trend_color};'><div style='{stat_label} color:{trend_color};'>AI FORECAST</div><div style='{stat_val} color:{trend_color};'>{trend_arrow} {abs(trend_pct):.2f}%</div></div>", unsafe_allow_html=True)

            # --- METRICS ---
            c_m1, c_m2, c_m3 = st.columns(3)
            box_style = "border:2px solid #00ff41; padding:10px; text-align:center; height:100%; display:flex; flex-direction:column; justify-content:center;"
            label_font = "font-family: 'Press Start 2P', cursive; font-size: 14px; margin-bottom: 5px; color: #00ff41;"
            value_font = "font-family: 'VT323', monospace; font-size: 40px; margin: 0; line-height: 1; color: #ffffff;"

            c_m1.markdown(f"<div style='{box_style}'><div style='{label_font}'>RMSE (Test)</div><div style='{value_font}'>{rmse:.2f}</div></div>", unsafe_allow_html=True)
            c_m2.markdown(f"<div style='{box_style}'><div style='{label_font}'>MAPE (Test)</div><div style='{value_font}'>{mape:.2f}%</div></div>", unsafe_allow_html=True)
            c_m3.markdown(f"<div style='border:2px solid #00ffff; padding:10px; text-align:center; height:100%; display:flex; flex-direction:column; justify-content:center;'><div style='font-family: \"Press Start 2P\", cursive; font-size: 14px; margin-bottom: 5px; color: #00ffff;'>AI ARCHITECTURE</div><div style='font-family: \"VT323\", monospace; font-size: 35px; margin: 0; line-height: 1; color: #ffffff;'>{info}</div></div>", unsafe_allow_html=True)

            st.write("")
            
            # ==================================================================
            # BIỂU ĐỒ TƯƠNG TÁC (PLOTLY) - ĐÃ CÓ FUTURE & LEGEND FIX
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
                mode='lines', name='ACTUAL (TEST)',
                line=dict(color='#00ff41', width=2)
            ))

            # 3. AI Prediction on Test
            fig.add_trace(go.Scatter(
                x=preds.index, y=preds.values,
                mode='lines', name='AI BACKTEST',
                line=dict(color='#ff00ff', width=2, dash='dot')
            ))

            # 4. Future
            if not future_series.empty:
                fig.add_trace(go.Scatter(
                    x=future_series.index, y=future_series.values,
                    mode='lines+markers', name=f'AI FUTURE ({future_days}D)',
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
                with st.expander("📋 VIEW FUTURE DATA POINTS"):
                    st.dataframe(future_series.to_frame(name="AI Prediction").T)

            # --- VS MODE (CẬP NHẬT LOGIC AI) ---
            st.markdown("---")
            st.markdown("<h3 style='text-align:center; color:#ffcc00; font-family:\"Press Start 2P\"'>VS MODE (AI BATTLE)</h3>", unsafe_allow_html=True)
            
            v1, v2, v3 = st.columns([1, 2, 1])
            with v2:
                rivals_input = st.text_input("ENTER RIVALS", value="AAPL, MSFT, GOOG", placeholder="EX: TSLA, AMZN")
                st.write("")
                btn_fight = st.button(">> START COMPARISON <<")

            if btn_fight:
                rivals = [r.strip().upper() for r in rivals_input.split(",") if r.strip()]
                all_tickers = [ticker] + rivals[:3] 
                results_map = {}
                progress_bar = st.progress(0)
                
                for i, t in enumerate(all_tickers):
                    try:
                        # Dùng cùng khung thời gian
                        d_t = yf.download(t, start="2020-11-23", end="2025-11-21", progress=False)
                        val = clean_yfinance_data(d_t)
                        if val is not None and not val.empty:
                            val = val.astype(float)
                            val = val.dropna()
                            if len(val) > test_size + window_size:
                                # Gọi AI dự báo cho từng đối thủ
                                _, _, pred_t, _, _, _ = get_forecast(val, model_display, test_size, window_size, future_days=0)
                                if not pred_t.isna().all(): results_map[t] = pred_t
                    except Exception: pass
                    progress_bar.progress((i + 1) / len(all_tickers))
                progress_bar.empty()

                if len(results_map) > 0:
                    st.markdown("<h4 style='text-align:center; font-family:VT323; margin-top:20px'>AI GROWTH PREDICTION COMPARISON</h4>", unsafe_allow_html=True)
                    
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
            AI NEURAL NETWORK READY...<br>[ WAITING FOR INPUT ]
        </div>
        <style>@keyframes blinker { 50% { opacity: 0; } }</style>
    """, unsafe_allow_html=True)
