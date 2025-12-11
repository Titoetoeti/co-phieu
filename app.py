import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import warnings
import time
import base64
import os

# --- CÁC THƯ VIỆN AI ---
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from pmdarima import auto_arima
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout

# ==============================================================================
# 1. CẤU HÌNH & HÀM LOAD DATA "SIÊU BỀN"
# ==============================================================================
warnings.filterwarnings("ignore")
st.set_page_config(page_title="PIXEL TRADER DEBUG", layout="wide", page_icon="🛠️")
plt.style.use('dark_background') 

DATA_FILE = "Data_1.xlsx - Tong_Hop_log_return.csv"

@st.cache_data
def load_and_inspect_data(filepath):
    """
    Hàm đọc dữ liệu 'bất chấp lỗi' để đảm bảo khớp Excel 100%
    """
    try:
        # 1. Đọc file CSV thuần túy
        df = pd.read_csv(filepath)
        
        # 2. [FIX 100%] Xóa khoảng trắng thừa ở tên cột (Ví dụ: " Date " -> "Date")
        df.columns = df.columns.str.strip()
        
        # 3. Tìm cột Date (kể cả khi nó viết hoa/thường khác nhau)
        date_col = None
        for col in df.columns:
            if col.lower() == 'date':
                date_col = col
                break
        
        if date_col is None:
            st.error("❌ Không tìm thấy cột 'Date' trong file CSV!")
            return None

        # 4. [FIX 100%] Ép kiểu ngày tháng an toàn
        # errors='coerce': Nếu dòng nào lỗi ngày tháng, biến nó thành NaT chứ không báo lỗi
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # 5. Loại bỏ các dòng mà ngày tháng bị lỗi (NaT)
        df = df.dropna(subset=[date_col])
        
        # 6. Sắp xếp lại chuẩn chỉ
        df = df.sort_values(by=date_col)
        
        # 7. Set Index
        df.set_index(date_col, inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Lỗi nghiêm trọng khi đọc file: {e}")
        return None

# --- INTRO VIDEO (GIỮ NGUYÊN) ---
def show_intro_video(video_file, duration=8):
    if 'intro_done' not in st.session_state: st.session_state['intro_done'] = False
    if st.session_state['intro_done']: return
    if not os.path.exists(video_file):
        st.session_state['intro_done'] = True; return
    try:
        with open(video_file, "rb") as f: video_bytes = f.read()
        video_str = base64.b64encode(video_bytes).decode()
        st.markdown(f"""<style>.stApp {{overflow:hidden}} #intro {{position:fixed;top:0;left:0;width:100vw;height:100vh;background:#000;z-index:999999}}</style><div id="intro"><video style="width:100%;height:100%;object-fit:cover" autoplay muted playsinline><source src="data:video/mp4;base64,{video_str}" type="video/mp4"></video></div>""", unsafe_allow_html=True)
        time.sleep(duration); st.empty(); st.session_state['intro_done'] = True; st.rerun()
    except: st.session_state['intro_done'] = True

show_intro_video("intro1.mp4", duration=6)

# ==============================================================================
# 2. CSS & GIAO DIỆN
# ==============================================================================
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');
        .stApp { background-color: #0d0d0d; color: #00ff41; font-family: 'VT323', monospace; font-size: 20px; }
        h1 { font-family: 'Press Start 2P'; text-align: center; color: #00ff41; font-size: 50px; }
        .stDataFrame { border: 1px solid #333; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. AI CORE LOGIC
# ==============================================================================
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length]); y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def get_forecast(full_data, model_type, test_size, window_size, future_days=0):
    train_data = full_data.iloc[:-test_size]
    test_data = full_data.iloc[-test_size:]
    preds = pd.Series(index=test_data.index, dtype='float64')
    future_series = pd.Series(dtype='float64')
    info = ""
    
    try:
        if model_type == "ARIMA":
            model = auto_arima(train_data, start_p=1, start_q=1, max_p=3, max_q=3, seasonal=False, stepwise=True, error_action='ignore')
            preds[:] = model.predict(n_periods=len(test_data)).values
            if future_days > 0:
                model_full = auto_arima(full_data, start_p=1, start_q=1, max_p=3, max_q=3, seasonal=False)
                future_vals = model_full.predict(n_periods=future_days)
                future_dates = pd.bdate_range(start=full_data.index[-1], periods=future_days + 1)[1:]
                future_series = pd.Series(future_vals.values, index=future_dates)
            info = f"ARIMA{model.order}"

        elif model_type == "XGBoost":
            def create_lags(s, lag=3):
                d = pd.DataFrame(s)
                for i in range(1, lag+1): d[f'lag_{i}'] = d.iloc[:,0].shift(i)
                return d.dropna()
            df_lags = create_lags(full_data, window_size)
            X, y = df_lags.drop(columns=[df_lags.columns[0]]), df_lags.iloc[:,0]
            X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
            y_train = y.iloc[:-test_size]
            
            model = XGBRegressor(objective='reg:squarederror', n_estimators=100); model.fit(X_train, y_train)
            preds[:] = model.predict(X_test)
            
            if future_days > 0:
                full_model = XGBRegressor(objective='reg:squarederror', n_estimators=100); full_model.fit(X, y)
                curr = list(X.iloc[-1].values); fut = []
                for _ in range(future_days):
                    p = full_model.predict(np.array(curr).reshape(1,-1))[0]
                    fut.append(p); curr.pop(-1); curr.insert(0, p)
                future_series = pd.Series(fut, index=pd.bdate_range(start=full_data.index[-1], periods=future_days+1)[1:])
            info = "XGBoost"

        elif model_type in ["LSTM", "GRU"]:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled = scaler.fit_transform(full_data.values.reshape(-1, 1))
            X, y = create_sequences(scaled, window_size)
            split = len(X) - test_size
            if split > 0:
                X_train, y_train = X[:split], y[:split]; X_test = X[split:]
                X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                
                m = Sequential()
                layer = LSTM if model_type == 'LSTM' else GRU
                m.add(layer(50, return_sequences=True, input_shape=(X_train.shape[1],1))); m.add(Dropout(0.2))
                m.add(layer(50)); m.add(Dropout(0.2)); m.add(Dense(1))
                m.compile(optimizer='adam', loss='mse')
                m.fit(X_train, y_train, epochs=15, batch_size=16, verbose=0)
                
                p_scaled = m.predict(X_test, verbose=0)
                p_inv = scaler.inverse_transform(p_scaled).flatten()
                preds.iloc[-len(p_inv):] = p_inv
                
                if future_days > 0:
                    curr = scaled[-window_size:].reshape(1, window_size, 1); fut_sc = []
                    for _ in range(future_days):
                        p = m.predict(curr, verbose=0)[0,0]; fut_sc.append(p)
                        curr = np.append(curr[:,1:,:], [[[p]]], axis=1)
                    future_series = pd.Series(scaler.inverse_transform(np.array(fut_sc).reshape(-1,1)).flatten(), index=pd.bdate_range(start=full_data.index[-1], periods=future_days+1)[1:])
            info = f"{model_type}"

    except Exception as e: info = "ERROR"; preds[:] = np.nan
    return preds, future_series, info

# ==============================================================================
# 4. GIAO DIỆN CHÍNH
# ==============================================================================
st.markdown("<h1>PIXEL TRADER AI</h1>", unsafe_allow_html=True)

# LOAD DATA
df_full = load_and_inspect_data(DATA_FILE)

if df_full is None:
    st.error(f"❌ Vui lòng upload file: {DATA_FILE}"); st.stop()

tickers = list(df_full.columns)

# --- DEBUGGING AREA (PHẦN QUAN TRỌNG ĐỂ FIX 100% LỖI) ---
with st.expander("🔍 KIỂM TRA DỮ LIỆU GỐC (SO SÁNH VỚI EXCEL)", expanded=True):
    st.info("Hãy nhìn vào dòng cuối cùng (Tail) dưới đây. Nếu nó khớp với Excel, biểu đồ sẽ đúng.")
    d_head, d_tail = st.columns(2)
    with d_head: st.write("5 Dòng Đầu (Head):"); st.dataframe(df_full.head())
    with d_tail: st.write("5 Dòng Cuối (Tail):"); st.dataframe(df_full.tail())

with st.container():
    c1, c2 = st.columns([1, 2])
    with c1:
        ticker = st.selectbox("CHỌN MÃ", tickers)
        model_display = st.selectbox("MÔ HÌNH", ("ARIMA", "XGBoost", "LSTM", "GRU"))
        
        # [QUAN TRỌNG] Cho phép người dùng tự chọn ngày kết thúc
        min_date = df_full.index.min().date()
        max_date = df_full.index.max().date()
        
        st.write("---")
        st.write("⏳ PHẠM VI DỮ LIỆU:")
        end_date_input = st.date_input("Ngày Kết Thúc", value=max_date, min_value=min_date, max_value=max_date)
        
        btn_run = st.button(">> CHẠY DỰ BÁO <<")

# ==============================================================================
# 5. XỬ LÝ
# ==============================================================================
if btn_run:
    with st.spinner(f"Đang xử lý {ticker}..."):
        # Cắt dữ liệu đúng theo ngày người dùng chọn
        raw_series = df_full[ticker]
        
        # Fill số liệu trống
        data = raw_series.fillna(method='ffill').fillna(method='bfill')
        
        # Cắt đến ngày được chọn
        data = data.loc[:str(end_date_input)]
        
        # Hiển thị dữ liệu thực tế được đưa vào mô hình
        st.success(f"Dữ liệu được lấy từ {data.index[0].date()} đến {data.index[-1].date()}")
        st.write(f"Giá trị cuối cùng được dùng để tính toán: **{data.iloc[-1]:,.2f}** (Check xem khớp Excel chưa?)")

        if len(data) < 30: st.error("Dữ liệu quá ngắn!"); st.stop()

        # Chạy dự báo
        test_sz = 30; win_sz = 30; fut_days = 30
        preds, fut_series, info = get_forecast(data, model_display, test_sz, win_sz, fut_days)
        
        # Vẽ biểu đồ
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data.values, name='LỊCH SỬ', line=dict(color='#555')))
        fig.add_trace(go.Scatter(x=data.index[-test_sz:], y=data.iloc[-test_sz:], name='THỰC TẾ (TEST)', line=dict(color='#00ff41', width=2)))
        fig.add_trace(go.Scatter(x=preds.index, y=preds.values, name='AI BACKTEST', line=dict(color='#ff00ff', dash='dot')))
        if not fut_series.empty:
            fig.add_trace(go.Scatter(x=fut_series.index, y=fut_series.values, name='TƯƠNG LAI', line=dict(color='#ffff00', width=3)))

        fig.update_layout(
            font=dict(family="Courier New", color="white"),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='#333'), yaxis=dict(gridcolor='#333'),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Hiện bảng giá tương lai
        if not fut_series.empty:
            st.write("📋 GIÁ DỰ BÁO TƯƠNG LAI:")
            st.dataframe(fut_series.to_frame("Giá Dự Báo").T)
