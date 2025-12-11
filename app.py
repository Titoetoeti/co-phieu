import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import warnings
import time
import base64
import os

# --- THƯ VIỆN AI ---
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
from pmdarima import auto_arima

# ==============================================================================
# 1. CẤU HÌNH & HÀM ĐỌC DỮ LIỆU "NGUYÊN BẢN"
# ==============================================================================
warnings.filterwarnings("ignore")
st.set_page_config(page_title="PIXEL TRADER RAW", layout="wide", page_icon="💾")
plt.style.use('dark_background') 

DATA_FILE = "Data_1.xlsx - Tong_Hop_log_return.csv"

@st.cache_data
def load_raw_data(filepath):
    try:
        # Đọc file thuần túy
        df = pd.read_csv(filepath)
        
        # 1. Xóa khoảng trắng thừa trong tên cột (nếu có)
        df.columns = df.columns.str.strip()
        
        # 2. Tìm cột Date
        date_col = next((c for c in df.columns if c.lower() == 'date'), None)
        if not date_col: return None
        
        # 3. Ép kiểu ngày tháng NHƯNG KHÔNG ĐỔI MÚI GIỜ
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # 4. Sắp xếp tăng dần
        df = df.sort_values(by=date_col)
        
        # 5. Set Index
        df.set_index(date_col, inplace=True)
        
        # [QUAN TRỌNG] Loại bỏ Timezone để tránh bị lệch ngày
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        return df
    except Exception:
        return None

# --- INTRO (GIỮ NGUYÊN) ---
def show_intro_video(video_file, duration=8):
    if 'intro_done' not in st.session_state: st.session_state['intro_done'] = False
    if st.session_state['intro_done']: return
    if os.path.exists(video_file):
        try:
            with open(video_file, "rb") as f: v = base64.b64encode(f.read()).decode()
            st.markdown(f"""<style>.stApp {{overflow:hidden}} #intro {{position:fixed;top:0;left:0;width:100%;height:100%;background:#000;z-index:999}}</style><div id="intro"><video style="width:100%;height:100%;object-fit:cover" autoplay muted playsinline><source src="data:video/mp4;base64,{v}" type="video/mp4"></video></div>""", unsafe_allow_html=True)
            time.sleep(duration); st.empty(); st.session_state['intro_done'] = True; st.rerun()
        except: st.session_state['intro_done'] = True
    else: st.session_state['intro_done'] = True

show_intro_video("intro1.mp4", duration=6)

# ==============================================================================
# 2. CORE AI (GIỮ NGUYÊN LOGIC TÍNH TOÁN)
# ==============================================================================
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length]); y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def get_forecast(series_data, model_type, test_size, window_size, future_days):
    # series_data lúc này đã là dữ liệu SẠCH, KHÔNG FILL ẢO
    train = series_data.iloc[:-test_size]
    test = series_data.iloc[-test_size:]
    preds = pd.Series(index=test.index, dtype='float64')
    future_series = pd.Series(dtype='float64')
    info = ""

    # (Logic dự báo giữ nguyên như cũ để đảm bảo tính năng)
    try:
        if model_type == "ARIMA":
            model = auto_arima(train, start_p=1, start_q=1, max_p=3, max_q=3, seasonal=False, error_action='ignore')
            preds[:] = model.predict(n_periods=len(test)).values
            if future_days > 0:
                full_model = auto_arima(series_data, start_p=1, start_q=1, max_p=3, max_q=3, seasonal=False)
                future_series = pd.Series(full_model.predict(n_periods=future_days).values, index=pd.bdate_range(start=series_data.index[-1], periods=future_days+1)[1:])
            info = f"ARIMA{model.order}"
            
        elif model_type == "XGBoost":
            def mk_lags(s, w):
                d = pd.DataFrame(s); 
                for i in range(1, w+1): d[f'l{i}'] = d.iloc[:,0].shift(i)
                return d.dropna()
            df_l = mk_lags(series_data, window_size)
            X, y = df_l.drop(columns=[df_l.columns[0]]), df_l.iloc[:,0]
            X_tr, X_te = X.iloc[:-test_size], X.iloc[-test_size:]
            y_tr = y.iloc[:-test_size]
            m = XGBRegressor(n_estimators=100); m.fit(X_tr, y_tr)
            preds[:] = m.predict(X_te)
            if future_days > 0:
                m.fit(X, y)
                curr = list(X.iloc[-1].values); fut = []
                for _ in range(future_days):
                    p = m.predict(np.array(curr).reshape(1,-1))[0]; fut.append(p); curr.pop(-1); curr.insert(0, p)
                future_series = pd.Series(fut, index=pd.bdate_range(start=series_data.index[-1], periods=future_days+1)[1:])
            info = "XGBoost"

        elif model_type in ["LSTM", "GRU"]:
            sc = MinMaxScaler((0,1)); scaled = sc.fit_transform(series_data.values.reshape(-1,1))
            X, y = create_sequences(scaled, window_size)
            split = len(X) - test_size
            if split > 0:
                Xt, yt = X[:split], y[:split]; Xte = X[split:]
                Xt = Xt.reshape((Xt.shape[0], Xt.shape[1], 1)); Xte = Xte.reshape((Xte.shape[0], Xte.shape[1], 1))
                m = Sequential()
                ly = LSTM if model_type=='LSTM' else GRU
                m.add(ly(50, return_sequences=True, input_shape=(Xt.shape[1],1))); m.add(Dropout(0.2))
                m.add(ly(50)); m.add(Dropout(0.2)); m.add(Dense(1)); m.compile('adam', 'mse')
                m.fit(Xt, yt, epochs=15, batch_size=16, verbose=0)
                preds.iloc[:] = sc.inverse_transform(m.predict(Xte, verbose=0)).flatten()[-len(preds):]
                if future_days > 0:
                    curr = scaled[-window_size:].reshape(1, window_size, 1); fut = []
                    for _ in range(future_days):
                        p = m.predict(curr, verbose=0)[0,0]; fut.append(p); curr = np.append(curr[:,1:,:], [[[p]]], axis=1)
                    future_series = pd.Series(sc.inverse_transform(np.array(fut).reshape(-1,1)).flatten(), index=pd.bdate_range(start=series_data.index[-1], periods=future_days+1)[1:])
            info = model_type

    except: preds[:] = np.nan; info="Err"
    return preds, future_series, info

# ==============================================================================
# 3. GIAO DIỆN CHÍNH & DEBUG AREA
# ==============================================================================
st.markdown("<h1>PIXEL TRADER RAW</h1>", unsafe_allow_html=True)
st.markdown("""
<style>
.stApp {background:#0d0d0d; color:#00ff41; font-family:'VT323', monospace; font-size:18px}
h1 {font-family:'Press Start 2P'; text-align:center; color:#00ff41}
.debug-box {border: 1px solid #ffcc00; padding: 10px; color: #ffcc00; font-family: monospace;}
</style>
""", unsafe_allow_html=True)

df_full = load_raw_data(DATA_FILE)

if df_full is None: st.error(f"❌ K lỗi file {DATA_FILE}"); st.stop()

# --- KHU VỰC "SOI" DỮ LIỆU (DEBUGGER) ---
with st.expander("🔍 CLICK ĐỂ SOI DỮ LIỆU THỰC TẾ (CHECK VỚI EXCEL)", expanded=True):
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.write("👉 **3 Dòng ĐẦU TIÊN trong file:**")
        st.dataframe(df_full.head(3))
    with col_d2:
        st.write("👉 **3 Dòng CUỐI CÙNG trong file (Check kỹ ngày và giá):**")
        st.dataframe(df_full.tail(3))
    
    st.caption("⚠️ Lưu ý: Nếu bảng trên hiển thị khác Excel, hãy kiểm tra lại file CSV của bạn có dòng trống ở cuối hay không.")

tickers = list(df_full.columns)
with st.container():
    c1, c2 = st.columns([1, 2])
    with c1:
        ticker = st.selectbox("CHỌN MÃ", tickers)
        model = st.selectbox("MODEL", ["ARIMA", "XGBoost", "LSTM", "GRU"])
        
        # CHỌN PHẠM VI NGÀY CỨNG (KHÔNG AUTO)
        min_d, max_d = df_full.index.min().date(), df_full.index.max().date()
        st.write("---")
        d_range = st.date_input("KHOẢNG THỜI GIAN", [min_d, max_d], min_value=min_d, max_value=max_d)
        
        btn = st.button(">> CHẠY LOGIC <<")

if btn:
    if len(d_range) == 2:
        start_date, end_date = d_range
        # Cắt dữ liệu thô, KHÔNG FILL
        data = df_full[ticker].loc[str(start_date):str(end_date)].dropna()
        
        if data.empty: st.error("Không có dữ liệu trong khoảng này!"); st.stop()
        
        # HIỂN THỊ GIÁ CUỐI CÙNG CHÍNH XÁC
        last_date = data.index[-1].strftime('%d/%m/%Y')
        last_price = data.iloc[-1]
        
        st.markdown(f"""
        <div class='debug-box'>
            DATA CHECKPOINT:<br>
            • Ngày cuối cùng Code lấy được: <b>{last_date}</b><br>
            • Giá trị tại ngày đó: <b>{last_price:,.4f}</b><br>
            (Hãy so sánh 2 số này với Excel của bạn ngay bây giờ!)
        </div>
        """, unsafe_allow_html=True)
        
        # Chạy dự báo
        with st.spinner("AI Computing..."):
            # Chỉ fillna nhẹ khi đưa vào model để tránh crash, nhưng không ảnh hưởng data hiển thị gốc
            model_data = data.fillna(method='ffill') 
            preds, fut, info = get_forecast(model_data, model, 30, 30, 30)
            
            # Vẽ biểu đồ
            fig = go.Figure()
            # Vẽ dữ liệu GỐC (Có lỗ hổng thì để lỗ hổng, k tự vẽ dây nối)
            fig.add_trace(go.Scatter(x=data.index, y=data.values, name='DATA GỐC (EXCEL)', line=dict(color='#888')))
            fig.add_trace(go.Scatter(x=preds.index, y=preds.values, name='AI BACKTEST', line=dict(color='#f0f', dash='dot')))
            if not fut.empty:
                fig.add_trace(go.Scatter(x=fut.index, y=fut.values, name='TƯƠNG LAI', line=dict(color='#ff0', width=2)))
                
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)
            
            if not fut.empty:
                st.write("Bảng giá Tương lai:"); st.dataframe(fut.to_frame("Dự báo").T)
    else:
        st.error("Vui lòng chọn đủ ngày bắt đầu và kết thúc.")
