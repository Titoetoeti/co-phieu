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

# Tắt cảnh báo
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Stock Forecast App", layout="wide")

# ==============================================================================
# 1. MODULE TỐI ƯU HÓA THAM SỐ
# ==============================================================================
def find_optimal_params(train_data, model_type, seasonal_periods=None):
    bounds_limit = (0.01, 0.99)
    
    def loss_function(params):
        try:
            if model_type == 'SES':
                model = SimpleExpSmoothing(train_data).fit(smoothing_level=params[0], optimized=False)
            elif model_type == 'Holt':
                model = ExponentialSmoothing(train_data, trend='add', seasonal=None, damped_trend=False).fit(
                    smoothing_level=params[0], smoothing_trend=params[1], optimized=False)
            elif model_type == 'Holt-Winters':
                model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=seasonal_periods).fit(
                    smoothing_level=params[0], smoothing_trend=params[1], smoothing_seasonal=params[2], optimized=False)
            
            return np.sqrt(mean_squared_error(train_data, model.fittedvalues))
        except:
            return 1e10

    if model_type == 'SES':
        initial_guess = [0.5]
        bounds = [bounds_limit]
    elif model_type == 'Holt':
        initial_guess = [0.5, 0.1]
        bounds = [bounds_limit, bounds_limit]
    elif model_type == 'Holt-Winters':
        initial_guess = [0.5, 0.1, 0.1]
        bounds = [bounds_limit, bounds_limit, bounds_limit]
    else:
        return []

    result = minimize(loss_function, initial_guess, bounds=bounds, method='L-BFGS-B')
    return result.x

# ==============================================================================
# 2. GIAO DIỆN VÀ XỬ LÝ CHÍNH
# ==============================================================================

st.title("📈 Ứng Dụng Dự Báo Giá Cổ Phiếu Chuyên Sâu")
st.markdown("---")

# --- SIDEBAR: INPUT NGƯỜI DÙNG ---
st.sidebar.header("Cấu hình Dự báo")

ticker = st.sidebar.text_input("Nhập mã cổ phiếu (Ví dụ: AAPL, TSLA, VNM.VN):", value="AAPL")

freq_option = st.sidebar.selectbox(
    "Chọn khung thời gian dữ liệu:",
    ("Ngày (Daily)", "Tháng (Monthly)", "Quý (Quarterly)")
)

model_option = st.sidebar.selectbox(
    "Chọn kỹ thuật dự báo:",
    ("Naive (Ngây thơ)", "Moving Average (Trung bình trượt)", "Simple Exponential Smoothing (SES)", 
     "Holt's Linear (Trend)", "Holt-Winters (Trend + Seasonality)")
)

window_size = 0
if model_option == "Moving Average (Trung bình trượt)":
    window_size = st.sidebar.slider("Chọn cửa sổ trượt (Window):", min_value=2, max_value=50, value=3)

test_size = st.sidebar.slider("Số điểm dữ liệu dùng để Test (Backtest):", min_value=4, max_value=60, value=12)

if st.sidebar.button("🚀 Phân tích & Dự báo"):
    
    with st.spinner('Đang tải và xử lý dữ liệu...'):
        try:
            # Tải dữ liệu
            df = yf.download(ticker, period="5y", progress=False)
            
            if df.empty:
                st.error("Không tìm thấy dữ liệu cổ phiếu. Vui lòng kiểm tra lại mã.")
                st.stop()
            
            # --- PHẦN SỬA LỖI Ở ĐÂY ---
            # Kiểm tra xem có cột 'Adj Close' hay không, nếu không thì dùng 'Close'
            # Đôi khi yfinance trả về MultiIndex, cần xử lý flat lại
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if 'Adj Close' in df.columns:
                data = df['Adj Close']
            elif 'Close' in df.columns:
                data = df['Close']
            else:
                # Nếu không tìm thấy tên cột quen thuộc, lấy cột đầu tiên
                data = df.iloc[:, 0]
            # ---------------------------
            
            # Resample dữ liệu
            if freq_option == "Tháng (Monthly)":
                data
