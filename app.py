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

st.title("Dự Báo Giá Cổ Phiếu")
st.markdown("---")

import streamlit as st
import yfinance as yf

# ==============================================================================
# 1. SETUP GIAO DIỆN DARK MODE & TECH STYLE (CSS)
# ==============================================================================
# Lưu ý: Đặt dòng này ngay đầu file app.py, sau các lệnh import
st.markdown("""
    <style>
        /* 1. Nền tổng thể và Sidebar màu đen sâu */
        .stApp {
            background-color: #0e1117; /* Màu nền chính tối */
        }
        [data-testid="stSidebar"] {
            background-color: #000000; /* Sidebar đen tuyền */
            border-right: 1px solid #222222; /* Viền mỏng tinh tế */
        }

        /* 2. Tùy chỉnh Tiêu đề Sidebar */
        .sidebar-title {
            color: #ffffff;
            font-family: 'Courier New', monospace; /* Font kiểu code */
            font-size: 24px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 20px;
            border-bottom: 2px solid #00ff41; /* Gạch chân xanh neon */
            padding-bottom: 10px;
        }

        /* 3. Tùy chỉnh Input và Selectbox */
        div[data-baseweb="input"] > div {
            background-color: #111111 !important;
            color: #00ff41 !important; /* Chữ xanh neon khi gõ */
            border: 1px solid #333333;
            border-radius: 4px;
        }
        div[data-baseweb="select"] > div {
            background-color: #111111 !important;
            color: white !important;
            border: 1px solid #333333;
            border-radius: 4px;
        }
        label {
            color: #aaaaaa !important; /* Màu nhãn xám nhạt hiện đại */
            font-size: 12px !important;
            text-transform: uppercase;
            font-weight: 600;
        }

        /* 4. Nút bấm phong cách Cyberpunk */
        div.stButton > button {
            width: 100%;
            background: linear-gradient(90deg, #000000, #1a1a1a);
            color: #00ff41; /* Chữ xanh neon */
            border: 1px solid #00ff41;
            padding: 12px 24px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-weight: bold;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        div.stButton > button:hover {
            background-color: #00ff41;
            color: #000000;
            box-shadow: 0 0 15px rgba(0, 255, 65, 0.5); /* Hiệu ứng phát sáng */
            border-color: #00ff41;
        }
        
        /* 5. Slider */
        div[data-baseweb="slider"] div {
            background-color: #00ff41 !important; /* Thanh trượt xanh */
        }

    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CODE SIDEBAR ĐÃ TỐI GIẢN
# ==============================================================================

# Tạo tiêu đề thủ công bằng HTML để đẹp hơn st.header
st.sidebar.markdown('<div class="sidebar-title">⚡ STOCK.AI</div>', unsafe_allow_html=True)

# Nhập liệu (Dùng placeholder để gọn gàng hơn)
ticker = st.sidebar.text_input(
    "Mã tài sản (Symbol)", 
    value="META", 
    placeholder="VD: AAPL, BTC-USD..."
).upper()

st.sidebar.markdown("---") # Đường kẻ phân cách mờ

# Nhóm các cấu hình vào 2 cột nhỏ hoặc để trơn cho thoáng
col1, col2 = st.sidebar.columns(2)
with col1:
    freq_option = st.selectbox("Khung thời gian", ("Ngày", "Tháng", "Quý"))

with col2:
    # Logic mapping đơn giản để code gọn hơn
    model_map = {
        "Naive": "Naive", 
        "Moving Average": "MA", 
        "Simple Exponential Smoothing": "SES", 
        "Holt's Linear": "Holt", 
        "Holt-Winters": "HW"
    }
    # Hiển thị tên đầy đủ, nhưng lấy giá trị viết tắt để xử lý logic
    model_display = st.selectbox("Thuật toán", list(model_map.keys()))
    model_option = model_display # Giữ nguyên biến cũ để không hỏng code dưới

# Cấu hình nâng cao (Ẩn bớt để tối giản, chỉ hiện khi cần)
with st.sidebar.expander("⚙️ Cấu hình nâng cao", expanded=True):
    window_size = 0
    if model_option == "Moving Average":
        window_size = st.slider("Cửa sổ trượt (Window)", 2, 50, 3)
    
    test_size = st.slider("Backtest Size (Kỳ)", 4, 60, 12)

st.sidebar.markdown("<br>", unsafe_allow_html=True) # Khoảng trắng

# Nút bấm (Đã được CSS biến thành nút Cyberpunk)
if st.sidebar.button("KHỞI CHẠY PHÂN TÍCH"):
    
    with st.spinner('SYSTEM PROCESSING...'):
        try:
            # Code tải dữ liệu giữ nguyên
            df = yf.download(ticker, period="5y", progress=False)
            
            if df.empty:
                st.error("❌ DATA NOT FOUND.")
                st.stop()
            
            # ... (Phần xử lý tiếp theo của bạn giữ nguyên) ...
            
        except Exception as e:
            st.error(f"SYSTEM ERROR: {e}")
            
            # --- XỬ LÝ DỮ LIỆU CHỐNG LỖI (FIXED) ---
            # 1. Xử lý MultiIndex (trường hợp yfinance trả về 2 tầng cột)
            if isinstance(df.columns, pd.MultiIndex):
                # Chỉ lấy tầng tên cột, bỏ tầng mã chứng khoán
                df.columns = df.columns.get_level_values(0)

            # 2. Tìm cột giá phù hợp
            if 'Adj Close' in df.columns:
                data = df['Adj Close']
            elif 'Close' in df.columns:
                data = df['Close']
            else:
                # Nếu bí quá thì lấy cột số đầu tiên
                data = df.iloc[:, 0]
            
            # Đảm bảo data là Series 1 chiều, không phải DataFrame
            if isinstance(data, pd.DataFrame):
                data = data.iloc[:, 0]
            # ----------------------------------------
            
            # Resample dữ liệu
            if freq_option == "Tháng (Monthly)":
                data = data.resample('M').last()
                seasonal_p = 12
            elif freq_option == "Quý (Quarterly)":
                data = data.resample('Q').last()
                seasonal_p = 4
            else: # Daily
                data = data.asfreq('B').fillna(method='ffill')
                seasonal_p = 5

            # Chia Train/Test
            if len(data) < test_size + 5:
                 st.error(f"Dữ liệu quá ngắn ({len(data)} dòng) không đủ để dự báo.")
                 st.stop()

            train = data.iloc[:-test_size]
            test = data.iloc[-test_size:]
            
            st.success(f"Đã tải dữ liệu {ticker}. Kích thước Train: {len(train)}, Test: {len(test)}")
            
        except Exception as e:
            st.error(f"Chi tiết lỗi: {e}")
            st.stop()

    # --- BƯỚC 2: CHẠY MÔ HÌNH DỰ BÁO ---
    st.subheader(f"Kết quả Dự báo: {model_option}")
    
    predictions = pd.Series(index=test.index, dtype='float64')
    params_info = ""
    
    try:
        if model_option == "Naive (Ngây thơ)":
            pred_values = pd.concat([train.iloc[-1:], test[:-1]]).values
            predictions[:] = pred_values.ravel()
            params_info = "Dùng giá trị phiên trước đó"

        elif model_option == "Moving Average":
            rolling_ma = data.rolling(window=window_size).mean().shift(1)
            predictions = rolling_ma.loc[test.index]
            params_info = f"Window size = {window_size}"

        elif model_option == "Simple Exponential Smoothing (SES)":
            alpha_opt = find_optimal_params(train, 'SES')[0]
            model = SimpleExpSmoothing(train).fit(smoothing_level=alpha_opt, optimized=False)
            predictions = model.forecast(len(test))
            params_info = f"Alpha tối ưu = {alpha_opt:.4f}"

        elif model_option == "Holt's Linear (Trend)":
            params = find_optimal_params(train, 'Holt')
            model = ExponentialSmoothing(train, trend='add', seasonal=None, damped_trend=False).fit(
                smoothing_level=params[0], smoothing_trend=params[1], optimized=False)
            predictions = model.forecast(len(test))
            params_info = f"Alpha={params[0]:.4f}, Beta={params[1]:.4f}"

        elif model_option == "Holt-Winters (Trend + Seasonality)":
            params = find_optimal_params(train, 'Holt-Winters', seasonal_periods=seasonal_p)
            model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=seasonal_p).fit(
                smoothing_level=params[0], smoothing_trend=params[1], smoothing_seasonal=params[2], optimized=False)
            predictions = model.forecast(len(test))
            params_info = f"Alpha={params[0]:.2f}, Beta={params[1]:.2f}, Gamma={params[2]:.2f}"

    except Exception as e:
        st.error(f"Lỗi khi chạy mô hình: {e}")
        st.stop()

    # --- BƯỚC 3: ĐÁNH GIÁ VÀ HIỂN THỊ ---
    
    # Làm sạch NaN
    valid_idx = ~np.isnan(predictions) & ~np.isnan(test)
    if valid_idx.sum() > 0:
        rmse = np.sqrt(mean_squared_error(test[valid_idx], predictions[valid_idx]))
        mae = mean_absolute_error(test[valid_idx], predictions[valid_idx])
        mape = mean_absolute_percentage_error(test[valid_idx], predictions[valid_idx]) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RMSE", f"{rmse:.2f}")
        col2.metric("MAE", f"{mae:.2f}")
        col3.metric("MAPE (Sai số %)", f"{mape:.2f}%")
        col4.info(f"Tham số: {params_info}")
    else:
        st.warning("Không đủ dữ liệu để tính sai số.")

    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(14, 7))
    
    display_train = 100 if len(train) > 100 else len(train)
    ax.plot(train.index[-display_train:], train.iloc[-display_train:], label='Dữ liệu Huấn luyện (Train)', color='gray', alpha=0.5)
    ax.plot(test.index, test, label='Thực tế (Actual)', color='black', linewidth=2)
    ax.plot(test.index, predictions, label=f'Dự báo ({model_option})', color='red', linestyle='--', linewidth=2, marker='o')
    
    ax.set_title(f'Biểu đồ So sánh Thực tế vs Dự báo: {ticker}', fontsize=16)
    ax.set_ylabel('Giá Cổ phiếu')
    ax.set_xlabel('Thời gian')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

    with st.expander("Xem dữ liệu chi tiết"):
        results_df = pd.DataFrame({'Thực tế': test, 'Dự báo': predictions})
        results_df['Sai lệch'] = results_df['Thực tế'] - results_df['Dự báo']
        st.dataframe(results_df)

else:
    st.info("👈 Vui lòng nhập mã cổ phiếu và nhấn nút 'Phân tích & Dự báo' ở thanh bên trái.")

