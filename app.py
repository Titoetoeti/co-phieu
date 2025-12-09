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
# 1. MODULE TỐI ƯU HÓA THAM SỐ (DỰA TRÊN CODE PDF)
# ==============================================================================
# Logic dựa trên hàm find_optimal_params trong file PDF 
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

    # Thiết lập giá trị khởi tạo và biên [cite: 556-562, 617-619]
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

    # Chạy tối ưu hóa L-BFGS-B [cite: 564]
    result = minimize(loss_function, initial_guess, bounds=bounds, method='L-BFGS-B')
    return result.x

# ==============================================================================
# 2. GIAO DIỆN VÀ XỬ LÝ CHÍNH
# ==============================================================================

st.title("📈 Ứng Dụng Dự Báo Giá Cổ Phiếu Chuyên Sâu")
st.markdown("---")

# --- SIDEBAR: INPUT NGƯỜI DÙNG ---
st.sidebar.header("Cấu hình Dự báo")

# 1. Nhập mã cổ phiếu
ticker = st.sidebar.text_input("Nhập mã cổ phiếu (Ví dụ: AAPL, TSLA, VNM.VN):", value="AAPL")

# 2. Chọn khung thời gian (Daily/Monthly/Quarterly)
# Mapping logic resample theo file PDF [cite: 619, 634]
freq_option = st.sidebar.selectbox(
    "Chọn khung thời gian dữ liệu:",
    ("Ngày (Daily)", "Tháng (Monthly)", "Quý (Quarterly)")
)

# 3. Chọn mô hình dự báo
model_option = st.sidebar.selectbox(
    "Chọn kỹ thuật dự báo:",
    ("Naive (Ngây thơ)", "Moving Average (Trung bình trượt)", "Simple Exponential Smoothing (SES)", 
     "Holt's Linear (Trend)", "Holt-Winters (Trend + Seasonality)")
)

# Tham số phụ cho MA
window_size = 0
if model_option == "Moving Average (Trung bình trượt)":
    window_size = st.sidebar.slider("Chọn cửa sổ trượt (Window):", min_value=2, max_value=50, value=3)

# Số lượng điểm dự báo kiểm thử (Test size)
test_size = st.sidebar.slider("Số điểm dữ liệu dùng để Test (Backtest):", min_value=4, max_value=60, value=12)

# Nút chạy
if st.sidebar.button("🚀 Phân tích & Dự báo"):
    
    # --- BƯỚC 1: TẢI VÀ XỬ LÝ DỮ LIỆU ---
    with st.spinner('Đang tải và xử lý dữ liệu...'):
        try:
            # Tải dữ liệu từ Yahoo Finance thay vì upload file để linh hoạt hơn
            df = yf.download(ticker, period="5y")
            if df.empty:
                st.error("Không tìm thấy dữ liệu cổ phiếu. Vui lòng kiểm tra lại mã.")
                st.stop()
            
            # Lấy giá đóng cửa điều chỉnh (Adj Close) như file PDF [cite: 435]
            data = df['Adj Close']
            
            # Resample dữ liệu theo yêu cầu 
            if freq_option == "Tháng (Monthly)":
                data = data.resample('M').last()
                seasonal_p = 12
            elif freq_option == "Quý (Quarterly)":
                data = data.resample('Q').last()
                seasonal_p = 4
            else: # Daily
                data = data.asfreq('B').fillna(method='ffill') # Business days
                seasonal_p = 5 # Tuần làm việc 5 ngày [cite: 529]

            # Chia Train/Test [cite: 576, 620]
            train = data.iloc[:-test_size]
            test = data.iloc[-test_size:]
            
            st.success(f"Đã tải dữ liệu {ticker}. Kích thước Train: {len(train)}, Test: {len(test)}")
            
        except Exception as e:
            st.error(f"Lỗi xử lý dữ liệu: {e}")
            st.stop()

    # --- BƯỚC 2: CHẠY MÔ HÌNH DỰ BÁO ---
    st.subheader(f"Kết quả Dự báo: {model_option}")
    
    predictions = pd.Series(index=test.index)
    params_info = ""
    
    try:
        # A. Naive Model 
        if model_option == "Naive (Ngây thơ)":
            # Dự báo t = thực tế t-1
            pred_values = pd.concat([train.iloc[-1:], test[:-1]]).values
            predictions[:] = pred_values.ravel() # Flatten array
            params_info = "Dùng giá trị phiên trước đó"

        # B. Moving Average [cite: 590, 593]
        elif model_option == "Moving Average (Trung bình trượt)":
            rolling_ma = data.rolling(window=window_size).mean().shift(1)
            predictions = rolling_ma.loc[test.index]
            params_info = f"Window size = {window_size}"

        # C. Simple Exponential Smoothing (SES) [cite: 600-604]
        elif model_option == "Simple Exponential Smoothing (SES)":
            alpha_opt = find_optimal_params(train, 'SES')[0]
            model = SimpleExpSmoothing(train).fit(smoothing_level=alpha_opt, optimized=False)
            # Forecast cho tập test
            predictions = model.forecast(len(test))
            params_info = f"Alpha tối ưu = {alpha_opt:.4f}"

        # D. Holt's Linear [cite: 608, 622]
        elif model_option == "Holt's Linear (Trend)":
            params = find_optimal_params(train, 'Holt')
            model = ExponentialSmoothing(train, trend='add', seasonal=None, damped_trend=False).fit(
                smoothing_level=params[0], smoothing_trend=params[1], optimized=False)
            predictions = model.forecast(len(test))
            params_info = f"Alpha={params[0]:.4f}, Beta={params[1]:.4f}"

        # E. Holt-Winters [cite: 609, 623]
        elif model_option == "Holt-Winters (Trend + Seasonality)":
            params = find_optimal_params(train, 'Holt-Winters', seasonal_periods=seasonal_p)
            model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=seasonal_p).fit(
                smoothing_level=params[0], smoothing_trend=params[1], smoothing_seasonal=params[2], optimized=False)
            predictions = model.forecast(len(test))
            params_info = f"Alpha={params[0]:.2f}, Beta={params[1]:.2f}, Gamma={params[2]:.2f}"

    except Exception as e:
        st.error(f"Mô hình không hội tụ hoặc lỗi tính toán: {e}")
        st.stop()

    # --- BƯỚC 3: ĐÁNH GIÁ VÀ HIỂN THỊ ---
    
    # 1. Tính toán Metrics [cite: 610, 624]
    # Làm sạch NaN (đối với MA)
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

    # 2. Vẽ biểu đồ [cite: 612, 625, 641]
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Vẽ Train (chỉ lấy phần cuối cho đỡ rối)
    display_train = 100 if len(train) > 100 else len(train)
    ax.plot(train.index[-display_train:], train.iloc[-display_train:], label='Dữ liệu Huấn luyện (Train)', color='gray', alpha=0.5)
    
    # Vẽ Test (Actual)
    ax.plot(test.index, test, label='Thực tế (Actual)', color='black', linewidth=2)
    
    # Vẽ Dự báo (Forecast)
    ax.plot(test.index, predictions, label=f'Dự báo ({model_option})', color='red', linestyle='--', linewidth=2, marker='o')
    
    ax.set_title(f'Biểu đồ So sánh Thực tế vs Dự báo: {ticker}', fontsize=16)
    ax.set_ylabel('Giá Cổ phiếu')
    ax.set_xlabel('Thời gian')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

    # --- BƯỚC 4: DỮ LIỆU CHI TIẾT ---
    with st.expander("Xem dữ liệu chi tiết"):
        results_df = pd.DataFrame({'Thực tế': test, 'Dự báo': predictions})
        results_df['Sai lệch'] = results_df['Thực tế'] - results_df['Dự báo']
        st.dataframe(results_df)

else:
    st.info("👈 Vui lòng nhập mã cổ phiếu và nhấn nút 'Phân tích & Dự báo' ở thanh bên trái.")
