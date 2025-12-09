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
# HÀM TỐI ƯU HÓA (GIỮ NGUYÊN)
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
        initial_guess = [0.5]; bounds = [bounds_limit]
    elif model_type == 'Holt':
        initial_guess = [0.5, 0.1]; bounds = [bounds_limit, bounds_limit]
    elif model_type == 'Holt-Winters':
        initial_guess = [0.5, 0.1, 0.1]; bounds = [bounds_limit, bounds_limit, bounds_limit]
    else:
        return []

    result = minimize(loss_function, initial_guess, bounds=bounds, method='L-BFGS-B')
    return result.x

# ==============================================================================
# GIAO DIỆN CHÍNH
# ==============================================================================

st.title("📈 Ứng Dụng Dự Báo Giá Cổ Phiếu (Phiên bản Fix Lỗi)")
st.markdown("---")

st.sidebar.header("Cấu hình")
ticker = st.sidebar.text_input("Nhập mã cổ phiếu:", value="AAPL")
freq_option = st.sidebar.selectbox("Khung thời gian:", ("Ngày (Daily)", "Tháng (Monthly)", "Quý (Quarterly)"))
model_option = st.sidebar.selectbox("Mô hình:", ("Naive", "Moving Average", "SES", "Holt's Linear", "Holt-Winters"))

window_size = 3
if model_option == "Moving Average":
    window_size = st.sidebar.slider("Cửa sổ trượt:", 2, 50, 3)
test_size = st.sidebar.slider("Số kỳ Test:", 4, 60, 12)

if st.sidebar.button("🚀 Chạy Dự báo"):
    with st.spinner('Đang tải dữ liệu...'):
        try:
            # 1. Tải dữ liệu
            df = yf.download(ticker, period="5y", progress=False)
            
            if df.empty:
                st.error(f"Không tìm thấy dữ liệu cho mã: {ticker}")
                st.stop()

            # --- DEBUG INFO (Hiện ra để kiểm tra) ---
            with st.expander("🔍 Kiểm tra dữ liệu thô (Debug)"):
                st.write("Dữ liệu gốc từ Yahoo:", df.head())
                st.write("Tên các cột:", df.columns.tolist())

            # 2. Xử lý MultiIndex (Vấn đề chính gây lỗi)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # 3. Chọn cột giá (Ưu tiên Adj Close -> Close -> Cột đầu tiên)
            # Chuyển tên cột về chữ thường để so sánh cho dễ
            df.columns = [str(c).lower().strip() for c in df.columns]
            
            col_name = None
            if 'adj close' in df.columns:
                col_name = 'adj close'
            elif 'close' in df.columns:
                col_name = 'close'
            else:
                col_name = df.columns[0] # Lấy cột đầu tiên nếu không tìm thấy tên
            
            data = df[col_name]

            # 4. Xử lý Timezone và kiểu dữ liệu
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            data = data.astype(float) # Ép kiểu số thực
            data = data.asfreq('B').fillna(method='ffill') # Lấp đầy ngày nghỉ
            data = data.dropna() # Xóa NaN còn sót

            # 5. Resample theo yêu cầu
            if freq_option == "Tháng (Monthly)":
                data = data.resample('M').last()
                seasonal_p = 12
            elif freq_option == "Quý (Quarterly)":
                data = data.resample('Q').last()
                seasonal_p = 4
            else:
                seasonal_p = 5

            # 6. Chia Train/Test
            if len(data) < test_size + 2 * seasonal_p:
                st.error("Dữ liệu quá ngắn để chạy mô hình này. Hãy chọn mã khác hoặc giảm số kỳ Test.")
                st.stop()

            train = data.iloc[:-test_size]
            test = data.iloc[-test_size:]

            # 7. CHẠY MÔ HÌNH
            predictions = pd.Series(index=test.index, dtype='float64')
            msg = ""

            if model_option == "Naive":
                pred_val = np.array([train.iloc[-1]] * len(test)) # Naive đơn giản: lấy giá cuối cùng
                predictions[:] = pred_val
                msg = "Naive (Last Value)"

            elif model_option == "Moving Average":
                rolling = data.rolling(window=window_size).mean().shift(1)
                predictions = rolling.loc[test.index]
                msg = f"MA Window={window_size}"

            elif model_option == "SES":
                alpha = find_optimal_params(train, 'SES')[0]
                model = SimpleExpSmoothing(train).fit(smoothing_level=alpha, optimized=False)
                predictions = model.forecast(len(test))
                msg = f"Alpha={alpha:.3f}"

            elif model_option == "Holt's Linear":
                p = find_optimal_params(train, 'Holt')
                model = ExponentialSmoothing(train, trend='add').fit(
                    smoothing_level=p[0], smoothing_trend=p[1], optimized=False)
                predictions = model.forecast(len(test))
                msg = f"Alpha={p[0]:.3f}, Beta={p[1]:.3f}"

            elif model_option == "Holt-Winters":
                p = find_optimal_params(train, 'Holt-Winters', seasonal_periods=seasonal_p)
                model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=seasonal_p).fit(
                    smoothing_level=p[0], smoothing_trend=p[1], smoothing_seasonal=p[2], optimized=False)
                predictions = model.forecast(len(test))
                msg = f"Params: {p}"

            # 8. HIỂN THỊ KẾT QUẢ
            # Xóa NaN trong dự báo (nếu có)
            valid_mask = ~np.isnan(predictions) & ~np.isnan(test)
            
            if valid_mask.sum() == 0:
                st.warning("Không tính được sai số (Dữ liệu dự báo toàn NaN).")
            else:
                rmse = np.sqrt(mean_squared_error(test[valid_mask], predictions[valid_mask]))
                mape = mean_absolute_percentage_error(test[valid_mask], predictions[valid_mask]) * 100
                
                c1, c2 = st.columns(2)
                c1.metric("RMSE", f"{rmse:.2f}")
                c2.metric("MAPE", f"{mape:.2f}%")
                st.info(f"Thông tin mô hình: {msg}")

            # Vẽ biểu đồ
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(train.index[-100:], train.iloc[-100:], label='Train', color='gray', alpha=0.5)
            ax.plot(test.index, test, label='Actual', color='black', linewidth=2)
            ax.plot(test.index, predictions, label='Forecast', color='red', linestyle='--', marker='o')
            ax.set_title(f"Dự báo: {ticker}")
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error("CÓ LỖI XẢY RA:")
            st.code(e) # Hiện chi tiết lỗi để dễ sửa
            st.stop()
