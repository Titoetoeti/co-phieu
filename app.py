from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.api import SimpleExpSmoothing, ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")
app = Flask(__name__)

# ==========================================
# LOGIC TÍNH TOÁN (Dựa trên PDF)
# ==========================================

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        ticker = data.get('ticker')
        model_option = data.get('model')
        freq_option = data.get('timeframe')
        test_size = int(data.get('test_size', 12))

        # 1. Tải dữ liệu
        df = yf.download(ticker, period="5y", progress=False)
        if df.empty:
            return jsonify({'error': 'Không tìm thấy mã cổ phiếu'}), 400
        
        # Lấy cột Adj Close (nếu không có thì lấy Close)
        series_data = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
        
        # Resample logic
        if freq_option == 'monthly':
            series_data = series_data.resample('M').last()
            seasonal_p = 12
        elif freq_option == 'quarterly':
            series_data = series_data.resample('Q').last()
            seasonal_p = 4
        else: # daily
            series_data = series_data.asfreq('B').fillna(method='ffill')
            seasonal_p = 5

        # Chia Train/Test
        train = series_data.iloc[:-test_size]
        test = series_data.iloc[-test_size:]
        
        predictions = []
        params_info = ""

        # 2. Chạy Model
        if model_option == 'naive':
            pred_values = pd.concat([train.iloc[-1:], test[:-1]]).values.ravel()
            predictions = pred_values
            params_info = "Naive (Phiên trước)"

        elif model_option == 'ma':
            window = 3
            predictions = series_data.rolling(window=window).mean().shift(1).loc[test.index].values
            params_info = f"MA Window={window}"

        elif model_option == 'ses':
            alpha = find_optimal_params(train, 'SES')[0]
            model = SimpleExpSmoothing(train).fit(smoothing_level=alpha, optimized=False)
            predictions = model.forecast(len(test)).values
            params_info = f"α={alpha:.3f}"

        elif model_option == 'holt':
            params = find_optimal_params(train, 'Holt')
            model = ExponentialSmoothing(train, trend='add').fit(
                smoothing_level=params[0], smoothing_trend=params[1], optimized=False)
            predictions = model.forecast(len(test)).values
            params_info = f"α={params[0]:.3f}, β={params[1]:.3f}"

        elif model_option == 'hw':
            params = find_optimal_params(train, 'Holt-Winters', seasonal_periods=seasonal_p)
            model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=seasonal_p).fit(
                smoothing_level=params[0], smoothing_trend=params[1], smoothing_seasonal=params[2], optimized=False)
            predictions = model.forecast(len(test)).values
            params_info = f"α={params[0]:.2f}, β={params[1]:.2f}, γ={params[2]:.2f}"

        # 3. Tính Metrics
        # Xử lý NaN cho metrics
        clean_idx = ~np.isnan(test.values) & ~np.isnan(predictions)
        if np.sum(clean_idx) > 0:
            rmse = np.sqrt(mean_squared_error(test.values[clean_idx], predictions[clean_idx]))
            mae = mean_absolute_error(test.values[clean_idx], predictions[clean_idx])
            mape = mean_absolute_percentage_error(test.values[clean_idx], predictions[clean_idx]) * 100
        else:
            rmse, mae, mape = 0, 0, 0

        # Chuẩn bị dữ liệu gửi về JSON (phải chuyển numpy array thành list)
        response = {
            'labels': test.index.strftime('%Y-%m-%d').tolist(),
            'actual': np.where(np.isnan(test.values), None, test.values).tolist(),
            'forecast': np.where(np.isnan(predictions), None, predictions).tolist(),
            'rmse': f"{rmse:.2f}",
            'mae': f"{mae:.2f}",
            'mape': f"{mape:.2f}",
            'params': params_info
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)