import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import ta_py as ta
import streamlit as st
import base64
import os

# Define constants
scaler = MinMaxScaler()
period = 5

# Load data function
def load_data():
    data = yf.download("ANTM.JK", start="2021-01-01", end="2023-12-31")
    data = data.interpolate(method='linear')
    data['return'] = [None] + [np.log(i / j) for (i, j) in zip(data['Close'][1:], data['Close'][0:-1])]
    data['sma'] = [None for _ in range(period - 1)] + list(ta.sma(data['Adj Close'], period))
    data['ema'] = [None for _ in range(period - 1)] + list(ta.ema(data['Adj Close'], period))
    data['rsi'] = [None for _ in range(period - 1)] + list(ta.rsi(data['Adj Close'], period))
    data = data.dropna()
    features = ['sma', 'ema', 'rsi']
    data[features] = scaler.fit_transform(data[features])
    return data, features

# Train model function
def train_model(data, features, step):
    X = data[features].iloc[:step].values
    y = data["return"].iloc[:step].values
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

# Evaluate performance function
def evaluate_performance(predicted_prices, actual_prices):
    mse = mean_squared_error(actual_prices, predicted_prices)
    r2 = r2_score(actual_prices, predicted_prices)
    return mse, r2

# Calculate volatility function
def calculate_volatility(data):
    daily_returns = np.log(data["Adj Close"].pct_change() + 1)
    annualized_volatility = daily_returns.std() * np.sqrt(252)
    return annualized_volatility

# GBM simulation function
def gbm_sim(spot_price, volatility, time_horizon, steps, model, features, data, num_simulations):
    dt = 1
    actual = [spot_price] + list(data['Adj Close'].iloc[:].values)
    drift = model.predict(scaler.fit_transform(data.loc[:][features]))
    all_paths = []

    for _ in range(num_simulations):
        paths = [spot_price]
        for i in range(1, time_horizon + 1):
            if i >= len(data):
                break
            paths.append(actual[i] * np.exp((drift[i-1] - 0.5 * (volatility/252)**2) * dt + (volatility/252) * np.random.normal(scale=np.sqrt(1/252))))
        all_paths.append(paths)

    return all_paths, drift

# Calculate confidence intervals, monte carlo
def calculate_confidence_intervals(paths, confidence_level=0.95):
    lower_percentile = (1.0 - confidence_level) / 2.0 * 100
    upper_percentile = (1.0 + confidence_level) / 2.0 * 100
    lower_bounds = np.percentile(paths, lower_percentile, axis=0)
    upper_bounds = np.percentile(paths, upper_percentile, axis=0)
    return lower_bounds, upper_bounds

# Save results to CSV function
def save_results_to_csv(results_df):
    csv_file = st.button('Save results to CSV')
    if csv_file:
        file_path = st.text_input('Enter file path to save CSV', 'predicted_results.csv')
        with open(file_path, 'w') as f:
            results_df.to_csv(f, index=False)
        st.success(f'Results saved to {file_path}')
        st.markdown(get_binary_file_downloader_html(file_path, 'CSV file'), unsafe_allow_html=True)

# Function to download CSV
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:file/csv;base64,{bin_str}" download="{os.path.basename(bin_file)}">{file_label}</a>'
    return href

# Streamlit web
st.set_page_config(page_title="Prediksi Harga Saham Aneka Tambang Tbk PT", page_icon=":chart_with_upwards_trend:")
st.title('Prediksi Harga Saham (ANTM)')
st.header('Disusun Oleh Grup 7')
st.subheader(
  """
  -
  -
  -
  -
  """
)

# Load data
data, features = load_data()
steps = int(len(data) / 2)
model = train_model(data, features, steps)
spot_price = data["Adj Close"].iloc[steps - 1]
volatility = calculate_volatility(data.iloc[0:steps])

# Sidebar inputs
num_simulations = st.sidebar.number_input('Number of Simulations', min_value=1, max_value=100, value=5)
time_horizon = st.sidebar.number_input('Time Horizon (days)', min_value=1, max_value=252, value=252)

# Simulate paths
simulated_paths, drifts = gbm_sim(spot_price, volatility, time_horizon, steps, model, features, data.iloc[steps:], num_simulations)

# Trim data['Adj Close'] to match simulated_paths length
actual_prices = data['Adj Close'][steps - 1:steps - 1 + len(simulated_paths[0])].values

# Evaluate model performance
mse, r2 = evaluate_performance(simulated_paths[0], actual_prices)
st.write(f"Mean Squared Error (MSE): {mse:.4f}")
st.write(f"R-squared (R²): {r2:.4f}")

# Plot results and confidence intervals
labels = ['Predicted Drift', 'Actual Drift', 'Absolute Error']
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].plot(drifts, label='Predicted Drift', color='blue')
ax[0].set_title('Predicted Drift')
ax[0].set_xlabel('Time Step')
ax[0].legend()

ax[1].plot(data['return'].iloc[steps:].values, label='Actual Drift', color='green')
ax[1].set_title('Actual Drift')
ax[1].set_xlabel('Time Step')
ax[1].legend()

ax[2].plot([abs(i - j) for (i, j) in zip(drifts, data['return'].iloc[steps:].values)], label='Absolute Error', color='red')
ax[2].set_title('Absolute Error')
ax[2].set_xlabel('Time Step')
ax[2].legend()

plt.tight_layout()
st.pyplot(fig)

# Calculate confidence intervals
all_simulated_paths = np.array(simulated_paths)
lower_bounds, upper_bounds = calculate_confidence_intervals(all_simulated_paths)

# Display predicted prices in a table for all simulations
results_df = pd.DataFrame(simulated_paths).transpose()
results_df.columns = [f"Simulation {i+1}" for i in range(num_simulations)]
st.write(results_df)

save_results_to_csv(results_df)

# Plot results with confidence intervals
index = data.index[steps - 1:steps - 1 + len(simulated_paths[0])]
fig, ax = plt.subplots(figsize=(10, 6))
for i, path in enumerate(simulated_paths):
    ax.plot(index, path, label=f'Predicted {i+1}', alpha=0.3)
ax.plot(index, actual_prices, label='Actual', color='black', linewidth=2)
ax.fill_between(index, lower_bounds, upper_bounds, color='grey', alpha=0.2, label='95% Confidence Interval')
ax.set_xlabel("Time Step")
ax.set_ylabel("Stock Price")
ax.set_title("Simulated Stock Price Paths")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# Plot absolute and relative errors
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(index, [abs(i - j) for (i, j) in zip(simulated_paths[0], actual_prices)], '.-')
ax[0].set_title('Absolute Error of Prediction Price')
ax[1].plot(index, [abs(i - j) / j * 100 for (i, j) in zip(simulated_paths[0], actual_prices)], '.-')
ax[1].set_title('Relative Absolute Error of Prediction Price (in %)')
_ = [ax[i].tick_params(axis='x', labelrotation=40) for i in [0, 1]]
st.pyplot(fig)
