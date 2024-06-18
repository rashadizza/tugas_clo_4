import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import ta_py as ta
import streamlit as st

# Define constants
scaler = MinMaxScaler()
period = 5

# Load data function
def load_data():
    data = yf.download("BBCA.JK", start="2021-01-01", end="2023-12-31")
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

# Calculate confidence intervals
def calculate_confidence_intervals(paths, confidence_level=0.95):
    lower_percentile = (1.0 - confidence_level) / 2.0 * 100
    upper_percentile = (1.0 + confidence_level) / 2.0 * 100
    lower_bounds = np.percentile(paths, lower_percentile, axis=0)
    upper_bounds = np.percentile(paths, upper_percentile, axis=0)
    return lower_bounds, upper_bounds

# Streamlit app
st.title('Prediksi Harga Saham')
st.write('Disusun Oleh Grup 7')
st.write(
  """
  - rashad
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

# Calculate confidence intervals
all_simulated_paths = np.array(simulated_paths)
lower_bounds, upper_bounds = calculate_confidence_intervals(all_simulated_paths)

# Plot results
index = data.index[steps - 1:steps - 1 + len(simulated_paths[0])]

# Plot simulated and actual stock prices
plt.figure(figsize=(10, 6))
plt.plot(index, simulated_paths[0], label='Predicted')
plt.plot(index, actual_prices, label='Actual', color='black', linewidth=2)
plt.xlabel("Time Step")
plt.ylabel("Stock Price")
plt.title("Simulated vs Actual Stock Price Paths")
plt.grid(True)
plt.legend()
st.pyplot()

# Plot drifts and errors
labels = ['Predicted Drift', 'Actual Drift', 'Absolute Error']
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(drifts, label='Predicted Drift')
ax[0].set_title('Predicted Drift')
ax[0].set_xlabel('Time Step')
ax[0].set_ylabel('Drift Value')
ax[0].legend()

ax[1].plot(data['return'].iloc[steps:].values, label='Actual Drift', color='green')
ax[1].set_title('Actual Drift')
ax[1].set_xlabel('Time Step')
ax[1].set_ylabel('Drift Value')
ax[1].legend()

ax[2].plot(index, [abs(i - j) for (i, j) in zip(drifts, data['return'].iloc[steps:].values)], '.-')
ax[2].set_title('Absolute Error of Drift Prediction')
ax[2].set_xlabel('Time Step')
ax[2].set_ylabel('Absolute Error')
ax[2].tick_params(axis='x', rotation=40)

plt.tight_layout()
st.pyplot(fig)
