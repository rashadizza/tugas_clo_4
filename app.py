import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import ta_py as ta
import streamlit as st
