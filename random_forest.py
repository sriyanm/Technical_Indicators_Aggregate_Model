import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# fetch stock data
def fetch_data(ticker):
    data = yf.download(ticker, period='5y')
    return data

# calculate technical indicators
def calculate_indicators(data):
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    
    delta = data['Close'].diff().dropna()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    
    roll_up = up.ewm(com=13-1, adjust=False).mean()
    roll_down = down.ewm(com=13-1, adjust=False).mean().abs()
    
    RS = roll_up / roll_down
    RSI = 100.0 - (100.0 / (1.0 + RS))
    
    data['RSI'] = RSI
    
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # generate signals based on indicators
    data['SMA_Signal'] = np.where(data['SMA_50'] > data['SMA_200'], 1, 0)
    data['RSI_Signal'] = np.where(data['RSI'] < 30, 1, np.where(data['RSI'] > 70, -1, 0))
    data['MACD_Signal'] = np.where(data['MACD'] > data['Signal_Line'], 1, -1)
    
    # signal for model: Buy (1), Sell (-1), Hold (0)
    data['Final_Signal'] = np.where(data['Close'].shift(-1) > data['Close'], 1,
                                    np.where(data['Close'].shift(-1) < data['Close'], -1, 0))
    
    return data

# prepare data for model
def prepare_data(data):
    indicators = ['SMA_Signal', 'RSI_Signal', 'MACD_Signal']
    X = data[indicators].dropna()
    y = data['Final_Signal'].dropna()
    
    # ensure X and y have the same length
    min_len = min(len(X), len(y))
    X = X.head(min_len)
    y = y.head(min_len)
    
    return X, y

# train model w/ hyperparameter tuning
def train_model(X, y):
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    rf = RandomForestClassifier(random_state=42)
    
    rf_random = RandomizedSearchCV(estimator=rf,
                                   param_distributions=param_grid,
                                   n_iter=50,
                                   cv=3,
                                   verbose=2,
                                   random_state=42,
                                   n_jobs=-1)
    
    rf_random.fit(X, y)
    
    print("Best Parameters:", rf_random.best_params_)
    
    return rf_random.best_estimator_

# backtest
def backtest(data, signals):
    # ensure signals length matches data length
    if len(signals) > len(data):
        signals = signals[:len(data)]
    elif len(signals) < len(data):
        signals = np.pad(signals, (0, len(data) - len(signals)), mode='constant', constant_values=np.nan)
    
    data['Signal'] = signals
    
    data['Daily_Return'] = data['Close'].pct_change()
    
    data['Strategy_Return'] = data['Signal'].shift(1) * data['Daily_Return']
    
    data['Cumulative_Strategy_Return'] = (1 + data['Strategy_Return']).cumprod()
    data['Cumulative_Market_Return'] = (1 + data['Daily_Return']).cumprod()
    
    return data[['Cumulative_Strategy_Return', 'Cumulative_Market_Return']]


def plot_cumulative_returns(backtest_results):
    plt.figure(figsize=(10, 6))
    
    plt.plot(backtest_results.index, backtest_results['Cumulative_Strategy_Return'], label='Strategy Return', marker='o')
    plt.plot(backtest_results.index, backtest_results['Cumulative_Market_Return'], label='Market Return', marker='o')
    
    plt.title('Cumulative Returns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid()
    
    plt.show()

if __name__ == "__main__":
    ticker = 'WM'  
    data = fetch_data(ticker)
    
    data = calculate_indicators(data)
    
    X, y = prepare_data(data)
    
    model = train_model(X, y)
    
    signals = model.predict(data[['SMA_Signal', 'RSI_Signal', 'MACD_Signal']].dropna())
    
    backtest_results = backtest(data.dropna(), signals)
    
    print(backtest_results.tail())
    
    plot_cumulative_returns(backtest_results)
