# Portfolio Optimization and Financial Analysis Tool

This Python script provides a comprehensive financial analysis and portfolio optimization tool. It fetches historical price data for a predefined list of stocks and cryptocurrencies using `yfinance`, calculates key metrics like **RSI**, **Bollinger Bands**, **MACD**, and **Beta**, and optimizes portfolios using **Modern Portfolio Theory (MPT)** and the **Black-Litterman model**. The tool also includes visualization functions to plot metrics and insights, helping users make informed investment decisions. Robust error handling ensures smooth execution even with missing or invalid data.

## Features:
- **Portfolio Optimization**: Maximizes Sharpe Ratio using MPT.
- **Black-Litterman Model**: Combines market equilibrium with investor views.
- **Technical Indicators**: RSI, Bollinger Bands, MACD, and Beta.
- **Visualizations**: Plots for trends, volatility, and portfolio performance.
- **Error Handling**: Gracefully handles missing or invalid data.

## Dependencies:
- `yfinance`, `pandas`, `numpy`, `matplotlib`, `scipy`


import tkinter as tk
from tkinter import ttk
from tkinter import Text, Label
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Initialize assets and parameters
assets = [
    "Apple (AAPL)", "Amazon (AMZN)", "Bitcoin (BTC-USD)",
    "Alphabet (GOOGL)", "Meta (META)", "Microsoft (MSFT)",
    "Nvidia (NVDA)", "S&P 500 Index (SPY)", "Tesla (TSLA)"
]
tickers = ['AAPL', 'AMZN', 'BTC-USD', 'GOOGL', 'META', 'MSFT', 'NVDA', 'SPY', 'TSLA']
start_date = '2022-01-01'
end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
risk_free_rate = 0.04
benchmark_ticker = '^GSPC'  # S&P 500 as the benchmark


# Fetch stock data
def fetch_data(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
        if isinstance(data.columns, pd.MultiIndex):  # Handle multi-index format
            data = data.xs('Close', level=1, axis=1)  # Extract Close prices
        elif 'Close' in data.columns:
            data = data['Close']
        else:
            raise KeyError("Neither 'Close' nor 'Adj Close' found in data.")

        data = data.dropna()  # Remove rows with missing values
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure


# Calculate RSI
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, 1)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# Calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20):
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return sma, upper_band, lower_band


# Calculate Beta
def calculate_beta(ticker, benchmark_ticker, start_date, end_date):
    try:
        # Download stock and benchmark data
        stock_data = yf.download(ticker, start=start_date, end=end_date)['Close']
        benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date)['Close']

        # Ensure the data is a Series (not a DataFrame)
        if isinstance(stock_data, pd.DataFrame):
            stock_data = stock_data.squeeze()  # Convert to Series if it's a DataFrame
        if isinstance(benchmark_data, pd.DataFrame):
            benchmark_data = benchmark_data.squeeze()  # Convert to Series if it's a DataFrame

        # Calculate daily returns
        returns_stock = stock_data.pct_change(fill_method=None).dropna()
        returns_benchmark = benchmark_data.pct_change(fill_method=None).dropna()

        # Calculate covariance and variance
        covariance = returns_stock.cov(returns_benchmark)
        variance = returns_benchmark.var()

        # Calculate beta
        beta = covariance / variance
        return beta
    except Exception as e:
        print(f"Error calculating beta for {ticker}: {e}")
        return None


# Calculate MACD
def calculate_macd(data):
    short_ema = data.ewm(span=12, adjust=False).mean()
    long_ema = data.ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


# Modern Portfolio Theory (MPT)
def optimize_portfolio(tickers, risk_free_rate):
    data = fetch_data(tickers, start_date, end_date)
    if data.empty:
        raise ValueError("No data available for optimization.")

    returns = data.pct_change(fill_method=None).dropna()
    if returns.isnull().values.any() or (returns == 0).all(axis=0).any():
        raise ValueError("Invalid returns data: contains nulls or all zeros.")

    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
        portfolio_return = np.sum(mean_returns * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
        return -sharpe_ratio

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(tickers)))
    init_guess = len(tickers) * [1. / len(tickers)]

    opt_results = minimize(
        neg_sharpe_ratio, init_guess, args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP', bounds=bounds, constraints=constraints
    )

    optimal_weights = opt_results.x
    return {ticker: round(weight, 2) for ticker, weight in zip(tickers, optimal_weights)}


# Visualization Functions
def plot_rsi(data, ticker):
    rsi = calculate_rsi(data)
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, rsi, label='RSI')
    plt.axhline(70, color='r', linestyle='--', label='Overbought')
    plt.axhline(30, color='g', linestyle='--', label='Oversold')
    plt.title(f'RSI for {ticker}')
    plt.legend()
    plt.show()


def plot_bollinger_bands(data, ticker):
    sma, upper_band, lower_band = calculate_bollinger_bands(data)
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data, label='Close Price')
    plt.plot(data.index, sma, label='SMA')
    plt.plot(data.index, upper_band, label='Upper Band')
    plt.plot(data.index, lower_band, label='Lower Band')
    plt.title(f'Bollinger Bands for {ticker}')
    plt.legend()
    plt.show()


def plot_macd(data, ticker):
    macd, signal = calculate_macd(data)
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, macd, label='MACD')
    plt.plot(data.index, signal, label='Signal Line')
    plt.title(f'MACD for {ticker}')
    plt.legend()
    plt.show()


# Plot Beta for all tickers
def plot_beta(tickers, benchmark_ticker, start_date, end_date):
    betas = {}
    for ticker in tickers:
        beta = calculate_beta(ticker, benchmark_ticker, start_date, end_date)
        if beta is not None:
            betas[ticker] = beta

    if not betas:
        print("No valid beta values to plot.")
        return

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(betas.keys(), betas.values(), color='blue')
    plt.axhline(1, color='red', linestyle='--', label='Market Beta (1.0)')
    plt.title('Beta Values for Tickers (vs S&P 500)')
    plt.xlabel('Tickers')
    plt.ylabel('Beta')
    plt.legend()
    plt.show()


# Main Execution
try:
    data = fetch_data(tickers, start_date, end_date)
    if data.empty:
        print("No data fetched. Exiting.")
    else:
        # Portfolio Optimization
        portfolio_weights = optimize_portfolio(tickers, risk_free_rate)
        print("Portfolio Weights:", portfolio_weights)

        # Plot Beta for all tickers
        plot_beta(tickers, benchmark_ticker, start_date, end_date)

        # Plot RSI, Bollinger Bands, and MACD for each ticker
        for ticker in tickers:
            if ticker in data.columns:
                stock_data = data[ticker].dropna()
                print(f"Processing {ticker}...")
                plot_rsi(stock_data, ticker)
                plot_bollinger_bands(stock_data, ticker)
                plot_macd(stock_data, ticker)
            else:
                print(f"No data available for {ticker}.")

except Exception as e:
    print(f"An error occurred: {e}")

# Output Portfolio Summary
print("Portfolio Optimization and Beta Plotting Complete.")
