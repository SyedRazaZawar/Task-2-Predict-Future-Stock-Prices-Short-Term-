import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def main():
    # 1. Select a stock
    ticker_symbol = 'AAPL'
    print(f"Fetching data for {ticker_symbol}...")
    
    # 2. Load historical data
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period="5y")
    
    if df.empty:
        print("No data fetched. Exiting.")
        return

    print("Data loaded successfully.")
    
    # 3. Feature engineering
    # We use Open, High, Low, and Volume to predict the next Close price.
    features = ['Open', 'High', 'Low', 'Volume']
    
    # Target variable: next day's Close price
    df['Target_Close'] = df['Close'].shift(-1)
    
    # Drop the last row since it will have a NaN target
    df = df.dropna()
    
    X = df[features]
    y = df['Target_Close']
    
    # 4. Train/Test Split
    # Since it's time series data, we split chronologically (no shuffling)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 5. Train a model (Random Forest Regressor)
    print("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Model Evaluation:\n MSE: {mse:.2f}\n R2 Score: {r2:.2f}")
    
    # 6. Plot actual vs predicted closing prices
    print("Generating plot...")
    plt.figure(figsize=(12, 6))
    
    # The index of y_test contains the dates
    plt.plot(y_test.index, y_test.values, label='Actual Next-Day Close Price', color='blue', alpha=0.7)
    plt.plot(y_test.index, predictions, label='Predicted Next-Day Close Price', color='red', alpha=0.7)
    
    plt.title(f'{ticker_symbol} Stock Price Prediction (Short-Term)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plot_filename = f'{ticker_symbol}_prediction_plot.png'
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")

if __name__ == "__main__":
    main()
