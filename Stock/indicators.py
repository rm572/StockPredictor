import pandas as pd

class Indicators:
    DAYS = 50

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()

    # 1
    # Simple Moving Average (SMA): Average price of a stock over a set period
    def simple_moving_average(self, days=DAYS):
        ma = self.data["Close"].rolling(window=days).mean()
        self.data[f"Simple_MA_{days}"] = ma
        return ma  
    

    # 2
    # Exponential Moving Average (EMA): Similar to Simple Moving Average;
    # Gives more weight to recent prices compared to older ones
    def exponential_moving_average(self, days=DAYS):
        ma = self.data["Close"].ewm(span=days, adjust=False).mean()
        self.data[f"Exponential_MA_{days}"] = ma
        return ma  

    # 3
    # Relative Strength Index (RSI): Momentum indicator that measures how fast
    # and strongly the price is moving up or down
    # Scale is 0-100: RSI > 70 means stock is "overbought", RSI < 30 means stock is "oversold"
    # Calculated: 100 - 100/(1 + RS) where RS = Average Gain/Average Loss
    def rsi(self, days=DAYS):
        differences = self.data["Close"].diff()
        gain = differences.clip(lower=0)
        loss = -differences.clip(upper=0)

        RS = gain/loss
        RSI = 100 - (100/(1 + RS))
        self.data[f"RSI_{days}"] = RSI
        return self.data[f"RSI_{days}"]


    # 4
    # Bollinger Bands: "Band" of lines plotted around a moving average:
    # Middle line = moving average, upper/lower bands = MA +- 2 standard deviations
    def bollinger_bands(self, days=DAYS, num_std=2):
        middle_band = self.simple_moving_average(days)
        standard_deviation = self.data["Close"].rolling(window=days).std()

        self.data[f"Bollinger_Upper_{days}"] = middle_band + (num_std * standard_deviation)
        self.data[f"Bollinger_Lower_{days}"] = middle_band - (num_std * standard_deviation)
        return self.data[f"Bollinger_Upper_{days}"], self.data[f"Bollinger_Lower_{days}"]

    # 5
    # Daily Returns: Percent change in stock from one day to the next
    # Formula = (P(t) - P(t-1))/P(t-1)
    def daily_returns(self):
        col = ""
        if "Adj Close" in self.data.columns:
            col = "Adj Close"
        else:
            col = "Close"
        self.data["Daily_Return"] = self.data[col].pct_change()
        # print("IMPORTANT")
        # print(self.data.head())
        # print(self.data.columns)
        return self.data["Daily_Return"]

    # 6
    # Rolling Volatility: Measure of how much the stock price moves up or down over a window
    # Calculated with standard deviation of daily returns over the window
    # High Volatility = more risk, unpredicatble; Low volatility = steadier
    def rolling_volatility(self, days=DAYS):
        if "Daily_Return" not in self.data.columns:
            self.daily_returns()
        self.data[f"Volatility_{days}"] = self.data["Daily_Return"].rolling(window=days).std()
        return self.data[f"Volatility_{days}"]

    # 7
    # Previous Market Data: Takes historical data from the S&P 500
    def spy_market_correlation(self, days=DAYS, spy_market_data=None):
        if spy_market_data is None:
            raise ValueError("SPY data not provided")
        
        # Align SPY to the stock data index
        spy_aligned = spy_market_data.reindex(self.data.index)
        stock_returns = 0

        # Compute daily returns
        if "Daily_Return" not in self.data.columns:
            stock_returns = self.daily_returns()
        else:
            stock_returns = self.data["Daily_Return"]

        spy_returns = spy_aligned["Close"].pct_change()
        market_correlation = stock_returns.rolling(window=days, min_periods=1).corr(spy_returns).fillna(0)       
        
        # Add to the DataFrame
        self.data[f"SPY_Market_Corr_{days}"] = market_correlation
        return market_correlation
        