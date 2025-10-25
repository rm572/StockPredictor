import pandas as pd
from indicators import Indicators

class Features:
    def __init__(self, price_data: pd.DataFrame, spy_market_data=None):
        self.price_data = price_data
        self.i = Indicators(price_data)
        self.spy_market_data = spy_market_data
    
    # Take all the indicators and put them into a DataFrame for the ML
    def compile_features(self) -> pd.DataFrame:
        data = pd.DataFrame(index=self.price_data.index)

        # Grab all the indicator data
        data["Daily_Return"] = self.i.daily_returns()
        data[f"Simple_MA_{Indicators.DAYS}"] = self.i.simple_moving_average(Indicators.DAYS)
        data[f"Exponential_MA_{Indicators.DAYS}"] = self.i.exponential_moving_average(Indicators.DAYS)
        data[f"RSI_{Indicators.DAYS}"] = self.i.rsi(Indicators.DAYS)
        data[f"Bollinger_Upper_{Indicators.DAYS}"], data[f"Bollinger_Lower_{Indicators.DAYS}"] = self.i.bollinger_bands(Indicators.DAYS, num_std=2)
        data["Lag_1"] = data["Daily_Return"].shift(1)
        data["Lag_2"] = data["Daily_Return"].shift(2)
        data["Lag_3"] = data["Daily_Return"].shift(3)
        data["Lag_4"] = data["Daily_Return"].shift(4)
        data["Lag_5"] = data["Daily_Return"].shift(5)
        data[f"Volatility_{Indicators.DAYS}"] = self.i.rolling_volatility(Indicators.DAYS)
        if self.spy_market_data is not None:
            data[f"SPY_Market_Corr_{Indicators.DAYS}"] = self.i.spy_market_correlation(Indicators.DAYS, self.spy_market_data)


        data["Future_Return_3"] = self.price_data["Close"].shift(-3) / self.price_data["Close"] - 1
        threshold = 0.005
        data["Target"] = (data["Future_Return_3"] > threshold).astype(int)
        
        data.dropna(inplace=True)

        return data