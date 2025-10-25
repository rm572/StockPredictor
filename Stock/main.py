from data import Data
from datetime import datetime
from features import Features
from model import Model
from indicators import Indicators

train_tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BRK.B", "NVDA", "META", "V", "JNJ",
    "WMT", "PG", "UNH", "HD", "DIS", "PYPL", "MA", "VZ", "INTC", "CSCO",
    "PEP", "KO", "MRK", "ABT", "T", "XOM", "CVX", "NKE", "MCD", "BA",
    "IBM", "GS", "AXP", "CAT", "MMM", "RTX", "LMT", "HON", "NEE", "UNP",
    "SPGI", "AMD", "MS", "UPS", "ADBE", "INTU", "ORCL", "AMGN", "MDT", "ISRG",
    "BMY", "VRTX", "CSX", "LOW", "DE", "SYK", "SBUX", "COP", "CVS", "CL",
    "CLX", "DHR", "DUK", "EMR", "GE", "GIS", "GM", "GILD", "GLW", "HCA",
    "HES", "HPE", "HSY", "ICE", "IDXX", "ILMN", "INCY", "JCI", "JPM", "KDP",
    "KHC", "KLAC", "LHX", "LIN", "LLY", "LRCX", "MAA", "MAR", "MDLZ", "MET",
    "MGM", "MMC", "MNST", "MO", "MU", "NEM", "NFLX", "NOC", "NUE", "NVR",
    "OTIS", "PAYX", "PFE", "PH", "PLD", "PM", "PNC", "POOL", "PPG", "PRU",
    "PSA", "PXD", "QCOM", "SCHW", "TGT", "TJX", "TMUS", "USB", "WBA"
]

csv_acc = {}
sharpe_ratio_train = {}

def training():
    # Determine the Dates
    start_time = "2010-01-01"
    today = datetime.today().strftime("%Y-%m-%d")

    # Load the data for the S&P 500
    spy_data = Data("SPY", start=start_time, end=today)
    spy_market_data = spy_data.load_data()


    for ticker in train_tickers:
        model = Model()
        print("="*40)
        print(f"Running model for {ticker}")
        print("="*40)
        
        
        # Load the stock's data
        data_load = Data(ticker, start=start_time, end=today)
        price_data = data_load.load_data()


        # Get the features using the data
        features = Features(price_data, spy_market_data)
        vals = features.compile_features()

        # These are all the indicators the model will be using
        predictors = vals[[
                           f"Simple_MA_{Indicators.DAYS}",
                           f"Exponential_MA_{Indicators.DAYS}", 
                           f"RSI_{Indicators.DAYS}", 
                           f"Bollinger_Upper_{Indicators.DAYS}", 
                           f"Bollinger_Lower_{Indicators.DAYS}",          
                           "Daily_Return", 
                           "Lag_1",
                           "Lag_2",
                           "Lag_3",
                           "Lag_4",
                           "Lag_5",
                           f"Volatility_{Indicators.DAYS}",
                           f"SPY_Market_Corr_{Indicators.DAYS}"                           
                        ]
                    ]
        target = vals["Target"]

        # Drop rows with NaN in predictors
        predictors = predictors.dropna()
        target = target.loc[predictors.index]

        if predictors.shape[0] < 10:
            continue

        prices = price_data["Close"].pct_change()

        # Train the model and evaluate it
        model.train_model(predictors, target, prices)

        # Map the accuracy and sharpe ratio so we can print out results later
        sharpe_ratio_train[ticker] = model.sharpe
        csv_acc[ticker] = model.cross_value_accuracy
    
    # Print out some results
    for key in csv_acc.keys():
        print(f"{key}: {csv_acc[key]}")
        print(" "*(len(key)-1), f": {sharpe_ratio_train[key]}")

    # Print out average accuracy and sharpe ratio
    print("Avg Accuracy: ", sum(csv_acc.values())/len(csv_acc))
    print("Avg Sharpe Ratio: ", sum(sharpe_ratio_train.values())/len(sharpe_ratio_train))

def main():
    training()

if __name__ == "__main__":
    main()


