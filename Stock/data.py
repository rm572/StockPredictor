import yfinance as yf
import pandas as pd
import os

class Data:
    def __init__(self, ticker, start, end, cache = False):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.cache = cache
        self.data = None
    
    def get_filename(self):
        folder = "data"
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, f"{self.ticker}_{self.start}_{self.end}.csv")

    def load_data(self):
        filename = self.get_filename()

        if self.cache and os.path.exists(filename):
            self.data = pd.read_csv(filename)
        else:
            self.data = yf.download(self.ticker, start=self.start, end=self.end)
            if self.cache:
                os.makedirs("data", exist_ok=True)
                self.data.to_csv(filename)
        
        return self.data