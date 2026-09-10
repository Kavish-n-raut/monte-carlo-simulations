import os
import yfinance as yf
import pandas as pd

def fetch_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
    cache_dir: str = './cache'
) -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{'_'.join(tickers)}_{start_date}_{end_date}.csv")
    
    # Load from cache only if it has valid data
    if os.path.exists(cache_file):
        cached_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        if not cached_data.empty and len(cached_data) > 5:
            return cached_data
            
    prices = yf.download(tickers, start=start_date, end=end_date)
    
    # Safely handle yfinance's changing MultiIndex column structure
    if isinstance(prices.columns, pd.MultiIndex):
        if 'Adj Close' in prices.columns.get_level_values(0):
            prices = prices['Adj Close']
        elif 'Close' in prices.columns.get_level_values(0):
            prices = prices['Close']
        else:
            prices = prices.xs('Close', level=1, axis=1)
            
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
        
    if prices.empty or prices.isnull().all().all():
        raise ValueError("Failed to download price data for the provided tickers.")
        
    # Clean data and drop fully missing columns
    prices = prices.ffill().bfill().dropna(axis=1, how='all')
    prices.to_csv(cache_file)
    
    return prices