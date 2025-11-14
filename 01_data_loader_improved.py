import os
from datetime import datetime

DATA_DIR = 'data'
RAW_OHLCV = os.path.join(DATA_DIR, 'raw_prices_ohlcv.csv')
RAW_PRICES_CLOSE = os.path.join(DATA_DIR, 'raw_prices.csv')

TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'JPM', 'JNJ', 'SPY'
]
START_DATE = '2010-01-01'
END_DATE = '2025-10-31'


def download_ohlcv(tickers=TICKERS, start=START_DATE, end=END_DATE, filepath=RAW_OHLCV):
    try:
        import yfinance as yf
    except Exception as e:
        raise ImportError('yfinance not installed. Install it or provide data/raw_prices_ohlcv.csv')

    print(f"Downloading OHLCV for {len(tickers)} tickers from {start} to {end}...")
    data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=False, progress=True, threads=True)

    # Flatten multiindex columns to single level like 'AAPL_Open'
    if isinstance(data.columns, tuple) or hasattr(data.columns, 'levels') and data.columns.nlevels > 1:
        new_cols = []
        for col in data.columns:
            if isinstance(col, tuple):
                new_cols.append(f"{col[0]}_{col[1]}")
            else:
                new_cols.append(str(col))
        data.columns = new_cols

    # Save OHLCV and also save close-only for backward compatibility
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data.to_csv(filepath)

    # Extract close-only table
    closes = {}
    for t in tickers:
        close_col = f"{t}_Close"
        if close_col in data.columns:
            closes[t] = data[close_col]
    if closes:
        close_df = __import__('pandas').DataFrame(closes)
        close_df.to_csv(RAW_PRICES_CLOSE)

    print(f"Saved OHLCV to {filepath} and close-only to {RAW_PRICES_CLOSE}")
    return data


def load_or_download():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if os.path.exists(RAW_OHLCV):
        print(f"Found existing OHLCV file: {RAW_OHLCV}")
        import pandas as pd
        return pd.read_csv(RAW_OHLCV, index_col=0, parse_dates=True)
    else:
        print('OHLCV file not found; attempting to download via yfinance')
        return download_ohlcv()


if __name__ == '__main__':
    load_or_download()
