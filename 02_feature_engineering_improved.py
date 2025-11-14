import os
import pandas as pd
import numpy as np

DATA_DIR = 'data'
RAW_OHLCV = os.path.join(DATA_DIR, 'raw_prices_ohlcv.csv')
RAW_PRICES_CLOSE = os.path.join(DATA_DIR, 'raw_prices.csv')
FEATURES_FILE = os.path.join(DATA_DIR, 'features.csv')

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'JPM', 'JNJ', 'SPY']
SHORT_WINDOW = 21
LONG_WINDOW = 63
FORWARD_WINDOW = 21
RSI_WINDOW = 14


def load_ohlcv():
    if os.path.exists(RAW_OHLCV):
        df = pd.read_csv(RAW_OHLCV, index_col=0, parse_dates=True)
        return df
    elif os.path.exists(RAW_PRICES_CLOSE):
        # Legacy: close-only
        close = pd.read_csv(RAW_PRICES_CLOSE, index_col=0, parse_dates=True)
        return close
    else:
        raise FileNotFoundError('No OHLCV or close-only file found. Run 01_data_loader_improved.py')


def compute_features(df_raw):
    # If flat columns like 'AAPL_Close' exist, convert to dict of DataFrames per ticker
    cols = df_raw.columns
    per_ticker = {}
    for t in TICKERS:
        expected = [f"{t}_Open", f"{t}_High", f"{t}_Low", f"{t}_Close", f"{t}_Volume"]
        if all(c in cols for c in expected):
            sub = df_raw[expected].copy()
            sub.columns = ['Open','High','Low','Close','Volume']
            per_ticker[t] = sub
        elif t in cols:
            # close-only
            sub = pd.DataFrame(df_raw[t]).copy()
            sub['Open'] = sub[t]
            sub['High'] = sub[t]
            sub['Low'] = sub[t]
            sub['Close'] = sub[t]
            sub['Volume'] = np.nan
            per_ticker[t] = sub[['Open','High','Low','Close','Volume']]
        else:
            print(f"Ticker {t} not in data; skipping")

    frames = []
    # build close cross-section
    close_cs = pd.DataFrame({t: per_ticker[t]['Close'] for t in per_ticker})

    for t, df in per_ticker.items():
        df = df.sort_index()
        f = pd.DataFrame(index=df.index)
        f['Close'] = df['Close']
        f['ret_1d'] = df['Close'].pct_change(1)
        f['ret_2d'] = df['Close'].pct_change(2)
        f['ret_5d'] = df['Close'].pct_change(5)
        f['ret_21d'] = df['Close'].pct_change(21)

        # Moving averages and ratios
        f['ma_5'] = df['Close'].rolling(5).mean()
        f['ma_21'] = df['Close'].rolling(21).mean()
        f['ma_63'] = df['Close'].rolling(63).mean()
        f['ma_ratio_5_21'] = f['ma_5'] / f['ma_21'] - 1
        f['mom_21'] = df['Close'] / f['ma_21'] - 1

        # ATR and volatility
        high_low = df['High'] - df['Low']
        high_close_prev = (df['High'] - df['Close'].shift(1)).abs()
        low_close_prev = (df['Low'] - df['Close'].shift(1)).abs()
        atr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1).rolling(14).mean()
        f['ATR_14'] = atr
        f['vol_21'] = df['Close'].pct_change().rolling(21).std()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta>0,0)).rolling(RSI_WINDOW).mean()
        loss = (-delta.where(delta<0,0)).rolling(RSI_WINDOW).mean()
        rs = gain / (loss + 1e-9)
        f['RSI_14'] = 100 - (100/(1+rs))

        # MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        f['MACD'] = ema12 - ema26
        f['MACD_sig'] = f['MACD'].ewm(span=9, adjust=False).mean()

        # Bollinger band width
        mb = df['Close'].rolling(20).mean()
        sd = df['Close'].rolling(20).std()
        f['BB_width'] = (mb + 2*sd - (mb - 2*sd)) / (mb + 1e-9)

        # Volume features
        if 'Volume' in df.columns:
            f['vol_1d'] = df['Volume'].pct_change(1)
            f['vol_21_mean'] = df['Volume'].rolling(21).mean()
            f['vol_21_z'] = (df['Volume'] - f['vol_21_mean']) / (df['Volume'].rolling(21).std() + 1e-9)
        else:
            f['vol_1d'] = np.nan; f['vol_21_mean'] = np.nan; f['vol_21_z'] = np.nan

        # Distributional
        f['ret_skew_21'] = df['Close'].pct_change().rolling(21).skew()
        f['ret_kurt_21'] = df['Close'].pct_change().rolling(21).kurt()

        # forward label
        f['forward_return_21d'] = df['Close'].pct_change(FORWARD_WINDOW).shift(-FORWARD_WINDOW)
        f['Ticker'] = t
        frames.append(f)

    features = pd.concat(frames).reset_index().rename(columns={'index':'Date'})

    # Cross-sectional rank of 21d returns
    pivot = features.pivot_table(index='Date', columns='Ticker', values='forward_return_21d')
    cs_rank = pivot.rank(axis=1, pct=True)

    def get_cs_rank(row):
        try:
            return cs_rank.loc[row['Date'], row['Ticker']]
        except Exception:
            return np.nan

    features['cs_rank_ret21'] = features.apply(get_cs_rank, axis=1)

    # labels
    features['label_dir'] = (features['forward_return_21d'] > 0).astype(int)
    features['label_quantile_top30'] = features.groupby('Date')['forward_return_21d'].transform(lambda x: (x >= x.quantile(0.7)).astype(int))

    # winsorize numeric columns at 1%-99%
    num_cols = features.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        lower = features[c].quantile(0.01)
        upper = features[c].quantile(0.99)
        features[c] = features[c].clip(lower, upper)

    # dropna on label and required features
    required = ['forward_return_21d','Ticker','Date']
    features.dropna(subset=required, inplace=True)
    features.dropna(inplace=True)

    features.to_csv(FEATURES_FILE, index=False)
    print(f"Saved features to {FEATURES_FILE} rows={len(features)}")
    return features


if __name__ == '__main__':
    df = load_ohlcv()
    features = compute_features(df)