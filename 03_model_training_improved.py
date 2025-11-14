# 03_model_training_improved.py
import os
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

# Try to use tqdm for progress bar; if unavailable, fallback to simple prints
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False

# --- 1. Paths and constants (kept consistent with your provided code) ---
DATA_DIR = 'data'
FEATURES_FILE = os.path.join(DATA_DIR, 'features.csv')
PREDICTIONS_FILE = os.path.join(DATA_DIR, 'predictions.csv')

FEATURE_NAMES = [
    'ret_1d','ret_2d','ret_5d','ret_21d','ma_ratio_5_21','mom_21',
    'ATR_14','vol_21','RSI_14','MACD','MACD_sig','BB_width',
    'vol_1d','vol_21_z','ret_skew_21','ret_kurt_21','cs_rank_ret21'
]
LABEL = 'forward_return_21d'
TRAIN_WINDOW = 504
RETRAIN_EVERY = 63

# --- 2. Model grid and definitions ---
PARAM_GRID = {
    'Ridge': {'ridge__alpha':[0.01,0.1,1.0,10.0,100.0]},
    'RF': {'n_estimators':[50,100],'max_depth':[3,5,7],'min_samples_leaf':[10,20]},
    'HGB': {'max_iter':[100,200],'max_depth':[3,5]}
}

MODELS = {
    'Ridge': Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(random_state=42))]),
    'RF': RandomForestRegressor(random_state=42, n_jobs=-1),
    'HGB': HistGradientBoostingRegressor(random_state=42)
}

# --- 3. Utility functions ---
def time_series_grid_search(X, y, model, param_grid):
    """
    Time-series grid search with TimeSeriesSplit.
    """
    tscv = TimeSeriesSplit(n_splits=3)
    gs = GridSearchCV(model, param_grid, cv=tscv,
                      scoring='neg_mean_squared_error', n_jobs=-1, error_score='raise')
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_

def compute_ic(y_true, y_pred):
    """
    Information Coefficient (Pearson corr) between predictions and true returns
    """
    if len(y_true) < 2:
        return np.nan
    # if constant, return nan safely
    if np.nanstd(y_pred) == 0 or np.nanstd(y_true) == 0:
        return np.nan
    return np.corrcoef(y_true, y_pred)[0,1]

# --- 4. Main prediction generation with progress reporting ---
def generate_predictions(features):
    """
    Rolling-window training & out-of-sample predictions across tickers.
    Progress is shown using tqdm if available; otherwise periodic prints.
    """
    # Ensure Date column is datetime
    features['Date'] = pd.to_datetime(features['Date'])
    tickers = features['Ticker'].unique().tolist()

    # Precompute total prediction steps for a global progress bar
    total_steps = 0
    per_ticker_lengths = {}
    for t in tickers:
        df_t = features[features['Ticker']==t].sort_values('Date')
        steps = max(0, len(df_t) - TRAIN_WINDOW)
        per_ticker_lengths[t] = len(df_t)
        total_steps += steps

    if total_steps == 0:
        print("No ticker has enough data to run rolling training (check TRAIN_WINDOW and your features.csv).")
        return pd.DataFrame()

    use_tqdm = TQDM_AVAILABLE
    iterator = tqdm(tickers, desc='Tickers') if use_tqdm else tickers

    out_frames = []
    global_step = 0
    for t in iterator:
        df = features[features['Ticker']==t].sort_values('Date').reset_index(drop=True).copy()
        n = len(df)
        X = df[FEATURE_NAMES]
        y = df[LABEL]

        # init prediction columns
        for name in list(MODELS.keys()):
            df[f'Prediction_{name}'] = np.nan

        trained = {}
        param_history = {name: [] for name in MODELS.keys()}

        # If ticker has less than TRAIN_WINDOW rows, skip
        if n <= TRAIN_WINDOW:
            out_frames.append(df)
            # advance progress bar by 0
            continue

        # Per-ticker progress (if tqdm available we can show inner progress)
        if use_tqdm:
            inner_iter = tqdm(range(TRAIN_WINDOW, n), desc=f'{t} steps', leave=False)
        else:
            inner_iter = range(TRAIN_WINDOW, n)

        for i in inner_iter:
            is_retrain = (i - TRAIN_WINDOW) % RETRAIN_EVERY == 0

            if is_retrain:
                train_start = i - TRAIN_WINDOW
                train_end = i
                X_train = X.iloc[train_start:train_end].fillna(0)
                y_train = y.iloc[train_start:train_end].fillna(0)

                for name, model in MODELS.items():
                    try:
                        # Only run grid search if param grid provided
                        pg = PARAM_GRID.get(name, {})
                        if pg:
                            best_model, best_params = time_series_grid_search(X_train, y_train, model, pg)
                            trained[name] = best_model
                            param_history[name].append(best_params)
                            # print a brief update for this retrain
                            print(f"[{t}] Retrain idx={i} model={name} best_params={best_params}")
                        else:
                            # No param grid; fit default
                            trained[name] = model.fit(X_train, y_train)
                            print(f"[{t}] Retrain idx={i} model={name} fitted with default params")
                    except Exception as e:
                        # fallback: fit without gridsearch
                        try:
                            trained[name] = model.fit(X_train, y_train)
                            print(f"[{t}] GridSearch failed for {name} at i={i}, fallback to fit (error: {e})")
                        except Exception as e2:
                            print(f"[{t}] Failed to train model {name} at i={i} (error: {e2}). Continuing.")
                            trained[name] = None

            # Make predictions for each trained model
            for name in MODELS.keys():
                if trained.get(name) is not None:
                    try:
                        X_test = X.iloc[i:i+1].fillna(0)
                        pred = trained[name].predict(X_test)[0]
                        df.loc[i, f'Prediction_{name}'] = pred
                    except Exception:
                        df.loc[i, f'Prediction_{name}'] = np.nan

            # update global progress
            global_step += 1
            if not use_tqdm and (global_step % max(1, total_steps // 50) == 0 or global_step == total_steps):
                pct = global_step / total_steps * 100
                print(f"Progress: {global_step}/{total_steps} steps ({pct:.1f}%)")

        # After all steps for ticker, create ensemble (mean of available model preds)
        pred_cols = [f'Prediction_{n}' for n in MODELS.keys()]
        df['Prediction_Ensemble'] = df[pred_cols].mean(axis=1)

        # append to outputs
        out_frames.append(df)

        # print parameter stability summary per ticker
        for name, history in param_history.items():
            if history:
                print(f"[{t}] {name} optimizations: {len(history)}")

    # concat and save
    final = pd.concat(out_frames, ignore_index=True)
    os.makedirs(os.path.dirname(PREDICTIONS_FILE), exist_ok=True)
    final.to_csv(PREDICTIONS_FILE, index=False)
    print(f"Saved predictions to {PREDICTIONS_FILE}")
    return final

# --- 5. Entry point: run generation and quick evaluation ---
if __name__ == '__main__':
    if not os.path.exists(FEATURES_FILE):
        raise FileNotFoundError(f'features.csv missing. Run feature engineering first. Expected at {FEATURES_FILE}')

    features = pd.read_csv(FEATURES_FILE, parse_dates=['Date'])
    predictions = generate_predictions(features)

    # Quick evaluation (validation window 2012-01-01 ~ 2022-12-31)
    val = predictions[(predictions['Date'] >= '2012-01-01') & (predictions['Date'] <= '2022-12-31')].dropna()
    if val.empty:
        print("No validation data available (after dropna). Skipping quick evaluation.")
    else:
        model_list = list(MODELS.keys()) + ['Ensemble']
        for name in model_list:
            col = f'Prediction_{name}' if name != 'Ensemble' else 'Prediction_Ensemble'
            if col not in val.columns:
                continue
            ic = compute_ic(val[LABEL], val[col])
            dir_acc = ((val[col] > 0) == (val[LABEL] > 0)).mean()
            r2 = r2_score(val[LABEL], val[col]) if len(val) > 1 else np.nan
            sample_size = len(val)
            print(f"{name}: IC={ic:.4f}, DirAcc={dir_acc:.3f}, R2={r2:.4f}, SampleSize={sample_size}")