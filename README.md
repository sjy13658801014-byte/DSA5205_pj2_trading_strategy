# Project 2: Leveraging Machine Learning for Quantitative Investment Strategies

This project implements a workflow to download price data, engineer features, train ML models, and backtest a trading strategy based on the model predictions.

## Requirements

Install all dependencies using the provided file:

```bash
pip install -r requirements.txt
````

## How to Run

Execute the scripts in sequential order. Each script depends on the output file from the previous one.

**1. Load Data**
Downloads and saves raw OHLCV price data.

```bash
python 01_data_loader_improved.py
```

> *Output: `data/raw_prices_ohlcv.csv`*

**2. Engineer Features**
Generates features and labels from the raw price data.

```bash
python 02_feature_engineering_improved.py
```

> *Output: `data/features.csv`*

**3. Train Models**
Runs a rolling-window training process to generate out-of-sample predictions.

```bash
python 03_model_training_improved.py
```

> *Output: `data/predictions.csv`*

**4. Run Backtest**
Uses `backtrader` to evaluate the ML strategy against benchmarks.

```bash
python 04_backtest_improved.py
```

> *Output: Console summary of strategy performance.*

```
```
