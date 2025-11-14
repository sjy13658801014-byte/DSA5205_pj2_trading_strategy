import pandas as pd
import numpy as np
import backtrader as bt
import os
from datetime import datetime


# --- 1. Custom data classes supporting predictive columns ---
class PredictionData(bt.feeds.PandasData):
    """
    Extend PandasData to support machine learning prediction columns
    """
    # Add a new data cable
    lines = ('prediction',)

    # Define column mapping
    params = (
        ('prediction', -1),  # Prediction column, -1 indicates automatic detection
        ('volume', -1),  # Set the volume column explicitly to -1 (does not exist)
    )


# --- 2. Strategy 1: Machine Learning-Based Predictive Strategy ---
class MLStrategy(bt.Strategy):
    """
    Trading strategies employing machine learning predictions
    """
    params = (
        ('buy_threshold', 0.01),  # Predicting a rise of over 1%, buy.
        ('sell_threshold', -0.005),  # Sell if a decline of over 1% is forecast
        ('position_size', 0.1),  # Use 10% of funds each time
        ('printlog', False),  # Reduce log output
    )

    def __init__(self):
        # Data citation
        self.dataclose = self.datas[0].close
        self.prediction = self.datas[0].prediction  # Machine learning prediction values

        # Order and Transaction Tracking
        self.order = None
        self.buyprice = None
        self.buycomm = None

        print(f" ML Strategy Initialisation - Buy Threshold: {self.params.buy_threshold}")

    def log(self, txt, dt=None):
        '''Logarithmic function'''
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}: {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'Buy order executed, price: {order.executed.price:.2f}')
            else:
                self.log(f'Sell execution, price: {order.executed.price:.2f}')
            self.bar_executed = len(self)

        self.order = None

    def next(self):
        # If there are any uncompleted orders, no new operations shall be performed.
        if self.order:
            return

        # Check for open positions
        if not self.position:
            # No positions held; check purchase conditions.
            if self.prediction[0] > self.params.buy_threshold:
                # Calculate the purchase quantity (10% of capital)
                size = int((self.broker.getcash() * self.params.position_size) / self.dataclose[0])
                if size > 0:
                    self.log(f' Buy signal: Forecast value {self.prediction[0]:.3f} > threshold {self.params.buy_threshold}')
                    # Execute at the opening price of the next bar
                    self.order = self.buy(size=size, exectype=bt.Order.Market)

        else:
            # Holdings exist; check conditions for selling.
            if self.prediction[0] < self.params.sell_threshold:
                self.log(f' Sell signal: Forecast value {self.prediction[0]:.3f} < threshold {self.params.sell_threshold}')
                self.order = self.sell()


# --- 3. Strategy 2: Classic Moving Average Strategy (as a benchmark)---
class MAStrategy(bt.Strategy):
    """
    Classic Double Moving Average Strategy - As a Benchmark Comparison
    """
    params = (
        ('fast', 20),  # Fast moving average period
        ('slow', 50),  # Slow moving average period
        ('printlog', False),
    )

    def __init__(self):
        # Calculate the moving average
        self.fast_ma = bt.indicators.SMA(self.datas[0].close, period=self.params.fast)
        self.slow_ma = bt.indicators.SMA(self.datas[0].close, period=self.params.slow)

        # Order Tracking
        self.order = None
        print(f" MA Strategy Initialisation - Fast Line: {self.params.fast}days, Slow Line: {self.params.slow}days")

    def log(self, txt, dt=None):
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}: {txt}')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.order = None

    def next(self):
        if self.order:
            return

        if not self.position:
            # Golden Cross Buy Signal: Fast Moving Average crosses above Slow Moving Average
            if self.fast_ma[0] > self.slow_ma[0] and self.fast_ma[-1] <= self.slow_ma[-1]:
                size = int(self.broker.getcash() * 0.1 / self.datas[0].close[0])
                if size > 0:
                    self.log(f' MA Golden Cross Buy Signal: Fast Line{self.fast_ma[0]:.2f} > Slow line{self.slow_ma[0]:.2f}')
                    self.order = self.buy(size=size)

        else:
            # Death Cross Sell Signal: Fast Moving Average crosses below Slow Moving Average
            if self.fast_ma[0] < self.slow_ma[0] and self.fast_ma[-1] >= self.slow_ma[-1]:
                self.log(f' MA Death Cross Sell Signal: Fast Moving Average{self.fast_ma[0]:.2f} < Slow line{self.slow_ma[0]:.2f}')
                self.order = self.sell()



# --- 5. Data Preparation Function ---
def prepare_data(ticker='TSLA'):
    """Prepare data in Backtrader format"""
    print(f" Prepare {ticker} data...")

    # Loading forecast data
    predictions_df = pd.read_csv('data/predictions.csv', parse_dates=['Date'])
    ticker_data = predictions_df[predictions_df['Ticker'] == ticker].copy()

    if ticker_data.empty:
        print(f"Error: No data found for {ticker}")
        return None

    # Index and sort
    ticker_data.set_index('Date', inplace=True)
    ticker_data.sort_index(inplace=True)

    # Ensure OHLC data is available
    for col in ['Open', 'High', 'Low']:
        if col not in ticker_data.columns:
            ticker_data[col] = ticker_data['Close']

    # Rename the prediction column to 'prediction' (as required by backtrader)
    if 'Prediction_Ridge' in ticker_data.columns:
        ticker_data['prediction'] = ticker_data['Prediction_Ridge']
    else:
        print("Error: Prediction_Ridge column not found")
        return None

    print(f" Data preparation complete: {len(ticker_data)} lines, {ticker_data.index[0]} to {ticker_data.index[-1]}")
    return ticker_data


# --- 6. Backtesting Execution Function ---
def run_backtest(strategy_class, data, strategy_name, cash=100000):
    """Run backtesting"""
    print(f"\n{'=' * 50}")
    print(f"Commencing backtesting: {strategy_name}")
    print(f"{'=' * 50}")

    # Create a backtesting engine
    cerebro = bt.Cerebro()
    # When creating a backtesting engine, configure the charting options directly.
    cerebro = bt.Cerebro(stdstats=False)

    # Add only the required indicators to the chart
    cerebro.addobserver(bt.observers.Broker)
    cerebro.addobserver(bt.observers.Trades)

    # Set up funding
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=0.002)  # 0.2%commission

    # Prepare data - using custom PredictionData
    datafeed = PredictionData(
        dataname=data,
        datetime=None,  # Using an index as time
        open='Open',
        high='High',
        low='Low',
        close='Close',
        volume=-1,
        prediction='prediction'  # Machine learning prediction values
    )

    cerebro.adddata(datafeed)
    cerebro.addstrategy(strategy_class)

    # Add analyser
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    # Run backtesting
    initial_value = cerebro.broker.getvalue()
    print(f' Initial capital: ${initial_value:,.2f}')

    results = cerebro.run()
    strategy = results[0]

    final_value = cerebro.broker.getvalue()
    total_return = (final_value / initial_value - 1) * 100

    # Obtain the analysis results
    sharpe_analysis = strategy.analyzers.sharpe.get_analysis()
    drawdown_analysis = strategy.analyzers.drawdown.get_analysis()
    returns_analysis = strategy.analyzers.returns.get_analysis()
    trade_analysis = strategy.analyzers.trades.get_analysis()

    print(f' Final funding: ${final_value:,.2f}')
    print(f' Total return rate: {total_return:.2f}%')
    sharpe_ratio = sharpe_analysis.get("sharperatio", 0)
    print(f' Sharp ratio: {sharpe_ratio:.2f}' if sharpe_ratio is not None else ' Sharp ratio: N/A')
    print(f' Maximum drawdown: {drawdown_analysis.get("max", {}).get("drawdown", 0):.2f}%')

    if 'total' in trade_analysis:
        total_trades = trade_analysis['total']['total']
        print(f' Total number of transactions: {total_trades}')
        if total_trades > 0:
            win_trades = trade_analysis.get('won', {}).get('total', 0)
            win_rate = (win_trades / total_trades) * 100
            print(f' Win rate: {win_rate:.1f}%')

    return {
        'strategy': strategy_name,
        'final_value': final_value,
        'total_return': total_return,
        'sharpe': sharpe_analysis.get('sharperatio', 0),
        'max_drawdown': drawdown_analysis.get('max', {}).get('drawdown', 0),
        'cerebro': cerebro,
        'trade_analysis': trade_analysis
    }


# --- 7. Main function ---
def main():
    print(" Commencing Backtrader backtesting analysis")

    # Prepare data
    data = prepare_data('AAPL')
    if data is None:
        return

    # Using only test set data (post-2023)
    test_data = data['2023-01-01':].copy()
    print(f" Test set data: {len(test_data)} lines")

    # Calculate buy-and-hold returns (as a benchmark)
    buy_hold_return = (test_data['Close'].iloc[-1] / test_data['Close'].iloc[0] - 1) * 100
    print(f" Buy-and-hold strategy returns: {buy_hold_return:.2f}%")

    results = []

    # ML strategy
    ml_result = run_backtest(MLStrategy, test_data, "Machine Learning Strategy")
    results.append(ml_result)

    # MA Strategy
    ma_result = run_backtest(MAStrategy, test_data, "Moving Average Strategy")
    results.append(ma_result)

    # Comparison of Results
    print(f"\n{'=' * 60}")
    print("Summary of Strategy Comparison")
    print(f"{'=' * 60}")

    best_strategy = max(results, key=lambda x: x['total_return'])

    for result in results:
        outperformance = result['total_return'] - buy_hold_return
        print(f"\n{result['strategy']}:")
        print(f"  Total return: {result['total_return']:.2f}%")
        print(f"  Compare with buy-and-hold: {outperformance:+.2f}%")
        sharpe = result['sharpe'] if result['sharpe'] is not None else 0
        print(f"  Sharp ratio: {sharpe:.2f}")
        print(f"  Maximum drawdown: {result['max_drawdown']:.2f}%")

        # Display the number of transactions (if any)
        if 'total' in result.get('trade_analysis', {}):
            trades = result['trade_analysis']['total']['total']
            print(f"  Number of transactions: {trades}")

    print(f"\n Optimal strategy: {best_strategy['strategy']} ({best_strategy['total_return']:.2f}%)")

    # Plotting charts (optional)
    plot = input("\nShould a chart be drawn?(y/n): ").lower()
    if plot == 'y':
        best_strategy['cerebro'].plot(style='candle', volume=False)
        # best_strategy['cerebro'].plot()


if __name__ == '__main__':
    main()