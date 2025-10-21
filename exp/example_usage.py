import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from backtest import Backtester, prepare_technical_columns, make_technical_buy_fn, make_threshold_sell_fn


def make_synth_ohlc(n=200, start_price=100.0, vol=0.02, seed=42):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, vol, n)
    prices = [start_price]
    for r in rets:
        prices.append(prices[-1] * (1 + r))
    prices = np.array(prices[1:])
    # Build OHLC from close-like series
    close = prices
    open_ = np.concatenate([[start_price], prices[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol/2, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol/2, n)))

    start = datetime(2020, 1, 1)
    idx = pd.date_range(start, periods=n, freq='D')

    df = pd.DataFrame({
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': close,
    }, index=idx)
    return df


def main():
    df = make_synth_ohlc(n=300)

    # Precompute indicators can be provided directly as a function to the engine

    buy_fn = make_technical_buy_fn(drop_trigger_pct=0.02)
    sell_fn = make_threshold_sell_fn(rise_threshold_pct=0.02, sell_pct_of_position=0.10)

    engine = Backtester(
        data=df,
        start_fund=10_000,
        target_leverage=1.5,
        fee_pct=0.0002,
        mmr=0.005,
        buy_on_start=True,
        print_daily=False,
        buy_at="close",
        buy_fn=buy_fn,
        sell_fn=sell_fn,
    precompute_fn=prepare_technical_columns,  # engine will call this once before run
    precompute_params=None,                   # or pass a dict to configure spans/windows
    )
    engine.run()

    # Print a compact summary
    print("Final fund:", f"{engine.final_fund:,.2f}")
    print("Max drawdown (%):", f"{engine.max_drawdown_pct:.2f}")
    print("Trades:", len(engine.trades))
    if engine.liquidation_event:
        print("Liquidated on:", engine.liquidation_event['date'])
    else:
        print("No liquidation.")

    # Optional: plot
    try:
        engine.plot_unrealized()
    except RuntimeError as e:
        # matplotlib missing; skip plotting in environments without it
        print(str(e))


if __name__ == "__main__":
    main()
