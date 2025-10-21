# gc-backtest

A tiny, flexible backtesting engine for long-only strategies with modular buy/sell functions and explicit liquidation checks using the Low of each bar.

## Quick start

- File: `backtest.py` provides the `Backtester` class and strategy helpers.
- Example: `example_usage.py` shows how to wire it up.

### Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

### Run the example

```bash
python example_usage.py
```

If you want to see plots, ensure matplotlib is installed (already in requirements.txt):

```bash
python3 -m pip install matplotlib
```

### Minimal usage

```python
from backtest import Backtester

# df must have columns: Open, High, Low, Close, indexed by datetime
engine = Backtester(
    data=df,
    start_fund=10_000,
    target_leverage=1.5,
    buy_on_start=True,
    print_daily=False,
    buy_fn=lambda df, i: (True, None),   # default: buy to target leverage
    sell_fn=lambda df, i: (False, None), # no sells
)
engine.run()
print(engine.final_fund)
```

### Action contract (buy_fn/sell_fn)
- Return either:
  - `(should_act: bool, trigger_price: float|None)`
    - Buy: adds to reach target leverage
    - Sell: sells entire position (to zero)
  - `(should_act, trigger_price, action: dict)` where `action['type']` is one of:
    - `'to_leverage'` with `{'leverage': float}`
    - `'percent'` with `{'pct': float}` (fraction of current notional)
    - `'to_zero'` (sell only)

### Helpers
- `prepare_technical_columns(df, ...)`
- `make_technical_buy_fn(drop_trigger_pct=0.02)`
- `make_threshold_sell_fn(rise_threshold_pct=0.01, sell_pct_of_position=0.10)`

## Lightweight tests

A simple self-contained test script uses no external frameworks:

```bash
python3 test_backtester.py
```

It validates:
- Initial buy to target leverage occurs
- to_zero sell empties the position
- percent sell reduces the position size
