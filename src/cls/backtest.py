import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional, List, Dict

class Backtester:
    """
    Backtesting class for testing trade strategy:
    - Long-only dip-buy with pyramiding to target leverage
    - Explicit liquidation using daily LOW (intraday approximation)

    Required inputs:
      - data (pd.DataFrame): must have columns ['Open','High','Low','Close'] and either a Date index or 'date' column
      - start_fund (float): starting equity/collateral
      - target_leverage (float): desired total leverage (Notional / Equity) to reach when adding
      - fee_pct (float): proportional fee applied to each add's notional (e.g., 0.0002 = 2 bps)
      - mmr (float): maintenance margin rate in (0, 1), e.g., 0.005 (0.5%)
      - buy_on_start (bool): if True, buy at the first day's OPEN to reach target leverage; if False, wait for trigger
      - print_daily (bool): if True, print daily updates (default: True)
      - buy_at (str): price to execute dip-buy, either 'close' (default) or 'trigger'.
          • 'close'  -> buy at the daily close when condition is met
          • 'trigger'-> buy at the trigger level price determined by buy_fn
    - buy_fn (Callable|None): function that decides whether to buy on a given day and returns an action
    - sell_fn (Callable|None): function that decides whether to sell on a given day and returns an action (optional; if None, no sells)
        - precompute_fn (Callable|None): optional function to compute indicators/columns on df before running; can mutate in-place or return a new DataFrame
      - start_date (str): optional start date for backtest (format: 'YYYY-MM-DD')
      - end_date (str): optional end date for backtest (format: 'YYYY-MM-DD')

    Signal function contracts (buy_fn and sell_fn):
      - Input: (df: pd.DataFrame, i: int) where i is the integer location of the current row in df
      - Output (two supported forms):
          1) Tuple[bool, Optional[float]] -> (should_act, trigger_price)
             - Backwards compatible: if True, buys target leverage; sells (if provided) exit to 0.
          2) Tuple[bool, Optional[float], Dict] -> (should_act, trigger_price, action)
             - action['type'] in {'to_leverage','percent','to_zero'}
             - for 'to_leverage': action['leverage'] (float) desired total leverage (Notional/Equity)
             - for 'percent':    action['pct'] (float in (0,inf)).
                  Buy: add pct * current_notional. Sell: reduce pct * current_notional.
             - for 'to_zero':    sells only; liquidate entire position (sets qty to 0)
                 - optional for SELLs: action['pre_liq'] == True -> attempt to execute this sell BEFORE liquidation check
                    using a conservative intrabar assumption: if bar's Low <= trigger_price, fill at min(trigger_price, Low).
      - trigger_price: execution level if provided; for buys, used when buy_at='trigger'; for sells, used if provided, else Close.

        Notes:
            - Any daily/weekly drop filters should be implemented inside the provided buy_fn/sell_fn.
            - If indicators are needed, compute them via precompute_fn before running.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        start_fund: float,
        target_leverage: float,
        fee_pct: float = 0.0002,
        mmr: float = 0.005,
        buy_on_start: bool = True,
        print_daily: bool = True,
        buy_at: str = "close",
        buy_fn: Optional[Callable[[pd.DataFrame, int], Tuple]] = None,
        sell_fn: Optional[Callable[[pd.DataFrame, int], Tuple]] = None,
        precompute_fn: Optional[Callable[..., Optional[pd.DataFrame]]] = None,
        start_date: str = None,
        end_date: str = None,
    ):
        # --- Validate & prepare data ---
        df = data.copy()
        # Accept 'date' column as index if present
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

        if start_date is not None:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df.index <= pd.to_datetime(end_date)]

        # Enforce required title-cased OHLC columns
        required_cols = {'Open', 'High', 'Low', 'Close'}
        if not required_cols.issubset(df.columns):
            raise ValueError("DataFrame must have columns: 'Open','High','Low','Close'")

        df = df.sort_index()

        # --- Validate parameters ---
        if mmr <= 0 or mmr >= 1:
            raise ValueError("mmr must be between 0 and 1 (e.g., 0.005 = 0.5%)")
        if target_leverage <= 0:
            raise ValueError("target_leverage must be positive")
        if fee_pct < 0:
            raise ValueError("fee_pct cannot be negative")
        if buy_at not in ("close", "trigger"):
            raise ValueError("buy_at must be either 'close' or 'trigger'")

        # --- Store inputs ---
        self.df = df
        self.start_fund = float(start_fund)
        self.target_leverage = float(target_leverage)
        self.fee_pct = float(fee_pct)
        self.mmr = float(mmr)
        self.buy_on_start = bool(buy_on_start)
        self.print_daily = bool(print_daily)
        self.buy_at = buy_at
        self.buy_fn = buy_fn
        self.sell_fn = sell_fn
        self.precompute_fn = precompute_fn

        # --- Outputs (initialized) ---
        self.equity_curve: Optional[pd.Series] = None
        self.unrealized_curve: Optional[pd.Series] = None
        self.position_qty: Optional[pd.Series] = None
        self.position_avg_price: Optional[pd.Series] = None
        self.unrealized_df: Optional[pd.DataFrame] = None
        self.max_drawdown_amount: Optional[float] = None
        self.max_drawdown_pct: Optional[float] = None
        self.final_fund: Optional[float] = None
        self.trades: List[Dict] = []
        self.liquidation_event: Optional[Dict] = None

        # --- Optional: precompute indicators/features ---
        if self.precompute_fn is not None:
            # Compute indicators/columns once before running
            maybe_df = self.precompute_fn(self.df)
            if isinstance(maybe_df, pd.DataFrame):
                # Allow function to return a new DF (otherwise assume in-place)
                self.df = maybe_df.sort_index()

    # --- Internal helper: cross-margin liquidation price ---
    def _liq_price_cross(self, qty: float, avg_price: float, fees_paid_total: float, realized_pnl_total: float) -> float:
        """
        Cross-margin liquidation price solving E = mmr * N.
        Returns -inf if no position (so it never triggers).
        """
        if qty <= 0 or not np.isfinite(avg_price):
            return -np.inf
        denom = (self.mmr - 1.0) * qty  # negative for long (since mmr < 1)
        if denom == 0:
            return np.inf
        # Include realized PnL in equity base
        return (self.start_fund + realized_pnl_total - fees_paid_total - qty * avg_price) / denom

    # --- Main run ---
    def run(self) -> None:
        """
        Execute the backtest over the provided data and strategy.
        """
        idx = self.df.index

        # State variables
        qty = 0.0
        avg_price = np.nan
        fees_paid_total = 0.0
        realized_pnl_total = 0.0

        # Time series collectors
        equity_ts, unrealized_ts, qty_ts, avg_price_ts = [], [], [], []

        first_bar = True
        for i, (dt, row) in enumerate(self.df.iterrows()):
            o = float(row['Open'])
            l = float(row['Low'])
            c = float(row['Close'])

            # --- Optional initial buy at OPEN on first day ---
            if first_bar and self.buy_on_start and qty == 0.0:
                target_notional = self.target_leverage * self.start_fund
                if target_notional > 0.0 and o > 0.0:
                    add_qty = target_notional / o
                    fee_paid = self.fee_pct * target_notional
                    fees_paid_total += fee_paid

                    avg_price = o
                    qty += add_qty

                    self.trades.append({
                        'date': dt,
                        'price': o,
                        'qty': float(add_qty),
                        'notional': float(target_notional),
                        'fee_paid': float(fee_paid),
                        'reason': 'init-buy'
                    })

                    # Immediate liquidation check within the same bar (using LOW)
                    liq_price_init = self._liq_price_cross(qty, avg_price, fees_paid_total, realized_pnl_total)
                    if l <= liq_price_init:
                        self.liquidation_event = {
                            'date': dt,
                            'liq_price': float(liq_price_init),
                            'equity_before': float(self.start_fund + realized_pnl_total - fees_paid_total),
                            'qty_before': float(qty),
                            'avg_price': float(avg_price),
                        }
                        equity = 0.0
                        unrealized = -(self.start_fund + realized_pnl_total - fees_paid_total)
                        qty = 0.0
                        avg_price = np.nan

                        if self.print_daily:
                            liq_str = f"{liq_price_init:.4f}" if np.isfinite(liq_price_init) else "NA"
                            print(f"{dt.date()} | PRICE: {c:.4f} | EQUITY: {equity:,.2f} | LIQ_PRICE: {liq_str} | *** LIQUIDATED ***")

                        equity_ts.append(equity)
                        unrealized_ts.append(unrealized)
                        qty_ts.append(qty)
                        avg_price_ts.append(avg_price)

                        self.trades.append({
                            'date': dt,
                            'price': float(liq_price_init),
                            'qty': 0.0,
                            'notional': 0.0,
                            'fee_paid': 0.0,
                            'reason': 'liquidation'
                        })
                        break
                first_bar = False

            # Compute equity and notional using CLOSE
            unrealized = qty * (c - avg_price) if qty != 0.0 and np.isfinite(avg_price) else 0.0
            equity = self.start_fund + realized_pnl_total + unrealized - fees_paid_total
            notional = qty * c

            # --- Optional pre-liquidation SELL (stop-like) ---
            # If sell_fn returns action with {'pre_liq': True} and a trigger price that is hit by today's range,
            # execute a sell BEFORE liquidation check using conservative intrabar fill at min(trigger_price, Low).
            preliq_executed = False
            if callable(self.sell_fn) and qty > 0.0:
                out_pre = self.sell_fn(self.df, i)
                if isinstance(out_pre, tuple) and len(out_pre) >= 3:
                    should_pre, pre_price, pre_action = out_pre[0], out_pre[1], out_pre[2]
                    if bool(should_pre) and isinstance(pre_action, dict) and pre_action.get('pre_liq', False):
                        if pre_price is not None and np.isfinite(pre_price):
                            trigger_price = float(pre_price)
                            # For a sell stop, if Low <= trigger, assume it triggered intrabar
                            if l <= trigger_price and qty > 0.0:
                                # Worst-case execution for a sell stop: at min(trigger, Low)
                                p_exec_sell_pre = float(min(trigger_price, l))

                                # Determine reduction amount based on action
                                reduce_notional = 0.0
                                act_type = pre_action.get('type') if isinstance(pre_action, dict) else None
                                if act_type == 'to_leverage':
                                    target_lev = float(pre_action.get('leverage', 0.0))
                                    target_notional = max(0.0, target_lev * equity)
                                    reduce_notional = max(0.0, notional - target_notional)
                                elif act_type == 'percent':
                                    pct = float(pre_action.get('pct', 0.0))
                                    reduce_notional = max(0.0, pct * notional)
                                elif act_type == 'to_zero':
                                    reduce_notional = max(0.0, notional)

                                if reduce_notional > 0.0 and p_exec_sell_pre > 0.0:
                                    reduce_qty = min(qty, reduce_notional / p_exec_sell_pre)
                                    realized_pnl = reduce_qty * (p_exec_sell_pre - avg_price)
                                    realized_pnl_total += realized_pnl
                                    fee_paid = self.fee_pct * (reduce_qty * p_exec_sell_pre)
                                    fees_paid_total += fee_paid
                                    qty -= reduce_qty
                                    if qty <= 0.0:
                                        qty = 0.0
                                        avg_price = np.nan

                                    self.trades.append({
                                        'date': dt,
                                        'price': p_exec_sell_pre,
                                        'qty': float(-reduce_qty),
                                        'notional': float(reduce_qty * p_exec_sell_pre),
                                        'fee_paid': float(fee_paid),
                                        'reason': 'pre-liq-sell'
                                    })

                                    # Recompute equity and notional post pre-liq sell
                                    unrealized = qty * (c - avg_price) if qty != 0.0 and np.isfinite(avg_price) else 0.0
                                    equity = self.start_fund + realized_pnl_total + unrealized - fees_paid_total
                                    notional = qty * c
                                    preliq_executed = True

            # --- Liquidation check using LOW (intraday) ---
            liq_price = self._liq_price_cross(qty, avg_price, fees_paid_total, realized_pnl_total)
            if qty > 0.0 and l <= liq_price:
                self.liquidation_event = {
                    'date': dt,
                    'liq_price': float(liq_price),
                    'equity_before': float(equity),
                    'qty_before': float(qty),
                    'avg_price': float(avg_price) if np.isfinite(avg_price) else None,
                }
                equity = 0.0
                unrealized = -(self.start_fund + realized_pnl_total - fees_paid_total)
                qty = 0.0
                avg_price = np.nan

                if self.print_daily:
                    liq_str = f"{liq_price:.4f}" if np.isfinite(liq_price) else "NA"
                    print(f"{dt.date()} | PRICE: {c:.4f} | EQUITY: {equity:,.2f} | LIQ_PRICE: {liq_str} | *** LIQUIDATED ***")

                equity_ts.append(equity)
                unrealized_ts.append(unrealized)
                qty_ts.append(qty)
                avg_price_ts.append(avg_price)

                self.trades.append({
                    'date': dt,
                    'price': float(liq_price),
                    'qty': 0.0,
                    'notional': 0.0,
                    'fee_paid': 0.0,
                    'reason': 'liquidation'
                })
                break
            
            # --- Signal handling (sell first, then buy) ---
            # SELL
            should_sell, sell_price, sell_action = (False, None, None)
            if not preliq_executed and callable(self.sell_fn):
                out = self.sell_fn(self.df, i)
                if isinstance(out, tuple):
                    if len(out) == 2:
                        should_sell, sell_price = out
                        sell_action = {'type': 'to_zero'}
                    elif len(out) >= 3:
                        should_sell, sell_price, sell_action = out[0], out[1], out[2]
            if bool(should_sell) and qty > 0.0:
                p_exec_sell = float(c)
                if sell_price is not None and np.isfinite(sell_price):
                    p_exec_sell = float(sell_price)

                reduce_notional = 0.0
                if isinstance(sell_action, dict):
                    act_type = sell_action.get('type')
                    if act_type == 'to_leverage':
                        target_lev = float(sell_action.get('leverage', 0.0))
                        target_notional = max(0.0, target_lev * equity)
                        reduce_notional = max(0.0, notional - target_notional)
                    elif act_type == 'percent':
                        pct = float(sell_action.get('pct', 0.0))
                        reduce_notional = max(0.0, pct * notional)
                    elif act_type == 'to_zero':
                        reduce_notional = max(0.0, notional)
                else:
                    reduce_notional = notional

                if reduce_notional > 0.0 and p_exec_sell > 0.0:
                    reduce_qty = min(qty, reduce_notional / p_exec_sell)
                    realized_pnl = reduce_qty * (p_exec_sell - avg_price)
                    realized_pnl_total += realized_pnl
                    fee_paid = self.fee_pct * (reduce_qty * p_exec_sell)
                    fees_paid_total += fee_paid
                    qty -= reduce_qty
                    if qty <= 0.0:
                        qty = 0.0
                        avg_price = np.nan

                    self.trades.append({
                        'date': dt,
                        'price': p_exec_sell,
                        'qty': float(-reduce_qty),
                        'notional': float(reduce_qty * p_exec_sell),
                        'fee_paid': float(fee_paid),
                        'reason': 'signal-sell'
                    })

                    unrealized = qty * (c - avg_price) if qty != 0.0 and np.isfinite(avg_price) else 0.0
                    equity = self.start_fund + realized_pnl_total + unrealized - fees_paid_total
                    notional = qty * c

            # BUY
            should_buy, buy_price, buy_action = (False, None, None)
            if callable(self.buy_fn):
                out = self.buy_fn(self.df, i)
                if isinstance(out, tuple):
                    if len(out) == 2:
                        should_buy, buy_price = out
                        buy_action = {'type': 'to_leverage', 'leverage': self.target_leverage}
                    elif len(out) >= 3:
                        should_buy, buy_price, buy_action = out[0], out[1], out[2]
            should_buy = bool(should_buy) and (equity > 0.0)

            if should_buy:
                add_notional = 0.0
                if isinstance(buy_action, dict):
                    act_type = buy_action.get('type')
                    if act_type == 'to_leverage':
                        target_lev = float(buy_action.get('leverage', self.target_leverage))
                        target_notional = max(0.0, target_lev * equity)
                        add_notional = max(0.0, target_notional - notional)
                    elif act_type == 'percent':
                        pct = float(buy_action.get('pct', 0.0))
                        add_notional = max(0.0, pct * notional)
                if add_notional > 0.0:
                    p_exec = c
                    if self.buy_at == 'trigger' and buy_price is not None and np.isfinite(buy_price):
                        p_exec = float(buy_price)
                    if p_exec <= 0.0:
                        p_exec = c

                    add_qty = add_notional / p_exec
                    fee_paid = self.fee_pct * add_notional
                    fees_paid_total += fee_paid

                    if qty == 0.0:
                        avg_price = p_exec
                    else:
                        avg_price = (qty * avg_price + add_qty * p_exec) / (qty + add_qty)
                    qty += add_qty

                    self.trades.append({
                        'date': dt,
                        'price': p_exec,
                        'qty': float(add_qty),
                        'notional': float(add_notional),
                        'fee_paid': float(fee_paid),
                        'reason': 'signal-buy'
                    })

                    unrealized = qty * (c - avg_price) if qty != 0.0 and np.isfinite(avg_price) else 0.0
                    equity = self.start_fund + realized_pnl_total + unrealized - fees_paid_total
                    notional = qty * c

            # --- Daily print ---
            if self.print_daily:
                liq_price_report = self._liq_price_cross(qty, avg_price, fees_paid_total, realized_pnl_total)
                liq_str = f"{liq_price_report:.4f}" if np.isfinite(liq_price_report) else "NA"
                print(f"{dt.date()} | EQUITY: {equity:,.2f} | PRICE: {c:.4f} | LIQ_PRICE: {liq_str}")

            # Collect outputs for this bar
            equity_ts.append(equity)
            unrealized_ts.append(unrealized)
            qty_ts.append(qty)
            avg_price_ts.append(avg_price)

            if first_bar:
                first_bar = False

        # --- Build time series outputs ---
        out_idx = idx[:len(equity_ts)]
        self.equity_curve = pd.Series(equity_ts, index=out_idx, name='equity')
        self.unrealized_curve = pd.Series(unrealized_ts, index=out_idx, name='unrealized_pnl')
        self.position_qty = pd.Series(qty_ts, index=out_idx, name='position_qty')
        self.position_avg_price = pd.Series(avg_price_ts, index=out_idx, name='avg_entry_price')

        # --- Drawdown metrics from equity curve ---
        roll_max = self.equity_curve.cummax()
        dd_curve = self.equity_curve - roll_max
        dd_pct_series = (self.equity_curve / roll_max) - 1.0

        self.max_drawdown_amount = float(dd_curve.min())
        with np.errstate(divide='ignore', invalid='ignore'):
            self.max_drawdown_pct = float((self.equity_curve / roll_max - 1.0).min()) * 100.0

        # Final fund at last processed date
        self.final_fund = float(self.equity_curve.iloc[-1])

        # Make a DataFrame for unrealized PnL for easy downstream use
        low_series = self.df.loc[out_idx, 'Low']
        qty_series = pd.Series(qty_ts, index=out_idx, name='qty')
        self.unrealized_df = pd.DataFrame(
            {
                'unrealized_pnl': self.unrealized_curve.values,
                'low': low_series.values,
                'qty': qty_series.values,
                'drawdown_pct': dd_pct_series.values
            },
            index=out_idx
        )

    def plot_unrealized(self) -> None:
        """
        Plot four stacked charts (price, unrealized PnL, position qty, drawdown%) over time using self.unrealized_df.
        """
        if self.unrealized_df is None or self.unrealized_df.empty:
            raise RuntimeError("Backtester has not been run yet or there is no data to plot. Call run() before plotting.")

        df = self.unrealized_df.copy()
        df.index = pd.to_datetime(df.index)

        # Determine price series (prefer Close from original df, fallback to low)
        if 'Close' in self.df.columns:
            price_series = self.df.loc[df.index, 'Close']
        else:
            price_series = df['low']

        fig, axes = plt.subplots(4, 1, figsize=(20, 10), sharex=True)

        # Price
        axes[0].plot(df.index, price_series, color='tab:blue', label='Price')
        axes[0].set_ylabel('Price')
        axes[0].legend(loc='upper left')
        axes[0].grid(True)

        # Unrealized PnL
        axes[1].plot(df.index, df['unrealized_pnl'], color='tab:green', label='Unrealized PnL')
        axes[1].set_ylabel('Unrealized PnL')
        axes[1].legend(loc='upper left')
        axes[1].grid(True)

        # Position Qty
        axes[2].bar(df.index, df['qty'], color='tab:orange', label='Position Qty')
        axes[2].set_ylabel('Qty')
        axes[2].set_xlabel('Date')
        axes[2].legend(loc='upper left')
        axes[2].grid(True)

        # Drawdown
        axes[3].plot(df.index, df['drawdown_pct'] * 100, color='tab:red', label='Drawdown (%)')
        axes[3].set_ylabel('Drawdown (%)')
        axes[3].set_xlabel('Date')
        axes[3].legend(loc='upper left')
        axes[3].grid(True)

        fig.suptitle('Price, Unrealized PnL, and Position Quantity')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()