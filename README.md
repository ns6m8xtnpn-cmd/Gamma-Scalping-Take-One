diff --git a/gamma_scalping.py b/gamma_scalping.py
new file mode 100644
index 0000000000000000000000000000000000000000..34e6a54c6eae2642baa7817454ced31c08a76eb6
--- /dev/null
+++ b/gamma_scalping.py
@@ -0,0 +1,295 @@
+"""Self-contained gamma scalping simulation with optional scientific stack support.
+
+The script simulates a delta-hedged long call position under geometric Brownian
+motion. If NumPy/pandas/matplotlib/scipy are available, they are used for speed
+and plotting; otherwise, a lightweight pure-Python fallback runs without external
+packages so the demo still executes in restricted environments.
+"""
+
+from __future__ import annotations
+
+import importlib.util
+import math
+import random
+from dataclasses import dataclass
+from typing import Dict, List, Sequence
+
+# -----------------------------
+# Optional third-party support
+# -----------------------------
+HAS_NUMPY = importlib.util.find_spec("numpy") is not None
+HAS_PANDAS = importlib.util.find_spec("pandas") is not None
+HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
+HAS_SCIPY = importlib.util.find_spec("scipy") is not None
+
+if HAS_NUMPY:
+    import numpy as np  # type: ignore
+
+if HAS_PANDAS:
+    import pandas as pd  # type: ignore
+
+if HAS_MATPLOTLIB:
+    import matplotlib.pyplot as plt  # type: ignore
+
+if HAS_SCIPY:
+    from scipy.stats import norm  # type: ignore
+
+# -----------------------------
+# Parameters and configuration
+# -----------------------------
+S0 = 100.0  # initial stock price
+K = 100.0  # strike price
+r = 0.02  # risk-free rate
+sigma = 0.20  # implied volatility used for pricing and hedging
+actual_vol = 0.25  # realized volatility used in the price simulation
+mu = r  # drift of the stock under the real-world measure
+T = 0.5  # time to maturity in years
+N = 252  # number of hedge steps
+option_position = 1  # long 1 call option
+random.seed(42)
+
+# Derived quantities
+DT = T / N
+sqrt_dt = math.sqrt(DT)
+time_grid: Sequence[float]
+if HAS_NUMPY:
+    time_grid = np.linspace(0.0, T, N + 1)
+else:
+    time_grid = [i * DT for i in range(N + 1)]
+
+
+# -----------------------
+# Black-Scholes functions
+# -----------------------
+def _norm_cdf(x: float) -> float:
+    if HAS_SCIPY:
+        return norm.cdf(x)  # type: ignore
+    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
+
+
+def _norm_pdf(x: float) -> float:
+    if HAS_SCIPY:
+        return norm.pdf(x)  # type: ignore
+    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
+
+
+def bs_call_price(S: float, strike: float, rate: float, vol: float, time: float) -> float:
+    """Return the Black-Scholes price of a European call option."""
+    if time <= 0:
+        return max(S - strike, 0.0)
+
+    sqrt_t = math.sqrt(time)
+    d1 = (math.log(S / strike) + (rate + 0.5 * vol ** 2) * time) / (vol * sqrt_t)
+    d2 = d1 - vol * sqrt_t
+    return S * _norm_cdf(d1) - strike * math.exp(-rate * time) * _norm_cdf(d2)
+
+
+def bs_call_delta(S: float, strike: float, rate: float, vol: float, time: float) -> float:
+    """Return the Black-Scholes delta of a European call option."""
+    if time <= 0:
+        if S > strike:
+            return 1.0
+        if S < strike:
+            return 0.0
+        return 0.5
+
+    sqrt_t = math.sqrt(time)
+    d1 = (math.log(S / strike) + (rate + 0.5 * vol ** 2) * time) / (vol * sqrt_t)
+    return _norm_cdf(d1)
+
+
+def bs_call_gamma(S: float, strike: float, rate: float, vol: float, time: float) -> float:
+    """Return the Black-Scholes gamma of a European call option."""
+    if time <= 0:
+        return 0.0
+
+    sqrt_t = math.sqrt(time)
+    d1 = (math.log(S / strike) + (rate + 0.5 * vol ** 2) * time) / (vol * sqrt_t)
+    return _norm_pdf(d1) / (S * vol * sqrt_t)
+
+
+# -----------------------
+# Simulation data classes
+# -----------------------
+@dataclass
+class Record:
+    time: float
+    stock: float
+    delta: float
+    gamma: float
+    option_value: float
+    shares: float
+    hedge_trade: float
+    cash: float
+    portfolio_value: float
+    option_pnl: float
+    hedge_pnl: float
+
+
+# -----------------------
+# Stock path simulation
+# -----------------------
+def simulate_price_path() -> Sequence[float]:
+    """Simulate a single GBM price path using NumPy if available, else pure Python."""
+    if HAS_NUMPY:
+        Z = np.random.standard_normal(N)
+        path = np.empty(N + 1)
+        path[0] = S0
+        for t in range(1, N + 1):
+            drift = (mu - 0.5 * actual_vol ** 2) * DT
+            diffusion = actual_vol * sqrt_dt * Z[t - 1]
+            path[t] = path[t - 1] * math.exp(drift + diffusion)
+        return path
+
+    path: List[float] = [0.0] * (N + 1)
+    path[0] = S0
+    for t in range(1, N + 1):
+        z = random.gauss(0.0, 1.0)
+        drift = (mu - 0.5 * actual_vol ** 2) * DT
+        diffusion = actual_vol * sqrt_dt * z
+        path[t] = path[t - 1] * math.exp(drift + diffusion)
+    return path
+
+
+# -------------------------------------------------
+# Gamma scalping / delta hedging simulation storage
+# -------------------------------------------------
+def run_simulation(price_path: Sequence[float]) -> List[Record]:
+    records: List[Record] = []
+
+    initial_option_value = bs_call_price(S0, K, r, sigma, T)
+    initial_delta = bs_call_delta(S0, K, r, sigma, T)
+    current_shares = option_position * initial_delta * -1.0  # short shares to be delta neutral
+    cash = -initial_option_value - current_shares * S0  # pay for option, receive/pay for shares
+
+    initial_portfolio_value = initial_option_value + current_shares * S0 + cash
+    records.append(
+        Record(
+            time=time_grid[0],
+            stock=price_path[0],
+            delta=initial_delta,
+            gamma=bs_call_gamma(S0, K, r, sigma, T),
+            option_value=initial_option_value,
+            shares=current_shares,
+            hedge_trade=current_shares,
+            cash=cash,
+            portfolio_value=initial_portfolio_value,
+            option_pnl=0.0,
+            hedge_pnl=0.0,
+        )
+    )
+
+    for t in range(1, N + 1):
+        S_t = price_path[t]
+        tau = T - t * DT
+        option_value = bs_call_price(S_t, K, r, sigma, tau)
+        delta = bs_call_delta(S_t, K, r, sigma, tau)
+        gamma = bs_call_gamma(S_t, K, r, sigma, tau)
+
+        target_shares = option_position * delta * -1.0
+        hedge_trade = target_shares - current_shares
+
+        cash -= hedge_trade * S_t
+        current_shares = target_shares
+
+        stock_value = current_shares * S_t
+        portfolio_value = option_value + stock_value + cash
+        option_pnl = option_value - initial_option_value
+        hedge_value = stock_value + cash
+        hedge_pnl = hedge_value + initial_option_value
+
+        records.append(
+            Record(
+                time=time_grid[t],
+                stock=S_t,
+                delta=delta,
+                gamma=gamma,
+                option_value=option_value,
+                shares=current_shares,
+                hedge_trade=hedge_trade,
+                cash=cash,
+                portfolio_value=portfolio_value,
+                option_pnl=option_pnl,
+                hedge_pnl=hedge_pnl,
+            )
+        )
+
+    return records
+
+
+# -----------------
+# P&L calculations
+# -----------------
+def summarize(records: Sequence[Record]) -> Dict[str, float]:
+    final = records[-1]
+    initial_option_value = records[0].option_value
+    option_pnl = final.option_value - initial_option_value
+    hedge_pnl = final.portfolio_value - option_pnl  # initial portfolio value is zero
+    return {
+        "final_option_pnl": option_pnl,
+        "final_hedge_pnl": hedge_pnl,
+        "total_pnl": final.portfolio_value,
+    }
+
+
+# -----------------
+# Visualization
+# -----------------
+def plot_if_available(records: Sequence[Record]) -> None:
+    if not HAS_MATPLOTLIB or not HAS_NUMPY:
+        print("matplotlib/numpy not installed; skipping plots. The simulation still ran.")
+        return
+
+    results = pd.DataFrame([r.__dict__ for r in records]) if HAS_PANDAS else None
+
+    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
+    times = [r.time for r in records]
+    axes[0].plot(times, [r.stock for r in records], label="Stock price")
+    axes[0].set_ylabel("Price")
+    axes[0].set_title("Simulated stock path")
+    axes[0].legend()
+
+    axes[1].plot(times, [r.delta for r in records], label="Call delta")
+    axes[1].plot(times, [r.shares for r in records], label="Hedge shares")
+    axes[1].set_ylabel("Delta / Shares")
+    axes[1].set_title("Delta and hedge position")
+    axes[1].legend()
+
+    opt_pnl_series = (
+        results["option_value"] - results.loc[0, "option_value"]
+        if results is not None
+        else [r.option_pnl for r in records]
+    )
+    axes[2].plot(times, [r.portfolio_value for r in records], label="Total portfolio")
+    axes[2].plot(times, opt_pnl_series, label="Option P&L")
+    axes[2].set_ylabel("P&L")
+    axes[2].set_xlabel("Time (years)")
+    axes[2].set_title("Portfolio and option P&L")
+    axes[2].legend()
+
+    plt.tight_layout()
+    plt.show()
+
+
+# -----------------
+# Main execution
+# -----------------
+def main() -> None:
+    price_path = simulate_price_path()
+    records = run_simulation(price_path)
+    summary = summarize(records)
+
+    print("--- Gamma Scalping Summary ---")
+    print(f"Final option P&L: {summary['final_option_pnl']: .4f}")
+    print(f"Final hedge P&L:  {summary['final_hedge_pnl']: .4f}")
+    print(f"Total P&L:        {summary['total_pnl']: .4f}")
+    print(
+        "Higher realized volatility than implied should make the hedge P&L positive "
+        "(buying low / selling high). If realized < implied, theta bleed dominates."
+    )
+
+    plot_if_available(records)
+
+
+if __name__ == "__main__":
+    main()
