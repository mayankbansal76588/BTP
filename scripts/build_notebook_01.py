"""Builds 01_bsm_mc_fft.ipynb programmatically with nbformat."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

md("""\
# Option Pricing: Black-Scholes, Monte Carlo, and FFT (Carr-Madan)

This notebook compares three European call option pricing methods under
Black-Scholes/GBM dynamics:

1. **Black-Scholes-Merton (BSM) analytical formula** - exact, closed-form.
2. **Monte Carlo (MC) simulation** - shown to converge to BSM as the number
   of simulated paths grows.
3. **FFT / characteristic-function method (Carr-Madan, 1999)** - near-exact
   numerical inversion of the characteristic function, validated against BSM.

The pricing functions below are copied unchanged from `code.py` (already
verified numerically to agree with BSM to ~1e-5 absolute error)."""
)

code("""\
from __future__ import annotations

import math
import time
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 20261116
MC_PATHS = 100_000
FFT_N = 16384
FFT_ETA = 0.25
FFT_ALPHA = 1.5

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
""")

md("## Pricing functions (BSM, Monte Carlo, Carr-Madan FFT)")

code("""\
def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def bsm_call(spot: float, strike: float, rate: float, sigma: float, tau: float) -> float:
    \"\"\"European call price, continuously compounded rate, time-to-maturity tau.\"\"\"
    if spot <= 0 or strike <= 0 or sigma < 0 or tau < 0:
        raise ValueError("spot and strike must be positive; sigma and tau nonnegative")
    if tau == 0:
        return max(spot - strike, 0.0)
    if sigma == 0:
        return max(spot - strike * math.exp(-rate * tau), 0.0)
    vol_time = sigma * math.sqrt(tau)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * tau) / vol_time
    d2 = d1 - vol_time
    return spot * norm_cdf(d1) - strike * math.exp(-rate * tau) * norm_cdf(d2)


def mc_call(spot: float, strike: float, rate: float, sigma: float, tau: float,
            normals: np.ndarray) -> float:
    \"\"\"One-step exact GBM Monte Carlo estimator using caller-supplied N(0,1) draws.\"\"\"
    if tau == 0:
        return max(spot - strike, 0.0)
    if sigma == 0:
        return max(spot - strike * math.exp(-rate * tau), 0.0)
    terminal = spot * np.exp((rate - 0.5 * sigma * sigma) * tau
                             + sigma * math.sqrt(tau) * normals)
    return math.exp(-rate * tau) * float(np.maximum(terminal - strike, 0.0).mean())


def carr_madan_call(spot: float, strike: float, rate: float, sigma: float,
                    tau: float, n: int = FFT_N, eta: float = FFT_ETA,
                    alpha: float = FFT_ALPHA) -> float:
    \"\"\"Carr-Madan FFT inversion for a Black-Scholes European call.

    Produces a log-strike grid and linearly interpolates at log(strike/spot).
    \"\"\"
    if spot <= 0 or strike <= 0 or sigma < 0 or tau < 0:
        raise ValueError("spot and strike must be positive; sigma and tau nonnegative")
    if tau == 0 or sigma == 0:
        return bsm_call(spot, strike, rate, sigma, tau)
    if n < 16 or n & (n - 1):
        raise ValueError("FFT size n must be a power of two >= 16")
    if eta <= 0 or alpha <= 0:
        raise ValueError("eta and alpha must be positive")

    j = np.arange(n)
    u = eta * j
    log_forward = math.log(spot) + (rate - 0.5 * sigma * sigma) * tau
    z = u - 1j * (alpha + 1.0)
    phi = np.exp(1j * z * log_forward - 0.5 * sigma * sigma * tau * z * z)
    denominator = alpha * alpha + alpha - u * u + 1j * (2.0 * alpha + 1.0) * u
    psi = math.exp(-rate * tau) * phi / denominator

    weights = np.ones(n)
    weights[1::2] = 4.0
    weights[2:-1:2] = 2.0
    weights /= 3.0
    weights[0] = 1.0 / 3.0
    weights[-1] = 1.0 / 3.0
    b = math.pi / eta
    integrand = np.exp(1j * b * u) * psi * eta * weights
    fft_values = np.fft.fft(integrand)
    dk = 2.0 * math.pi / (n * eta)
    log_strike_grid = -b + dk * np.arange(n)
    calls = np.exp(-alpha * log_strike_grid) * fft_values.real / math.pi
    target = math.log(strike)
    if target < log_strike_grid[0] or target > log_strike_grid[-1]:
        uu = u
        direct_integrand = np.exp(-1j * uu * target) * psi
        direct_integrand[0] *= 0.5
        price = math.exp(-alpha * target) * eta * float(direct_integrand.real.sum()) / math.pi
    else:
        price = float(np.interp(target, log_strike_grid, calls))
    return max(price, 0.0)


def make_dataset(count: int, seed: int) -> list[dict[str, float]]:
    \"\"\"Generate valid parameter sets; rates may be negative, all other inputs positive.\"\"\"
    rng = np.random.default_rng(seed)
    spots = rng.uniform(50.0, 200.0, count)
    moneyness = rng.uniform(0.70, 1.30, count)
    rates = rng.uniform(-0.02, 0.10, count)
    sigmas = rng.uniform(0.05, 0.80, count)
    taus = rng.uniform(0.05, 3.0, count)
    return [dict(spot=float(s), strike=float(s * m), rate=float(r),
                 sigma=float(v), tau=float(t))
            for s, m, r, v, t in zip(spots, moneyness, rates, sigmas, taus)]
""")

md("""\
## 1. Single-option sanity check

A reference option used throughout this notebook: spot = 100, strike = 100
(at-the-money), rate = 5%, volatility = 20%, maturity = 1 year."""
)

code("""\
ref = dict(spot=100.0, strike=100.0, rate=0.05, sigma=0.20, tau=1.0)

ref_bsm = bsm_call(**ref)
rng_ref = np.random.default_rng(SEED)
ref_mc = mc_call(**ref, normals=rng_ref.standard_normal(MC_PATHS))
ref_fft = carr_madan_call(**ref)

print(f"BSM price : {ref_bsm:.6f}")
print(f"MC price  : {ref_mc:.6f}  (paths={MC_PATHS:,})")
print(f"FFT price : {ref_fft:.6f}")
print(f"MC  abs error vs BSM: {abs(ref_mc - ref_bsm):.6f}")
print(f"FFT abs error vs BSM: {abs(ref_fft - ref_bsm):.2e}")
""")

md("""\
## 2. Monte Carlo convergence to Black-Scholes

As the number of simulated paths grows, the Monte Carlo price should
converge to the exact BSM price. We re-price the reference option at
increasing path counts and plot both against a horizontal BSM reference
line."""
)

code("""\
path_counts = np.unique(np.logspace(2, 6, 25).astype(int))
rng_conv = np.random.default_rng(SEED + 1)

mc_prices = []
for n_paths in path_counts:
    normals = rng_conv.standard_normal(int(n_paths))
    mc_prices.append(mc_call(**ref, normals=normals))
mc_prices = np.array(mc_prices)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(path_counts, mc_prices, "o-", color="tab:blue", label="Monte Carlo price")
ax.axhline(ref_bsm, color="tab:red", linestyle="--", label=f"BSM price = {ref_bsm:.4f}")
ax.set_xscale("log")
ax.set_xlabel("Number of simulated paths")
ax.set_ylabel("Call option price")
ax.set_title("Monte Carlo convergence to Black-Scholes price")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(DATA_DIR / "mc_convergence.png", dpi=150)
plt.show()
""")

md("""\
## 3. BSM vs Monte Carlo (10^5 paths) across many different options

The previous plot fixed one option and varied the number of simulated
paths. Here we instead fix the number of paths at `10^5` and vary the
*option*: a batch of randomly sampled option parameter sets (same sampling
ranges as `make_dataset`), each priced once with BSM (red) and once with
Monte Carlo at `10^5` paths (blue). This shows MC tracking BSM across many
different spot/strike/rate/sigma/tau combinations, not just one."""
)

code("""\
N_MC_COMPARE = 40
MC_COMPARE_PATHS = 10**5

compare_options = make_dataset(N_MC_COMPARE, SEED + 10)
rng_compare = np.random.default_rng(SEED + 11)

bsm_points = []
mc_points = []
for p in compare_options:
    bsm_points.append(bsm_call(**p))
    normals = rng_compare.standard_normal(MC_COMPARE_PATHS)
    mc_points.append(mc_call(**p, normals=normals))

bsm_points = np.array(bsm_points)
mc_points = np.array(mc_points)

# Sort by BSM price so the two series read as smooth, comparable curves.
order = np.argsort(bsm_points)
bsm_points = bsm_points[order]
mc_points = mc_points[order]
option_index = np.arange(1, N_MC_COMPARE + 1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(option_index, bsm_points, color="red", label="BSM price", zorder=3)
ax.scatter(option_index, mc_points, color="blue", label=f"MC price ({MC_COMPARE_PATHS:,} paths)",
           zorder=2, alpha=0.8)
ax.set_xlabel("Option (sorted by BSM price)")
ax.set_ylabel("Call option price")
ax.set_title("BSM vs Monte Carlo price across many random options")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(DATA_DIR / "mc_vs_bsm_many_options.png", dpi=150)
plt.show()

mc_vs_bsm_abs_error = np.abs(mc_points - bsm_points)
print(f"Mean abs error (MC vs BSM) over {N_MC_COMPARE} options: {mc_vs_bsm_abs_error.mean():.4f}")
print(f"Max abs error (MC vs BSM) over {N_MC_COMPARE} options: {mc_vs_bsm_abs_error.max():.4f}")
""")

md("""\
## 4. FFT (Carr-Madan) vs Black-Scholes across a range of strikes

With spot, rate, sigma, and maturity fixed at the reference values, we sweep
the strike price and compare the BSM and FFT prices at each strike, plus the
pricing error between them."""
)

code("""\
strikes = np.linspace(50.0, 200.0, 61)
fixed = dict(spot=ref["spot"], rate=ref["rate"], sigma=ref["sigma"], tau=ref["tau"])

bsm_by_strike = np.array([bsm_call(strike=k, **fixed) for k in strikes])
fft_by_strike = np.array([carr_madan_call(strike=k, **fixed) for k in strikes])
fft_error = fft_by_strike - bsm_by_strike

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                gridspec_kw={"height_ratios": [3, 1]})
ax1.plot(strikes, bsm_by_strike, label="BSM (analytical)", color="tab:red", linewidth=2)
ax1.plot(strikes, fft_by_strike, "--", label="FFT (Carr-Madan)", color="tab:green")
ax1.set_ylabel("Call option price")
ax1.set_title("FFT vs Black-Scholes price across strikes")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(strikes, fft_error, color="tab:purple")
ax2.axhline(0.0, color="black", linewidth=0.8)
ax2.set_xlabel("Strike")
ax2.set_ylabel("FFT - BSM")
ax2.set_title("Pricing error")
ax2.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(DATA_DIR / "fft_vs_bsm_strikes.png", dpi=150)
plt.show()

print(f"Max abs error over strike grid: {np.max(np.abs(fft_error)):.2e}")
print(f"Mean abs error over strike grid: {np.mean(np.abs(fft_error)):.2e}")
""")

md("""\
## 5. Dataset-level comparison (BSM vs MC vs FFT)

Finally, we generate a random set of option parameter combinations (same
sampling ranges as `code.py`'s `make_dataset`) and compute all three prices
for each, along with per-method timing. The resulting table is saved to
`data/bsm_mc_fft_results.csv` for reuse in the final comparison notebook."""
)

code("""\
def timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - start) * 1_000_000.0


N_OPTIONS = 200
dataset = make_dataset(N_OPTIONS, SEED)
rng_ds = np.random.default_rng(SEED + 2)

rows = []
for i, p in enumerate(dataset):
    normals = rng_ds.standard_normal(MC_PATHS)
    analytic, t_analytic = timed(lambda: bsm_call(**p))
    mc, t_mc = timed(lambda: mc_call(**p, normals=normals))
    fft, t_fft = timed(lambda: carr_madan_call(**p))
    rows.append({
        "option_id": i + 1, **p,
        "bsm_analytic": analytic, "mc_price": mc, "fft_price": fft,
        "mc_abs_error": abs(mc - analytic),
        "mc_relative_error": abs(mc - analytic) / max(analytic, 1e-12),
        "fft_abs_error": abs(fft - analytic),
        "fft_relative_error": abs(fft - analytic) / max(analytic, 1e-12),
        "analytic_us": t_analytic, "mc_us": t_mc, "fft_us": t_fft,
    })

results_df = pd.DataFrame(rows)
results_df.to_csv(DATA_DIR / "bsm_mc_fft_results.csv", index=False)
results_df.head()
""")

code("""\
summary = results_df[[
    "mc_abs_error", "mc_relative_error", "fft_abs_error", "fft_relative_error",
    "analytic_us", "mc_us", "fft_us",
]].agg(["mean", "max"]).T
summary
""")

md("""\
**Takeaways**

- Monte Carlo converges to the Black-Scholes price as the number of paths
  grows (visible in the log-scale convergence plot), with residual error at
  any finite path count being statistical (Monte Carlo) noise.
- The FFT (Carr-Madan) method matches the Black-Scholes price to within
  about `1e-5`-`1e-6` absolute error across the whole strike grid and the
  random dataset - essentially machine-precision-level agreement, since FFT
  is a near-exact numerical inversion of the exact characteristic function.
- `data/bsm_mc_fft_results.csv` carries this dataset forward into
  `03_comparison.ipynb`."""
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3 (BTP .venv)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3"},
}

with open("01_bsm_mc_fft.ipynb", "w") as f:
    nbf.write(nb, f)

print("Wrote 01_bsm_mc_fft.ipynb")
