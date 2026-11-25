#!/usr/bin/env python3
"""Compare Black–Scholes analytic, Monte Carlo, and Carr–Madan FFT prices.

Requires Python 3.10+ and NumPy. Run:
    python outputs/black_scholes_pricing.py
    python outputs/black_scholes_pricing.py --n-options 25 --output results.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from pathlib import Path

import numpy as np


SEED = 20261116
MC_PATHS = 100_000
FFT_N = 16384
FFT_ETA = 0.25
FFT_ALPHA = 1.5


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def bsm_call(spot: float, strike: float, rate: float, sigma: float, tau: float) -> float:
    """European call price, continuously compounded rate, time-to-maturity tau."""
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
    """One-step exact GBM Monte Carlo estimator using caller-supplied N(0,1) draws."""
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
    """Carr–Madan FFT inversion for a Black–Scholes European call.

    Produces a log-strike grid and linearly interpolates at log(strike/spot).
    """
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
    # Characteristic function of X_T = log(S_T), evaluated at u-i(alpha+1).
    z = u - 1j * (alpha + 1.0)
    phi = np.exp(1j * z * log_forward - 0.5 * sigma * sigma * tau * z * z)
    denominator = alpha * alpha + alpha - u * u + 1j * (2.0 * alpha + 1.0) * u
    psi = math.exp(-rate * tau) * phi / denominator

    # Simpson weights for the Fourier integral; k-grid spacing is 2*pi/(N*eta).
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
        # A direct Carr–Madan quadrature fallback avoids extrapolating beyond the FFT grid.
        uu = u
        direct_integrand = np.exp(-1j * uu * target) * psi
        direct_integrand[0] *= 0.5
        price = math.exp(-alpha * target) * eta * float(direct_integrand.real.sum()) / math.pi
    else:
        price = float(np.interp(target, log_strike_grid, calls))
    # Numerical roundoff can create tiny negative prices at extreme strikes.
    return max(price, 0.0)


def make_dataset(count: int, seed: int) -> list[dict[str, float]]:
    """Generate valid parameter sets; rates may be negative, all other inputs positive."""
    rng = np.random.default_rng(seed)
    spots = rng.uniform(50.0, 200.0, count)
    moneyness = rng.uniform(0.70, 1.30, count)
    rates = rng.uniform(-0.02, 0.10, count)
    sigmas = rng.uniform(0.05, 0.80, count)
    taus = rng.uniform(0.05, 3.0, count)
    return [dict(spot=float(s), strike=float(s * m), rate=float(r),
                 sigma=float(v), tau=float(t))
            for s, m, r, v, t in zip(spots, moneyness, rates, sigmas, taus)]


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - start) * 1_000_000.0


def run(count: int, seed: int, paths: int, output: Path) -> None:
    dataset = make_dataset(count, seed)
    rng = np.random.default_rng(seed + 1)
    # Shared standardized draws per option ensure the MC run itself is repeatable.
    rows = []
    for i, p in enumerate(dataset):
        normals = rng.standard_normal(paths)
        analytic, t_analytic = timed(lambda: bsm_call(**p))
        mc, t_mc = timed(lambda: mc_call(**p, normals=normals))
        fft, t_fft = timed(lambda: carr_madan_call(**p))
        rows.append({
            "option_id": i + 1, **p,
            "bsm_analytic": analytic, "mc_price": mc, "fft_price": fft,
            "mc_abs_error": abs(mc - analytic), "mc_relative_error": abs(mc - analytic) / max(analytic, 1e-12),
            "fft_abs_error": abs(fft - analytic), "fft_relative_error": abs(fft - analytic) / max(analytic, 1e-12),
            "analytic_us": t_analytic, "mc_us": t_mc, "fft_us": t_fft,
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} options to {output}")
    print(f"seed={seed}; paths/option={paths}; FFT N={FFT_N}, eta={FFT_ETA}, alpha={FFT_ALPHA}")
    for label, err, timing in (("MC", "mc_abs_error", "mc_us"), ("FFT", "fft_abs_error", "fft_us")):
        print(f"{label}: mean abs error={statistics.mean(r[err] for r in rows):.6g}; "
              f"max abs error={max(r[err] for r in rows):.6g}; "
              f"mean runtime={statistics.mean(r[timing] for r in rows):.1f} us/option")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-options", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--paths", type=int, default=MC_PATHS)
    parser.add_argument("--output", type=Path, default=Path("outputs/pricing_results.csv"))
    args = parser.parse_args()
    if args.n_options < 1 or args.paths < 1:
        parser.error("--n-options and --paths must be positive")
    run(args.n_options, args.seed, args.paths, args.output)


if __name__ == "__main__":
    main()