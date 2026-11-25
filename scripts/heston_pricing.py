"""Shared Heston pricing helpers for notebooks 05-07.

All prices are European call prices under the risk-neutral Heston model.
Spot is fixed to 100 in the project experiment; the ML feature uses
log-forward-moneyness, log(K / (S exp((r-q)T))).
"""

from __future__ import annotations

import math
import time
from typing import Iterable

import numpy as np


SPOT = 100.0
FEATURE_NAMES = [
    "log_forward_moneyness", "T", "r", "q", "kappa", "rho", "xi", "theta", "v0"
]
PARAMETER_NAMES = ["T", "r", "q", "kappa", "rho", "xi", "theta", "v0"]


def bounded_exponential(rng: np.random.Generator, n: int, low: float,
                        high: float, scale: float = 0.025) -> np.ndarray:
    """Sample a truncated exponential, concentrating more draws near low."""
    u = rng.random(n)
    mass = 1.0 - math.exp(-(high - low) / scale)
    return low - scale * np.log1p(-u * mass)


def sample_contexts(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Draw independent Heston market/model contexts from the paper ranges."""
    return {
        "T": rng.uniform(11.0 / 12.0, 1.0, n),
        "r": rng.uniform(0.015, 0.025, n),
        "q": rng.uniform(0.0, 0.05, n),
        "kappa": rng.uniform(1.4, 2.6, n),
        "rho": rng.uniform(-0.85, -0.55, n),
        # Paper notation: theta is vol-of-vol; eta is long-run variance.
        # This project uses xi for vol-of-vol and theta for long-run variance.
        "xi": rng.uniform(0.45, 0.75, n),
        "theta": rng.uniform(0.01, 0.10, n),
        "v0": bounded_exponential(rng, n, 0.01, 0.10, scale=0.025),
    }


def sample_fresh_options(n: int, seed: int = 2027) -> list[dict[str, float]]:
    """Draw unseen comparison options inside the paper's narrower test ranges."""
    rng = np.random.default_rng(seed)
    ctx = sample_contexts(n, rng)
    ctx["kappa"] = rng.uniform(1.5, 2.5, n)
    ctx["rho"] = rng.uniform(-0.8, -0.6, n)
    ctx["xi"] = rng.uniform(0.5, 0.7, n)
    ctx["theta"] = rng.uniform(0.02, 0.10, n)
    ctx["v0"] = bounded_exponential(rng, n, 0.02, 0.10, scale=0.025)
    moneyness = rng.uniform(0.5, 1.5, n)
    options = []
    for i in range(n):
        p = {name: float(values[i]) for name, values in ctx.items()}
        forward = SPOT * math.exp((p["r"] - p["q"]) * p["T"])
        p.update(S=SPOT, forward_moneyness=float(moneyness[i]), K=float(moneyness[i] * forward))
        options.append(p)
    return options


def feature_matrix(options: Iterable[dict[str, float]]) -> np.ndarray:
    """Convert option dictionaries to the fixed 9-feature model schema."""
    rows = []
    for p in options:
        log_m = math.log(p["K"] / (p["S"] * math.exp((p["r"] - p["q"]) * p["T"])))
        rows.append([log_m, p["T"], p["r"], p["q"], p["kappa"], p["rho"],
                     p["xi"], p["theta"], p["v0"]])
    return np.asarray(rows, dtype=np.float64)


def heston_characteristic(u: np.ndarray | complex, *, S: float, r: float, q: float,
                          T: float, kappa: float, theta: float, xi: float,
                          rho: float, v0: float) -> np.ndarray:
    """Characteristic function of log(S_T), using the stable little-trap form."""
    u = np.asarray(u, dtype=np.complex128)
    beta = kappa - rho * xi * 1j * u
    d = np.sqrt(beta * beta + xi * xi * (u * u + 1j * u))
    d = np.where(d.real < 0.0, -d, d)
    denom = beta + d
    g = (beta - d) / denom
    exp_dt = np.exp(-d * T)
    log_ratio = np.log1p(-g * exp_dt) - np.log1p(-g)
    c = (1j * u * (math.log(S) + (r - q) * T)
         + (kappa * theta / (xi * xi)) * ((beta - d) * T - 2.0 * log_ratio))
    D = ((beta - d) / (xi * xi)) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))
    return np.exp(c + D * v0)


def heston_fft_calls(context: dict[str, float], strikes: np.ndarray | list[float],
                     n: int = 4096, eta: float = 0.25,
                     alpha: float = 1.25) -> np.ndarray:
    """Carr-Madan damped FFT prices for many strikes under one Heston context."""
    if n < 256 or n & (n - 1):
        raise ValueError("FFT size must be a power of two of at least 256")
    strikes = np.asarray(strikes, dtype=np.float64)
    if np.any(strikes <= 0.0):
        raise ValueError("Strikes must be positive")
    S = float(context.get("S", SPOT))
    r, q, T = (float(context[k]) for k in ("r", "q", "T"))
    kappa, theta, xi, rho, v0 = (float(context[k]) for k in ("kappa", "theta", "xi", "rho", "v0"))

    v = eta * np.arange(n, dtype=np.float64)
    lam = 2.0 * np.pi / (n * eta)
    b = 0.5 * n * lam
    u = v - 1j * (alpha + 1.0)
    phi = heston_characteristic(u, S=S, r=r, q=q, T=T, kappa=kappa,
                                theta=theta, xi=xi, rho=rho, v0=v0)
    denom = alpha * alpha + alpha - v * v + 1j * (2.0 * alpha + 1.0) * v
    psi = np.exp(-r * T) * phi / denom

    # Composite-Simpson weights in the standard Carr-Madan FFT layout.
    weights = np.full(n, 2.0 / 3.0)
    weights[1::2] = 4.0 / 3.0
    weights[0] = 1.0 / 3.0
    fft_input = np.exp(1j * b * v) * psi * eta * weights
    k_grid = -b + lam * np.arange(n)
    call_grid = np.exp(-alpha * k_grid) * np.fft.fft(fft_input).real / np.pi
    price = np.interp(np.log(strikes), k_grid, call_grid)

    # Apply only the elementary lower/upper call bounds to suppress tiny FFT
    # ringing outside the useful strike grid.
    lower = np.maximum(S * math.exp(-q * T) - strikes * math.exp(-r * T), 0.0)
    upper = S * math.exp(-q * T)
    return np.clip(price, lower, upper)


def heston_mc_call(option: dict[str, float], n_paths: int = 20_000,
                   n_steps: int = 126, seed: int | None = None) -> tuple[float, float]:
    """Full-truncation Euler Monte Carlo with antithetic Brownian increments.

    Returns (discounted call estimate, standard error based on pair averages).
    """
    if n_paths < 2 or n_paths % 2:
        raise ValueError("n_paths must be an even integer >= 2")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    rng = np.random.default_rng(seed)
    S0, K = float(option.get("S", SPOT)), float(option["K"])
    r, q, T = (float(option[k]) for k in ("r", "q", "T"))
    kappa, theta, xi, rho, v0 = (float(option[k]) for k in ("kappa", "theta", "xi", "rho", "v0"))
    half = n_paths // 2
    dt, sqrt_dt = T / n_steps, math.sqrt(T / n_steps)
    s_a = np.full(half, S0, dtype=np.float64)
    s_b = np.full(half, S0, dtype=np.float64)
    v_a = np.full(half, v0, dtype=np.float64)
    v_b = np.full(half, v0, dtype=np.float64)
    orth = math.sqrt(max(0.0, 1.0 - rho * rho))

    for _ in range(n_steps):
        z1 = rng.standard_normal(half)
        z2 = rng.standard_normal(half)
        zv_a = rho * z1 + orth * z2
        zv_b = -zv_a
        vpa, vpb = np.maximum(v_a, 0.0), np.maximum(v_b, 0.0)
        s_a *= np.exp((r - q - 0.5 * vpa) * dt + np.sqrt(vpa) * sqrt_dt * z1)
        s_b *= np.exp((r - q - 0.5 * vpb) * dt - np.sqrt(vpb) * sqrt_dt * z1)
        v_a = np.maximum(v_a + kappa * (theta - vpa) * dt + xi * np.sqrt(vpa) * sqrt_dt * zv_a, 0.0)
        v_b = np.maximum(v_b + kappa * (theta - vpb) * dt + xi * np.sqrt(vpb) * sqrt_dt * zv_b, 0.0)

    pair_payoffs = 0.5 * (np.maximum(s_a - K, 0.0) + np.maximum(s_b - K, 0.0))
    discounted = math.exp(-r * T) * pair_payoffs
    return float(discounted.mean()), float(discounted.std(ddof=1) / math.sqrt(half))


def timed_call(function, *args, **kwargs):
    """Return function result and elapsed wall seconds."""
    start = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - start
