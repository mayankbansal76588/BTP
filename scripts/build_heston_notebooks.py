"""Create the three Heston workflow notebooks from maintained cell sources."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def md(source):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source):
    return nbf.v4.new_code_cell(source.strip())


def save(name, cells):
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.update({
        "kernelspec": {"display_name": "Python 3 (.venv)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    })
    nbf.write(notebook, ROOT / name)


save("05_heston_mc_fft.ipynb", [
    md(r"""
# Heston Dynamics: Monte Carlo and FFT

This notebook introduces the Heston risk-neutral dynamics and implements the two numerical pricing methods used to create the surrogate training data. Heston has no Black-Scholes elementary pricing formula; its characteristic function supports Fourier pricing, and Carr-Madan FFT is our numerical reference. Monte Carlo estimates are stochastic and are reported with standard errors.

The four-way comparison is in `07_heston_comparison.ipynb`. All prices here are European calls; spot is fixed at 100 and the remaining inputs use the paper's Heston parameter ranges.
"""),
    code(r"""
from pathlib import Path
import sys
import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if not (ROOT / "scripts").exists():
    raise FileNotFoundError("Open this notebook with the BTP project root as the working directory.")
sys.path.insert(0, str(ROOT / "scripts"))
from heston_pricing import SPOT, heston_fft_calls, heston_mc_call

SEED = 2027
FFT_N = 4096
FFT_ETA = 0.25
FFT_ALPHA = 1.25
plt.style.use("seaborn-v0_8-whitegrid")
print(f"Spot={SPOT:g}; Carr-Madan FFT N={FFT_N}, eta={FFT_ETA}, alpha={FFT_ALPHA}")
"""),
    md(r"""
## Risk-neutral Heston model

\[
\frac{dS_t}{S_t}=(r-q)dt+\sqrt{v_t}\,dW^S_t,\qquad
dv_t=\kappa(\theta-v_t)dt+\xi\sqrt{v_t}\,dW^v_t,
\]
\[
\mathrm{corr}(dW^S_t,dW^v_t)=\rho.
\]

Here `v0` is initial variance, `theta` is long-run variance, `kappa` is mean-reversion speed, `xi` is volatility of variance, and `rho` is the Brownian correlation. The paper uses `eta` for long-run variance and `theta` for vol-of-vol; the code uses the common `theta`/`xi` names.

Monte Carlo uses correlated Brownian increments and full-truncation Euler for variance. FFT uses the Heston characteristic function and Carr-Madan damping. Both price the same discounted European call payoff.
"""),
    md(r"""
## Numerical example and MC convergence

Run the MC at increasing path counts for one at-the-money-forward call. The FFT estimate is the deterministic reference line; the MC error bars show two standard errors from antithetic pair averages.
"""),
    code(r"""
demo = {
    "S": SPOT, "T": 0.95, "r": 0.02, "q": 0.015,
    "kappa": 1.9, "rho": -0.70, "xi": 0.58,
    "theta": 0.045, "v0": 0.050,
}
demo["K"] = demo["S"] * math.exp((demo["r"] - demo["q"]) * demo["T"])
fft_price = heston_fft_calls(demo, [demo["K"]], n=FFT_N, eta=FFT_ETA, alpha=FFT_ALPHA)[0]

mc_rows = []
for paths in (10_000, 30_000, 80_000):
    start = time.perf_counter()
    estimate, standard_error = heston_mc_call(demo, n_paths=paths, n_steps=1000, seed=SEED + paths)
    seconds = time.perf_counter() - start
    mc_rows.append({"paths": paths, "mc_price": estimate, "standard_error": standard_error,
                    "seconds": seconds, "fft_price": fft_price})
mc_convergence = pd.DataFrame(mc_rows)
display(mc_convergence.style.format({"mc_price": "{:.5f}", "standard_error": "{:.5f}",
                                     "seconds": "{:.2f}", "fft_price": "{:.5f}"}))

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.errorbar(mc_convergence["paths"], mc_convergence["mc_price"],
            yerr=2 * mc_convergence["standard_error"], fmt="o-", capsize=4,
            color="tab:orange", label="Heston MC (estimate ± 2 SE)")
ax.axhline(fft_price, color="tab:green", linestyle="--", label="Heston Carr-Madan FFT")
ax.set_xscale("log")
ax.set_xlabel("Monte Carlo paths")
ax.set_ylabel("European call price")
ax.set_title("Heston MC convergence toward FFT")
ax.legend()
fig.tight_layout()
plt.show()
"""),
    md(r"""
## Pricing functions used downstream

The implementation is shared in `scripts/heston_pricing.py` so data generation and the comparison call identical numerical routines. `heston_fft_calls` prices a strike vector for one market/model context; batching strikes this way makes the 100,000-row training dataset practical. `heston_mc_call` returns both a price and a Monte Carlo standard error.

The FFT grid, damping parameter, MC time steps, path count, and sampling ranges remain visible and configurable in the later notebooks. Check grid stability by increasing `FFT_N`; check the MC estimate by increasing both paths and time steps.
"""),
])


save("06_heston_dataset_ml.ipynb", [
    md(r"""
# FFT-Labeled Heston Dataset: NN and GPR

Generate 100,000 European-call prices with the Heston Carr-Madan FFT, then train a feed-forward neural network and a paper-style Gaussian Process Regression surrogate on the same FFT target. Training and validation/test splits are by independent Heston contexts, so all strikes from a context stay together and held-out contexts are unseen.

The NN uses the full training split. Exact GPR has cubic fitting cost, so the default GPR subset is 5,000 rows (configurable, and comparable with the paper's 5k/10k/20k experiments). GPR uses a quadratic trend, ARD squared-exponential kernel, positive observation noise, and standardized features/targets.
"""),
    code(r"""
from pathlib import Path
import sys
import time
import copy

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
from heston_pricing import (SPOT, FEATURE_NAMES, sample_contexts,
                            heston_fft_calls, feature_matrix)

DATA_DIR, MODEL_DIR = ROOT / "data", ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
SEED = 2027
N_CONTEXTS, STRIKES_PER_CONTEXT = 5_000, 20
N_TOTAL = N_CONTEXTS * STRIKES_PER_CONTEXT
FFT_N, FFT_ETA, FFT_ALPHA = 4096, 0.25, 1.25
GPR_N = 5_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
plt.style.use("seaborn-v0_8-whitegrid")
print(f"Rows={N_TOTAL:,}; device={DEVICE}; GPR subset={GPR_N:,}")
"""),
    md(r"""
## 1. Generate the FFT-labeled data

The paper fixes spot at one and samples strike as a percentage of the forward price. We fix spot at 100 and sample the same forward moneyness, so prices remain in currency units. Training ranges follow its Heston experiment: forward moneyness 0.4-1.6; maturity 11-12 months; rate 1.5-2.5%; dividend yield 0-5%; `kappa` 1.4-2.6; `rho` -0.85 to -0.55; `xi` 0.45-0.75; long-run variance `theta` 0.01-0.1; and initial variance `v0` 0.01-0.1. `v0` is sampled from a bounded exponential distribution to give more observations to smaller starting variances.

Each context has 20 independently sampled strikes. The FFT is computed once per context for its 20 strikes. The cached CSV includes the complete parameters and context id.
"""),
    code(r"""
DATA_PATH = DATA_DIR / "heston_fft_dataset_100k.csv"
REBUILD_DATA = False

if DATA_PATH.exists() and not REBUILD_DATA:
    heston_df = pd.read_csv(DATA_PATH)
    print(f"Loaded cached dataset: {len(heston_df):,} rows")
else:
    rng = np.random.default_rng(SEED)
    contexts = sample_contexts(N_CONTEXTS, rng)
    rows = []
    for context_id in range(N_CONTEXTS):
        p = {name: float(values[context_id]) for name, values in contexts.items()}
        p["S"] = SPOT
        forward = SPOT * np.exp((p["r"] - p["q"]) * p["T"])
        m = rng.uniform(0.4, 1.6, STRIKES_PER_CONTEXT)
        strikes = forward * m
        fft_prices = heston_fft_calls(p, strikes, n=FFT_N, eta=FFT_ETA, alpha=FFT_ALPHA)
        for j in range(STRIKES_PER_CONTEXT):
            rows.append({"context_id": context_id, **p,
                         "forward_moneyness": float(m[j]), "K": float(strikes[j]),
                         "fft_price": float(fft_prices[j])})
        if (context_id + 1) % 500 == 0:
            print(f"Priced {context_id + 1:,}/{N_CONTEXTS:,} Heston contexts")
    heston_df = pd.DataFrame(rows)
    heston_df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(heston_df):,} rows to {DATA_PATH}")

display(heston_df.head())
print("Rows / columns:", heston_df.shape)
"""),
    md(r"""
## 2. Split by context and prepare features

Input features are log-forward-moneyness, maturity, rate, dividend yield, and the five Heston parameters. Spot is fixed at 100. Splitting context ids (rather than rows) prevents the model from seeing the same Heston context at different strikes in both train and test.
"""),
    code(r"""
context_ids = np.sort(heston_df["context_id"].unique())
train_ids, holdout_ids = train_test_split(context_ids, test_size=0.30, random_state=SEED)
val_ids, test_ids = train_test_split(holdout_ids, test_size=0.50, random_state=SEED)
train_df = heston_df[heston_df.context_id.isin(train_ids)].reset_index(drop=True)
val_df = heston_df[heston_df.context_id.isin(val_ids)].reset_index(drop=True)
test_df = heston_df[heston_df.context_id.isin(test_ids)].reset_index(drop=True)

X_train = feature_matrix(train_df.to_dict("records"))
X_val = feature_matrix(val_df.to_dict("records"))
X_test = feature_matrix(test_df.to_dict("records"))
y_train = train_df.fft_price.to_numpy(dtype=np.float64)
y_val = val_df.fft_price.to_numpy(dtype=np.float64)
y_test = test_df.fft_price.to_numpy(dtype=np.float64)

print(f"train={len(train_df):,} rows/{len(train_ids):,} contexts; "
      f"validation={len(val_df):,}/{len(val_ids):,}; test={len(test_df):,}/{len(test_ids):,}")
"""),
    md(r"""
## 3. Feed-forward neural network

The NN is a 9-128-128-64-32-1 MLP with ReLU activations, standardized inputs/target, Adam, mini-batches, and early stopping on validation MSE. It uses every training row.
"""),
    code(r"""
sx = StandardScaler().fit(X_train)
sy = StandardScaler().fit(y_train.reshape(-1, 1))

def tensor(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))

Xtr_t = tensor(sx.transform(X_train)).to(DEVICE)
Xva_t = tensor(sx.transform(X_val)).to(DEVICE)
Xte_t = tensor(sx.transform(X_test)).to(DEVICE)
ytr_t = tensor(sy.transform(y_train.reshape(-1, 1))).to(DEVICE)
yva_t = tensor(sy.transform(y_val.reshape(-1, 1))).to(DEVICE)

class HestonMLP(nn.Module):
    def __init__(self, n_features=len(FEATURE_NAMES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x)

torch.manual_seed(SEED)
fnn = HestonMLP().to(DEVICE)
optimizer = torch.optim.Adam(fnn.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
dataset = torch.utils.data.TensorDataset(Xtr_t, ytr_t)
loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)
best_state, best_val, wait = None, float("inf"), 0
PATIENCE, MAX_EPOCHS = 18, 200
history = {"train_mse": [], "val_mse": []}
start = time.perf_counter()

for epoch in range(1, MAX_EPOCHS + 1):
    fnn.train()
    train_sum = 0.0
    for xb, yb in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(fnn(xb), yb)
        loss.backward()
        optimizer.step()
        train_sum += loss.item() * len(xb)
    fnn.eval()
    with torch.no_grad():
        val_loss = sum(loss_fn(fnn(Xva_t[i:i+4096]), yva_t[i:i+4096]).item() * len(Xva_t[i:i+4096])
                       for i in range(0, len(Xva_t), 4096)) / len(Xva_t)
    train_loss = train_sum / len(Xtr_t)
    history["train_mse"].append(train_loss)
    history["val_mse"].append(val_loss)
    if val_loss < best_val:
        best_val, wait = val_loss, 0
        best_state = copy.deepcopy({k: v.detach().cpu() for k, v in fnn.state_dict().items()})
    else:
        wait += 1
    if epoch == 1 or epoch % 20 == 0:
        print(f"epoch {epoch:3d}: train={train_loss:.3e}, val={val_loss:.3e}")
    if wait >= PATIENCE:
        print(f"Early stopping at epoch {epoch}")
        break

fnn.load_state_dict(best_state)
fnn = fnn.to(DEVICE).eval()
fnn_train_seconds = time.perf_counter() - start
print(f"NN fit time: {fnn_train_seconds:.1f}s; best standardized validation MSE={best_val:.3e}")

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(history["train_mse"], label="train")
ax.plot(history["val_mse"], label="validation")
ax.set_yscale("log")
ax.set_xlabel("Epoch")
ax.set_ylabel("Standardized MSE")
ax.set_title("Heston NN training curve")
ax.legend()
fig.tight_layout()
plt.show()
"""),
    md(r"""
## 4. Paper-style exact GPR

The reference paper models Heston European call prices calculated by FFT. Its Heston inputs are strike, maturity, rates/dividend yield, and the Heston parameters. It uses a quadratic basis for the mean, a squared-exponential covariance, and strictly positive observation noise; it reports training sizes of 5k, 10k, and 20k. Here the default 5k subset keeps exact GPR practical, while both models use the same dataset and held-out contexts. Increase `GPR_N` to 10,000 or 20,000 if memory and runtime allow.

The polynomial trend uses a quadratic expansion of the standardized inputs. GPR is fit to the residuals around that trend. Kernel hyperparameters are optimized on at most 1,500 rows, then fixed while the exact GP is refit on the configured subset.
"""),
    code(r"""
from sklearn.base import clone

gpr_rng = np.random.default_rng(SEED + 1)
gpr_idx = gpr_rng.choice(len(train_df), size=min(GPR_N, len(train_df)), replace=False)
X_gpr_raw, y_gpr_raw = X_train[gpr_idx], y_train[gpr_idx]
gpr_sx = StandardScaler().fit(X_gpr_raw)
gpr_sy = StandardScaler().fit(y_gpr_raw.reshape(-1, 1))
X_gpr = gpr_sx.transform(X_gpr_raw)
y_gpr = gpr_sy.transform(y_gpr_raw.reshape(-1, 1)).ravel()
poly = PolynomialFeatures(degree=2, include_bias=True)
Phi = poly.fit_transform(X_gpr)
trend = Ridge(alpha=1e-6, fit_intercept=False).fit(Phi, y_gpr)
residual = y_gpr - trend.predict(Phi)

def make_kernel():
    return (ConstantKernel(1.0, (1e-2, 1e2))
            * RBF(length_scale=np.ones(X_gpr.shape[1]), length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-1)))

fit_n = min(1_500, len(X_gpr))
tune_rng = np.random.default_rng(SEED + 2)
tune_idx = np.sort(tune_rng.choice(len(X_gpr), size=fit_n, replace=False))
tune_trend = Ridge(alpha=1e-6, fit_intercept=False).fit(Phi[tune_idx], y_gpr[tune_idx])
tune_residual = y_gpr[tune_idx] - tune_trend.predict(Phi[tune_idx])
tune_gp = GaussianProcessRegressor(kernel=make_kernel(), alpha=1e-10,
                                   normalize_y=False, n_restarts_optimizer=0,
                                   random_state=SEED)
gpr_start = time.perf_counter()
tune_gp.fit(X_gpr[tune_idx], tune_residual)
gpr_kernel = clone(tune_gp.kernel_)
gpr = GaussianProcessRegressor(kernel=gpr_kernel, alpha=1e-10,
                               optimizer=None, normalize_y=False)
gpr.fit(X_gpr, residual)
gpr_train_seconds = time.perf_counter() - gpr_start
print(f"Exact GPR rows={len(X_gpr):,}; fit time={gpr_train_seconds:.1f}s")
print("Fixed kernel:", gpr.kernel_)

def predict_gpr(X_raw, return_std=False):
    Xs = gpr_sx.transform(X_raw)
    trend_pred = trend.predict(poly.transform(Xs))
    if return_std:
        mean_resid, std_resid = gpr.predict(Xs, return_std=True)
        mean = gpr_sy.inverse_transform((trend_pred + mean_resid).reshape(-1, 1)).ravel()
        std = std_resid * float(gpr_sy.scale_[0])
        return mean, std
    residual_pred = gpr.predict(Xs)
    return gpr_sy.inverse_transform((trend_pred + residual_pred).reshape(-1, 1)).ravel()
"""),
    md(r"""
## 5. Held-out evaluation and saved artifacts

Report mean, maximum, and root-mean-square absolute error against the FFT label, plus R² and the fractions within 0.10 and 0.25 price units. The test contexts were not used for fitting either model. Save weights, scalers, exact GP state, the cached FFT dataset, and summary metrics for the final comparison notebook.
"""),
    code(r"""
def predict_fnn(X_raw):
    Xt = tensor(sx.transform(X_raw)).to(DEVICE)
    with torch.no_grad():
        pred_scaled = fnn(Xt).cpu().numpy()
    return sy.inverse_transform(pred_scaled).ravel()

fnn_test_pred = predict_fnn(X_test)
gpr_test_pred, gpr_test_sd = predict_gpr(X_test, return_std=True)

def metric_row(name, pred):
    err = pred - y_test
    return {"model": name, "train_rows": len(X_train) if name == "FNN" else len(X_gpr),
            "MAE": mean_absolute_error(y_test, pred),
            "max_abs_error": np.abs(err).max(),
            "RMSE": mean_squared_error(y_test, pred) ** 0.5,
            "R2": r2_score(y_test, pred),
            "within_0.10": np.mean(np.abs(err) <= 0.10),
            "within_0.25": np.mean(np.abs(err) <= 0.25)}

metrics = pd.DataFrame([metric_row("FNN", fnn_test_pred), metric_row("GPR", gpr_test_pred)])
display(metrics.style.format({"MAE": "{:.5f}", "max_abs_error": "{:.5f}",
                              "RMSE": "{:.5f}", "R2": "{:.6f}",
                              "within_0.10": "{:.1%}", "within_0.25": "{:.1%}"}))

model_rows = [{"model": "FNN", "train_rows": len(X_train), "fit_seconds": fnn_train_seconds},
              {"model": "GPR", "train_rows": len(X_gpr), "fit_seconds": gpr_train_seconds}]
training_times = pd.DataFrame(model_rows)
metrics.to_csv(DATA_DIR / "heston_ml_test_metrics.csv", index=False)
training_times.to_csv(DATA_DIR / "heston_ml_training_times.csv", index=False)
pd.DataFrame(history).to_csv(DATA_DIR / "heston_fnn_training_curve.csv", index=False)

torch.save(fnn.state_dict(), MODEL_DIR / "heston_fnn.pt")
joblib.dump({"scaler_X": sx, "scaler_y": sy, "feature_names": FEATURE_NAMES,
             "train_seconds": fnn_train_seconds}, MODEL_DIR / "heston_fnn_preprocessing.joblib")
joblib.dump({"scaler_X": gpr_sx, "scaler_y": gpr_sy, "polynomial": poly,
             "trend": trend, "gp": gpr, "feature_names": FEATURE_NAMES,
             "train_seconds": gpr_train_seconds}, MODEL_DIR / "heston_gpr.joblib")
print("Saved Heston NN/GPR artifacts and metric tables.")
"""),
])


save("07_heston_comparison.ipynb", [
    md(r"""
# Heston: Four-Method Comparison on New Options

This notebook draws fresh option/model combinations that are independent of the FFT training dataset, then prices each European call with all four methods: full-truncation Monte Carlo, Carr-Madan FFT, the trained NN, and the trained GPR. The requested plot uses option index on the x-axis and predicted option price on the y-axis, with a distinct color for each method.

FFT is the numerical reference for error metrics. MC error is stochastic, so the output table includes its standard error. This notebook requires the trained artifacts from `06_heston_dataset_ml.ipynb`.
"""),
    code(r"""
from pathlib import Path
import sys
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
from heston_pricing import (FEATURE_NAMES, SPOT, feature_matrix, sample_fresh_options,
                            heston_fft_calls, heston_mc_call)

DATA_DIR, MODEL_DIR = ROOT / "data", ROOT / "models"
FFT_N, FFT_ETA, FFT_ALPHA = 4096, 0.25, 1.25
N_NEW = 100
MC_PATHS, MC_STEPS = 20_000, 1_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
plt.style.use("seaborn-v0_8-whitegrid")
"""),
    md(r"""
## 1. Load the trained NN and GPR

The NN was trained on the complete FFT training split. The exact GPR was trained on the configured 5,000-row subset with a quadratic trend and squared-exponential covariance.
"""),
    code(r"""
fnn_pre = joblib.load(MODEL_DIR / "heston_fnn_preprocessing.joblib")
gpr_payload = joblib.load(MODEL_DIR / "heston_gpr.joblib")
if fnn_pre["feature_names"] != FEATURE_NAMES or gpr_payload["feature_names"] != FEATURE_NAMES:
    raise ValueError("Saved model features do not match the Heston comparison schema")

class HestonMLP(nn.Module):
    def __init__(self, n_features=len(FEATURE_NAMES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x)

fnn = HestonMLP().to(DEVICE)
fnn.load_state_dict(torch.load(MODEL_DIR / "heston_fnn.pt", map_location=DEVICE, weights_only=True))
fnn.eval()

def tensor(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))

def predict_fnn(X):
    Xs = fnn_pre["scaler_X"].transform(X)
    with torch.no_grad():
        ys = fnn(tensor(Xs).to(DEVICE)).cpu().numpy()
    return fnn_pre["scaler_y"].inverse_transform(ys).ravel()

def predict_gpr(X):
    Xs = gpr_payload["scaler_X"].transform(X)
    trend = gpr_payload["trend"].predict(gpr_payload["polynomial"].transform(Xs))
    resid = gpr_payload["gp"].predict(Xs)
    return gpr_payload["scaler_y"].inverse_transform((trend + resid).reshape(-1, 1)).ravel()

print(f"Models loaded on {DEVICE}; NN trained on all training rows, GPR on its exact subset.")
"""),
    md(r"""
## 2. Price fresh options with all four methods

Fresh test inputs use the paper's narrower Heston test ranges and a new random seed. Each is priced independently by MC and FFT; NN and GPR predictions are batched. The plot below compares prices on exactly the same option rows.
"""),
    code(r"""
options = sample_fresh_options(N_NEW, seed=2027 + 10_000)
X_new = feature_matrix(options)

def run_fft_batch():
    return np.array([heston_fft_calls(p, [p["K"]], n=FFT_N, eta=FFT_ETA,
                                      alpha=FFT_ALPHA)[0] for p in options])

fft_prices = run_fft_batch()
fft_samples = []
for _ in range(5):
    start = time.perf_counter(); run_fft_batch(); fft_samples.append(time.perf_counter() - start)
fft_seconds = float(np.median(fft_samples))

def run_mc_batch(seed_offset):
    prices, errors = [], []
    for i, option in enumerate(options):
        estimate, se = heston_mc_call(option, n_paths=MC_PATHS, n_steps=MC_STEPS,
                                      seed=2027 + 20_000 + seed_offset + i)
        prices.append(estimate); errors.append(se)
    return np.asarray(prices), np.asarray(errors)

mc_prices, mc_ses = run_mc_batch(0)
mc_samples = []
for repeat in range(3):
    start = time.perf_counter(); run_mc_batch((repeat + 1) * 1_000)
    mc_samples.append(time.perf_counter() - start)
mc_seconds = float(np.median(mc_samples))

fnn_prices = predict_fnn(X_new)
def sync_device():
    if DEVICE.type == "mps":
        torch.mps.synchronize()
    elif DEVICE.type == "cuda":
        torch.cuda.synchronize()
sync_device()
for _ in range(5):
    predict_fnn(X_new); sync_device()
fnn_samples = []
for _ in range(15):
    sync_device(); start = time.perf_counter(); predict_fnn(X_new); sync_device()
    fnn_samples.append(time.perf_counter() - start)
fnn_seconds = float(np.median(fnn_samples))

gpr_prices = predict_gpr(X_new)
for _ in range(5):
    predict_gpr(X_new)
gpr_samples = []
for _ in range(15):
    start = time.perf_counter(); predict_gpr(X_new)
    gpr_samples.append(time.perf_counter() - start)
gpr_seconds = float(np.median(gpr_samples))

comparison = pd.DataFrame({
    "data_point": np.arange(1, N_NEW + 1),
    "forward_moneyness": [p["forward_moneyness"] for p in options],
    "FFT": fft_prices, "MC": mc_prices, "MC_standard_error": mc_ses,
    "NN": fnn_prices, "GPR": gpr_prices,
})
comparison.to_csv(DATA_DIR / "heston_four_method_new_options.csv", index=False)
display(comparison.head(10).style.format({k: "{:.5f}" for k in comparison.columns
                                          if k != "data_point"}))
"""),
    md(r"""
## 3. Requested four-color price plot

The x-axis is the index of each new option data point. The y-axis is the predicted European call price. Each method uses its own color and all four series use the same option ordering. Markers are nudged slightly left or right within each data point only to make overlapping predictions visible; the tick labels remain the original option indices.
"""),
    code(r"""
fig, ax = plt.subplots(figsize=(13, 5.5))
styles = {
    "MC": {"color": "tab:orange", "marker": "o"},
    "FFT": {"color": "tab:green", "marker": "s"},
    "NN": {"color": "tab:blue", "marker": "^"},
    "GPR": {"color": "tab:purple", "marker": "D"},
}
offsets = {"MC": -0.24, "FFT": -0.08, "NN": 0.08, "GPR": 0.24}
for method, style in styles.items():
    ax.scatter(comparison.data_point + offsets[method], comparison[method], s=34, alpha=0.82,
               label=method, **style)
ax.set_xticks([1, 25, 50, 75, 100])
ax.set_xlabel("New option data point")
ax.set_ylabel("Predicted European call price")
ax.set_title("Heston option prices: MC vs FFT vs NN vs GPR")
ax.legend(title="Pricing method", ncols=4)
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(DATA_DIR / "heston_four_method_comparison.png", dpi=160)
plt.show()
"""),
    md(r"""
## 4. Accuracy, MC uncertainty, and speed

FFT is the numerical reference. Report absolute-error metrics against FFT for all four methods; MC also reports how many estimates fall within two estimated standard errors of FFT. Inference times use repeated batch timings after model warm-up over the same 100 new options, with NN/GPR batched as they are used in practice.
"""),
    code(r"""
predictions = {"MC": mc_prices, "FFT": fft_prices, "NN": fnn_prices, "GPR": gpr_prices}
metric_rows = []
for name, pred in predictions.items():
    err = pred - fft_prices
    metric_rows.append({"method": name,
                        "MAE_vs_FFT": mean_absolute_error(fft_prices, pred),
                        "max_abs_error_vs_FFT": np.max(np.abs(err)),
                        "RMSE_vs_FFT": mean_squared_error(fft_prices, pred) ** 0.5,
                        "R2_vs_FFT": r2_score(fft_prices, pred),
                        "MC_within_2SE": (np.mean(np.abs(err) <= 2 * mc_ses)
                                          if name == "MC" else np.nan)})
metrics = pd.DataFrame(metric_rows)
timings = pd.DataFrame([
    {"method": "MC", "total_seconds": mc_seconds},
    {"method": "FFT", "total_seconds": fft_seconds},
    {"method": "NN", "total_seconds": fnn_seconds},
    {"method": "GPR", "total_seconds": gpr_seconds},
])
timings["microseconds_per_option"] = timings.total_seconds / N_NEW * 1e6
fft_us = timings.loc[timings.method == "FFT", "microseconds_per_option"].iloc[0]
timings["speed_up_vs_FFT"] = fft_us / timings.microseconds_per_option
metrics.to_csv(DATA_DIR / "heston_four_method_metrics.csv", index=False)
timings.to_csv(DATA_DIR / "heston_four_method_timings.csv", index=False)
print("Accuracy versus FFT")
display(metrics.style.format({"MAE_vs_FFT": "{:.5f}", "max_abs_error_vs_FFT": "{:.5f}",
                              "RMSE_vs_FFT": "{:.5f}", "R2_vs_FFT": "{:.6f}",
                              "MC_within_2SE": "{:.1%}"}))
print("Inference timing")
display(timings.style.format({"total_seconds": "{:.4f}", "microseconds_per_option": "{:.2f}",
                              "speed_up_vs_FFT": "{:.2f}x"}))
print(f"MC settings: {MC_PATHS:,} paths, {MC_STEPS} time steps per option")
"""),
])

print("Built 05_heston_mc_fft.ipynb, 06_heston_dataset_ml.ipynb, and 07_heston_comparison.ipynb")
