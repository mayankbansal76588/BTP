"""Builds 04_large_scale_training.ipynb programmatically with nbformat.

Reuses the (unchanged) pricing-function cell from 03_comparison.ipynb so the
math stays single-sourced; the generated notebook is still self-contained.
"""
import json

import nbformat as nbf

SVGP_INDUCING = 1024
SVGP_EPOCHS = 30
FNN_EPOCHS = 200

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


nb03 = json.load(open("03_comparison.ipynb"))
pricing_src = next("".join(c["source"]) for c in nb03["cells"]
                   if c["cell_type"] == "code" and "def carr_madan_call" in "".join(c["source"]))

md(f"""\
# Large-Scale Training: FNN and GPR on 10^5+ FFT-Priced Options

Notebooks 02/03 trained the FNN on 35,000 rows and the exact GPR on at most
10,000 rows. Here **both models are trained on 105,000 rows** (a 150,000-row
FFT dataset split 70/15/15) and compared on:

- a held-out test split of 22,500 rows, and
- the **same 1,000-option shared test set as `03_comparison.ipynb`** (same
  seed), so results line up directly with the smaller models there.

**Why the GPR here is a *sparse* GP.** Exact GPR needs the full `n x n`
kernel matrix: at n = 105,000 that is 105,000^2 x 8 bytes = **~88 GB** of
memory and an `O(n^3)` Cholesky - impossible on this machine (8 GB RAM).
The standard way to scale GPs is a **Sparse Variational Gaussian Process
(SVGP, Hensman et al. 2013)**, implemented with GPyTorch: the posterior is
summarised by `M = {SVGP_INDUCING}` learned *inducing points*, and the model
is trained with mini-batches on the variational ELBO, so cost per step is
`O(batch x M^2 + M^3)` and **every one of the 105,000 points is used**. It
keeps the same kernel family as before (constant x ARD-RBF + Gaussian noise)
and still gives a predictive standard deviation.""")

code("""\
from __future__ import annotations

import math
import time
import warnings
from pathlib import Path

import gpytorch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 20261116
np.random.seed(SEED)
torch.manual_seed(SEED)

FFT_N = 16384
FFT_ETA = 0.25
FFT_ALPHA = 1.5

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_NAMES = ["spot", "strike", "rate", "sigma", "tau"]
COLORS = {"FNN": "tab:blue", "GPR": "tab:purple"}
device = torch.device("cpu")
""")

md("## Pricing functions (identical to notebooks 01-03)")
code(pricing_src)

md("""\
## 1. Generate a 150,000-row FFT dataset

Same sampling ranges as before (`make_dataset`), new seed (`SEED + 150`) so
it is a fresh draw. Each row is priced with Carr-Madan FFT (the training
label, as in notebook 02); the BSM price is stored too for checking.
Pricing takes a couple of minutes, so the result is cached to
`data/fft_dataset_150k.csv` and reused on re-runs.""")

code("""\
N_TOTAL = 150_000
DATASET_PATH = DATA_DIR / "fft_dataset_150k.csv"

if DATASET_PATH.exists():
    big_df = pd.read_csv(DATASET_PATH)
    print(f"Loaded cached dataset: {len(big_df):,} rows")
else:
    options = make_dataset(N_TOTAL, SEED + 150)
    t0 = time.perf_counter()
    fft_prices = np.empty(N_TOTAL)
    for i, p in enumerate(options):
        fft_prices[i] = carr_madan_call(**p)
        if (i + 1) % 25_000 == 0:
            print(f"  priced {i + 1:,} / {N_TOTAL:,}  ({time.perf_counter() - t0:.0f}s)")
    big_df = pd.DataFrame(options)
    big_df["fft_price"] = fft_prices
    big_df["bsm_price"] = [bsm_call(**p) for p in options]
    big_df.to_csv(DATASET_PATH, index=False)
    print(f"Generated and priced {N_TOTAL:,} options in {time.perf_counter() - t0:.0f}s")

print(f"max |FFT - BSM| over the whole dataset: {np.abs(big_df.fft_price - big_df.bsm_price).max():.2e}")
""")

md("## 2. Train / validation / test split (70 / 15 / 15) and data statistics")

code("""\
idx = np.arange(len(big_df))
idx_train, idx_tmp = train_test_split(idx, test_size=0.30, random_state=SEED)
idx_val, idx_test = train_test_split(idx_tmp, test_size=0.50, random_state=SEED)

X_big = big_df[FEATURE_NAMES].to_numpy(dtype=np.float64)
y_big = big_df["fft_price"].to_numpy(dtype=np.float64)
X_train, y_train = X_big[idx_train], y_big[idx_train]
X_val, y_val = X_big[idx_val], y_big[idx_val]
X_test, y_test = X_big[idx_test], y_big[idx_test]
y_test_bsm = big_df["bsm_price"].to_numpy()[idx_test]

stats = []
for name, ii in [("train", idx_train), ("validation", idx_val), ("test", idx_test)]:
    d = big_df.iloc[ii]
    m = d.strike / d.spot
    stats.append({"split": name, "rows": len(d),
                  "price_mean": d.fft_price.mean(), "price_std": d.fft_price.std(),
                  "price_median": d.fft_price.median(), "price_max": d.fft_price.max(),
                  "sigma_mean": d.sigma.mean(), "tau_mean": d.tau.mean(), "rate_mean": d.rate.mean(),
                  "ITM_%": (m < 0.97).mean() * 100, "ATM_%": ((m >= 0.97) & (m <= 1.03)).mean() * 100,
                  "OTM_%": (m > 1.03).mean() * 100, "price_below_1c": int((d.fft_price < 0.01).sum())})
split_stats = pd.DataFrame(stats).set_index("split")
split_stats.to_csv(DATA_DIR / "large_scale_split_stats.csv")
split_stats.round(3)
""")

code("""\
sx = StandardScaler().fit(X_train)
sy = StandardScaler().fit(y_train.reshape(-1, 1))


def tens(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


Xtr_t, Xval_t, Xte_t = (tens(sx.transform(a)) for a in (X_train, X_val, X_test))
ytr_t = tens(sy.transform(y_train.reshape(-1, 1)).ravel())
yval_t = tens(sy.transform(y_val.reshape(-1, 1)).ravel())


def unscale(pred_scaled: np.ndarray) -> np.ndarray:
    return sy.inverse_transform(np.asarray(pred_scaled).reshape(-1, 1)).ravel()


def error_summary(true: np.ndarray, pred: np.ndarray, floor: float = 0.01) -> dict[str, float]:
    abs_err = np.abs(pred - true)
    return {"mean_abs_error": float(abs_err.mean()), "max_abs_error": float(abs_err.max()),
            "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
            "mean_rel_error": float((abs_err / np.maximum(true, floor)).mean()),
            "r2": float(r2_score(true, pred)),
            "within_0.10": float((abs_err <= 0.10).mean()), "within_0.25": float((abs_err <= 0.25).mean())}
""")

md(f"""\
## 3. FNN on 105,000 rows

Same architecture as notebook 02 (5 -> 128 -> 128 -> 64 -> 32 -> 1, ReLU).
Adam (lr 1e-3) with cosine decay, batch 1,024, {FNN_EPOCHS} epochs; the weights
with the best validation loss are kept.""")

code(f"""\
class OptionPricerMLP(nn.Module):
    def __init__(self, n_features: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


FNN_EPOCHS = {FNN_EPOCHS}
torch.manual_seed(SEED)
fnn = OptionPricerMLP(len(FEATURE_NAMES))
loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr_t, ytr_t.unsqueeze(1)),
                                     batch_size=1024, shuffle=True)
opt = torch.optim.Adam(fnn.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FNN_EPOCHS * len(loader))
loss_fn = nn.MSELoss()

fnn_hist = {{"train": [], "val": []}}
best_val, best_state = float("inf"), None
t0 = time.perf_counter()
for epoch in range(1, FNN_EPOCHS + 1):
    fnn.train()
    running = 0.0
    for xb, yb in loader:
        opt.zero_grad()
        loss = loss_fn(fnn(xb), yb)
        loss.backward()
        opt.step(); sched.step()
        running += loss.item() * xb.size(0)
    fnn.eval()
    with torch.no_grad():
        val_loss = loss_fn(fnn(Xval_t), yval_t.unsqueeze(1)).item()
    fnn_hist["train"].append(running / len(Xtr_t)); fnn_hist["val"].append(val_loss)
    if val_loss < best_val:
        best_val, best_state = val_loss, {{k: v.clone() for k, v in fnn.state_dict().items()}}
    if epoch == 1 or epoch % 25 == 0:
        print(f"epoch {{epoch:4d}}  train={{fnn_hist['train'][-1]:.2e}}  val={{val_loss:.2e}}  ({{time.perf_counter() - t0:.0f}}s)")
fnn_fit_seconds = time.perf_counter() - t0
fnn.load_state_dict(best_state); fnn.eval()
print(f"FNN trained on {{len(X_train):,}} rows in {{fnn_fit_seconds:.0f}}s (best val loss {{best_val:.2e}})")
""")

md(f"""\
## 4. Sparse GPR (SVGP) on 105,000 rows

- `M = {SVGP_INDUCING}` inducing points, initialised at random training
  inputs and then **learned**.
- Kernel: `Scale x RBF(5 ARD length scales)`, constant mean, Gaussian
  likelihood (noise learned) - the same family as the exact GPR before.
- Mini-batch (1,024) Adam on the variational ELBO, lr 0.01 with cosine
  decay, {SVGP_EPOCHS} epochs over all 105,000 points. Validation MAE tracked.""")

code(f"""\
class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor):
        var_dist = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        var_strat = gpytorch.variational.VariationalStrategy(
            self, inducing_points, var_dist, learn_inducing_locations=True)
        super().__init__(var_strat)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=len(FEATURE_NAMES)))

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


def svgp_predict(model, likelihood, X_scaled: torch.Tensor, return_std: bool = False,
                 batch: int = 8192):
    model.eval(); likelihood.eval()
    mus, sds = [], []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(0, len(X_scaled), batch):
            out = likelihood(model(X_scaled[i:i + batch]))
            mus.append(out.mean.numpy())
            if return_std:
                sds.append(out.stddev.numpy())
    mu = unscale(np.concatenate(mus))
    if return_std:
        return mu, np.concatenate(sds) * sy.scale_[0]
    return mu


SVGP_INDUCING, SVGP_EPOCHS = {SVGP_INDUCING}, {SVGP_EPOCHS}
torch.manual_seed(SEED)
Z0 = Xtr_t[torch.randperm(len(Xtr_t))[:SVGP_INDUCING]].clone()
svgp = SVGPModel(Z0)
svgp_lik = gpytorch.likelihoods.GaussianLikelihood()
gp_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr_t, ytr_t),
                                        batch_size=1024, shuffle=True)
gp_opt = torch.optim.Adam(list(svgp.parameters()) + list(svgp_lik.parameters()), lr=0.01)
gp_sched = torch.optim.lr_scheduler.CosineAnnealingLR(gp_opt, T_max=SVGP_EPOCHS * len(gp_loader))
elbo = gpytorch.mlls.VariationalELBO(svgp_lik, svgp, num_data=len(ytr_t))

gp_hist = {{"neg_elbo": [], "val_mae": []}}
t0 = time.perf_counter()
for epoch in range(1, SVGP_EPOCHS + 1):
    svgp.train(); svgp_lik.train()
    running = 0.0
    for xb, yb in gp_loader:
        gp_opt.zero_grad()
        loss = -elbo(svgp(xb), yb)
        loss.backward()
        gp_opt.step(); gp_sched.step()
        running += loss.item() * xb.size(0)
    gp_hist["neg_elbo"].append(running / len(Xtr_t))
    gp_hist["val_mae"].append(float(np.abs(svgp_predict(svgp, svgp_lik, Xval_t) - y_val).mean()))
    if epoch == 1 or epoch % 5 == 0:
        print(f"epoch {{epoch:3d}}  -ELBO={{gp_hist['neg_elbo'][-1]:.4f}}  val MAE={{gp_hist['val_mae'][-1]:.4f}}  "
              f"({{time.perf_counter() - t0:.0f}}s)")
gpr_fit_seconds = time.perf_counter() - t0
print(f"SVGP trained on {{len(X_train):,}} rows in {{gpr_fit_seconds:.0f}}s")
print("Learned ARD length scales (standardized inputs):",
      dict(zip(FEATURE_NAMES, svgp.covar_module.base_kernel.lengthscale.detach().numpy().ravel().round(3))))
""")

code("""\
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
ax1.plot(fnn_hist["train"], label="train"); ax1.plot(fnn_hist["val"], label="validation")
ax1.set_yscale("log"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE (standardized)")
ax1.set_title("FNN training (105k rows)"); ax1.legend(); ax1.grid(True, which="both", alpha=0.3)
ax2.plot(gp_hist["val_mae"], color=COLORS["GPR"], marker="o", ms=3)
ax2.set_yscale("log"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Validation mean abs error ($)")
ax2.set_title("Sparse GPR training (105k rows)"); ax2.grid(True, which="both", alpha=0.3)
fig.tight_layout(); fig.savefig(DATA_DIR / "large_scale_training_curves.png", dpi=150); plt.show()
""")

md("""\
## 5. Evaluation

### 5a. Held-out test split (22,500 rows), errors vs FFT labels and vs BSM""")

code("""\
t0 = time.perf_counter()
with torch.no_grad():
    fnn_test = unscale(fnn(Xte_t).numpy().ravel())
fnn_pred_us = (time.perf_counter() - t0) / len(X_test) * 1e6

t0 = time.perf_counter()
gpr_test, gpr_test_sd = svgp_predict(svgp, svgp_lik, Xte_t, return_std=True)
t0b = time.perf_counter(); _ = svgp_predict(svgp, svgp_lik, Xte_t); gpr_pred_us = (time.perf_counter() - t0b) / len(X_test) * 1e6

test_rows = []
for name, pred in [("FNN", fnn_test), ("GPR", gpr_test)]:
    test_rows.append({"model": name, "vs": "FFT label", **error_summary(y_test, pred)})
    test_rows.append({"model": name, "vs": "BSM", **error_summary(y_test_bsm, pred)})
test_df = pd.DataFrame(test_rows)
test_df.to_csv(DATA_DIR / "large_scale_test_metrics.csv", index=False)
coverage = float((np.abs(gpr_test - y_test) <= 2 * gpr_test_sd).mean())
print(f"Sparse GPR: {coverage:.1%} of test labels within +/-2 sigma (ideal ~95%)")
test_df.style.format({c: "{:.4f}" for c in ["mean_abs_error", "max_abs_error", "rmse"]} |
                     {"mean_rel_error": "{:.2%}", "r2": "{:.6f}", "within_0.10": "{:.1%}", "within_0.25": "{:.1%}"})
""")

code("""\
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
lim = [0, y_test.max() * 1.03]
for ax, (name, pred) in zip(axes[:2], [("FNN", fnn_test), ("GPR", gpr_test)]):
    ax.scatter(y_test, pred, s=2, alpha=0.25, color=COLORS[name])
    ax.plot(lim, lim, "r--", lw=1.2)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("FFT price"); ax.set_ylabel(f"{name} price")
    ax.set_title(f"{name} (105k) parity, test split"); ax.grid(True, alpha=0.3)
bins = np.linspace(-1, 1, 81)
axes[2].hist(np.clip(fnn_test - y_test, -1, 1), bins=bins, alpha=0.6, color=COLORS["FNN"], label="FNN")
axes[2].hist(np.clip(gpr_test - y_test, -1, 1), bins=bins, alpha=0.6, color=COLORS["GPR"], label="Sparse GPR")
axes[2].set_xlabel("Predicted - FFT price (clipped to +/-1)"); axes[2].set_ylabel("Count")
axes[2].set_title("Test-split error distribution"); axes[2].legend(); axes[2].grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(DATA_DIR / "large_scale_test_parity.png", dpi=150); plt.show()
""")

md("""\
### 5b. Same 1,000-option shared test set as notebook 03 (errors vs BSM)

Rebuilt with the identical seed (`SEED + 2026`), so these numbers are
directly comparable to the small-data models measured in notebook 03
(`data/gpr_vs_fnn_bsm.csv`).""")

code("""\
shared = pd.DataFrame(make_dataset(1000, SEED + 2026))[FEATURE_NAMES]
X_sh = shared.to_numpy(dtype=np.float64)
true_sh = np.array([bsm_call(*r) for r in X_sh])
Xsh_t = tens(sx.transform(X_sh))
with torch.no_grad():
    fnn_sh = unscale(fnn(Xsh_t).numpy().ravel())
gpr_sh = svgp_predict(svgp, svgp_lik, Xsh_t)

fnn_bytes = sum(p.numel() for p in fnn.parameters()) * 4
gpr_bytes = sum(p.numel() for p in list(svgp.parameters()) + list(svgp_lik.parameters())) * 4

rows = [{"model": "FNN", "train_size": len(X_train), "label_source": "FFT", "approach": "MLP",
         **error_summary(true_sh, fnn_sh), "fit_seconds": fnn_fit_seconds,
         "predict_us_per_option": fnn_pred_us, "model_memory_MB": fnn_bytes / 1e6},
        {"model": "GPR", "train_size": len(X_train), "label_source": "FFT",
         "approach": f"sparse SVGP (M={SVGP_INDUCING})",
         **error_summary(true_sh, gpr_sh), "fit_seconds": gpr_fit_seconds,
         "predict_us_per_option": gpr_pred_us, "model_memory_MB": gpr_bytes / 1e6}]
large_df = pd.DataFrame(rows)

prev = pd.read_csv(DATA_DIR / "gpr_vs_fnn_bsm.csv")
prev["approach"] = np.where(prev["model"].str.startswith("GPR"), "exact GPR", "MLP")
prev["model"] = prev["model"].str.replace(" (nb 02)", "", regex=False)
combined = pd.concat([prev, large_df], ignore_index=True)[
    ["model", "approach", "train_size", "label_source", "mean_abs_error", "max_abs_error", "rmse",
     "r2", "within_0.10", "fit_seconds", "predict_us_per_option", "model_memory_MB"]]
combined = combined.sort_values(["model", "train_size"]).reset_index(drop=True)
combined.to_csv(DATA_DIR / "large_scale_vs_small_models.csv", index=False)
combined.style.format({"mean_abs_error": "{:.4f}", "max_abs_error": "{:.4f}", "rmse": "{:.4f}",
                       "r2": "{:.6f}", "within_0.10": "{:.1%}", "fit_seconds": "{:.1f}",
                       "predict_us_per_option": "{:.2f}", "model_memory_MB": "{:.2f}"}, na_rep="-")
""")

code("""\
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
for name in ["FNN", "GPR"]:
    d = combined[combined["model"] == name].sort_values("train_size")
    ax1.plot(d["train_size"], d["mean_abs_error"], "o-", color=COLORS[name], label=name)
    for _, r in d.iterrows():
        ax1.annotate(r["approach"].replace(" GPR", "").replace("sparse ", ""), (r["train_size"], r["mean_abs_error"]),
                     textcoords="offset points", xytext=(4, 6), fontsize=7)
    ax2.plot(d["train_size"], d["predict_us_per_option"], "o-", color=COLORS[name], label=name)
for ax, t, yl in [(ax1, "Mean abs error vs BSM (shared 1,000 options)", "Mean |price - BSM| ($)"),
                  (ax2, "Inference cost", "Microseconds per option")]:
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Training set size")
    ax.set_ylabel(yl); ax.set_title(t); ax.legend(); ax.grid(True, which="both", alpha=0.3)
fig.tight_layout(); fig.savefig(DATA_DIR / "large_scale_scaling.png", dpi=150); plt.show()
""")

md("## 6. Save models and summarise")

code("""\
torch.save(fnn.state_dict(), MODELS_DIR / "fnn_105k.pt")
torch.save({"svgp": svgp.state_dict(), "likelihood": svgp_lik.state_dict()}, MODELS_DIR / "svgp_105k.pt")
joblib.dump(sx, MODELS_DIR / "scaler_X_105k.joblib")
joblib.dump(sy, MODELS_DIR / "scaler_y_105k.joblib")

f, g = large_df.iloc[0], large_df.iloc[1]
print(f"Both models trained on {len(X_train):,} FFT-priced options.")
print(f"Shared 1,000-option set (vs BSM): FNN mean abs err {f.mean_abs_error:.4f} | sparse GPR {g.mean_abs_error:.4f}")
print(f"Training time: FNN {f.fit_seconds:.0f}s | sparse GPR {g.fit_seconds:.0f}s")
print(f"Inference: FNN {f.predict_us_per_option:.2f} us | sparse GPR {g.predict_us_per_option:.2f} us per option")
print(f"Model size: FNN {f.model_memory_MB:.2f} MB | sparse GPR {g.model_memory_MB:.2f} MB "
      f"(exact GPR at this n would need ~{len(X_train) ** 2 * 8 / 1e9:.0f} GB)")
for name in ["FNN", "GPR"]:
    d = combined[combined["model"] == name].sort_values("train_size")
    print(f"{name}: mean abs err {d.mean_abs_error.iloc[0]:.4f} (n={d.train_size.iloc[0]:,}) -> "
          f"{d.mean_abs_error.iloc[-1]:.4f} (n={d.train_size.iloc[-1]:,})")
""")

md("""\
**Takeaways**

- With 10^5 training rows the FNN keeps improving: more data is the main
  lever for an MLP, and its inference cost and model size don't change.
- Exact GPR cannot be trained at this size (memory ~88 GB); a sparse
  variational GP makes it feasible, using all 105k points through `M`
  inducing points. Its accuracy is then capped by `M` rather than `n`, which
  is the price paid for scalability - compare it with the exact 10k GPR
  in the table above.
- The sparse GP keeps GPR's advantages of a predictive uncertainty and a
  fixed-size model (inference `O(M)` instead of `O(n)` per option).""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3 (BTP .venv)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
with open("04_large_scale_training.ipynb", "w") as fh:
    nbf.write(nb, fh)
print("Wrote 04_large_scale_training.ipynb")
