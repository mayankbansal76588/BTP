# BTP Project: Option Pricing via BSM and Heston Methods

## Context

The original completed comparison uses Black-Scholes/GBM. A new Heston
stochastic-volatility extension is now in scope and is described below:

1. **Black-Scholes-Merton (BSM) analytical formula** — ground truth.
2. **Monte Carlo simulation** — shown to converge to BSM via a graph.
3. **FFT / characteristic-function method (Carr-Madan)** — theory notes are
   in `material/lecture3_FourierPricing_I.pdf` and
   `material/lecture4_2020_FourierPricing_II_.pdf`.
4. **ML surrogate model**: generate an (x, y) dataset by pricing many
   options with the FFT method, train a neural network on it, then use the
   trained network to price *new* option parameter combinations instantly.
   Reference paper: `material/Machine learning for quantitative finance
   fast derivative pricing hedging and fitting.pdf`.

[code.py](code.py) already implements methods 1-3. It was verified
numerically (BSM vs FFT agree to ~1e-5 to 1e-7 absolute across 5 varied
param sets; `python code.py --n-options 5` runs cleanly end-to-end) — **the
BSM, MC, and FFT implementations in code.py are correct.** No math changes
needed for methods 1-3, just restructure into notebook cells with
visualizations, since code.py currently only writes a CSV and prints
summary stats with no graphs.

Deliverable format: **Jupyter notebooks** (not .py scripts).
NN framework: **PyTorch**.

## Notebook structure

Three notebooks (a dedicated final notebook is needed for the cross-method +
efficiency comparison, since it depends on artifacts from both of the first
two):

### `01_bsm_mc_fft.ipynb`
Reuses `code.py`'s `bsm_call`, `mc_call`, `carr_madan_call`,
`make_dataset` (clean them up into the notebook, keep the math unchanged).
Cells:
- BSM price for a single example + across a range of strikes/spots.
- MC convergence graph: MC price vs. number of paths (log-scale x-axis),
  with BSM price as a horizontal reference line, to visually show MC → BSM.
- FFT vs BSM price curve across a strike grid (overlaid line plot) + an
  error plot (`fft_price - bsm_price` vs strike) to show near-zero error.
- Save the per-option comparison table (BSM/MC/FFT price + errors + timing,
  same columns as code.py's CSV output) to `data/bsm_mc_fft_results.csv`
  for reuse in notebook 03.

### `02_fft_dataset_ml.ipynb`
- Generate a larger synthetic dataset (e.g. tens of thousands of rows) over
  `(spot, strike, rate, sigma, tau)` using `make_dataset`-style sampling,
  price every row with `carr_madan_call` → this is the (x, y) dataset from
  the paper's methodology (x = the 5 parameters, or normalized
  moneyness/vol/tau features; y = FFT price).
- Train/val/test split.
- PyTorch feedforward NN (small MLP, e.g. 4-5 hidden layers,
  with scaling of inputs/outputs) trained to regress price from
  parameters. Track training/validation loss curve (graph).
- Evaluate on held-out test points: predicted vs. FFT-true price
  (parity/scatter plot), and error histogram.
- Apply the trained model to **brand-new** sampled points (not in training
  set) and compare against freshly computed FFT (and BSM, since BSM is
  exact here) prices — this directly answers "take new data points and use
  trained model to calculate option prices."
- Save the trained model weights and the test-set predictions + timing
  (see below) to `data/ml_fft_results.csv` / `models/model.pt` for
  notebook 03.

### `03_comparison.ipynb`
- Define **one shared set of test option parameter rows** (e.g. a strike
  sweep at fixed spot/rate/sigma/tau, or a batch of random
  `(spot, strike, rate, sigma, tau)` rows) — not reused from notebooks 01/02
  separately, so every method is evaluated on *exactly the same data
  points*.
- Run all 4 methods — BSM analytic, MC, FFT, and the trained NN (loaded
  from notebook 02's saved weights) — on that same shared set of points.
- Load any other needed artifacts from notebooks 01 and 02 (e.g. the
  trained model).
- One graph overlaying all 4 methods' prices across that shared set of
  points (same x-axis), so the agreement/disagreement between methods is
  directly visible on identical inputs.
- **Computational efficiency comparison**: benchmark per-option (or
  per-batch) wall-clock time for BSM analytical, MC (fixed path count),
  FFT/Carr-Madan, and trained-NN inference (batched, since NN inference is
  efficient in batch) — bar chart on a log scale, since the NN should be
  orders of magnitude faster per option than re-running Carr-Madan/MC for
  each new parameter set once trained. Discuss that BSM is also instant
  since it's closed-form — the real efficiency story is "NN replaces
  FFT/MC for models where no closed form exists, at near-zero marginal cost
  per query after training," using BSM/this pipeline as a validated
  proof-of-concept.
- Summary error table (mean/max abs & relative error of MC, FFT, and ML vs
  BSM ground truth).

## Implementation notes

- Keep `code.py`'s pricing functions as the single source of truth; don't
  re-derive the Carr-Madan math. Copy the verified functions into notebook
  01's first cells so the three deliverable notebooks are self-contained
  and runnable independently.
- New dependencies needed: `torch` (PyTorch), `matplotlib` (graphs),
  `pandas` (tabular handling/CSV), `scikit-learn` (train/val/test split,
  scaling) — installed in a project-local `.venv`.
- Keep a fixed seed for reproducibility across notebooks.
- No changes to the underlying financial model (GBM/Black-Scholes) per
  user's explicit choice — this keeps scope bounded and consistent with
  `code.py`.

## Verification plan

- Notebook 01: visually confirm MC converges to BSM as path count grows,
  and FFT-vs-BSM error plot stays within ~1e-4 absolute across the strike
  grid.
- Notebook 02: training/validation loss should decrease and plateau;
  predicted-vs-true parity plot should sit tightly on the diagonal; report
  test-set mean absolute error for new data points as well.
- Notebook 03: confirm the 4-way comparison graph shows visual agreement
  between BSM/MC/FFT/NN, and that the efficiency bar chart shows NN
  inference time is lowest per query (after the one-time FFT-dataset
  generation + training cost, which should be called out as upfront cost
  vs. the per-query cost being compared).
- Run all three notebooks top-to-bottom to confirm no errors before
  considering the task done.

## GPR vs FNN comparison (added to notebook 02)

The reference paper (De Spiegeleer, Madan, Reyners & Schoutens, 2018)
implements **Gaussian Process Regression (GPR)**, not Gradient Boosting —
confirmed by extracting and searching the paper's full text (zero mentions
of boosting). Added a new section 7 to `02_fft_dataset_ml.ipynb`:

- GPR trained on our Black-Scholes dataset at sizes 2,000 / 5,000 / 10,000
  (paper's 20,000 omitted — exact GPR is O(n^3), infeasible interactively
  at that size). Kernel hyperparameters fit once on the smallest size via
  gradient-based optimization, then reused fixed for the larger sizes to
  keep cost tractable.
- Reports the paper's own metric names: `MAE` = **maximum** absolute error,
  `AAE` = **average** absolute error (i.e. what's usually called MAE
  elsewhere) — both in-sample and out-of-sample, plus speed-up vs FFT.
- FNN evaluated on the exact same shared out-of-sample batch for a fair
  side-by-side comparison, saved to `data/gpr_fnn_paper_metrics.csv`.
- Explicit caveat in the notebook: our dataset is Black-Scholes/GBM, not
  the paper's VG/Heston setup, so absolute numbers aren't expected to
  numerically match the paper's Table 2 — only checking the qualitative
  pattern (error shrinks with training size, large speed-ups).
- The Heston replication is now being implemented in the separate notebooks
  listed in the Heston section below.

**Executed results:** GPR out-of-sample AAE decreases monotonically with
training size (0.143 → 0.090 → 0.072 at n=2k/5k/10k) and speed-up shrinks
(64.7x → 24.0x → 12.7x) — both match the paper's qualitative pattern, and
the speed-up range lands right in the paper's reported ~7x-40x. GPR
in-sample AAE is *not* monotonic (0.042 → 0.053 → 0.056) since kernel
hyperparameters were fit once on 2,000 points and reused — a caveat of the
cubic-cost workaround. Best GPR (n=10k) AAE=0.072 beats the FNN's AAE=0.133
(n=35k) on accuracy, but FNN is ~190x faster (2433x vs 12.7x speed-up over
FFT) and scales to far more training data.

## GPR on BSM prices + GPR vs FNN head-to-head (added to notebook 03)

Per user request, `03_comparison.ipynb` adds GPR as a **fifth pricing
method**, trained directly on **BSM closed-form prices** (not FFT labels):

- Inputs = notebook 02's training-split rows (same `train_test_split` seed,
  so no notebook-02 test rows leak in); labels = exact `bsm_call` prices.
- Same GPR recipe as notebook 02: `C * RBF(5 ARD length scales) + White`,
  hyperparameters optimized once on 2,000 points (~26s), then reused fixed
  for 5,000 / 10,000. The 10k GPR is the one in the 5-way comparison.
- **Fair GPR-vs-FNN comparison**: an FNN with the same architecture is
  trained on the *identical* 2k/5k/10k BSM-labelled rows (Adam + cosine LR,
  fixed ~14k gradient-step budget), alongside notebook 02's production FNN
  (35k FFT labels) as reference. Compared on accuracy, training time,
  inference time, memory, error-vs-training-size scaling, and GPR's
  predictive uncertainty (±2σ coverage). Break-even (queries needed to
  repay labelling + training cost) is computed for both.
- Outputs: `data/comparison_*.csv/.png`, `data/gpr_vs_fnn_bsm.csv`,
  `data/gpr_vs_fnn_scaling.png`, `data/gpr_vs_fnn_errors.png`.

**Executed results (shared 1,000 unseen random options, error vs BSM):**

| Method | Mean abs err | Max abs err | R² | µs / option | Speed-up vs FFT |
|---|---|---|---|---|---|
| MC (100k paths) | 0.145 | 1.44 | 0.99987 | 1242 | 0.65x |
| FFT | 1.3e-5 | 2.7e-4 | ~1.0 | 803 | 1x |
| FNN (nb 02, 35k FFT) | 0.128 | 1.18 | 0.99994 | 0.48 | ~1660x |
| GPR (10k BSM) | 0.071 | 1.22 | 0.99996 | 78 | ~10x |
| BSM (closed form) | 0 | 0 | 1 | 0.54 | ~1500x |

**GPR vs FNN on identical BSM data** (mean abs error / fit time / inference):

| n | GPR | FNN |
|---|---|---|
| 2,000 | 0.142 / 0.1s / 17 µs | 0.284 / 18s / ~1 µs |
| 5,000 | 0.093 / 1.2s / 43 µs | 0.195 / 18s / ~1 µs |
| 10,000 | 0.071 / 7.9s / 77 µs | 0.162 / 18s / ~1 µs |

- GPR is ~2x more accurate than the FNN at every equal training size, and
  its error falls monotonically with n (so does the FNN's).
- FNN inference is ~80x faster than GPR and constant in n; GPR inference
  grows O(n) and memory grows O(n²) (~800 MB at n=10k vs 0.11 MB for the
  FNN); exact GPR fit is O(n³), so it can't scale to the FNN's 35k+ rows.
- GPR's predictive σ is reasonably calibrated: 92.9% of true prices fall
  within ±2σ (ideal 95%), σ–|error| correlation 0.48. The FNN has no
  built-in uncertainty.
- Takeaway: GPR when data is scarce/expensive or uncertainty matters; FNN
  when large datasets are available and query throughput matters most.
- GPR fit times above exclude the one-time ~26s hyperparameter search.

## Large-scale training on 10^5+ points (`04_large_scale_training.ipynb`)

Per user request, both ML models trained on **105,000 rows**:

- New 150,000-option dataset (seed `SEED+150`, same sampling ranges), FFT
  labels (max |FFT−BSM| = 9.6e-4), cached at `data/fft_dataset_150k.csv`;
  split 70/15/15 → 105,000 / 22,500 / 22,500 (split stats in
  `data/large_scale_split_stats.csv` — splits are near-identical).
- **FNN**: same 128-128-64-32 architecture, Adam + cosine LR, batch 1024,
  200 epochs, best-val checkpoint. 94s to train.
- **GPR → sparse variational GP (SVGP, GPyTorch)**: exact GPR at n=105k
  needs ~88 GB for the kernel matrix (machine has 8 GB), so used SVGP with
  M=1,024 learned inducing points, ARD-RBF kernel, mini-batch Adam on the
  ELBO, 30 epochs over all 105k rows. 402s to train. (Natural-gradient
  variant was prototyped and didn't converge faster here.)
- Evaluated on the 22,500-row test split and on notebook 03's shared
  1,000-option set (same seed), so numbers line up with the small models.
- Saved: `models/fnn_105k.pt`, `models/svgp_105k.pt`,
  `models/scaler_{X,y}_105k.joblib`; results in `data/large_scale_*.csv/.png`.

**Executed results (shared 1,000 options, error vs BSM):**

| Model | n | Mean abs err | Max abs err | R² | Train | µs/option | Size |
|---|---|---|---|---|---|---|---|
| FNN | 35k (nb 02) | 0.128 | 1.18 | 0.99994 | — | 0.70 | 0.11 MB |
| **FNN** | **105k** | **0.094** | **0.59** | 0.99997 | 94s | 0.44 | 0.11 MB |
| exact GPR | 10k (nb 03) | 0.071 | 1.22 | 0.99996 | 7.9s | 77 | 800 MB |
| **sparse GPR** | **105k** | **0.134** | 2.35 | 0.99988 | 402s | 22 | 4.2 MB |

On the 22,500-row test split: FNN mean abs err 0.091 (63.5% within $0.10,
96.2% within $0.25); sparse GPR 0.132 (60.3% / 86.9%); sparse GPR's ±2σ
band covers 98.7% of labels (slightly conservative).

- FNN improves steadily with data (0.284 at 2k → 0.128 at 35k → 0.094 at
  105k) and is now the best ML model on max error.
- Sparse GPR uses all 105k points but its accuracy is capped by the 1,024
  inducing points: it's worse than the exact 10k GPR (0.134 vs 0.071),
  though it's 180x smaller and 3.5x faster to query. Its validation error
  was still falling at epoch 30 (0.136 → 0.131), so more epochs / more
  inducing points would improve it at extra training cost.

## Heston stochastic-volatility pricing (`05`-`07` notebooks)

The Heston extension is now in scope. Under risk-neutral dynamics, the
underlying follows correlated spot and variance diffusions:

- `dS/S = (r-q)dt + sqrt(v)dW_S`
- `dv = kappa(theta-v)dt + xi*sqrt(v)dW_v`
- `corr(dW_S,dW_v) = rho`

There is no Black-Scholes-style elementary closed-form price. The Heston
characteristic function is available and will be used with Carr-Madan FFT as
the numerical reference that labels the training set. Four methods will be
compared on the same new, held-out call options: risk-neutral Monte Carlo,
Heston FFT, an NN trained on FFT prices, and exact Gaussian Process
Regression trained on FFT prices. GPR will use the paper's scaled inputs,
ARD squared-exponential/RBF covariance with a noise term, and standardized
targets. Since exact GPR has cubic fit cost and quadratic memory, it will be
trained on a fixed subset of the FFT dataset (default 5,000 rows, configurable)
while the NN uses the full dataset. The paper uses a quadratic mean function,
a squared-exponential covariance, and strictly positive observation noise;
the Heston experiment will use that structure, with ARD length scales and
standardized features/targets. The notebook will report training sizes and
timings so the comparison is explicit.

New notebooks:

1. `05_heston_mc_fft.ipynb` — explain the Heston parameters, implement
   correlated full-truncation Euler Monte Carlo and Heston
   characteristic-function Carr-Madan FFT, and sanity-check convergence on
   representative options. No closed-form BSM value will be presented as a
   Heston truth.
2. `06_heston_dataset_ml.ipynb` — generate/cache a reproducible 100,000-row
   Heston parameter dataset priced by FFT; split train/validation/test; train
   and save the NN on the training split and exact GPR on its configured
   subset; evaluate both against FFT labels. Save data, models, scalers, and
   metric tables under `data/` and `models/` with `heston_` prefixes.
3. `07_heston_comparison.ipynb` — sample fresh parameter combinations not
   used for training, price them independently with all four methods, report
   MAE/RMSE/max absolute error vs FFT and inference timings, then plot all
   four price series using distinct colors. X-axis is the held-out option
   index (data point); y-axis is predicted call price. MC uncertainty will be
   shown/reported so stochastic noise is not mistaken for model error.

Use one fixed parameter schema across all methods. Fix spot at `S=100` and
use features `(log(K/F), T, r, q, kappa, rho, xi, theta, v0)`, where
`F=S*exp((r-q)T)` and `K/F` is forward moneyness. Here `xi` is vol-of-vol,
`theta` is long-run variance, matching common Heston notation; the paper
denotes those two parameters by `theta` and `eta`, respectively. Use the
paper's training ranges (K/F 0.4-1.6, T 11-12 months, r 1.5-2.5%, q 0-5%,
kappa 1.4-2.6, rho -0.85 to -0.55, xi 0.45-0.75, theta 0.01-0.1, v0
0.01-0.1) and its narrower fresh-test ranges (K/F 0.5-1.5, kappa 1.5-2.5,
rho -0.8 to -0.6, xi 0.5-0.7, theta/v0 0.02-0.1; other ranges unchanged).
Create 100,000 rows as 5,000 independent Heston market/model contexts with
20 strikes each, and split by context to keep validation/test contexts
unseen. Sampling ranges, seed, FFT damping/grid settings, MC path/time-step
counts, and GPR subset size will be defined once and reused. Keep MC and FFT
implementations in a shared Heston pricing utility imported by all new
notebooks. The new notebooks are separate from the completed BSM notebooks
and do not change their results.

Heston implementation checklist:

- [x] `05_heston_mc_fft.ipynb`: Heston MC and characteristic-function FFT.
- [x] `06_heston_dataset_ml.ipynb`: 100k FFT dataset, NN, and paper-style GPR.
- [x] `07_heston_comparison.ipynb`: fresh options, all four prices, metrics,
      timings, and the requested four-color indexed scatter plot.
- [x] Confirm numerical behavior: MC convergence, chosen FFT grid/damping
      neighborhood, and loading both saved ML models for comparison.
- [x] Record final settings, limitations, and measured results below.

**Executed Heston results (2026-11-23):**

- Data: 100,000 FFT-labeled calls from 5,000 independent contexts × 20
  strikes; split by context into 70,000 train / 15,000 validation / 15,000
  test. Spot is 100. Carr-Madan settings: N=4,096, eta=0.25, alpha=1.25.
- NN: trained on 70,000 rows on MPS in 235.0s. Held-out (15,000 rows)
  MAE=0.02961, max error=0.22928, RMSE=0.04059, R²=0.999995; 97.1% within
  $0.10 and 100% within $0.25 of FFT.
- Paper-style exact GPR: 5,000 rows (quadratic trend + ARD squared-exponential
  covariance + positive noise), fit in 12.6s. Held-out MAE=0.00855, max
  error=0.22965, RMSE=0.01599, R²=0.999999; 99.7% within $0.10 and 100%
  within $0.25 of FFT. The serialized exact model is about 191 MB.
- Four-method comparison: 100 independently sampled fresh options; MC uses
  20,000 paths × 1,000 full-truncation Euler steps. Errors versus FFT:

  | Method | MAE | Max abs error | RMSE | R² | µs / option | Speed vs FFT |
  |---|---:|---:|---:|---:|---:|---:|
  | MC | 0.04200 | 0.15602 | 0.05540 | 0.999989 | 336,383 | 0.00173x |
  | FFT | 0 | 0 | 0 | 1.000000 | 582.09 | 1x |
  | NN | 0.02683 | 0.14256 | 0.03747 | 0.999995 | 10.92 | 53.33x |
  | GPR | 0.00720 | 0.04535 | 0.01165 | 1.000000 | 46.35 | 12.56x |

- MC estimates were within 2 estimated standard errors of FFT for 93% of
  the 100 options. On the demonstration call, increasing paths at 1,000
  time steps gave MC prices 7.45716 / 7.37945 / 7.38028 for 10k / 30k /
  80k paths, versus FFT 7.37857.
- FFT sensitivity on five strikes: N=4,096 vs 8,192 differed by at most
  4.5e-4; alpha=1.25 vs 1.5 was stable, while alpha=0.75 changed deep-OTM
  values more materially. Keep alpha=1.25.
- Comparison graph and raw outputs: `data/heston_four_method_comparison.png`,
  `data/heston_four_method_new_options.csv`,
  `data/heston_four_method_metrics.csv`,
  `data/heston_four_method_timings.csv`.

## Status

- [x] code.py (BSM, MC, FFT) verified correct.
- [x] Project-local `.venv` with torch/matplotlib/pandas/scikit-learn/jupyter.
      The old `.venv` pointed at another machine's Python
      (`/opt/miniconda3`) and was broken; it was moved to `.venv_broken_old`
      (safe to delete) and rebuilt from `/opt/anaconda3` Python 3.12
      (torch 2.15).
- [x] `01_bsm_mc_fft.ipynb` — built and executed end-to-end; MC converges to
      BSM, FFT matches BSM to ~1e-5; outputs saved to `data/`.
- [x] `02_fft_dataset_ml.ipynb` — built and executed end-to-end. 50,000-row
      FFT dataset; PyTorch MLP (128-128-64-32) trains to test MAE≈0.13,
      RMSE≈0.18; NN inference ~1225x faster per option than looped FFT;
      model/scalers/results saved to `models/` and `data/`. Fixed a bug
      where test-set relative error blew up to nonsense values (near-zero
      true price in the denominator for deep-OTM options) by flooring the
      denominator at $0.01 — corrected mean relative error is 43.16%
      (63/7500 options floored), which is expected since the NN's roughly
      constant ~$0.13 absolute error is large in percentage terms for
      cheap/OTM options. Added an accuracy cell: R^2 = 0.999928;
      tolerance-band accuracy 46.65% within $0.10, 86.81% within $0.25,
      98.80% within $0.50, 99.93% within $1.00.
      Re-executed 2026-11-23 in the rebuilt venv; numbers shifted slightly
      (new torch version): test MAE 0.126, RMSE 0.170, R² 0.999935, within
      $0.10/$0.25/$0.50/$1.00 = 50.3%/88.8%/99.0%/99.95%, FNN ~800x faster
      than looped FFT. GPR section-7 results unchanged (AAE 0.143→0.090→
      0.072); its speed-ups are timing-noisy run to run (now 50x→19x→7x).
- [x] `03_comparison.ipynb` — built (`scripts/build_notebook_03.py`) and
      executed end-to-end: shared strike sweep + 1,000 random options,
      5 methods (BSM/MC/FFT/FNN/GPR), overlay + error plots, error table,
      log-scale timing bar chart, break-even, and the GPR-vs-FNN section
      above.
- [x] All three notebooks run top-to-bottom with no errors (01 → 02 → 03).
- [x] `04_large_scale_training.ipynb` — FNN and sparse GPR trained on 105k
      rows; executed end-to-end without errors (~11 min; ~9 min on re-run
      since the dataset is cached). Needs `gpytorch` (installed in `.venv`).
- [x] Heston workflow notebooks and shared Heston pricing functions authored.
- [x] Notebooks 05-07 executed; generated the dataset, trained models,
      four-way graph, metrics, and measured results recorded above.
