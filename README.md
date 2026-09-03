[README_hawkes_ofi_market_microstructure.md](https://github.com/user-attachments/files/31768168/README_hawkes_ofi_market_microstructure.md)
# Hawkes–OFI Market Microstructure

<p align="center">
  <img src="assets/header.svg" alt="Hawkes–OFI Market Microstructure" width="100%">
</p>

<p align="center">
  <strong>Event-time modeling of BTCUSDT trade arrivals + order-flow imbalance for short-horizon price prediction.</strong>
</p>

<p align="center">
  <a href="#research-status">Research Status</a> ·
  <a href="#methodology">Methodology</a> ·
  <a href="#key-results">Key Results</a> ·
  <a href="#repository">Repository</a> ·
  <a href="#reproducibility">Reproducibility</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/MATLAB-R2026a-orange?logo=mathworks&logoColor=white">
  <img src="https://img.shields.io/badge/Market-BTCUSDT-F7931A">
  <img src="https://img.shields.io/badge/Data-Binance-003087">
  <img src="https://img.shields.io/badge/Validation-Leave--One--Capture--Out-6f42c1">
  <img src="https://img.shields.io/badge/Replication-Python%20%2B%20MATLAB-2ea44f">
</p>

---

## Overview

This project studies whether the **temporal clustering of buy- and sell-side trade arrivals** contains predictive information about short-horizon BTCUSDT mid-price returns beyond a conventional order-flow imbalance (OFI) benchmark.

The core idea is simple:

\[
\lambda_B(t) \neq \text{constant}, \qquad
\lambda_S(t) \neq \text{constant}
\]

Instead, trade arrivals are modeled with a bivariate Hawkes process. The resulting signed pressure signal is

\[
H_t = \lambda_B(t)-\lambda_S(t),
\]

where positive values indicate relatively stronger conditional buy intensity and negative values indicate relatively stronger conditional sell intensity.

The project is deliberately built as an **evidence-first microstructure research pipeline** rather than a black-box prediction benchmark.

## Research Status

### Completed

- ✅ Live BTCUSDT trade + depth capture
- ✅ Snapshot/depth synchronization validation
- ✅ Deterministic order-book reconstruction
- ✅ L1 and multi-level OFI construction
- ✅ Bivariate same-side Hawkes estimation
- ✅ Poisson benchmark
- ✅ Walk-forward validation
- ✅ Cross-capture validation
- ✅ Three-capture leave-one-capture-out evaluation
- ✅ Temporal-resolution sensitivity
- ✅ Block-bootstrap uncertainty analysis
- ✅ Independent Python/MATLAB replication
- ✅ Publication-quality figures
- ✅ Consolidated results tables

### Current dataset

| Capture | Duration | Trades | Buy | Sell | Reconstructed book states |
|---|---:|---:|---:|---:|---:|
| `capture_02` | ~600 s | 13,117 | 8,182 | 4,935 | 6,000 |
| `capture_03` | ~600 s | 18,595 | 8,925 | 9,670 | 6,002 |
| `capture_04` | ~600 s | 7,730 | 4,845 | 2,885 | 6,000 |
| **Total** | **~1,800 s** | **39,442** | **21,952** | **17,490** | **18,002** |

---

## Research Pipeline

<p align="center">
  <img src="assets/research_pipeline.svg" alt="Animated research pipeline" width="100%">
</p>

The pipeline is:

```text
Binance live streams
        │
        ▼
Snapshot + depth synchronization
        │
        ▼
Validated order-book reconstruction
        │
        ├───────────────► OFI features
        │
        ▼
Trade-side classification
        │
        ▼
100-ms event grid
        │
        ▼
Bivariate Hawkes estimation
        │
        ▼
Hawkes pressure = λ_buy − λ_sell
        │
        ├───────────────► 1 s forecast
        │
        └───────────────► 5 s forecast
                         │
                         ▼
              Out-of-sample validation
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       LOO-CV       Bootstrap       Resolution
          │              │              │
          └──────────────┼──────────────┘
                         ▼
              Python ↔ MATLAB check
```

---

## Methodology

### 1. Trade arrival process

For buy and sell event counts \(N_B(t)\) and \(N_S(t)\), the primary model uses same-side exponential excitation:

\[
\lambda_B(t)
=
\mu_B+
\alpha_B
\int_0^t e^{-\beta(t-s)}\,dN_B(s),
\]

\[
\lambda_S(t)
=
\mu_S+
\alpha_S
\int_0^t e^{-\beta(t-s)}\,dN_S(s).
\]

The branching ratios are

\[
n_B = \frac{\alpha_B}{\beta},
\qquad
n_S = \frac{\alpha_S}{\beta}.
\]

For the restricted diagonal excitation matrix, the spectral radius is

\[
\rho=\max(n_B,n_S).
\]

All three fitted captures satisfy \(\rho<1\).

### 2. Hawkes pressure

The research signal is

\[
H_t=\lambda_B(t)-\lambda_S(t).
\]

This converts the two conditional event intensities into a single signed pressure measure.

### 3. OFI benchmark

L1 OFI is used as the primary benchmark. Additional L10 and depth-weighted variants were evaluated during development.

### 4. Forecast target

For horizon \(h\),

\[
r_{t,t+h} = \log P_{t+h}-\log P_t.
\]

Primary horizons:

- 1 second
- 5 seconds

### 5. Validation design

The primary generalization test is leave-one-capture-out:

```text
capture_02 + capture_03 → capture_04
capture_02 + capture_04 → capture_03
capture_03 + capture_04 → capture_02
```

The held-out capture is excluded from model fitting.

---

## Key Results

### Event clustering

The pooled one-second trade-count process has:

\[
E[N] = 21.91
\]

\[
Var(N) = 4049.85
\]

and therefore:

\[
F = \frac{Var(N)}{E[N]} = 184.82.
\]

The Poisson benchmark has \(F=1\).

Capture-level Fano factors:

| Capture | Fano factor |
|---|---:|
| `capture_02` | 227.87 |
| `capture_03` | 176.38 |
| `capture_04` | 120.37 |
| **Pooled** | **184.82** |

This provides strong empirical motivation for modeling event history rather than assuming independent Poisson arrivals.

### Hawkes parameter estimates

| Capture | \(\mu_B\) | \(\mu_S\) | \(\beta\) | \(n_B\) | \(n_S\) | \(\rho\) |
|---|---:|---:|---:|---:|---:|---:|
| `capture_02` | 7.538 | 7.074 | 5.175 | 0.447 | 0.140 | 0.447 |
| `capture_03` | 6.138 | 10.575 | 3.057 | 0.587 | 0.344 | 0.587 |
| `capture_04` | 3.374 | 2.820 | 1.032 | 0.582 | 0.417 | 0.582 |

### Leave-one-capture-out prediction

Average OOS \(R^2\):

| Horizon | L1 OFI | Hawkes pressure | OFI + Hawkes |
|---|---:|---:|---:|
| 1 s | -0.19% | **1.12%** | 1.03% |
| 5 s | 0.01% | **1.62%** | 1.54% |

Average prediction/return correlation:

| Horizon | L1 OFI | Hawkes pressure | OFI + Hawkes |
|---|---:|---:|---:|
| 1 s | 0.060 | **0.122** | 0.121 |
| 5 s | 0.060 | **0.161** | 0.159 |

The important result is not that Hawkes produces a huge \(R^2\). It is that the signal remains positive when an entire capture is held out and generally outperforms a simple L1 OFI benchmark.

### Pressure-response result

When observations are ranked into within-capture Hawkes-pressure deciles:

- lowest-pressure decile: approximately **−0.31 bps** mean future 5-second return
- highest-pressure decile: approximately **+0.37 bps**
- low-to-high spread: approximately **0.68 bps**

### Temporal-resolution sensitivity

At a 1-second forecast horizon:

| Resolution | OOS \(R^2\) |
|---|---:|
| 50 ms | **4.71%** |
| 100 ms | 3.38% |
| 250 ms | 2.45% |
| 500 ms | 1.79% |

At a 5-second horizon:

| Resolution | OOS \(R^2\) |
|---|---:|
| 50 ms | 2.24% |
| 100 ms | 6.49% |
| 250 ms | **6.54%** |
| 500 ms | 6.25% |

The temporal-resolution effect is horizon-dependent.

---

## Visual Results

### Trade-arrival clustering

<p align="center">
  <img src="figures/figure_01_event_clustering.png" alt="Trade arrival clustering versus Poisson benchmark" width="85%">
</p>

<p align="center">
  <img src="figures/figure_01b_fano_by_capture.png" alt="Fano factor by capture" width="85%">
</p>

### Hawkes pressure and future returns

<p align="center">
  <img src="figures/figure_02_hawkes_pressure_response.png" alt="Hawkes pressure response" width="85%">
</p>

<p align="center">
  <img src="figures/figure_02b_hawkes_pressure_by_capture.png" alt="Hawkes pressure response by capture" width="85%">
</p>

### Out-of-sample performance

<p align="center">
  <img src="figures/figure_03_oos_r2_comparison.png" alt="Out of sample R2 comparison" width="85%">
</p>

<p align="center">
  <img src="figures/figure_03b_oos_correlation.png" alt="Out of sample correlation comparison" width="85%">
</p>

<p align="center">
  <img src="figures/figure_03c_hawkes_by_capture.png" alt="Hawkes performance by held out capture" width="85%">
</p>

### Temporal-resolution robustness

<p align="center">
  <img src="figures/figure_04_resolution_r2.png" alt="Resolution sensitivity R2" width="85%">
</p>

<p align="center">
  <img src="figures/figure_04b_resolution_correlation.png" alt="Resolution sensitivity correlation" width="85%">
</p>

<p align="center">
  <img src="figures/figure_04c_resolution_rmse.png" alt="Resolution sensitivity RMSE" width="85%">
</p>

---

## Repository

```text
hawkes-ofi-market-microstructure/
│
├── data/
│   ├── live/
│   │   ├── capture_02.jsonl
│   │   ├── capture_03.jsonl
│   │   └── capture_04.jsonl
│   │
│   └── processed/
│       ├── all_capture_book_states.parquet
│       ├── all_capture_trade_events.parquet
│       ├── leave_one_capture_out.csv
│       ├── final_statistical_test.csv
│       ├── hawkes_resolution_prediction.csv
│       └── ...
│
├── src/
│   ├── data/
│   └── models/
│
├── experiments/
│   ├── build_multi_capture_trades.py
│   ├── leave_one_capture_out.py
│   ├── final_statistical_test.py
│   ├── figure_event_clustering.py
│   ├── figure_hawkes_pressure_response.py
│   ├── figure_oos_model_comparison.py
│   ├── figure_resolution_robustness.py
│   ├── finalize_figures.py
│   └── ...
│
├── matlab/
│   └── fit_hawkes_bivariate.m
│
├── figures/
│   ├── figure_01_*.png
│   ├── figure_01_*.pdf
│   ├── figure_02_*.png
│   ├── figure_02_*.pdf
│   ├── figure_03_*.png
│   ├── figure_03_*.pdf
│   └── figure_04_*.png
│
├── results/
│   ├── table_01_hawkes_parameters.*
│   ├── table_02_leave_one_capture_out.*
│   ├── table_03_statistical_comparison.*
│   ├── table_04_resolution_sensitivity.*
│   └── final_results_summary.txt
│
└── README.md
```

---

## Reproducibility

The project is designed so the empirical chain can be reconstructed from raw captures.

A typical workflow is:

```powershell
.\.venv\Scripts\python.exe experiments\build_multi_capture_trades.py
.\.venv\Scripts\python.exe experiments\process_all_captures.py
.\.venv\Scripts\python.exe experiments\leave_one_capture_out.py
.\.venv\Scripts\python.exe experiments\final_statistical_test.py
.\.venv\Scripts\python.exe experiments\figure_oos_model_comparison.py
.\.venv\Scripts\python.exe experiments\figure_resolution_robustness.py
.\.venv\Scripts\python.exe experiments\finalize_figures.py
```

MATLAB replication:

```powershell
matlab -batch "cd('C:\Users\annoy\hawkes-ofi-market-microstructure'); addpath('matlab'); results=fit_hawkes_bivariate();"
```

The Python and MATLAB implementations use the same exact 100-ms capture grid and binned likelihood.

---

## Scientific Interpretation

The project supports a deliberately narrow conclusion:

> **Temporal clustering in BTCUSDT trade arrivals contains short-horizon predictive information that is not fully captured by a simple contemporaneous L1 OFI measure in the observed independent capture episodes.**

The evidence should not be interpreted as proof of a universal profitable trading strategy.

Important limitations include:

- only three independent ten-minute captures;
- strong within-capture serial dependence;
- no transaction-cost or execution simulation;
- no long-horizon regime analysis;
- restricted same-side Hawkes specification;
- no claim of universal cryptocurrency predictability.

The leave-one-capture-out result is therefore the primary generalization benchmark.

---

## Research Stack

```text
Data
  └─ Binance trade + depth streams

Microstructure
  ├─ synchronized snapshot
  ├─ order-book reconstruction
  ├─ trade-side classification
  └─ OFI

Point-process modeling
  ├─ Poisson benchmark
  ├─ bivariate Hawkes
  ├─ branching ratios
  └─ Hawkes pressure

Prediction
  ├─ 1-second return
  ├─ 5-second return
  ├─ OOS R²
  ├─ RMSE
  └─ prediction correlation

Validation
  ├─ walk-forward
  ├─ cross-capture
  ├─ leave-one-capture-out
  ├─ block bootstrap
  └─ resolution sensitivity

Reproducibility
  ├─ Python
  └─ MATLAB
```

---

## Status

**Empirical pipeline:** complete for the current three-capture dataset.

**Primary model:** frozen.

**Primary validation:** complete.

**Independent implementation check:** complete.

**Next research direction:** expand the number of independent market episodes and test whether the effect survives across broader volatility and liquidity regimes.

---

## References

- Cont, R., Kukanov, A., & Stoikov, S. (2014). *The Price Impact of Order Book Events*. Journal of Financial Econometrics, 12(1), 47–88.
- Bacry, E., Delattre, S., Hoffmann, M., & Muzy, J.-F. (2013). *Modelling microstructure noise with mutually exciting point processes*. Quantitative Finance, 13(1), 65–77.
- Bacry, E., & Muzy, J.-F. (2014). *Hawkes model for price and trades high-frequency dynamics*. Quantitative Finance, 14(7), 1147–1166.
- Bacry, E., Mastromatteo, I., & Muzy, J.-F. (2015). *Hawkes Processes in Finance*. Market Microstructure and Liquidity, 1(1), 1550005.
- Anantha, A. N., & Jain, S. (2026). *Forecasting High Frequency Order Flow Imbalance using Hawkes Processes*. Computational Economics, 67(1), 279–312.
- Raffaelli, D., Cestari, R. G., Marazzina, D., & Formentin, S. (2026). *Forecasting Bitcoin price movements using multivariate Hawkes processes and limit order book data*. Decisions in Economics and Finance.
