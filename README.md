
# Hawkes–OFI Market Microstructure

<p align="center">
  <strong>BTCUSDT market-microstructure research using Hawkes processes, order-flow imbalance, and strict cross-capture out-of-sample validation.</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#research-question">Research Question</a> ·
  <a href="#data">Data</a> ·
  <a href="#methodology">Methodology</a> ·
  <a href="#results">Results</a> ·
  <a href="#reproducibility">Reproducibility</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/MATLAB-R2026a-orange?logo=mathworks&logoColor=white" alt="MATLAB">
  <img src="https://img.shields.io/badge/Market-BTCUSDT-F7931A" alt="BTCUSDT">
  <img src="https://img.shields.io/badge/Exchange-Binance-003087" alt="Binance">
  <img src="https://img.shields.io/badge/Model-Bivariate%20Hawkes-6f42c1" alt="Bivariate Hawkes">
  <img src="https://img.shields.io/badge/Validation-Leave--One--Capture--Out-2ea44f" alt="Leave-One-Capture-Out">
  <img src="https://img.shields.io/badge/Replication-Python%20%2B%20MATLAB-0f766e" alt="Python and MATLAB replication">
</p>

---

## Overview

This repository contains a reproducible market-microstructure study of BTCUSDT trade-arrival dynamics.

The project asks whether the **temporal clustering of buy- and sell-side trade arrivals** contains predictive information about short-horizon BTCUSDT mid-price returns beyond a conventional **Order-Flow Imbalance (OFI)** benchmark.

The central research signal is:

```text
Hawkes pressure
    = conditional buy-arrival intensity
    - conditional sell-arrival intensity
```

Positive Hawkes pressure means the conditional buy intensity is greater than the conditional sell intensity. Negative pressure means the reverse.

The project is designed around a complete empirical chain:

```text
raw Binance market data
        ↓
capture validation
        ↓
order-book reconstruction
        ↓
trade-side classification
        ↓
100-ms temporal alignment
        ↓
OFI benchmark + Hawkes estimation
        ↓
Hawkes pressure
        ↓
1 s / 5 s return prediction
        ↓
cross-capture validation
        ↓
bootstrap / sensitivity analysis
        ↓
Python ↔ MATLAB replication
```

---

# Research Question

> Does explicitly modeling the temporal clustering of buy- and sell-side trade arrivals provide predictive information about short-horizon BTCUSDT returns beyond a simple L1 OFI benchmark?

The project also examines:

- whether trade arrivals are consistent with a Poisson process;
- the strength of buy- and sell-side excitation;
- whether Hawkes pressure maps into future returns;
- whether the signal generalizes to an entirely held-out capture;
- whether OFI adds information beyond Hawkes pressure;
- how temporal resolution changes the result;
- whether the Hawkes estimator can be reproduced independently.

---

# Research Status

| Component | Status |
|---|---|
| Live BTCUSDT capture | ✅ Complete |
| Snapshot validation | ✅ Complete |
| Depth sequence validation | ✅ Complete |
| Trade validation | ✅ Complete |
| Reception timestamp validation | ✅ Complete |
| Order-book reconstruction | ✅ Complete |
| L1 OFI | ✅ Complete |
| Multi-level OFI | ✅ Complete |
| Depth-weighted OFI | ✅ Complete |
| Poisson benchmark | ✅ Complete |
| Bivariate Hawkes estimation | ✅ Complete |
| Hawkes pressure | ✅ Complete |
| Walk-forward analysis | ✅ Complete |
| Cross-capture analysis | ✅ Complete |
| Leave-one-capture-out validation | ✅ Complete |
| Paired prediction-error comparison | ✅ Complete |
| Block-bootstrap analysis | ✅ Complete |
| State-conditioned analysis | ✅ Complete |
| Temporal-resolution sensitivity | ✅ Complete |
| Python implementation | ✅ Complete |
| MATLAB implementation | ✅ Complete |
| Exact-grid Python/MATLAB verification | ✅ Complete |
| Final figures | ✅ Complete |
| Final result tables | ✅ Complete |
| Larger independent-capture study | ⏳ Future work |
| Execution / transaction-cost model | ⏳ Future work |

---

# Data

The current dataset contains three independently captured BTCUSDT market episodes of approximately ten minutes each.

| Capture | Duration | Trades | Buys | Sells | Book states |
|---|---:|---:|---:|---:|---:|
| `capture_02` | ~599 s | 13,117 | 8,182 | 4,935 | 6,000 |
| `capture_03` | ~600 s | 18,595 | 8,925 | 9,670 | 6,002 |
| `capture_04` | ~600 s | 7,730 | 4,845 | 2,885 | 6,000 |
| **Total** | **~1,800 s** | **39,442** | **21,952** | **17,490** | **18,002** |

### Trade composition

| Capture | Trades / second | Buy fraction | Sell fraction |
|---|---:|---:|---:|
| `capture_02` | 21.89 | 62.38% | 37.62% |
| `capture_03` | 31.01 | 47.99% | 52.00% |
| `capture_04` | 12.89 | 62.68% | 37.32% |

The captures intentionally differ in activity and trade-side composition. That heterogeneity is useful for the capture-level generalization test.

---

# Live Capture Validation

Each capture contains:

- Binance trade events;
- Binance depth events;
- an exchange order-book snapshot.

The validation pipeline checks:

```text
snapshot integrity
       ↓
depth sequence continuity
       ↓
update-ID consistency
       ↓
trade-record validity
       ↓
reception timestamps
```

The fourth live capture was independently collected and validated before being added to the final multi-capture dataset.

### Capture 04 validation

```text
Depth events:      6,000
Trade events:      7,730
Snapshot:          OK
Depth sequence:    OK
Trades:            OK
Timestamps:        OK
Validation:        PASSED
```

---

# Methodology

## 1. Order-book reconstruction

Each capture starts with an exchange-provided snapshot followed by sequential depth updates.

The reconstructed state contains variables used to compute:

- mid-price;
- spread;
- queue imbalance;
- L1 OFI;
- multi-level OFI;
- depth-weighted OFI.

The final combined reconstructed book data are stored in:

```text
data/processed/all_capture_book_states.parquet
```

---

## 2. Trade-side classification

Trade events are separated into buy- and sell-side arrivals using the exchange trade-stream maker indicator under the convention used throughout the project.

The classification rule is held fixed across captures.

---

## 3. Temporal alignment

The main Hawkes estimation uses a **100-ms research grid**.

The implementation distinguishes between:

```text
Conceptual model:
continuous-time Hawkes process

Actual estimator:
100-ms binned count likelihood
```

The final Python and MATLAB estimators use the same exact grid convention.

---

# Hawkes Model

## Bivariate same-side specification

The primary model contains two self-exciting event streams:

```text
Buy events  → excite future buy events

Sell events → excite future sell events
```

The model does **not** include buy-to-sell or sell-to-buy cross-excitation in the primary specification.

The conditional intensities are conceptually:

```text
lambda_buy
    = buy baseline
    + historical buy excitation

lambda_sell
    = sell baseline
    + historical sell excitation
```

The exponential kernel uses a common decay parameter.

---

## Branching ratios

The integrated excitation is summarized by:

```text
buy branching ratio  = alpha_buy / beta
sell branching ratio = alpha_sell / beta
```

For the diagonal excitation matrix used in this project:

```text
spectral radius
    = max(buy branching ratio,
          sell branching ratio)
```

A fitted process is stationary when the spectral radius is below one.

All three final captures satisfy this condition.

---

## Hawkes pressure

The research signal is:

```text
Hawkes pressure
    = lambda_buy - lambda_sell
```

Interpretation:

```text
Hawkes pressure > 0
    → relatively stronger conditional buy activity

Hawkes pressure < 0
    → relatively stronger conditional sell activity
```

This is intentionally distinct from OFI.

```text
OFI
    = order-book / displayed-flow imbalance

Hawkes pressure
    = conditional trade-arrival intensity imbalance
```

---

# Poisson Benchmark

A Poisson event-count process has:

```text
variance = mean
```

and therefore:

```text
Fano factor = variance / mean = 1
```

The pooled one-second BTCUSDT trade counts produce:

```text
Mean:       21.912222
Variance:   4049.847766
Fano:       184.821408
```

Capture-level Fano factors:

| Capture | Fano factor |
|---|---:|
| `capture_02` | 227.871 |
| `capture_03` | 176.382 |
| `capture_04` | 120.373 |
| **Pooled** | **184.821** |

The observed process is therefore dramatically more dispersed than the Poisson benchmark.

This motivates a history-dependent point-process model.

It does **not** by itself prove that Hawkes is the unique or optimal model.

---

# Final Hawkes Estimates

| Capture | Buy baseline | Sell baseline | Decay | Buy branching | Sell branching | Spectral radius |
|---|---:|---:|---:|---:|---:|---:|
| `capture_02` | 7.538247 | 7.074440 | 5.174913 | 0.447026 | 0.139642 | 0.447026 |
| `capture_03` | 6.138138 | 10.574907 | 3.057319 | 0.587241 | 0.343635 | 0.587241 |
| `capture_04` | 3.374296 | 2.820131 | 1.031680 | 0.582087 | 0.416523 | 0.582087 |

Negative log-likelihoods:

| Capture | Negative log-likelihood |
|---|---:|
| `capture_02` | 50,709.89 |
| `capture_03` | 63,004.05 |
| `capture_04` | 28,784.48 |

The fitted processes are stationary while showing meaningful episode-level self-excitation.

---

# Primary Validation

The primary generalization test is **leave-one-capture-out validation**.

```text
Train: capture_02 + capture_03
Test:  capture_04

Train: capture_02 + capture_04
Test:  capture_03

Train: capture_03 + capture_04
Test:  capture_02
```

Each held-out capture is excluded from model fitting.

This avoids the strongest leakage problem associated with randomly splitting adjacent high-frequency observations.

Primary forecast horizons:

```text
1 second
5 seconds
```

Primary comparison:

```text
L1 OFI
vs
Hawkes pressure
```

Additional comparison:

```text
OFI + Hawkes
```

---

# Results

## Mean leave-one-capture-out OOS R²

| Horizon | L1 OFI | Hawkes pressure | OFI + Hawkes |
|---|---:|---:|---:|
| **1 s** | **-0.190%** | **1.121%** | **1.027%** |
| **5 s** | **0.007%** | **1.622%** | **1.540%** |

## Mean prediction / return correlation

| Horizon | L1 OFI | Hawkes pressure | OFI + Hawkes |
|---|---:|---:|---:|
| **1 s** | 0.060 | **0.122** | 0.121 |
| **5 s** | 0.060 | **0.161** | 0.159 |

### Main result

Hawkes pressure produces positive average out-of-sample explanatory power at both horizons and exceeds the simple L1 OFI benchmark in the current three-capture experiment.

The effect is **modest but consistently positive across the three held-out captures**.

---

# Capture-Level Generalization

The final Hawkes OOS results are:

| Training captures | Held-out capture | 1 s | 5 s |
|---|---|---:|---:|
| `02 + 03` | `04` | 1.270% | 2.310% |
| `02 + 04` | `03` | 1.790% | 2.492% |
| `03 + 04` | `02` | 0.304% | 0.063% |

The effect is positive in every held-out capture, but its magnitude varies considerably.

This heterogeneity is important and is discussed explicitly rather than averaged away.

---

# Hawkes Pressure / Future Returns

Pressure is standardized within capture and observations are ranked into pressure deciles.

The pooled five-second response shows:

```text
Lowest pressure decile:
≈ -0.31 bps

Highest pressure decile:
≈ +0.37 bps
```

Low-to-high difference:

```text
≈ 0.68 bps
```

This is a small short-horizon effect.

It should be interpreted as evidence of directional predictive information, not as a guarantee of trading profitability.

---

# Temporal-Resolution Sensitivity

The project evaluates:

```text
50 ms
100 ms
250 ms
500 ms
```

## 1-second horizon

| Resolution | OOS R² | Correlation |
|---|---:|---:|
| 50 ms | **4.71%** | **0.226** |
| 100 ms | 3.38% | 0.143 |
| 250 ms | 2.45% | 0.101 |
| 500 ms | 1.79% | 0.074 |

## 5-second horizon

| Resolution | OOS R² | Correlation |
|---|---:|---:|
| 50 ms | 2.24% | 0.125 |
| 100 ms | 6.49% | 0.160 |
| 250 ms | **6.54%** | 0.138 |
| 500 ms | 6.25% | 0.128 |

Interpretation:

- finer resolution is associated with stronger performance at the one-second horizon;
- five-second performance is comparatively stable across 100–500 ms;
- these results are sensitivity analysis, not a universal optimum.

---

# Statistical Comparison

The paired comparison uses the same held-out observations for both models.

Average Hawkes improvement in OOS R² over L1 OFI:

```text
1 second:
+1.311 percentage points

5 seconds:
+1.615 percentage points
```

A secondary block-bootstrap analysis used:

```text
2,000 repetitions
5-second blocks
```

The bootstrap generally favored Hawkes pressure.

However, the primary study contains only three independent capture episodes. Bootstrap probabilities are therefore treated as supportive uncertainty analysis rather than definitive population-level significance tests.

---

# State-Conditioned Analysis

The project also examined whether the Hawkes-pressure relationship changes across queue-imbalance regimes.

The results were heterogeneous across captures and horizons.

This analysis is treated as **secondary evidence** and is not the main empirical claim.

---

# Python ↔ MATLAB Replication

The core Hawkes estimator was independently implemented in Python and MATLAB.

Both implementations use:

```text
same captures
same reconstructed timeline
same exact 100-ms grid
same Hawkes specification
same binned likelihood
same stationarity constraint
```

For `capture_04`, the final Python verification gives:

```text
mu_buy          3.3742947465
mu_sell         2.8201315938
beta            1.0316796766
branching_buy   0.5820871370
branching_sell  0.4165226456
negative_loglik 28784.4782933165
```

The MATLAB implementation reproduces the reported parameters to approximately `1e-7` or better.

The likelihood matches to numerical precision.

This is an **implementation-level reproducibility check**, not an independent replication of the entire research study.

---

# Figures

Final figures are generated under:

```text
figures/
```

The repository contains PNG and PDF versions for:

| Figure | Analysis |
|---|---|
| Figure 1 | Trade-arrival clustering vs Poisson |
| Figure 1b | Fano factor by capture |
| Figure 2 | Hawkes pressure vs future return |
| Figure 2b | Pressure response by capture |
| Figure 3 | Leave-one-capture-out OOS R² |
| Figure 3b | OOS prediction / return correlation |
| Figure 3c | Hawkes performance by held-out capture |
| Figure 4 | Temporal-resolution OOS R² |
| Figure 4b | Temporal-resolution correlation |
| Figure 4c | Temporal-resolution RMSE |

---

# Results Package

Final CSV tables:

```text
results/table_01_hawkes_parameters.csv
results/table_02_leave_one_capture_out.csv
results/table_03_statistical_comparison.csv
results/table_04_resolution_sensitivity.csv
```

LaTeX table exports:

```text
results/table_01_hawkes_parameters.tex
results/table_02_leave_one_capture_out.tex
results/table_03_statistical_comparison.tex
results/table_04_resolution_sensitivity.tex
```

Final summary:

```text
results/final_results_summary.txt
```

Major processed datasets:

```text
data/processed/all_capture_book_states.parquet
data/processed/all_capture_trade_events.parquet
```

---

# Repository Structure

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
│       ├── bootstrap_oos_results.csv
│       └── ...
│
├── src/
│   ├── data/
│   └── models/
│
├── experiments/
│   ├── build_multi_capture_trades.py
│   ├── process_all_captures.py
│   ├── leave_one_capture_out.py
│   ├── final_statistical_test.py
│   ├── build_final_results_tables.py
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
│   └── final PNG / PDF figures
│
├── results/
│   └── final CSV / LaTeX / summary outputs
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Reproducibility

## Python

Run experiments through the project virtual environment:

```powershell
.\.venv\Scripts\python.exe
```

Build trade-event data:

```powershell
.\.venv\Scripts\python.exe experiments\build_multi_capture_trades.py
```

Process captures:

```powershell
.\.venv\Scripts\python.exe experiments\process_all_captures.py
```

Run leave-one-capture-out evaluation:

```powershell
.\.venv\Scripts\python.exe experiments\leave_one_capture_out.py
```

Run the final statistical comparison:

```powershell
.\.venv\Scripts\python.exe experiments\final_statistical_test.py
```

Build final result tables:

```powershell
.\.venv\Scripts\python.exe experiments\build_final_results_tables.py
```

Generate model-comparison figures:

```powershell
.\.venv\Scripts\python.exe experiments\figure_oos_model_comparison.py
```

Generate temporal-resolution figures:

```powershell
.\.venv\Scripts\python.exe experiments\figure_resolution_robustness.py
```

Regenerate the complete final figure set:

```powershell
.\.venv\Scripts\python.exe experiments\finalize_figures.py
```

---

## MATLAB

The MATLAB replication can be run with:

```powershell
matlab -batch "cd('C:\Users\annoy\hawkes-ofi-market-microstructure'); addpath('matlab'); results=fit_hawkes_bivariate();"
```

---

# Scientific Interpretation

The current evidence supports a narrow conclusion:

> **Temporal clustering in BTCUSDT trade arrivals contains short-horizon predictive information that is not fully captured by a simple contemporaneous L1 OFI measure in the observed independent capture episodes.**

The result is interesting because the primary model is simple, interpretable, and tested on held-out market episodes.

The result is also limited.

The current study does **not** establish:

- a universally profitable trading strategy;
- guaranteed alpha;
- causal prediction;
- stable performance across every BTCUSDT regime;
- optimality of the same-side Hawkes specification;
- profitability after fees and execution costs;
- broad population-level significance.

---

# Limitations

## Three independent episodes

The study contains 39,442 trade events but only three independent ten-minute market captures.

The raw event count should not be interpreted as 39,442 independent observations.

## Short observation windows

The total capture duration is approximately thirty minutes.

The study therefore cannot establish long-run stability.

## Restricted Hawkes model

The primary specification uses:

- same-side excitation;
- one common decay parameter;
- no buy-to-sell excitation;
- no sell-to-buy excitation.

## No execution model

The analysis predicts mid-price returns.

It does not yet include:

- trading fees;
- spread crossing;
- slippage;
- latency;
- queue position;
- market impact;
- inventory constraints.

## Limited model comparison

The research is focused on the interpretable Hawkes-vs-OFI question rather than an exhaustive comparison of every modern high-frequency forecasting model.

---

# Why This Project Is Interesting

The project connects two different views of market microstructure.

### Order-book view

```text
What is the current imbalance
in displayed liquidity?
```

### Event-time view

```text
How does the history of arrivals
change the conditional intensity
of future buy/sell events?
```

The Hawkes pressure signal attempts to capture the second effect:

```text
trade history
    ↓
conditional intensity
    ↓
buy/sell intensity imbalance
    ↓
Hawkes pressure
    ↓
short-horizon return information
```

The empirical question is whether this temporal dimension adds information beyond L1 OFI.

In the current dataset, the answer is:

```text
Yes, modestly.
```

---

# Research Position

This work does not claim to be the first application of Hawkes processes to finance or cryptocurrency.

Related research already covers:

- Hawkes processes in financial microstructure;
- order-flow imbalance and price impact;
- Hawkes models for order-flow;
- cryptocurrency limit-order-book dynamics;
- Hawkes-based OFI forecasting.

The contribution here is deliberately narrower:

1. a parsimonious same-side Hawkes pressure statistic;
2. direct comparison against L1 OFI;
3. strict capture-level out-of-sample validation;
4. temporal-resolution sensitivity;
5. independent Python/MATLAB estimator replication.

---

# Future Work

The most important next experiment is to increase the number of **independent market episodes**.

### Data expansion

- many more BTCUSDT captures;
- multiple sessions;
- multiple days;
- different volatility regimes;
- different liquidity regimes.

### Model expansion

- cross-exciting Hawkes processes;
- cancellation events;
- order submissions;
- richer event-type systems;
- deeper order-book features.

### Statistical expansion

- inference across many independent episodes;
- regime-conditional analysis;
- longer-horizon validation;
- broader model comparison.

### Economic expansion

- transaction-cost modeling;
- spread and slippage;
- latency;
- market impact;
- execution simulation;
- inventory-aware evaluation.

The next research question is not simply:

> Can the R² be made larger?

It is:

> **Does the Hawkes-pressure effect survive when the number of independent market regimes becomes substantially larger?**

---

# Key Numbers

```text
DATA
39,442       total trade events
18,002       reconstructed book states
3            independent captures
~1,800 s     total captured market time

CLUSTERING
184.82       pooled one-second Fano factor
1.00         Poisson benchmark Fano factor

PRIMARY OOS
1.12%        mean Hawkes OOS R² at 1 s
1.62%        mean Hawkes OOS R² at 5 s

L1 OFI
-0.19%       mean L1 OFI OOS R² at 1 s
0.01%        mean L1 OFI OOS R² at 5 s

CORRELATION
0.122        mean Hawkes prediction correlation at 1 s
0.161        mean Hawkes prediction correlation at 5 s

PRESSURE RESPONSE
≈0.68 bps    lowest-to-highest pressure response

REPLICATION
Python ≈ MATLAB
```

---

# Project Philosophy

```text
Capture
  ↓
Validate
  ↓
Reconstruct
  ↓
Model
  ↓
Hold out
  ↓
Quantify uncertainty
  ↓
Replicate
  ↓
Interpret conservatively
```

The goal is not to maximize a backtest statistic.

The goal is to determine whether a market-microstructure relationship survives:

- data validation;
- strict out-of-sample testing;
- capture-level heterogeneity;
- robustness analysis;
- independent implementation.

---

# Status

**Quant Project 1 — Empirical pipeline complete.**

The current three-capture experiment is frozen as the baseline result.

The next research phase is to collect substantially more independent market episodes and test whether the Hawkes-pressure relationship remains stable across broader market conditions.

---

# References

- Hawkes, A. G. (1971). *Spectra of some self-exciting and mutually exciting point processes*. Biometrika.
- Bacry, E., Delattre, S., Hoffmann, M., & Muzy, J.-F. (2013). *Modelling microstructure noise with mutually exciting point processes*. Quantitative Finance.
- Bacry, E., & Muzy, J.-F. (2014). *Hawkes model for price and trades high-frequency dynamics*. Quantitative Finance.
- Bacry, E., Mastromatteo, I., & Muzy, J.-F. (2015). *Hawkes Processes in Finance*. Market Microstructure and Liquidity.
- Cont, R., Kukanov, A., & Stoikov, S. (2014). *The Price Impact of Order Book Events*. Journal of Financial Econometrics.
- Wu, P., Rambaldi, M., Muzy, J.-F., & Bacry, E. (2019). *Queue-reactive Hawkes models for the order flow*.
- Anantha, A. N., & Jain, S. (2026). *Forecasting High Frequency Order Flow Imbalance using Hawkes Processes*. Computational Economics.
- Raffaelli, D., Cestari, R. G., Marazzina, D., & Formentin, S. (2026). *Forecasting Bitcoin price movements using multivariate Hawkes processes and limit order book data*. Decisions in Economics and Finance.

---

## Disclaimer

This repository is for research and educational purposes.

The empirical results are not investment advice and do not imply guaranteed trading profitability.
