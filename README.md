
# Hawkes–OFI Market Microstructure

<p align="center">
  <strong>Event-time modeling of BTCUSDT trade arrivals and order-flow imbalance for short-horizon price prediction.</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#research-paper">Research Paper</a> ·
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

This repository contains a reproducible market-microstructure study of **BTCUSDT trade-arrival dynamics**.

The project investigates whether the **temporal clustering of buy- and sell-side trade arrivals** contains predictive information about short-horizon BTCUSDT mid-price returns beyond a conventional **Order-Flow Imbalance (OFI)** benchmark.

The central research signal is:

```text
Hawkes pressure
    = conditional buy-arrival intensity
    - conditional sell-arrival intensity
```

The work combines:

```text
live market capture
        ↓
capture validation
        ↓
order-book reconstruction
        ↓
trade-event classification
        ↓
100-ms event-time grid
        ↓
Hawkes estimation + OFI benchmark
        ↓
out-of-sample prediction
        ↓
cross-capture validation
        ↓
bootstrap / resolution analysis
        ↓
Python ↔ MATLAB replication
```

The project is deliberately focused on **transparent empirical research**, not a black-box trading system.

---

# Research Paper

<p align="center">
  <strong>📄 Complete Research Paper</strong>
</p>

<p align="center">
  <a href="docs/hawkes_ofi_research_paper.pdf">
    <strong>Read the full research paper →</strong>
  </a>
</p>

The paper documents the complete study:

- BTCUSDT market capture and validation;
- deterministic order-book reconstruction;
- trade-arrival clustering;
- bivariate Hawkes estimation;
- Hawkes pressure construction;
- L1 OFI benchmark;
- leave-one-capture-out evaluation;
- statistical comparison;
- temporal-resolution sensitivity;
- Python/MATLAB replication;
- limitations and future work.

**Paper:** `docs/hawkes_ofi_research_paper.pdf`

> If the paper is stored under a different filename in the repository, update the link above to match the committed PDF path exactly.

---

# Research Question

> **Does explicitly modeling the temporal clustering of buy- and sell-side trade arrivals provide predictive information about short-horizon BTCUSDT returns beyond a simple L1 OFI benchmark?**

Supporting questions:

- Are trade arrivals adequately described by a Poisson process?
- How strong is buy- and sell-side self-excitation?
- Does Hawkes pressure contain directional information about future returns?
- Does the signal generalize to an entirely held-out capture?
- Does adding OFI materially improve Hawkes pressure?
- How sensitive is the result to temporal resolution?
- Can the core Hawkes estimator be independently reproduced?

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
| Research paper | ✅ Complete |
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

### Trade activity

| Capture | Trades / second | Buy fraction | Sell fraction |
|---|---:|---:|---:|
| `capture_02` | 21.89 | 62.38% | 37.62% |
| `capture_03` | 31.01 | 47.99% | 52.00% |
| `capture_04` | 12.89 | 62.68% | 37.32% |

The captures intentionally differ in activity and trade-side composition, allowing the main analysis to test generalization across distinct market episodes.

---

# Live Capture Validation

Each capture contains:

- Binance trade events;
- Binance depth events;
- an exchange order-book snapshot.

Validation checks include:

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

The final `capture_04` live validation reported:

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

# Research Pipeline

```text
┌──────────────────────┐
│ Binance trade stream │
└──────────┬───────────┘
           │
┌──────────────────────┐
│ Binance depth stream │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────┐
│ Snapshot + sequence checks   │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Deterministic book           │
│ reconstruction               │
└──────────────┬───────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌──────────────┐  ┌─────────────────┐
│ OFI features │  │ Trade-side      │
│ L1 / L10     │  │ classification  │
└──────┬───────┘  └────────┬────────┘
       │                   │
       └─────────┬─────────┘
                 ▼
        ┌─────────────────┐
        │ 100-ms grid     │
        └────────┬────────┘
                 │
     ┌───────────┼───────────┐
     ▼           ▼           ▼
  OFI        Poisson      Hawkes
benchmark    benchmark    estimation
                            │
                            ▼
                    Hawkes pressure
                            │
                            ▼
                     1 s / 5 s returns
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
             LOCO        Bootstrap     Resolution
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                    Final comparison
                            │
                            ▼
                    Python ↔ MATLAB
```

---

# Methodology

## 1. Order-book reconstruction

Each capture begins with an exchange-provided snapshot followed by sequential depth updates.

The reconstructed book is used to obtain:

- mid-price;
- spread;
- queue imbalance;
- L1 OFI;
- multi-level OFI;
- depth-weighted OFI.

---

## 2. Trade-side classification

Trades are separated into buy- and sell-side arrivals using the exchange trade-stream maker indicator under the convention used throughout the project.

The classification rule is fixed across captures.

---

## 3. Temporal alignment

The principal Hawkes estimator operates on a **100-ms event grid**.

The conceptual and computational representations are kept separate:

```text
Conceptual model
    = continuous-time Hawkes process

Actual estimator
    = 100-ms binned count likelihood
```

---

# Bivariate Hawkes Model

The primary model uses same-side exponential self-excitation:

```text
Buy events  → future buy-event intensity

Sell events → future sell-event intensity
```

There is no buy-to-sell or sell-to-buy cross-excitation in the primary specification.

Conceptually:

```text
lambda_buy
    = buy baseline
    + historical buy excitation

lambda_sell
    = sell baseline
    + historical sell excitation
```

The branching ratios are:

```text
buy branching ratio
    = alpha_buy / beta

sell branching ratio
    = alpha_sell / beta
```

For the diagonal excitation matrix:

```text
spectral radius
    = max(buy branching ratio,
          sell branching ratio)
```

All three final fitted captures have spectral radius below one.

---

# Hawkes Pressure

The central research feature is:

```text
Hawkes pressure
    = lambda_buy - lambda_sell
```

Interpretation:

```text
Hawkes pressure > 0
    stronger conditional buy activity

Hawkes pressure < 0
    stronger conditional sell activity
```

This is distinct from OFI:

```text
OFI
    = order-book / displayed-flow imbalance

Hawkes pressure
    = conditional trade-arrival intensity imbalance
```

---

# Poisson Benchmark

For a Poisson event-count process:

```text
variance = mean
```

therefore:

```text
Fano factor = variance / mean = 1
```

Observed pooled one-second trade-count statistics:

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

This provides strong motivation for a history-dependent point-process model.

It does not, by itself, establish Hawkes as the unique or optimal model.

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

---

# Primary Validation

The main generalization experiment is **leave-one-capture-out validation**.

```text
Train: capture_02 + capture_03
Test:  capture_04

Train: capture_02 + capture_04
Test:  capture_03

Train: capture_03 + capture_04
Test:  capture_02
```

The held-out capture is excluded from model fitting.

Primary horizons:

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

Additional model:

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

### Main finding

Hawkes pressure has positive mean out-of-sample explanatory power at both horizons and exceeds the simple L1 OFI benchmark in the current three-capture experiment.

The improvement is modest but positive across all three held-out captures.

---

# Capture-Level Generalization

| Training captures | Held-out capture | 1 s | 5 s |
|---|---|---:|---:|
| `02 + 03` | `04` | 1.270% | 2.310% |
| `02 + 04` | `03` | 1.790% | 2.492% |
| `03 + 04` | `02` | 0.304% | 0.063% |

The signal is positive in every held-out capture, but its magnitude varies substantially.

That heterogeneity is an important part of the result.

---

# Hawkes Pressure and Future Returns

Pressure is standardized within capture and observations are ranked into pressure deciles.

The pooled five-second response includes:

```text
Lowest pressure decile
    ≈ -0.31 bps

Highest pressure decile
    ≈ +0.37 bps
```

Low-to-high difference:

```text
≈ 0.68 bps
```

The effect is economically small and should be interpreted as short-horizon predictive information rather than guaranteed trading profitability.

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

These results indicate horizon-dependent sensitivity rather than a universal optimal grid.

---

# Statistical Comparison

The paired comparison evaluates Hawkes pressure and L1 OFI on the same held-out observations.

Average Hawkes improvement in OOS R²:

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

Because the primary study contains only three independent captures, bootstrap probabilities are treated as supporting uncertainty analysis rather than definitive population-level significance tests.

---

# Python / MATLAB Replication

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

The MATLAB implementation reproduces the parameters to approximately `1e-7` or better.

The likelihood agrees to numerical precision.

This is an **implementation-level reproducibility check**, not an independent replication of the entire empirical study.

---

# Results and Figures

## Main figure set

| Figure | Content |
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

Files are stored under:

```text
figures/
```

with both `.png` and `.pdf` versions.

---

# Final Results Package

```text
results/
├── table_01_hawkes_parameters.csv
├── table_02_leave_one_capture_out.csv
├── table_03_statistical_comparison.csv
├── table_04_resolution_sensitivity.csv
├── table_01_hawkes_parameters.tex
├── table_02_leave_one_capture_out.tex
├── table_03_statistical_comparison.tex
├── table_04_resolution_sensitivity.tex
└── final_results_summary.txt
```

Processed datasets:

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
├── docs/
│   └── hawkes_ofi_research_paper.pdf
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

## Python environment

Run the project through the repository virtual environment:

```powershell
.\.venv\Scripts\python.exe
```

### Build trade events

```powershell
.\.venv\Scripts\python.exe experiments\build_multi_capture_trades.py
```

### Process captures

```powershell
.\.venv\Scripts\python.exe experiments\process_all_captures.py
```

### Leave-one-capture-out evaluation

```powershell
.\.venv\Scripts\python.exe experiments\leave_one_capture_out.py
```

### Final statistical comparison

```powershell
.\.venv\Scripts\python.exe experiments\final_statistical_test.py
```

### Build final tables

```powershell
.\.venv\Scripts\python.exe experiments\build_final_results_tables.py
```

### Model-comparison figures

```powershell
.\.venv\Scripts\python.exe experiments\figure_oos_model_comparison.py
```

### Resolution figures

```powershell
.\.venv\Scripts\python.exe experiments\figure_resolution_robustness.py
```

### Regenerate final figures

```powershell
.\.venv\Scripts\python.exe experiments\finalize_figures.py
```

## MATLAB replication

```powershell
matlab -batch "cd('C:\Users\annoy\hawkes-ofi-market-microstructure'); addpath('matlab'); results=fit_hawkes_bivariate();"
```

---

# Scientific Interpretation

The strongest defensible conclusion is:

> **Temporal clustering in BTCUSDT trade arrivals contains short-horizon predictive information that is not fully captured by a simple contemporaneous L1 OFI measure in the observed independent capture episodes.**

The evidence comes from:

```text
strong trade-arrival overdispersion
        +
stationary Hawkes estimates
        +
positive cross-capture OOS performance
        +
higher average prediction correlation
        +
independent Python/MATLAB agreement
```

The current evidence is interesting but deliberately limited.

---

# Limitations

## Independent sample size

There are 39,442 trade events but only three independent ten-minute market episodes.

The raw event count should not be interpreted as 39,442 independent observations.

## Short observation windows

The current dataset represents approximately thirty minutes of captured market time.

It is not sufficient to establish long-run stability across all market regimes.

## Restricted Hawkes specification

The primary model uses:

- same-side excitation;
- one common decay parameter;
- no buy-to-sell excitation;
- no sell-to-buy excitation.

## No execution model

The study predicts mid-price returns.

It does not yet model:

- fees;
- spread crossing;
- slippage;
- latency;
- queue position;
- market impact;
- inventory constraints.

## Limited benchmark scope

The project focuses on the interpretable Hawkes-pressure vs L1 OFI comparison rather than an exhaustive benchmark against every modern high-frequency forecasting model.

---

# What This Project Does Not Claim

This repository does **not** claim:

- guaranteed alpha;
- a universally profitable trading strategy;
- causal prediction;
- universal superiority of Hawkes models;
- optimality of the same-side Hawkes specification;
- profitability after trading costs;
- broad population-level significance;
- stability across every BTCUSDT market regime.

---

# Why the Result Is Interesting

The project connects two distinct market-microstructure perspectives.

### Order-book perspective

```text
What is the current imbalance
in displayed liquidity?
```

### Event-time perspective

```text
How does the history of arrivals
change the conditional intensity
of future buy/sell activity?
```

The Hawkes-pressure construction turns this event-time information into a directly interpretable signal:

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

---

# Research Position

This work does not claim to be the first application of Hawkes processes to finance or cryptocurrency.

Related research already covers:

- Hawkes processes in financial microstructure;
- order-flow imbalance and price impact;
- Hawkes models for order-flow;
- cryptocurrency limit-order-book dynamics;
- Hawkes-based OFI forecasting.

The contribution here is narrower:

1. parsimonious same-side Hawkes pressure;
2. direct comparison with L1 OFI;
3. strict capture-level out-of-sample validation;
4. temporal-resolution sensitivity;
5. independent Python/MATLAB estimator replication.

---

# Future Work

The next major experiment is to increase the number of **independent market episodes**.

### Data

- many more BTCUSDT captures;
- multiple sessions;
- multiple days;
- different volatility regimes;
- different liquidity regimes.

### Modeling

- cross-exciting Hawkes processes;
- cancellations;
- order submissions;
- richer event-type systems;
- deeper order-book features.

### Statistics

- inference across many independent episodes;
- regime-conditional analysis;
- longer-horizon validation;
- broader model comparison.

### Economics

- transaction-cost modeling;
- spread/slippage;
- latency;
- market impact;
- execution simulation;
- inventory-aware evaluation.

The next question is not simply:

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

The objective is not to maximize a backtest number.

The objective is to determine whether a market-microstructure relationship survives:

- data validation;
- strict out-of-sample testing;
- capture-level heterogeneity;
- robustness analysis;
- independent implementation.

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
