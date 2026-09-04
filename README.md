[README (3).md](https://github.com/user-attachments/files/31817169/README.3.md)
<div align="center">

![status](https://img.shields.io/badge/status-research--complete-2b2b2b?style=flat-square)
![type](https://img.shields.io/badge/type-non--production-2b2b2b?style=flat-square)
![python](https://img.shields.io/badge/python-3.10%2B-2b2b2b?style=flat-square)
![matlab](https://img.shields.io/badge/matlab-cross--validated-2b2b2b?style=flat-square)

# Hawkes–OFI Market Microstructure

**Does temporal clustering in trade arrivals carry short-horizon predictive information beyond conventional order-flow imbalance?**

```
┌────────────────────────────────────────────────────────────────────┐
│  BTCUSDT · MARKET MICROSTRUCTURE RESEARCH                          │
│                                                                    │
│  trade tape    ·╎│╎·│╎││·╎│·╎│··╎│╎·│╎·│      buy · sell           │
│                        │                                           │
│                        ▼                                           │
│  λ_buy (t)     ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░   conditional           │
│  λ_sell(t)     ▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░   intensity             │
│                        │                                           │
│                        ▼                                           │
│  Hawkes pressure   =   λ_buy(t) − λ_sell(t)                        │
│                        │                                           │
│                        ▼                                           │
│  forecast horizon      1s  ·  5s   →   predicted direction         │
└────────────────────────────────────────────────────────────────────┘
```

<sub>Conceptual signal flow — trade events excite conditional buy/sell intensity, the imbalance between them ("Hawkes pressure") is read off as a short-horizon directional signal.</sub>

</div>

---

Hawkes–OFI Market Microstructure is a reproducible market-microstructure research pipeline. It models BTCUSDT trade arrivals as a bivariate same-side self-exciting (Hawkes) point process and tests whether the resulting conditional buy/sell intensity imbalance — **Hawkes pressure** — improves short-horizon return prediction relative to L1 OFI. It is a research pipeline, not a production trading system and not a black-box ML project.

<div align="center">

<a href="docs/hawkes_ofi_research_paper.pdf"><strong>📄 Read the full research paper</strong></a>
<br>
<sub>docs/hawkes_ofi_research_paper.pdf — resolves once the compiled paper is committed to the repository</sub>

</div>

---

> **Research question.** Does temporal clustering in BTCUSDT buy/sell trade arrivals contain short-horizon predictive information beyond conventional level-1 Order-Flow Imbalance (OFI)?

<div align="center">

| 1s OOS R² | 5s OOS R² | Poisson Fano factor |
|:---:|:---:|:---:|
| **1.12%** | **1.62%** | **184.82** |

**Hawkes pressure outperforms L1 OFI on average under three-capture leave-one-capture-out validation, while the effect remains modest and heterogeneous.**

</div>

**Status:** research complete, reproducibly validated · **Not** a trading system · **Not** investment advice · Independent Python/MATLAB implementation check

## Contents

[Pipeline](#pipeline) · [Data](#data) · [Methodology](#methodology) · [Results](#results) · [Validation](#validation) · [Replication](#replication) · [Repository](#repository) · [Reproducibility](#reproducibility) · [Limitations](#limitations) · [Future Work](#future-work) · [References](#references) · [Disclaimer](#disclaimer)

---

## Pipeline

```
capture  →  reconstruction & validation  →  Hawkes estimation (Python + MATLAB)
         →  leave-one-capture-out OOS evaluation  →  temporal-resolution robustness
         →  block-bootstrap comparison  →  figures & tables
```

Each stage operates on the same three independently captured BTCUSDT episodes described below, so every downstream number in this README traces back to the same reconstructed timeline.

## Data

Three independently captured BTCUSDT episodes (Binance trade stream + depth stream + order-book snapshot), each roughly ten minutes long.

| Capture | Duration | Trades | Buys | Sells | Book states |
|---|---:|---:|---:|---:|---:|
| `capture_02` | ≈599s | 13,117 | 8,182 | 4,935 | 6,000 |
| `capture_03` | ≈600s | 18,595 | 8,925 | 9,670 | 6,002 |
| `capture_04` | ≈600s | 7,730 | 4,845 | 2,885 | 6,000 |
| **Combined** | **≈1,800s** | **39,442** | **21,952** | **17,490** | **18,002** |

<details>
<summary><strong>Live capture validation</strong></summary>

<br>

Each capture is checked for snapshot integrity, depth sequence continuity, update-ID consistency, trade-record validation, and reception-timestamp sanity before it enters the pipeline.

Example — `capture_04`:

| Check | Result |
|---|---|
| Depth events | 6,000 |
| Trade events | 7,730 |
| Snapshot validation | passed |
| Depth sequence validation | passed |
| Trade validation | passed |
| Reception timestamp validation | passed |

</details>

## Methodology

**Model.** A bivariate *same-side* exponential Hawkes process: buy events excite future buy events, sell events excite future sell events. There is no buy→sell or sell→buy cross-excitation in the primary specification.

```
λ_buy (t)  = μ_buy  + Σ  α_buy  · exp( −β · (t − tᵢ) )     over past buy events tᵢ
λ_sell(t)  = μ_sell + Σ  α_sell · exp( −β · (t − tⱼ) )     over past sell events tⱼ

branching_buy   = α_buy  / β
branching_sell  = α_sell / β
spectral radius = max(branching_buy, branching_sell)        (excitation matrix is diagonal)
```

All final captures satisfy `spectral radius < 1` (stationary).

**Hawkes pressure.**

```
Hawkes pressure(t) = λ_buy(t) − λ_sell(t)

positive  →  stronger conditional buy activity
negative  →  stronger conditional sell activity
```

`Hawkes pressure` is kept conceptually distinct from **OFI**: OFI reflects the current order-book / displayed-flow imbalance, while Hawkes pressure reflects the *conditional trade-arrival intensity* imbalance implied by the fitted point process.

**Temporal grid.** The conceptual model is a continuous-time Hawkes process; the actual estimator is a **100 ms binned count likelihood**. Estimation runs on this 100 ms grid throughout.

**Poisson benchmark.** For a Poisson count process, variance equals the mean (Fano factor = 1). Pooled one-second BTCUSDT trade counts are strongly overdispersed:

| | Mean | Variance | Fano factor |
|---|---:|---:|---:|
| Pooled | 21.912222 | 4049.847766 | **184.821408** |

| Capture | Fano factor |
|---|---:|
| `capture_02` | 227.871248 |
| `capture_03` | 176.382325 |
| `capture_04` | 120.372916 |

This overdispersion motivates a history-dependent point-process model — it does not, on its own, prove Hawkes is the unique or correct such model.

**Fitted parameters** (final, per capture):

| Capture | μ_buy | μ_sell | β | branching_buy | branching_sell | spectral radius | NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| `capture_02` | 7.538247 | 7.074440 | 5.174913 | 0.447026 | 0.139642 | 0.447026 | 50709.888975 |
| `capture_03` | 6.138138 | 10.574907 | 3.057319 | 0.587241 | 0.343635 | 0.587241 | 63004.051723 |
| `capture_04` | 3.374296 | 2.820131 | 1.031680 | 0.582087 | 0.416523 | 0.582087 | 28784.478293 |

<details>
<summary>Full-precision parameter values</summary>

```
capture_02:  mu_buy=7.53824703  mu_sell=7.07443977  beta=5.17491277
             branching_buy=0.44702649  branching_sell=0.13964165
             spectral_radius=0.44702649  nll=50709.88897476

capture_03:  mu_buy=6.13813848  mu_sell=10.57490675  beta=3.05731929
             branching_buy=0.58724142  branching_sell=0.34363529
             spectral_radius=0.58724142  nll=63004.05172302

capture_04:  mu_buy=3.37429552  mu_sell=2.82013128  beta=1.03167992
             branching_buy=0.58208693  branching_sell=0.41652259
             spectral_radius=0.58208693  nll=28784.47829332
```

</details>

## Results

**Primary comparison** — mean leave-one-capture-out out-of-sample R², L1 OFI vs. Hawkes pressure vs. the combination:

| Horizon | L1 OFI | Hawkes pressure | OFI + Hawkes |
|---|---:|---:|---:|
| 1s | −0.19% | **1.12%** | 1.03% |
| 5s | 0.01% | **1.62%** | 1.54% |

Mean prediction/return correlation:

| Horizon | L1 OFI | Hawkes pressure | OFI + Hawkes |
|---|---:|---:|---:|
| 1s | 0.0603 | **0.1216** | 0.1212 |
| 5s | 0.0600 | **0.1611** | 0.1590 |

<details>
<summary>Full-precision values</summary>

```
OOS R²           1s: L1 OFI=-0.189918%  Hawkes=1.121453%  OFI+Hawkes=1.026555%
                  5s: L1 OFI=0.006638%   Hawkes=1.621601%  OFI+Hawkes=1.540160%

correlation       1s: L1 OFI=0.060261   Hawkes=0.121554   OFI+Hawkes=0.121201
                  5s: L1 OFI=0.060024   Hawkes=0.161076   OFI+Hawkes=0.159012
```

</details>

**Capture-level breakdown.** Hawkes pressure is positive in every held-out capture at both horizons, though the magnitude is heterogeneous:

| Fold | Train | Test | 1s R² | 5s R² |
|---|---|---|---:|---:|
| 1 | `capture_02` + `capture_03` | `capture_04` | 1.27% | 2.31% |
| 2 | `capture_02` + `capture_04` | `capture_03` | 1.79% | 2.49% |
| 3 | `capture_03` + `capture_04` | `capture_02` | 0.30% | 0.06% |

**Pressure–response.** Pressure is standardized within capture and ranked into deciles:

| Decile | Mean future 5s return |
|---|---:|
| Lowest pressure | ≈ −0.31 bps |
| Highest pressure | ≈ +0.37 bps |
| Spread (low → high) | **≈ 0.68 bps** |

**Temporal-resolution sensitivity.** The Hawkes estimator was re-fit at four grid resolutions. Read this as a sensitivity analysis, not evidence of a single optimal resolution:

| Resolution | 1s R² | 1s corr. | 5s R² | 5s corr. |
|---|---:|---:|---:|---:|
| 50 ms | 4.71% | 0.2256 | 2.24% | 0.1250 |
| 100 ms | 3.38% | 0.1427 | 6.49% | 0.1599 |
| 250 ms | 2.45% | 0.1010 | 6.54% | 0.1376 |
| 500 ms | 1.79% | 0.0742 | 6.25% | 0.1277 |

### Figures

The pipeline generates a fixed set of exploratory and validation figures (event clustering, Fano factors by capture, pressure-response curves, OOS R² and correlation comparisons, and resolution-robustness plots). They are not embedded here to avoid broken links in environments where `figures/` hasn't been committed — generate them locally with the scripts in [Reproducibility](#reproducibility), or browse `figures/` directly in the repository once populated.

## Validation

**Leave-one-capture-out (LOCO)** is the primary validation scheme: each of the three captures is held out in turn, the Hawkes model is fit on the remaining two, and predictions are scored out-of-sample at 1s and 5s horizons (see the fold table above).

**Statistical comparison.** Mean improvement in OOS R² of Hawkes pressure over L1 OFI: **+1.311 percentage points** at 1s, **+1.615 percentage points** at 5s. A secondary block-bootstrap analysis (2,000 repetitions, 5-second blocks) was used to characterize the sampling variability of this comparison. The resulting bootstrap probabilities are **not** p-values and are not used as a classical significance test.

**State-conditioned analysis.** A secondary analysis examined the Hawkes-pressure/return relationship across queue-imbalance regimes. Results varied across captures and horizons; this is treated as secondary, exploratory evidence, not the paper's main result.

## Replication

The Hawkes estimator was implemented **independently** in Python and MATLAB, sharing the same captures, reconstructed timeline, 100 ms grid, Hawkes specification, binned likelihood, and stationarity restriction.

Python `capture_04` verification:

```
mu_buy=3.3742947465   mu_sell=2.8201315938   beta=1.0316796766
branching_buy=0.5820871370   branching_sell=0.4165226456
negative_loglik=28784.4782933165
```

The MATLAB implementation reproduces these parameters to approximately `1e-7` or better. This is best described as an **implementation-level reproducibility check** — it confirms the two codebases agree on the same optimization problem, not an independent replication of the full empirical study.

## Repository

```
.
├── data/                                  # raw + reconstructed capture data
├── src/                                   # core estimation & pipeline code
├── experiments/                           # scripts listed under Reproducibility
├── matlab/                                # independent MATLAB implementation
├── figures/                               # generated exploratory & validation figures
├── results/                               # tables and fitted-parameter outputs
├── docs/
│   └── hawkes_ofi_research_paper.pdf      # must be committed for the link above to work
├── requirements.txt
└── README.md
```

## Reproducibility

Using the project virtual environment (`.\.venv\Scripts\python.exe`):

```
.\.venv\Scripts\python.exe experiments\build_multi_capture_trades.py
.\.venv\Scripts\python.exe experiments\process_all_captures.py
.\.venv\Scripts\python.exe experiments\leave_one_capture_out.py
.\.venv\Scripts\python.exe experiments\final_statistical_test.py
.\.venv\Scripts\python.exe experiments\build_final_results_tables.py
.\.venv\Scripts\python.exe experiments\figure_oos_model_comparison.py
.\.venv\Scripts\python.exe experiments\figure_resolution_robustness.py
.\.venv\Scripts\python.exe experiments\finalize_figures.py
```

MATLAB cross-check:

```
matlab -batch "cd('C:\Users\annoy\hawkes-ofi-market-microstructure'); addpath('matlab'); results=fit_hawkes_bivariate();"
```

<sub>Adjust the `cd(...)` path to your local checkout.</sub>

## Limitations

- Only three independent ten-minute captures
- Many event observations within a capture are serially dependent
- The effective independent sample is three market episodes, not 39,442 individual observations
- Short observation window
- Restricted same-side Hawkes specification — no cross-excitation in the primary model
- No transaction-cost model
- No execution simulation
- No latency model
- No queue-position model
- No market-impact model
- No profitability claim
- Limited regime analysis
- Limited population-level inference

## Future Work

- Extend the specification to allow buy↔sell cross-excitation and compare against the same-side baseline
- Add independent captures — across venues, symbols, and market regimes — to move from three episodes toward population-level inference
- Layer in transaction costs, latency, queue position, and market impact before any execution-oriented claim is entertained
- Deepen the state-conditioned (queue-imbalance regime) analysis with more captures per regime

## References

1. Hawkes, A. G. (1971). Spectra of some self-exciting and mutually exciting point processes. *Biometrika*.
2. Bacry, E., Delattre, S., Hoffmann, M., & Muzy, J.-F. (2013). Modelling microstructure noise with mutually exciting point processes. *Quantitative Finance*.
3. Bacry, E., & Muzy, J.-F. (2014). Hawkes model for price and trades high-frequency dynamics. *Quantitative Finance*.
4. Bacry, E., Mastromatteo, I., & Muzy, J.-F. (2015). Hawkes Processes in Finance. *Market Microstructure and Liquidity*.
5. Cont, R., Kukanov, A., & Stoikov, S. (2014). The Price Impact of Order Book Events. *Journal of Financial Econometrics*.
6. Wu, P., Rambaldi, M., Muzy, J.-F., & Bacry, E. (2019). Queue-reactive Hawkes models for the order flow.
7. Anantha, A. N., & Jain, S. (2026). Forecasting High Frequency Order Flow Imbalance using Hawkes Processes. *Computational Economics*.
8. Raffaelli, D., Cestari, R. G., Marazzina, D., & Formentin, S. (2026). Forecasting Bitcoin price movements using multivariate Hawkes processes and limit order book data. *Decisions in Economics and Finance*.

## Disclaimer

This repository is a research artifact, not investment advice and not a trading system. Nothing here constitutes a recommendation to trade any asset. Out-of-sample results are modest, heterogeneous across captures, and drawn from a short observation window — see [Limitations](#limitations). Use at your own risk and do your own diligence before drawing any decision-relevant conclusion from this work.
