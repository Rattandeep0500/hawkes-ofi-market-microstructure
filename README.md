
<div align="center">

<svg width="100%" viewBox="0 0 1200 420" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Animated Hawkes OFI market microstructure research pipeline">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#07111f"/>
      <stop offset="55%" stop-color="#0b1626"/>
      <stop offset="100%" stop-color="#101827"/>
    </linearGradient>
    <linearGradient id="line" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#67e8f9"/>
      <stop offset="50%" stop-color="#a5b4fc"/>
      <stop offset="100%" stop-color="#5eead4"/>
    </linearGradient>
  </defs>

  <rect width="1200" height="420" rx="24" fill="url(#bg)"/>
  <rect x="1" y="1" width="1198" height="418" rx="23" fill="none" stroke="#26364d"/>

  <text x="60" y="64" fill="#f8fafc" font-size="31" font-family="Arial, Helvetica, sans-serif" font-weight="700">
    BTCUSDT MARKET MICROSTRUCTURE
  </text>
  <text x="60" y="94" fill="#8fa6c1" font-size="16" font-family="Arial, Helvetica, sans-serif">
    Hawkes–OFI • trade clustering → conditional intensity → pressure → short-horizon return
  </text>

  <g font-family="Arial, Helvetica, sans-serif">
    <rect x="60" y="126" width="210" height="62" rx="12" fill="#111f31" stroke="#2b425e"/>
    <text x="165" y="151" text-anchor="middle" fill="#8fa6c1" font-size="12">EVENT FLOW</text>
    <text x="165" y="174" text-anchor="middle" fill="#f8fafc" font-size="17" font-weight="700">BUY • SELL</text>

    <line x1="278" y1="157" x2="340" y2="157" stroke="#4d6685" stroke-width="2"/>
    <circle cx="290" cy="157" r="4" fill="#67e8f9">
      <animate attributeName="cx" values="290;330;290" dur="2.4s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0.2;1;0.2" dur="2.4s" repeatCount="indefinite"/>
    </circle>

    <rect x="350" y="126" width="220" height="62" rx="12" fill="#111f31" stroke="#2b425e"/>
    <text x="460" y="151" text-anchor="middle" fill="#8fa6c1" font-size="12">CONDITIONAL INTENSITY</text>
    <text x="460" y="174" text-anchor="middle" fill="#f8fafc" font-size="17" font-weight="700">λ_buy / λ_sell</text>

    <line x1="578" y1="157" x2="640" y2="157" stroke="#4d6685" stroke-width="2"/>
    <circle cx="590" cy="157" r="4" fill="#a5b4fc">
      <animate attributeName="cx" values="590;630;590" dur="2.4s" begin="0.35s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0.2;1;0.2" dur="2.4s" begin="0.35s" repeatCount="indefinite"/>
    </circle>

    <rect x="650" y="126" width="220" height="62" rx="12" fill="#111f31" stroke="#2b425e"/>
    <text x="760" y="151" text-anchor="middle" fill="#8fa6c1" font-size="12">HAWKES PRESSURE</text>
    <text x="760" y="174" text-anchor="middle" fill="#f8fafc" font-size="17" font-weight="700">λ_buy − λ_sell</text>

    <line x1="878" y1="157" x2="940" y2="157" stroke="#4d6685" stroke-width="2"/>
    <circle cx="890" cy="157" r="4" fill="#5eead4">
      <animate attributeName="cx" values="890;930;890" dur="2.4s" begin="0.7s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0.2;1;0.2" dur="2.4s" begin="0.7s" repeatCount="indefinite"/>
    </circle>

    <rect x="950" y="126" width="190" height="62" rx="12" fill="#111f31" stroke="#2b425e"/>
    <text x="1045" y="151" text-anchor="middle" fill="#8fa6c1" font-size="12">FORECAST</text>
    <text x="1045" y="174" text-anchor="middle" fill="#f8fafc" font-size="17" font-weight="700">1s / 5s return</text>
  </g>

  <rect x="60" y="218" width="1080" height="128" rx="14" fill="#091522" stroke="#24364c"/>
  <text x="78" y="241" fill="#7f96af" font-size="11" font-family="Arial, Helvetica, sans-serif">
    TRADE ARRIVALS / MARKET-PRICE TRAJECTORY
  </text>

  <path d="M90 316 L140 307 L190 311 L240 286 L290 294 L340 263 L390 273 L440 255 L490 279 L540 262 L590 236 L640 249 L690 241 L740 262 L790 250 L840 225 L890 239 L940 230 L990 244 L1040 214 L1090 222"
        fill="none" stroke="url(#line)" stroke-width="3" stroke-dasharray="1500" stroke-dashoffset="1500">
    <animate attributeName="stroke-dashoffset" values="1500;0;0" keyTimes="0;0.72;1" dur="5.4s" repeatCount="indefinite"/>
  </path>

  <g fill="#5eead4">
    <circle cx="240" cy="286" r="4">
      <animate attributeName="r" values="3;8;3" dur="1.8s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0.35;1;0.35" dur="1.8s" repeatCount="indefinite"/>
    </circle>
    <circle cx="590" cy="236" r="4">
      <animate attributeName="r" values="3;9;3" dur="2.1s" begin="0.4s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0.35;1;0.35" dur="2.1s" begin="0.4s" repeatCount="indefinite"/>
    </circle>
    <circle cx="1040" cy="214" r="4">
      <animate attributeName="r" values="3;8;3" dur="1.9s" begin="0.8s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0.35;1;0.35" dur="1.9s" begin="0.8s" repeatCount="indefinite"/>
    </circle>
  </g>

  <g fill="#fb7185">
    <circle cx="390" cy="273" r="4">
      <animate attributeName="r" values="3;8;3" dur="2s" begin="0.5s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0.35;1;0.35" dur="2s" begin="0.5s" repeatCount="indefinite"/>
    </circle>
    <circle cx="740" cy="262" r="4">
      <animate attributeName="r" values="3;9;3" dur="2.2s" begin="0.2s" repeatCount="indefinite"/>
      <animate attributeName="opacity" values="0.35;1;0.35" dur="2.2s" begin="0.2s" repeatCount="indefinite"/>
    </circle>
  </g>

  <text x="78" y="382" fill="#5eead4" font-size="12" font-family="Arial, Helvetica, sans-serif">● buy-arrival pulse</text>
  <text x="225" y="382" fill="#fb7185" font-size="12" font-family="Arial, Helvetica, sans-serif">● sell-arrival pulse</text>
  <text x="914" y="382" fill="#8fa6c1" font-size="12" font-family="Arial, Helvetica, sans-serif">39,442 trades • 3 captures</text>
</svg>

# Hawkes–OFI Market Microstructure

**A reproducible quantitative-finance study of whether temporal clustering in BTCUSDT buy/sell trade arrivals contains short-horizon predictive information beyond conventional L1 Order-Flow Imbalance.**

**Central signal:** Hawkes pressure = conditional buy-arrival intensity − conditional sell-arrival intensity.

<sub>Research pipeline • capture-level out-of-sample validation • no black-box ML • no production-trading claim</sub>

</div>

---

## Research Paper

<div align="center">

<a href="docs/hawkes_ofi_research_paper.pdf"><kbd>📄 <strong>Read the full research paper</strong></kbd></a>

<br/><br/>

<strong>Hawkes Pressure versus Order-Flow Imbalance: A Capture-Level Comparison for Short-Horizon BTCUSDT Return Prediction</strong>

<sub>The link works when <code>docs/hawkes_ofi_research_paper.pdf</code> is committed at that exact repository path.</sub>

</div>

---

## Primary Out-of-Sample Result

<div align="center">

<table>
<tr>
<td align="center"><strong>1s Hawkes OOS R²</strong><br/><br/><strong>1.12%</strong></td>
<td align="center"><strong>5s Hawkes OOS R²</strong><br/><br/><strong>1.62%</strong></td>
<td align="center"><strong>Pooled Fano factor</strong><br/><br/><strong>184.82</strong></td>
</tr>
<tr>
<td align="center"><strong>1s L1 OFI OOS R²</strong><br/><br/>−0.19%</td>
<td align="center"><strong>5s L1 OFI OOS R²</strong><br/><br/>0.01%</td>
<td align="center"><strong>Independent captures</strong><br/><br/>3</td>
</tr>
</table>

</div>

> **Hawkes pressure outperforms L1 OFI on average under three-capture leave-one-capture-out validation, while the effect remains modest and heterogeneous.**

The pressure signal is positive in **every held-out capture at both the 1-second and 5-second horizons**, but its magnitude varies materially across episodes.

---

## Research Question

**Does temporal clustering in aggressive BTCUSDT trade arrivals contain predictive information about short-horizon returns beyond contemporaneous L1 OFI?**

The comparison deliberately separates two different microstructure objects:

| Signal | Interpretation |
|---|---|
| **L1 OFI** | Current order-book / displayed-flow imbalance at the best bid and ask |
| **Hawkes pressure** | Model-implied imbalance in conditional buy/sell trade-arrival intensity generated by recent event history |

Hawkes pressure is **not** renamed OFI. The two signals use different information sets and answer different questions.

---

## Research Status

| Component | Status |
|---|---|
| Binance trade/depth live capture | Complete |
| Snapshot and sequence validation | Complete |
| Order-book reconstruction | Complete |
| 100-ms Hawkes estimator | Complete |
| L1 OFI benchmark | Complete |
| Three-capture leave-one-capture-out validation | Complete |
| Pressure-response analysis | Complete |
| Temporal-resolution sensitivity | Complete |
| Block-bootstrap comparison | Complete |
| Python/MATLAB estimator check | Complete |
| Production execution system | **Not part of this project** |

This is a **completed empirical research pipeline**. It is designed for transparent market-microstructure analysis and reproducibility rather than automated trading deployment.

---

## Focused Contribution

The project does **not** claim to be the first Hawkes model in finance, the first Hawkes model for Bitcoin, or the first Hawkes application to market microstructure.

Its focused contribution is the combination of:

1. a **parsimonious same-side Hawkes-pressure statistic**;
2. a direct comparison against **L1 OFI**;
3. strict **capture-level out-of-sample evaluation**;
4. **temporal-resolution sensitivity analysis**; and
5. **Python/MATLAB implementation-level reproducibility** on the same estimator.

---

## Research Pipeline

```text
Binance trade stream ────────┐
                             │
Binance depth stream ────────┼─> capture validation
                             │        │
Order-book snapshot ─────────┘        v
                                book reconstruction
                                       │
                         ┌─────────────┴─────────────┐
                         v                           v
                  trade classification        L1 book states
                         │                           │
                         v                           v
                  100-ms event grid              L1 OFI
                         │
                         v
                Hawkes estimation
                         │
              ┌──────────┴──────────┐
              v                     v
           λ_buy                  λ_sell
              └──────────┬──────────┘
                         v
              Hawkes pressure
                  λ_buy − λ_sell
                         │
                         v
               1s / 5s forecasts
                         │
                         v
             leave-one-capture-out
                         │
          ┌──────────────┼──────────────┐
          v              v              v
        L1 OFI       Hawkes pressure   OFI + Hawkes
                         │
                         v
      OOS R² • correlation • robustness
```

---

# Data

## Three Independent BTCUSDT Captures

| Capture | Duration | Trades | Buys | Sells | Reconstructed book states |
|---|---:|---:|---:|---:|---:|
| `capture_02` | ≈ 599 s | 13,117 | 8,182 | 4,935 | 6,000 |
| `capture_03` | ≈ 600 s | 18,595 | 8,925 | 9,670 | 6,002 |
| `capture_04` | ≈ 600 s | 7,730 | 4,845 | 2,885 | 6,000 |
| **Combined** | **≈ 1,800 s** | **39,442** | **21,952** | **17,490** | **18,002** |

### The important sample-size distinction

The project contains **39,442 trade events**, but it does **not** contain 39,442 independent observations for population-level inference.

The effective independent experimental unit is the **market episode**:

> **3 independently captured BTCUSDT episodes**

Trade events, book states, and returns within each capture are strongly serially dependent.

---

## Live-Capture Validation

The capture pipeline records:

- Binance trade stream;
- Binance depth stream;
- initial order-book snapshot;
- exchange update identifiers; and
- local reception timestamps.

Validation checks include:

- snapshot integrity;
- snapshot depth consistency;
- depth-sequence continuity;
- update-ID consistency;
- trade-record validation; and
- reception-timestamp validation.

### `capture_04` validation

| Validation item | Result |
|---|---:|
| Depth events | 6,000 |
| Trade events | 7,730 |
| Snapshot validation | Passed |
| Depth sequence validation | Passed |
| Trade validation | Passed |
| Reception timestamp validation | Passed |

All final captures used by the study satisfy the capture-validation pipeline before entering the empirical analysis.

---

# Methodology

## 1. Poisson Benchmark

A homogeneous Poisson count process satisfies:

```text
variance = mean

Fano factor = variance / mean = 1
```

Observed pooled one-second BTCUSDT trade counts:

| Quantity | Value |
|---|---:|
| Mean | 21.912222 |
| Variance | 4049.847766 |
| Pooled Fano factor | **184.821408** |

Capture-specific Fano factors:

| Capture | Fano factor |
|---|---:|
| `capture_02` | 227.871248 |
| `capture_03` | 176.382325 |
| `capture_04` | 120.372916 |
| **Pooled** | **184.821408** |

The extreme overdispersion provides strong motivation for a history-dependent arrival model.

**It does not prove that a Hawkes process is the unique or true data-generating model.**

---

## 2. Bivariate Same-Side Exponential Hawkes Process

The conceptual model is continuous-time:

```text
lambda_buy(t)
    = mu_buy
    + alpha_buy * SUM over previous buy events:
      exp(-beta * time_since_buy_event)

lambda_sell(t)
    = mu_sell
    + alpha_sell * SUM over previous sell events:
      exp(-beta * time_since_sell_event)
```

Primary specification:

```text
buy event  -> future buy intensity
sell event -> future sell intensity

buy event  -X-> sell intensity
sell event -X-> buy intensity
```

There is **no buy→sell or sell→buy cross-excitation** in the primary specification. This is a parsimony and estimation-stability choice, not a claim that cross-side excitation is absent from real markets.

### Branching ratios and stationarity

```text
branching_buy  = alpha_buy / beta
branching_sell = alpha_sell / beta

spectral_radius = max(branching_buy, branching_sell)

stationary if spectral_radius < 1
```

All final capture-level fits satisfy the stationarity restriction.

---

## 3. Continuous-Time Concept vs. Actual Estimator

This distinction is central to the project.

| Layer | Specification |
|---|---|
| **Conceptual model** | Continuous-time bivariate exponential Hawkes process |
| **Actual estimator** | 100-ms binned count likelihood |

The primary computational grid uses:

```text
dt = 0.1 seconds
```

For each side, exponential excitation is propagated recursively:

```text
R_buy[k]
    = exp(-beta * dt) * R_buy[k - 1]
    + buy_count[k - 1]

lambda_buy[k]
    = mu_buy + alpha_buy * R_buy[k]

buy_count[k]
    ~ Poisson(lambda_buy[k] * dt)
```

The sell side is constructed analogously.

Every final parameter result should therefore be interpreted as a result from the **100-ms binned Hawkes estimator**, not from an irregular-event continuous-time maximum-likelihood implementation.

---

## 4. Hawkes Pressure

The primary research signal is:

```text
Hawkes pressure = lambda_buy - lambda_sell
```

Interpretation:

```text
positive pressure -> stronger conditional buy-arrival activity
negative pressure -> stronger conditional sell-arrival activity
```

Hawkes pressure summarizes **recent temporal clustering in executed trade arrivals**.

It remains conceptually distinct from OFI, which is constructed from order-book state changes.

---

## 5. Forecast Target and Evaluation

Forecast target:

```text
future_return(t, h)
    = log(mid_price at t+h)
    - log(mid_price at t)

h = 1 second or 5 seconds
```

Three forecasting specifications are compared:

```text
1. future return ~ L1 OFI
2. future return ~ Hawkes pressure
3. future return ~ L1 OFI + Hawkes pressure
```

The primary metrics are:

- out-of-sample R² relative to the training-sample-mean forecast benchmark; and
- Pearson correlation between prediction and realized future return.

OOS R² is an explanatory/predictive statistic. It is **not a profitability metric**.

---

# Hawkes Parameter Estimates

| Capture | `mu_buy` | `mu_sell` | `beta` | Buy branching | Sell branching | Spectral radius | Negative log-likelihood |
|---|---:|---:|---:|---:|---:|---:|---:|
| `capture_02` | 7.53824703 | 7.07443977 | 5.17491277 | 0.44702649 | 0.13964165 | 0.44702649 | 50,709.88897476 |
| `capture_03` | 6.13813848 | 10.57490675 | 3.05731929 | 0.58724142 | 0.34363529 | 0.58724142 | 63,004.05172302 |
| `capture_04` | 3.37429552 | 2.82013128 | 1.03167992 | 0.58208693 | 0.41652259 | 0.58208693 | 28,784.47829332 |

All three estimates satisfy:

```text
spectral_radius < 1
```

The parameters are visibly heterogeneous across captures, which is consistent with the episode-level heterogeneity later observed in predictive performance.

---

# Primary Validation

## Leave-One-Capture-Out Design

The final validation holds out an **entire independently captured market episode**.

| Fold | Training captures | Held-out test capture |
|---|---|---|
| 1 | `capture_02` + `capture_03` | `capture_04` |
| 2 | `capture_02` + `capture_04` | `capture_03` |
| 3 | `capture_03` + `capture_04` | `capture_02` |

The held-out capture is excluded from both:

- Hawkes parameter estimation; and
- forecast-model fitting.

This design avoids the major leakage risk created by randomly splitting adjacent high-frequency observations that are only milliseconds apart.

---

# Results

## Mean Leave-One-Capture-Out OOS R²

| Model | 1 second | 5 seconds |
|---|---:|---:|
| **L1 OFI** | **−0.189918%** | **0.006638%** |
| **Hawkes pressure** | **1.121453%** | **1.621601%** |
| **OFI + Hawkes** | **1.026555%** | **1.540160%** |

Hawkes pressure has the highest mean OOS R² at both horizons.

The joint OFI + Hawkes model does **not** materially improve on Hawkes pressure alone in these data.

---

## Mean Prediction / Return Correlation

| Model | 1 second | 5 seconds |
|---|---:|---:|
| L1 OFI | 0.060261 | 0.060024 |
| **Hawkes pressure** | **0.121554** | **0.161076** |
| OFI + Hawkes | 0.121201 | 0.159012 |

The correlation ranking is consistent with the OOS R² comparison.

---

## Capture-Level Hawkes OOS R²

| Train | Test | 1 second | 5 seconds |
|---|---|---:|---:|
| `02 + 03` | `04` | **1.2702%** | **2.3101%** |
| `02 + 04` | `03` | **1.7902%** | **2.4917%** |
| `03 + 04` | `02` | **0.3040%** | **0.0630%** |

The sign is stable:

> **Hawkes pressure is positive in every held-out capture at both horizons.**

The magnitude is not stable. The weakest 5-second result is only **0.0630%**, while the strongest is **2.4917%**.

That heterogeneity is a first-order limitation, not a footnote.

---

## Pressure–Return Response

Hawkes pressure is standardized within each capture and ranked into deciles.

Reported pooled endpoint behavior at the 5-second horizon:

| Pressure region | Mean future 5s return |
|---|---:|
| Lowest pressure decile | ≈ **−0.31 bps** |
| Highest pressure decile | ≈ **+0.37 bps** |
| Low-to-high difference | ≈ **0.68 bps** |

The economically relevant low-to-high spread is **0.68 bps**.

The direction is consistent with the signal definition, but the magnitude is small and should not be read as evidence of a profitable strategy.

---

# Temporal-Resolution Sensitivity

The Hawkes pipeline was re-run at:

```text
50 ms
100 ms
250 ms
500 ms
```

This analysis tests sensitivity to discretization. It is **not** presented as proof of a universally optimal grid.

## 1-Second Horizon

| Resolution | OOS R² | Correlation |
|---|---:|---:|
| 50 ms | **4.710763%** | **0.225598** |
| 100 ms | 3.379128% | 0.142695 |
| 250 ms | 2.450840% | 0.101040 |
| 500 ms | 1.790584% | 0.074175 |

At 1 second, the finer grids perform better in this sensitivity exercise.

## 5-Second Horizon

| Resolution | OOS R² | Correlation |
|---|---:|---:|
| 50 ms | 2.243830% | 0.124963 |
| 100 ms | 6.487510% | **0.159877** |
| 250 ms | **6.543242%** | 0.137616 |
| 500 ms | 6.254697% | 0.127720 |

At 5 seconds, the pattern differs: 100–500 ms are broadly similar in OOS R² while 50 ms performs materially worse.

The asymmetric resolution response across horizons is itself useful evidence that different forecasting horizons may interact differently with the discretization of event clustering.

---

# Statistical Comparison

Mean Hawkes-minus-L1-OFI OOS R² improvement:

| Horizon | Mean improvement |
|---|---:|
| 1 second | **+1.311 percentage points** |
| 5 seconds | **+1.615 percentage points** |

A paired block-bootstrap analysis used:

```text
2,000 repetitions
5-second blocks
```

For the final three-capture comparison, the reported bootstrap probabilities that Hawkes pressure has lower MSE than L1 OFI are:

| Horizon | Bootstrap P(MSE Hawkes < MSE OFI) |
|---|---:|
| 1 second | 0.856 |
| 5 seconds | 0.768 |

These are **bootstrap probabilities, not classical p-values**. With only three independent market episodes, they should not be presented as conventional population-level significance tests.

<details>
<summary><strong>Secondary historical two-capture bootstrap analysis</strong></summary>

<br/>

An earlier two-capture pilot used the same 2,000-repetition, 5-second-block bootstrap framework before the final three-capture LOCO design was established.

| Train → Test | Horizon | Point OOS R² | 95% interval | Bootstrap P(R² > 0) | Correlation |
|---|---:|---:|---:|---:|---:|
| `02 → 03` | 1s | 0.0724 | [0.0407, 0.0983] | 1.000 | 0.2875 |
| `02 → 03` | 5s | 0.1378 | [0.0460, 0.2024] | 0.996 | 0.4049 |
| `03 → 02` | 1s | 0.0435 | [−0.0229, 0.0752] | 0.912 | 0.2486 |
| `03 → 02` | 5s | 0.1358 | [0.0010, 0.2239] | 0.976 | 0.4294 |

This pilot is retained only as secondary robustness context. It must not be conflated with the final three-capture leave-one-capture-out design.

</details>

---

# State-Conditioned Analysis

A secondary analysis examined the Hawkes-pressure relationship across **queue-imbalance regimes**.

The relationship varied across captures and forecast horizons. Because there are only three independent episodes, the regime results are treated as **secondary evidence** rather than a headline conclusion.

No broad claim is made that a specific queue state universally strengthens or weakens Hawkes pressure.

---

# Python / MATLAB Reproducibility

The Hawkes estimator was implemented independently in **Python** and **MATLAB** using:

- the same captures;
- the same reconstructed timeline;
- the exact same 100-ms grid;
- the same same-side Hawkes specification;
- the same binned Poisson-count likelihood; and
- the same stationarity restriction.

### Python `capture_04` verification

| Parameter | Verification value |
|---|---:|
| `mu_buy` | 3.3742947465 |
| `mu_sell` | 2.8201315938 |
| `beta` | 1.0316796766 |
| `branching_buy` | 0.5820871370 |
| `branching_sell` | 0.4165226456 |
| Negative log-likelihood | 28784.4782933165 |

The MATLAB implementation reproduces the parameters to approximately **1e-7 or better** at the parameter level and agrees at numerical precision in the likelihood.

> This is an **implementation-level reproducibility check** of the estimator. It is not an independent replication of the entire empirical study.

---

# Figures

The project generates a full research figure set. The README intentionally does **not** embed these paths unless the corresponding PNG files are committed, preventing broken GitHub image links.

<details>
<summary><strong>Generated figure set</strong></summary>

```text
figures/figure_01_event_clustering.png
figures/figure_01b_fano_by_capture.png
figures/figure_02_hawkes_pressure_response.png
figures/figure_02b_hawkes_pressure_by_capture.png
figures/figure_03_oos_r2_comparison.png
figures/figure_03b_oos_correlation.png
figures/figure_03c_hawkes_by_capture.png
figures/figure_04_resolution_r2.png
figures/figure_04b_resolution_correlation.png
figures/figure_04c_resolution_rmse.png
```

Expected content includes:

- trade-arrival clustering versus the Poisson benchmark;
- capture-level Fano factors;
- Hawkes-pressure / future-return response;
- out-of-sample R² comparison;
- prediction/return correlation;
- capture-level Hawkes heterogeneity; and
- temporal-resolution robustness.

</details>

---

# Repository

```text
hawkes-ofi-market-microstructure/
│
├── data/
│   ├── live/
│   └── processed/
│
├── src/
│
├── experiments/
│   ├── build_multi_capture_trades.py
│   ├── process_all_captures.py
│   ├── leave_one_capture_out.py
│   ├── final_statistical_test.py
│   ├── build_final_results_tables.py
│   ├── figure_oos_model_comparison.py
│   ├── figure_resolution_robustness.py
│   └── finalize_figures.py
│
├── matlab/
│   └── fit_hawkes_bivariate.m
│
├── figures/
│
├── results/
│
├── docs/
│   └── hawkes_ofi_research_paper.pdf
│
├── requirements.txt
└── README.md
```

The paper link in this README assumes that:

```text
docs/hawkes_ofi_research_paper.pdf
```

is actually committed to the repository.

---

# Reproducibility

The project uses its local Windows virtual environment:

```text
.\.venv\Scripts\python.exe
```

## Core experiment commands

```powershell
.\.venv\Scripts\python.exe experiments/build_multi_capture_trades.py
```

```powershell
.\.venv\Scripts\python.exe experiments/process_all_captures.py
```

```powershell
.\.venv\Scripts\python.exe experiments/leave_one_capture_out.py
```

```powershell
.\.venv\Scripts\python.exe experiments/final_statistical_test.py
```

```powershell
.\.venv\Scripts\python.exe experiments/build_final_results_tables.py
```

```powershell
.\.venv\Scripts\python.exe experiments/figure_oos_model_comparison.py
```

```powershell
.\.venv\Scripts\python.exe experiments/figure_resolution_robustness.py
```

```powershell
.\.venv\Scripts\python.exe experiments/finalize_figures.py
```

## MATLAB estimator check

```powershell
matlab -batch "cd('C:\Users\annoy\hawkes-ofi-market-microstructure'); addpath('matlab'); results=fit_hawkes_bivariate();"
```

The MATLAB command above preserves the original project path used for the replication run and is therefore machine-specific.

---

# Reproducibility Contract

A successful reproduction should preserve all of the following:

```text
same raw captures
same validated book reconstruction
same trade-direction convention
same research timeline
same primary 100-ms grid
same diagonal Hawkes specification
same binned count likelihood
same stationarity restriction
same capture-level train/test separation
same 1s and 5s return horizons
```

Headline values that should emerge from the final pipeline include:

```text
Pooled Fano factor:
184.821408

Mean Hawkes OOS R²:
1s = 1.121453%
5s = 1.621601%

Mean L1 OFI OOS R²:
1s = -0.189918%
5s = 0.006638%

Pressure decile spread:
approximately 0.68 bps
```

---

# Limitations

The limitations are part of the result, not boilerplate.

- **Only three independent episodes.** The study uses three independently captured approximately ten-minute BTCUSDT market episodes.
- **Serial dependence is substantial.** The 39,442 trade events are not 39,442 independent statistical observations.
- **Short observation window.** Approximately 1,800 seconds of total market time cannot represent the full BTCUSDT regime distribution.
- **Restricted Hawkes structure.** The primary specification allows same-side self-excitation only.
- **No cross-excitation.** Buy→sell and sell→buy excitation are excluded from the primary model.
- **No transaction-cost model.** Fees, spread crossing, and slippage are not modeled as a trading P&L system.
- **No execution simulation.** The project does not model actual order placement or fill mechanics.
- **No latency model.** Signal computation, exchange latency, networking, and reaction time are not modeled.
- **No queue-position model.** Position in the limit-order queue and fill probability are not modeled.
- **No market-impact model.** Hypothetical strategy impact on the market is absent.
- **No profitability claim.** Positive OOS R² is not evidence of a profitable executable strategy.
- **Limited regime analysis.** State-conditioned results vary across captures and horizons.
- **Limited population-level inference.** Three independent captures are insufficient for strong general claims about BTCUSDT as a whole.
- **Trade-sign convention is exchange-dependent.** Buy/sell labels depend on the exchange trade-stream maker indicator used in the data pipeline.

---

# Future Work

The next research extensions are deliberately aimed at testing whether the observed relationship survives richer models and stronger evaluation:

- collect **substantially more independent captures** across volatility, liquidity, and time-of-day regimes;
- introduce **buy→sell and sell→buy cross-excitation**;
- test **marked Hawkes processes** using trade size;
- investigate **state-dependent baseline intensities**;
- evaluate pressure under richer **queue / order-book state conditioning**;
- compare against additional microstructure baselines, including multi-level OFI variants;
- strengthen uncertainty quantification at the **capture level**;
- study stability across longer market windows and different crypto instruments; and
- only after statistical robustness is established, add transaction costs, latency, queue position, execution, and market impact.

The research order matters: **establish generalization first; evaluate tradability second.**

---

# References

1. **Hawkes, A. G. (1971).** “Spectra of some self-exciting and mutually exciting point processes.” *Biometrika*, 58(1), 83–90.

2. **Bacry, E., Delattre, S., Hoffmann, M., & Muzy, J.-F. (2013).** “Modelling microstructure noise with mutually exciting point processes.” *Quantitative Finance*, 13(1), 65–77.

3. **Bacry, E., & Muzy, J.-F. (2014).** “Hawkes model for price and trades high-frequency dynamics.” *Quantitative Finance*, 14(7), 1147–1166.

4. **Bacry, E., Mastromatteo, I., & Muzy, J.-F. (2015).** “Hawkes Processes in Finance.” *Market Microstructure and Liquidity*, 1(1), 1550005.

5. **Cont, R., Kukanov, A., & Stoikov, S. (2014).** “The Price Impact of Order Book Events.” *Journal of Financial Econometrics*, 12(1), 47–88.

6. **Wu, P., Rambaldi, M., Muzy, J.-F., & Bacry, E. (2019).** “Queue-reactive Hawkes models for the order flow.”

7. **Nittur Anantha, A., & Jain, S. (2026).** “Forecasting High Frequency Order Flow Imbalance using Hawkes Processes.” *Computational Economics*, 67(1), 279–312.

8. **Raffaelli, D., Cestari, R. G., Marazzina, D., & Formentin, S. (2026).** “Forecasting Bitcoin price movements using multivariate Hawkes processes and limit order book data.” *Decisions in Economics and Finance*.

<details>
<summary><strong>Additional methodological context</strong></summary>

<br/>

**Hawkes, A. G. (1971).** “Point spectra of some mutually exciting point processes.” *Journal of the Royal Statistical Society: Series B (Methodological)*, 33(3), 438–443.

**Huang, W., Lehalle, C.-A., & Rosenbaum, M. (2015).** “Simulating and analyzing order book data: The queue-reactive model.” *Journal of the American Statistical Association*, 110(509), 107–122.

</details>

---

# Interpretation

The empirical result is narrow but meaningful:

> In the three observed BTCUSDT market episodes, a parsimonious measure of **trade-arrival intensity imbalance** carries more short-horizon out-of-sample information than the L1 OFI benchmark on average.

That statement is intentionally narrower than:

- “Hawkes models beat the market”;
- “Hawkes pressure is guaranteed alpha”;
- “this is a profitable trading strategy”; or
- “three captures establish population-level superiority.”

The current evidence supports a **microstructure predictability result**, not a production-trading claim.

---

# Disclaimer

This repository is for **quantitative research and educational purposes**.

It does not provide investment advice, does not recommend a trading strategy, and does not establish that the reported statistical relationships survive transaction costs, latency, queue position, slippage, market impact, or unseen market regimes.

Past empirical relationships in three short BTCUSDT captures are not guarantees of future performance.

---

<div align="center">

**Hawkes–OFI Market Microstructure**

<sub>Temporal clustering • conditional intensity • order-flow imbalance • capture-level validation</sub>

</div>
