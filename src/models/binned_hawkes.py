from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import poisson


INPUT_FILE = Path(
    "data/processed/trade_events.parquet"
)

OUTPUT_FILE = Path(
    "data/processed/binned_hawkes.csv"
)

BIN_SIZE_SECONDS = 0.1
STATIONARITY_LIMIT = 0.999
N_STARTS = 12


def load_events():
    data = pd.read_parquet(
        INPUT_FILE,
        columns=[
            "event_time_s",
        ],
        engine="pyarrow",
    )

    times = (
        data["event_time_s"]
        .to_numpy(dtype=float)
    )

    if len(times) == 0:
        raise RuntimeError(
            "No events found."
        )

    times = times - times[0]

    return times


def build_counts(times):
    duration = times[-1]

    n_bins = int(
        np.ceil(
            duration
            / BIN_SIZE_SECONDS
        )
    )

    edges = (
        np.arange(n_bins + 1)
        * BIN_SIZE_SECONDS
    )

    counts, _ = np.histogram(
        times,
        bins=edges,
    )

    return counts.astype(float)


def unpack_parameters(x):
    mu = np.exp(
        np.clip(x[0], -30.0, 30.0)
    )

    beta = np.exp(
        np.clip(x[1], -20.0, 20.0)
    )

    branching_ratio = (
        STATIONARITY_LIMIT
        * expit(x[2])
    )

    alpha = (
        branching_ratio
        * beta
    )

    return (
        mu,
        alpha,
        beta,
        branching_ratio,
    )


def conditional_means(
    counts,
    mu,
    alpha,
    beta,
):
    decay = np.exp(
        -beta
        * BIN_SIZE_SECONDS
    )

    excitation_scale = (
        alpha / beta
    ) * (
        1.0 - decay
    )

    means = np.empty(
        len(counts),
        dtype=float,
    )

    state = 0.0

    for i in range(len(counts)):
        means[i] = (
            mu * BIN_SIZE_SECONDS
            + excitation_scale * state
        )

        state = (
            decay * state
            + counts[i]
        )

    return means


def negative_log_likelihood(
    x,
    counts,
):
    mu, alpha, beta, branching_ratio = (
        unpack_parameters(x)
    )

    means = conditional_means(
        counts,
        mu,
        alpha,
        beta,
    )

    if not np.isfinite(
        means
    ).all():
        return 1e100

    if (
        means <= 0
    ).any():
        return 1e100

    log_likelihood = np.sum(
        poisson.logpmf(
            counts.astype(int),
            means,
        )
    )

    if not np.isfinite(
        log_likelihood
    ):
        return 1e100

    return -log_likelihood


def poisson_model(counts):
    mean_count = counts.mean()

    log_likelihood = np.sum(
        poisson.logpmf(
            counts.astype(int),
            np.full(
                len(counts),
                mean_count,
            ),
        )
    )

    return (
        mean_count,
        log_likelihood,
    )


def fit_hawkes(counts):
    mean_rate = (
        counts.mean()
        / BIN_SIZE_SECONDS
    )

    starts = []

    branching_values = [
        0.05,
        0.15,
        0.30,
        0.50,
        0.70,
        0.85,
    ]

    beta_values = [
        1.0,
        5.0,
        10.0,
        25.0,
        50.0,
        100.0,
    ]

    for branching in branching_values:
        for beta in beta_values:
            logit_n = np.log(
                branching
                / (
                    STATIONARITY_LIMIT
                    - branching
                )
            )

            starts.append(
                [
                    np.log(
                        max(
                            mean_rate
                            * 0.1,
                            1e-6,
                        )
                    ),
                    np.log(beta),
                    logit_n,
                ]
            )

    starts = starts[:N_STARTS]

    results = []

    for x0 in starts:
        result = minimize(
            negative_log_likelihood,
            np.asarray(
                x0,
                dtype=float,
            ),
            args=(counts,),
            method="L-BFGS-B",
            options={
                "maxiter": 3000,
                "ftol": 1e-12,
                "gtol": 1e-8,
                "maxls": 50,
            },
        )

        if not np.isfinite(
            result.fun
        ):
            continue

        mu, alpha, beta, n = (
            unpack_parameters(
                result.x
            )
        )

        results.append(
            {
                "negative_log_likelihood": result.fun,
                "log_likelihood": -result.fun,
                "mu": mu,
                "alpha": alpha,
                "beta": beta,
                "branching_ratio": n,
                "iterations": result.nit,
                "success": result.success,
            }
        )

    if not results:
        raise RuntimeError(
            "No Hawkes optimization run succeeded."
        )

    results = pd.DataFrame(
        results
    )

    return results.loc[
        results[
            "negative_log_likelihood"
        ].idxmin()
    ]


def information_criteria(
    log_likelihood,
    n_parameters,
    n_observations,
):
    aic = (
        2 * n_parameters
        - 2 * log_likelihood
    )

    bic = (
        n_parameters
        * np.log(n_observations)
        - 2 * log_likelihood
    )

    return aic, bic


def main():
    times = load_events()

    counts = build_counts(
        times
    )

    duration = (
        len(counts)
        * BIN_SIZE_SECONDS
    )

    total_events = int(
        counts.sum()
    )

    occupied_bins = int(
        (counts > 0).sum()
    )

    poisson_mean_count, poisson_ll = (
        poisson_model(
            counts
        )
    )

    hawkes = fit_hawkes(
        counts
    )

    poisson_aic, poisson_bic = (
        information_criteria(
            poisson_ll,
            1,
            len(counts),
        )
    )

    hawkes_aic, hawkes_bic = (
        information_criteria(
            hawkes["log_likelihood"],
            3,
            len(counts),
        )
    )

    result = pd.DataFrame(
        [
            {
                "model": "poisson",
                "bin_size_seconds": BIN_SIZE_SECONDS,
                "duration_seconds": duration,
                "total_events": total_events,
                "occupied_bins": occupied_bins,
                "mu": poisson_mean_count
                / BIN_SIZE_SECONDS,
                "alpha": 0.0,
                "beta": np.nan,
                "branching_ratio": 0.0,
                "log_likelihood": poisson_ll,
                "aic": poisson_aic,
                "bic": poisson_bic,
            },
            {
                "model": "binned_hawkes",
                "bin_size_seconds": BIN_SIZE_SECONDS,
                "duration_seconds": duration,
                "total_events": total_events,
                "occupied_bins": occupied_bins,
                "mu": hawkes["mu"],
                "alpha": hawkes["alpha"],
                "beta": hawkes["beta"],
                "branching_ratio": hawkes[
                    "branching_ratio"
                ],
                "log_likelihood": hawkes[
                    "log_likelihood"
                ],
                "aic": hawkes_aic,
                "bic": hawkes_bic,
            },
        ]
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Bin size: "
        f"{BIN_SIZE_SECONDS:.3f} s"
    )

    print(
        f"Duration: "
        f"{duration:.6f} s"
    )

    print(
        f"Total events: "
        f"{total_events:,}"
    )

    print(
        f"Occupied bins: "
        f"{occupied_bins:,}"
    )

    print()

    print(
        "Poisson benchmark:"
    )

    print(
        f"Rate: "
        f"{poisson_mean_count / BIN_SIZE_SECONDS:.8f}"
    )

    print(
        f"Log-likelihood: "
        f"{poisson_ll:.8f}"
    )

    print(
        f"AIC: "
        f"{poisson_aic:.8f}"
    )

    print(
        f"BIC: "
        f"{poisson_bic:.8f}"
    )

    print()

    print(
        "Binned Hawkes:"
    )

    print(
        f"mu: "
        f"{hawkes['mu']:.8f}"
    )

    print(
        f"alpha: "
        f"{hawkes['alpha']:.8f}"
    )

    print(
        f"beta: "
        f"{hawkes['beta']:.8f}"
    )

    print(
        f"Branching ratio: "
        f"{hawkes['branching_ratio']:.8f}"
    )

    print(
        f"Log-likelihood: "
        f"{hawkes['log_likelihood']:.8f}"
    )

    print(
        f"AIC: "
        f"{hawkes_aic:.8f}"
    )

    print(
        f"BIC: "
        f"{hawkes_bic:.8f}"
    )

    print()

    print(
        result[
            [
                "model",
                "log_likelihood",
                "aic",
                "bic",
                "mu",
                "alpha",
                "beta",
                "branching_ratio",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()