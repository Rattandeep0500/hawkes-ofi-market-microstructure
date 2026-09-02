from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


INPUT_FILE = Path(
    "data/processed/trade_events.parquet"
)

OUTPUT_FILE = Path(
    "data/processed/hawkes_univariate.csv"
)

STATIONARITY_LIMIT = 0.999
STARTS = 8


def load_events():
    data = pd.read_parquet(
        INPUT_FILE,
        columns=[
            "event_time_s",
            "trade_id",
        ],
        engine="pyarrow",
    )

    data = (
        data
        .sort_values(
            ["event_time_s", "trade_id"]
        )
        .reset_index(drop=True)
    )

    if data.empty:
        raise RuntimeError(
            "No trade events found."
        )

    grouped = (
        data.groupby(
            "event_time_s",
            sort=True,
        )
        .size()
        .rename("count")
        .reset_index()
    )

    times = grouped[
        "event_time_s"
    ].to_numpy(dtype=float)

    counts = grouped[
        "count"
    ].to_numpy(dtype=float)

    return times, counts


def hawkes_parameters(x):
    mu = np.exp(x[0])
    beta = np.exp(x[2])

    branching_ratio = (
        STATIONARITY_LIMIT
        / (
            1.0 + np.exp(-x[1])
        )
    )

    alpha = branching_ratio * beta

    return mu, alpha, beta, branching_ratio


def excitation(times, counts, beta):
    n = len(times)

    g = np.zeros(n)

    if n <= 1:
        return g

    previous = 0.0

    for i in range(1, n):
        delta = times[i] - times[i - 1]

        previous *= np.exp(
            -beta * delta
        )

        previous += counts[i - 1]

        g[i] = previous

    return g


def negative_log_likelihood(x, times, counts):
    mu, alpha, beta, branching_ratio = (
        hawkes_parameters(x)
    )

    if not (
        np.isfinite(mu)
        and np.isfinite(alpha)
        and np.isfinite(beta)
        and np.isfinite(branching_ratio)
    ):
        return 1e100

    g = excitation(
        times,
        counts,
        beta,
    )

    intensity = (
        mu + alpha * g
    )

    if (
        np.any(intensity <= 0)
        or not np.all(np.isfinite(intensity))
    ):
        return 1e100

    duration = times[-1] - times[0]

    integral = (
        mu * duration
        + (alpha / beta)
        * np.sum(
            counts
            * (
                1.0
                - np.exp(
                    -beta
                    * (
                        times[-1]
                        - times
                    )
                )
            )
        )
    )

    log_likelihood = (
        np.sum(
            counts
            * np.log(intensity)
        )
        - integral
    )

    if not np.isfinite(
        log_likelihood
    ):
        return 1e100

    return -log_likelihood


def poisson_log_likelihood(
    times,
    counts,
):
    duration = (
        times[-1] - times[0]
    )

    total_events = counts.sum()

    rate = (
        total_events
        / duration
    )

    log_likelihood = (
        total_events
        * np.log(rate)
        - rate * duration
    )

    return (
        log_likelihood,
        rate,
    )


def fit_hawkes(times, counts):
    duration = (
        times[-1] - times[0]
    )

    total_events = counts.sum()

    poisson_rate = (
        total_events
        / duration
    )

    estimates = []

    initial_branching = [
        0.05,
        0.15,
        0.30,
        0.50,
        0.70,
        0.85,
        0.95,
        0.98,
    ]

    beta_scales = [
        0.25,
        0.5,
        1.0,
        2.0,
    ]

    for branching in initial_branching:
        for beta_scale in beta_scales:

            beta0 = (
                beta_scale
                * max(
                    poisson_rate,
                    1e-6,
                )
            )

            alpha0 = (
                branching
                * beta0
            )

            x0 = np.array(
                [
                    np.log(
                        poisson_rate
                    ),
                    np.log(
                        branching
                        / (
                            STATIONARITY_LIMIT
                            - branching
                        )
                    ),
                    np.log(beta0),
                ]
            )

            result = minimize(
                negative_log_likelihood,
                x0,
                args=(
                    times,
                    counts,
                ),
                method="L-BFGS-B",
                options={
                    "maxiter": 2000,
                    "ftol": 1e-12,
                    "gtol": 1e-8,
                },
            )

            if result.success and np.isfinite(
                result.fun
            ):
                mu, alpha, beta, n = (
                    hawkes_parameters(
                        result.x
                    )
                )

                estimates.append(
                    {
                        "negative_log_likelihood": result.fun,
                        "log_likelihood": -result.fun,
                        "mu": mu,
                        "alpha": alpha,
                        "beta": beta,
                        "branching_ratio": n,
                        "converged": result.success,
                        "iterations": result.nit,
                    }
                )

    if not estimates:
        raise RuntimeError(
            "Hawkes optimization failed."
        )

    estimates = pd.DataFrame(
        estimates
    )

    best_index = (
        estimates[
            "negative_log_likelihood"
        ]
        .idxmin()
    )

    return estimates.loc[
        best_index
    ]


def calculate_information_criteria(
    log_likelihood,
    n_events,
):
    parameters = 3

    aic = (
        2 * parameters
        - 2 * log_likelihood
    )

    bic = (
        parameters
        * np.log(n_events)
        - 2 * log_likelihood
    )

    return aic, bic


def main():
    times, counts = load_events()

    n_events = int(
        counts.sum()
    )

    n_timestamps = len(
        times
    )

    duration = (
        times[-1] - times[0]
    )

    print(
        f"Unique timestamps: "
        f"{n_timestamps:,}"
    )

    print(
        f"Total events: "
        f"{n_events:,}"
    )

    print(
        f"Duration: "
        f"{duration:.6f} s"
    )

    print()

    poisson_ll, poisson_rate = (
        poisson_log_likelihood(
            times,
            counts,
        )
    )

    poisson_aic = (
        2 - 2 * poisson_ll
    )

    poisson_bic = (
        np.log(n_events)
        - 2 * poisson_ll
    )

    print(
        "Poisson benchmark:"
    )

    print(
        f"Rate: "
        f"{poisson_rate:.8f}"
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

    best = fit_hawkes(
        times,
        counts,
    )

    aic, bic = (
        calculate_information_criteria(
            best["log_likelihood"],
            n_events,
        )
    )

    result = pd.DataFrame(
        [
            {
                "model": "poisson",
                "mu": poisson_rate,
                "alpha": 0.0,
                "beta": np.nan,
                "branching_ratio": 0.0,
                "log_likelihood": poisson_ll,
                "aic": poisson_aic,
                "bic": poisson_bic,
                "n_events": n_events,
                "unique_timestamps": n_timestamps,
                "duration_seconds": duration,
            },
            {
                "model": "hawkes",
                "mu": best["mu"],
                "alpha": best["alpha"],
                "beta": best["beta"],
                "branching_ratio": best[
                    "branching_ratio"
                ],
                "log_likelihood": best[
                    "log_likelihood"
                ],
                "aic": aic,
                "bic": bic,
                "n_events": n_events,
                "unique_timestamps": n_timestamps,
                "duration_seconds": duration,
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
        "Hawkes estimate:"
    )

    print(
        f"mu: "
        f"{best['mu']:.8f}"
    )

    print(
        f"alpha: "
        f"{best['alpha']:.8f}"
    )

    print(
        f"beta: "
        f"{best['beta']:.8f}"
    )

    print(
        f"Branching ratio: "
        f"{best['branching_ratio']:.8f}"
    )

    print(
        f"Log-likelihood: "
        f"{best['log_likelihood']:.8f}"
    )

    print(
        f"AIC: "
        f"{aic:.8f}"
    )

    print(
        f"BIC: "
        f"{bic:.8f}"
    )

    print()

    print(
        "Model comparison:"
    )

    print(
        result[
            [
                "model",
                "log_likelihood",
                "aic",
                "bic",
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