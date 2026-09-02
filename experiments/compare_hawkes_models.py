from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import chi2, poisson


INPUT_FILE = Path(
    "data/processed/trade_events.parquet"
)

OUTPUT_FILE = Path(
    "data/processed/hawkes_model_comparison.csv"
)

BIN_SIZE_SECONDS = 0.1
STATIONARITY_LIMIT = 0.999


def load_events():
    data = pd.read_parquet(
        INPUT_FILE,
        columns=[
            "event_time_s",
            "side",
        ],
        engine="pyarrow",
    )

    if data.empty:
        raise RuntimeError(
            "No trade events found."
        )

    data = (
        data
        .sort_values("event_time_s")
        .reset_index(drop=True)
    )

    data["relative_time_s"] = (
        data["event_time_s"]
        - data["event_time_s"].iloc[0]
    )

    return data


def build_counts(data):
    duration = data[
        "relative_time_s"
    ].max()

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

    buy_times = data.loc[
        data["side"] == "buy",
        "relative_time_s",
    ].to_numpy(dtype=float)

    sell_times = data.loc[
        data["side"] == "sell",
        "relative_time_s",
    ].to_numpy(dtype=float)

    buy_counts, _ = np.histogram(
        buy_times,
        bins=edges,
    )

    sell_counts, _ = np.histogram(
        sell_times,
        bins=edges,
    )

    return np.column_stack(
        [
            buy_counts.astype(float),
            sell_counts.astype(float),
        ]
    )


def unpack_full(x):
    mu = np.exp(
        np.clip(
            x[:2],
            -20.0,
            20.0,
        )
    )

    beta = np.exp(
        np.clip(
            x[2],
            -10.0,
            10.0,
        )
    )

    raw = np.exp(
        np.clip(
            x[3:7],
            -20.0,
            20.0,
        )
    ).reshape(2, 2)

    spectral_raw = np.max(
        np.abs(
            np.linalg.eigvals(raw)
        )
    )

    if not np.isfinite(
        spectral_raw
    ):
        return None

    if spectral_raw <= 0:
        branching = np.zeros(
            (2, 2),
            dtype=float,
        )
        spectral_radius = 0.0
    else:
        spectral_radius = (
            STATIONARITY_LIMIT
            * expit(x[7])
        )

        branching = (
            raw
            * spectral_radius
            / spectral_raw
        )

    return (
        mu,
        beta,
        branching,
        spectral_radius,
    )


def unpack_restricted(x):
    mu = np.exp(
        np.clip(
            x[:2],
            -20.0,
            20.0,
        )
    )

    beta = np.exp(
        np.clip(
            x[2],
            -10.0,
            10.0,
        )
    )

    n_bb = (
        STATIONARITY_LIMIT
        * expit(x[3])
    )

    n_ss = (
        STATIONARITY_LIMIT
        * expit(x[4])
    )

    branching = np.array(
        [
            [n_bb, 0.0],
            [0.0, n_ss],
        ],
        dtype=float,
    )

    spectral_radius = max(
        n_bb,
        n_ss,
    )

    return (
        mu,
        beta,
        branching,
        spectral_radius,
    )


def conditional_means(
    counts,
    mu,
    beta,
    branching,
):
    decay = np.exp(
        -beta * BIN_SIZE_SECONDS
    )

    scale = 1.0 - decay

    means = np.zeros_like(
        counts,
        dtype=float,
    )

    state = np.zeros(
        2,
        dtype=float,
    )

    for i in range(
        len(counts)
    ):
        means[i] = (
            mu * BIN_SIZE_SECONDS
            + branching
            @ (
                scale * state
            )
        )

        state = (
            decay * state
            + counts[i]
        )

    return means


def full_nll(x, counts):
    unpacked = unpack_full(x)

    if unpacked is None:
        return 1e100

    mu, beta, branching, _ = (
        unpacked
    )

    means = conditional_means(
        counts,
        mu,
        beta,
        branching,
    )

    if not np.isfinite(means).all():
        return 1e100

    if (means <= 0).any():
        return 1e100

    value = 0.0

    for side in range(2):
        value += np.sum(
            poisson.logpmf(
                counts[:, side].astype(int),
                means[:, side],
            )
        )

    if not np.isfinite(value):
        return 1e100

    return -value


def restricted_nll(x, counts):
    mu, beta, branching, _ = (
        unpack_restricted(x)
    )

    means = conditional_means(
        counts,
        mu,
        beta,
        branching,
    )

    if not np.isfinite(means).all():
        return 1e100

    if (means <= 0).any():
        return 1e100

    value = 0.0

    for side in range(2):
        value += np.sum(
            poisson.logpmf(
                counts[:, side].astype(int),
                means[:, side],
            )
        )

    if not np.isfinite(value):
        return 1e100

    return -value


def initial_full_points(counts):
    rates = (
        counts.mean(axis=0)
        / BIN_SIZE_SECONDS
    )

    points = []

    branches = [
        0.10,
        0.30,
        0.50,
        0.70,
    ]

    betas = [
        1.0,
        5.0,
        10.0,
        25.0,
    ]

    matrices = [
        [1.0, 0.2, 0.2, 1.0],
        [1.0, 0.5, 0.1, 1.0],
        [1.0, 0.1, 0.5, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ]

    for branching in branches:
        for beta in betas:
            for matrix in matrices:
                x = np.array(
                    [
                        np.log(
                            max(
                                rates[0] * 0.5,
                                1e-6,
                            )
                        ),
                        np.log(
                            max(
                                rates[1] * 0.5,
                                1e-6,
                            )
                        ),
                        np.log(beta),
                        np.log(matrix[0]),
                        np.log(matrix[1]),
                        np.log(matrix[2]),
                        np.log(matrix[3]),
                        np.log(
                            branching
                            / (
                                STATIONARITY_LIMIT
                                - branching
                            )
                        ),
                    ],
                    dtype=float,
                )

                points.append(x)

    return points


def initial_restricted_points(counts):
    rates = (
        counts.mean(axis=0)
        / BIN_SIZE_SECONDS
    )

    points = []

    branches = [
        0.10,
        0.30,
        0.50,
        0.70,
    ]

    betas = [
        1.0,
        5.0,
        10.0,
        25.0,
    ]

    for branching in branches:
        for beta in betas:
            logit = np.log(
                branching
                / (
                    STATIONARITY_LIMIT
                    - branching
                )
            )

            x = np.array(
                [
                    np.log(
                        max(
                            rates[0] * 0.5,
                            1e-6,
                        )
                    ),
                    np.log(
                        max(
                            rates[1] * 0.5,
                            1e-6,
                        )
                    ),
                    np.log(beta),
                    logit,
                    logit,
                ],
                dtype=float,
            )

            points.append(x)

    return points


def fit_model(
    objective,
    starts,
    unpack,
    counts,
):
    candidates = []

    for x0 in starts:
        result = minimize(
            objective,
            x0,
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

        try:
            values = unpack(
                result.x
            )
        except Exception:
            continue

        if not np.isfinite(
            np.asarray(values[0])
        ).all():
            continue

        candidates.append(
            (
                result.fun,
                result,
                values,
            )
        )

    if not candidates:
        raise RuntimeError(
            "Optimization failed for model."
        )

    return min(
        candidates,
        key=lambda x: x[0],
    )


def information_criteria(
    log_likelihood,
    parameters,
    observations,
):
    aic = (
        2 * parameters
        - 2 * log_likelihood
    )

    bic = (
        parameters
        * np.log(observations)
        - 2 * log_likelihood
    )

    return aic, bic


def main():
    data = load_events()

    counts = build_counts(
        data
    )

    total_events = int(
        counts.sum()
    )

    duration = (
        len(counts)
        * BIN_SIZE_SECONDS
    )

    observations = (
        len(counts) * 2
    )

    restricted_fit = fit_model(
        restricted_nll,
        initial_restricted_points(counts),
        unpack_restricted,
        counts,
    )

    restricted_value, _, restricted_values = (
        restricted_fit
    )

    (
        restricted_mu,
        restricted_beta,
        restricted_branching,
        restricted_radius,
    ) = restricted_values

    full_fit = fit_model(
        full_nll,
        initial_full_points(counts),
        unpack_full,
        counts,
    )

    full_value, _, full_values = (
        full_fit
    )

    (
        full_mu,
        full_beta,
        full_branching,
        full_radius,
    ) = full_values

    restricted_ll = -restricted_value
    full_ll = -full_value

    restricted_aic, restricted_bic = (
        information_criteria(
            restricted_ll,
            5,
            observations,
        )
    )

    full_aic, full_bic = (
        information_criteria(
            full_ll,
            8,
            observations,
        )
    )

    lr_stat = 2.0 * (
        full_ll
        - restricted_ll
    )

    df = 3

    lr_pvalue = chi2.sf(
        lr_stat,
        df,
    )

    result = pd.DataFrame(
        [
            {
                "model": "restricted_independent_hawkes",
                "log_likelihood": restricted_ll,
                "aic": restricted_aic,
                "bic": restricted_bic,
                "mu_buy": restricted_mu[0],
                "mu_sell": restricted_mu[1],
                "beta": restricted_beta,
                "n_bb": restricted_branching[0, 0],
                "n_bs": 0.0,
                "n_sb": 0.0,
                "n_ss": restricted_branching[1, 1],
                "spectral_radius": restricted_radius,
            },
            {
                "model": "full_bivariate_hawkes",
                "log_likelihood": full_ll,
                "aic": full_aic,
                "bic": full_bic,
                "mu_buy": full_mu[0],
                "mu_sell": full_mu[1],
                "beta": full_beta,
                "n_bb": full_branching[0, 0],
                "n_bs": full_branching[0, 1],
                "n_sb": full_branching[1, 0],
                "n_ss": full_branching[1, 1],
                "spectral_radius": full_radius,
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

    print()

    print(
        "Restricted Hawkes:"
    )

    print(
        f"Log-likelihood: "
        f"{restricted_ll:.8f}"
    )

    print(
        f"AIC: "
        f"{restricted_aic:.8f}"
    )

    print(
        f"BIC: "
        f"{restricted_bic:.8f}"
    )

    print(
        f"n_bb: "
        f"{restricted_branching[0, 0]:.8f}"
    )

    print(
        f"n_ss: "
        f"{restricted_branching[1, 1]:.8f}"
    )

    print(
        f"Spectral radius: "
        f"{restricted_radius:.8f}"
    )

    print()

    print(
        "Full bivariate Hawkes:"
    )

    print(
        f"Log-likelihood: "
        f"{full_ll:.8f}"
    )

    print(
        f"AIC: "
        f"{full_aic:.8f}"
    )

    print(
        f"BIC: "
        f"{full_bic:.8f}"
    )

    print(
        f"n_bb: "
        f"{full_branching[0, 0]:.8f}"
    )

    print(
        f"n_bs: "
        f"{full_branching[0, 1]:.8f}"
    )

    print(
        f"n_sb: "
        f"{full_branching[1, 0]:.8f}"
    )

    print(
        f"n_ss: "
        f"{full_branching[1, 1]:.8f}"
    )

    print(
        f"Spectral radius: "
        f"{full_radius:.8f}"
    )

    print()

    print(
        "Likelihood-ratio test:"
    )

    print(
        f"LR statistic: "
        f"{lr_stat:.8f}"
    )

    print(
        f"Degrees of freedom: "
        f"{df}"
    )

    print(
        f"p-value: "
        f"{lr_pvalue:.8e}"
    )

    print()

    print(
        result.to_string(
            index=False
        )
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()