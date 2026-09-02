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
    "data/processed/bivariate_hawkes.csv"
)

BIN_SIZE_SECONDS = 0.1
STATIONARITY_LIMIT = 0.999
N_STARTS = 16


def load_events():
    data = pd.read_parquet(
        INPUT_FILE,
        columns=[
            "event_time_s",
            "side",
        ],
        engine="pyarrow",
    )

    data = data.sort_values(
        "event_time_s"
    ).reset_index(drop=True)

    if data.empty:
        raise RuntimeError(
            "No trade events found."
        )

    start = data["event_time_s"].iloc[0]

    data["relative_time_s"] = (
        data["event_time_s"] - start
    )

    return data


def build_counts(data):
    duration = (
        data["relative_time_s"].max()
    )

    n_bins = int(
        np.ceil(
            duration / BIN_SIZE_SECONDS
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

    counts = np.column_stack(
        [
            buy_counts.astype(float),
            sell_counts.astype(float),
        ]
    )

    return counts


def unpack_parameters(x):
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

    raw_matrix = np.exp(
        np.clip(
            x[3:7],
            -20.0,
            20.0,
        )
    ).reshape(2, 2)

    eigenvalues = np.linalg.eigvals(
        raw_matrix
    )

    rho_raw = np.max(
        np.abs(eigenvalues)
    )

    if not np.isfinite(rho_raw):
        raise RuntimeError(
            "Invalid branching matrix."
        )

    if rho_raw <= 0:
        branching_ratio = 0.0
        branching_matrix = np.zeros(
            (2, 2),
            dtype=float,
        )
    else:
        branching_ratio = (
            STATIONARITY_LIMIT
            * expit(x[7])
        )

        branching_matrix = (
            branching_ratio
            * raw_matrix
            / rho_raw
        )

    alpha = (
        branching_matrix
        * beta
    )

    return (
        mu,
        alpha,
        beta,
        branching_matrix,
        branching_ratio,
    )


def conditional_means(
    counts,
    mu,
    branching_matrix,
    beta,
):
    decay = np.exp(
        -beta
        * BIN_SIZE_SECONDS
    )

    excitation_scale = (
        1.0 - decay
    )

    n_bins = len(counts)

    means = np.zeros(
        (n_bins, 2),
        dtype=float,
    )

    state = np.zeros(
        2,
        dtype=float,
    )

    for t in range(n_bins):
        means[t] = (
            mu * BIN_SIZE_SECONDS
            + branching_matrix
            @ (
                excitation_scale
                * state
            )
        )

        state = (
            decay * state
            + counts[t]
        )

    return means


def negative_log_likelihood(
    x,
    counts,
):
    try:
        (
            mu,
            alpha,
            beta,
            branching_matrix,
            branching_ratio,
        ) = unpack_parameters(x)
    except Exception:
        return 1e100

    if not (
        np.isfinite(mu).all()
        and np.isfinite(alpha).all()
        and np.isfinite(beta)
        and np.isfinite(branching_matrix).all()
        and np.isfinite(branching_ratio)
    ):
        return 1e100

    means = conditional_means(
        counts,
        mu,
        branching_matrix,
        beta,
    )

    if not np.isfinite(means).all():
        return 1e100

    if (means <= 0).any():
        return 1e100

    log_likelihood = 0.0

    for side in range(2):
        log_likelihood += np.sum(
            poisson.logpmf(
                counts[:, side].astype(int),
                means[:, side],
            )
        )

    if not np.isfinite(
        log_likelihood
    ):
        return 1e100

    return -log_likelihood


def poisson_model(counts):
    means = counts.mean(axis=0)

    log_likelihood = 0.0

    for side in range(2):
        expected = np.full(
            len(counts),
            means[side],
        )

        log_likelihood += np.sum(
            poisson.logpmf(
                counts[:, side].astype(int),
                expected,
            )
        )

    return (
        means,
        log_likelihood,
    )


def make_initial_points(
    counts,
):
    mean_rates = (
        counts.mean(axis=0)
        / BIN_SIZE_SECONDS
    )

    points = []

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

    matrices = [
        [1.0, 0.3, 0.3, 1.0],
        [1.0, 0.8, 0.2, 1.0],
        [1.0, 0.2, 0.8, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ]

    for branching in branching_values:
        for beta in beta_values:
            for matrix in matrices:

                logit_branching = np.log(
                    branching
                    / (
                        STATIONARITY_LIMIT
                        - branching
                    )
                )

                x0 = np.array(
                    [
                        np.log(
                            max(
                                mean_rates[0]
                                * 0.5,
                                1e-6,
                            )
                        ),
                        np.log(
                            max(
                                mean_rates[1]
                                * 0.5,
                                1e-6,
                            )
                        ),
                        np.log(beta),
                        np.log(matrix[0]),
                        np.log(matrix[1]),
                        np.log(matrix[2]),
                        np.log(matrix[3]),
                        logit_branching,
                    ],
                    dtype=float,
                )

                points.append(x0)

    return points[:N_STARTS]


def fit_hawkes(counts):
    starts = make_initial_points(
        counts
    )

    results = []

    for x0 in starts:
        result = minimize(
            negative_log_likelihood,
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
            (
                mu,
                alpha,
                beta,
                branching_matrix,
                branching_ratio,
            ) = unpack_parameters(
                result.x
            )
        except Exception:
            continue

        results.append(
            {
                "negative_log_likelihood": result.fun,
                "log_likelihood": -result.fun,
                "mu_buy": mu[0],
                "mu_sell": mu[1],
                "alpha_bb": alpha[0, 0],
                "alpha_bs": alpha[0, 1],
                "alpha_sb": alpha[1, 0],
                "alpha_ss": alpha[1, 1],
                "beta": beta,
                "n_bb": branching_matrix[0, 0],
                "n_bs": branching_matrix[0, 1],
                "n_sb": branching_matrix[1, 0],
                "n_ss": branching_matrix[1, 1],
                "spectral_radius": branching_ratio,
                "iterations": result.nit,
                "success": result.success,
            }
        )

    if not results:
        raise RuntimeError(
            "Bivariate Hawkes optimization failed."
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

    duration = (
        len(counts)
        * BIN_SIZE_SECONDS
    )

    total_buy = int(
        counts[:, 0].sum()
    )

    total_sell = int(
        counts[:, 1].sum()
    )

    total_events = (
        total_buy
        + total_sell
    )

    poisson_means, poisson_ll = (
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
            2,
            len(counts) * 2,
        )
    )

    hawkes_aic, hawkes_bic = (
        information_criteria(
            hawkes["log_likelihood"],
            7,
            len(counts) * 2,
        )
    )

    result = pd.DataFrame(
        [
            {
                "model": "poisson",
                "duration_seconds": duration,
                "bin_size_seconds": BIN_SIZE_SECONDS,
                "total_events": total_events,
                "buy_events": total_buy,
                "sell_events": total_sell,
                "mu_buy": (
                    poisson_means[0]
                    / BIN_SIZE_SECONDS
                ),
                "mu_sell": (
                    poisson_means[1]
                    / BIN_SIZE_SECONDS
                ),
                "alpha_bb": 0.0,
                "alpha_bs": 0.0,
                "alpha_sb": 0.0,
                "alpha_ss": 0.0,
                "beta": np.nan,
                "n_bb": 0.0,
                "n_bs": 0.0,
                "n_sb": 0.0,
                "n_ss": 0.0,
                "spectral_radius": 0.0,
                "log_likelihood": poisson_ll,
                "aic": poisson_aic,
                "bic": poisson_bic,
            },
            {
                "model": "bivariate_hawkes",
                "duration_seconds": duration,
                "bin_size_seconds": BIN_SIZE_SECONDS,
                "total_events": total_events,
                "buy_events": total_buy,
                "sell_events": total_sell,
                "mu_buy": hawkes["mu_buy"],
                "mu_sell": hawkes["mu_sell"],
                "alpha_bb": hawkes["alpha_bb"],
                "alpha_bs": hawkes["alpha_bs"],
                "alpha_sb": hawkes["alpha_sb"],
                "alpha_ss": hawkes["alpha_ss"],
                "beta": hawkes["beta"],
                "n_bb": hawkes["n_bb"],
                "n_bs": hawkes["n_bs"],
                "n_sb": hawkes["n_sb"],
                "n_ss": hawkes["n_ss"],
                "spectral_radius": hawkes[
                    "spectral_radius"
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
        f"Buy events: "
        f"{total_buy:,}"
    )

    print(
        f"Sell events: "
        f"{total_sell:,}"
    )

    print()

    print(
        "Poisson benchmark:"
    )

    print(
        f"Buy rate: "
        f"{poisson_means[0] / BIN_SIZE_SECONDS:.8f}"
    )

    print(
        f"Sell rate: "
        f"{poisson_means[1] / BIN_SIZE_SECONDS:.8f}"
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
        "Bivariate Hawkes:"
    )

    print(
        f"mu_buy: "
        f"{hawkes['mu_buy']:.8f}"
    )

    print(
        f"mu_sell: "
        f"{hawkes['mu_sell']:.8f}"
    )

    print(
        f"alpha_bb: "
        f"{hawkes['alpha_bb']:.8f}"
    )

    print(
        f"alpha_bs: "
        f"{hawkes['alpha_bs']:.8f}"
    )

    print(
        f"alpha_sb: "
        f"{hawkes['alpha_sb']:.8f}"
    )

    print(
        f"alpha_ss: "
        f"{hawkes['alpha_ss']:.8f}"
    )

    print(
        f"beta: "
        f"{hawkes['beta']:.8f}"
    )

    print(
        f"n_bb: "
        f"{hawkes['n_bb']:.8f}"
    )

    print(
        f"n_bs: "
        f"{hawkes['n_bs']:.8f}"
    )

    print(
        f"n_sb: "
        f"{hawkes['n_sb']:.8f}"
    )

    print(
        f"n_ss: "
        f"{hawkes['n_ss']:.8f}"
    )

    print(
        f"Spectral radius: "
        f"{hawkes['spectral_radius']:.8f}"
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
                "mu_buy",
                "mu_sell",
                "n_bb",
                "n_bs",
                "n_sb",
                "n_ss",
                "spectral_radius",
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