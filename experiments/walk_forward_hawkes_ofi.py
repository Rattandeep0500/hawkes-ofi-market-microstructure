from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import expit
from scipy.stats import poisson


INPUT_FILE = Path(
    "data/processed/aligned_market_data.parquet"
)

OUTPUT_FILE = Path(
    "data/processed/walk_forward_hawkes_ofi.csv"
)

BIN_SIZE = 0.1
STATIONARITY_LIMIT = 0.999

HORIZONS = {
    "1s": 10,
    "5s": 50,
}

FOLDS = [
    (0.50, 0.65),
    (0.65, 0.80),
    (0.80, 1.00),
]


def load_data():
    data = pd.read_parquet(
        INPUT_FILE,
        engine="pyarrow",
    )

    required = [
        "time_bin_ms",
        "mid_price",
        "buy_count",
        "sell_count",
        "ofi_1",
        "ofi_2",
        "ofi_3",
        "ofi_4",
        "ofi_5",
        "ofi_6",
        "ofi_7",
        "ofi_8",
        "ofi_9",
        "ofi_10",
        "ofi_normalized",
        "ofi_depth_weighted",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    return (
        data
        .sort_values("time_bin_ms")
        .reset_index(drop=True)
    )


def hawkes_parameters(x):
    mu_buy = np.exp(
        np.clip(x[0], -20.0, 20.0)
    )

    mu_sell = np.exp(
        np.clip(x[1], -20.0, 20.0)
    )

    beta = np.exp(
        np.clip(x[2], -10.0, 10.0)
    )

    branching_buy = (
        STATIONARITY_LIMIT
        * expit(x[3])
    )

    branching_sell = (
        STATIONARITY_LIMIT
        * expit(x[4])
    )

    return (
        mu_buy,
        mu_sell,
        beta,
        branching_buy,
        branching_sell,
    )


def hawkes_states(
    buy,
    sell,
    beta,
):
    decay = np.exp(
        -beta * BIN_SIZE
    )

    buy_input = np.zeros(
        len(buy),
        dtype=float,
    )

    sell_input = np.zeros(
        len(sell),
        dtype=float,
    )

    if len(buy) > 1:
        buy_input[1:] = buy[:-1]
        sell_input[1:] = sell[:-1]

    buy_state = lfilter(
        [1.0],
        [1.0, -decay],
        buy_input,
    )

    sell_state = lfilter(
        [1.0],
        [1.0, -decay],
        sell_input,
    )

    return (
        buy_state,
        sell_state,
        decay,
    )


def negative_log_likelihood(
    x,
    buy,
    sell,
):
    (
        mu_buy,
        mu_sell,
        beta,
        branching_buy,
        branching_sell,
    ) = hawkes_parameters(x)

    buy_state, sell_state, decay = (
        hawkes_states(
            buy,
            sell,
            beta,
        )
    )

    scale = 1.0 - decay

    mean_buy = (
        mu_buy * BIN_SIZE
        + branching_buy
        * scale
        * buy_state
    )

    mean_sell = (
        mu_sell * BIN_SIZE
        + branching_sell
        * scale
        * sell_state
    )

    if (
        not np.isfinite(mean_buy).all()
        or not np.isfinite(mean_sell).all()
    ):
        return 1e100

    if (
        (mean_buy <= 0).any()
        or (mean_sell <= 0).any()
    ):
        return 1e100

    log_likelihood = (
        poisson.logpmf(
            buy.astype(np.int64),
            mean_buy,
        ).sum()
        + poisson.logpmf(
            sell.astype(np.int64),
            mean_sell,
        ).sum()
    )

    if not np.isfinite(
        log_likelihood
    ):
        return 1e100

    return -log_likelihood


def fit_hawkes(
    buy,
    sell,
):
    buy_rate = (
        buy.mean()
        / BIN_SIZE
    )

    sell_rate = (
        sell.mean()
        / BIN_SIZE
    )

    starts = []

    for beta in [
        1.0,
        5.0,
        10.0,
        25.0,
    ]:
        for branching in [
            0.20,
            0.40,
            0.60,
            0.75,
        ]:
            logit = np.log(
                branching
                / (
                    STATIONARITY_LIMIT
                    - branching
                )
            )

            starts.append(
                np.array(
                    [
                        np.log(
                            max(
                                buy_rate * 0.5,
                                1e-6,
                            )
                        ),
                        np.log(
                            max(
                                sell_rate * 0.5,
                                1e-6,
                            )
                        ),
                        np.log(beta),
                        logit,
                        logit,
                    ],
                    dtype=float,
                )
            )

    best = None

    for x0 in starts:
        result = minimize(
            negative_log_likelihood,
            x0,
            args=(
                buy,
                sell,
            ),
            method="L-BFGS-B",
            options={
                "maxiter": 500,
                "ftol": 1e-10,
                "gtol": 1e-7,
                "maxls": 25,
            },
        )

        if not np.isfinite(
            result.fun
        ):
            continue

        if (
            best is None
            or result.fun < best.fun
        ):
            best = result

    if best is None:
        raise RuntimeError(
            "Hawkes optimization failed."
        )

    return hawkes_parameters(
        best.x
    )


def generate_intensity(
    buy,
    sell,
    parameters,
):
    (
        mu_buy,
        mu_sell,
        beta,
        branching_buy,
        branching_sell,
    ) = parameters

    buy_state, sell_state, decay = (
        hawkes_states(
            buy,
            sell,
            beta,
        )
    )

    scale = 1.0 - decay

    buy_mean = (
        mu_buy * BIN_SIZE
        + branching_buy
        * scale
        * buy_state
    )

    sell_mean = (
        mu_sell * BIN_SIZE
        + branching_sell
        * scale
        * sell_state
    )

    return (
        buy_mean / BIN_SIZE,
        sell_mean / BIN_SIZE,
    )


def future_returns(
    prices,
    horizon_bins,
):
    values = prices.to_numpy(
        dtype=float
    )

    result = np.full(
        len(values),
        np.nan,
    )

    if horizon_bins >= len(values):
        return result

    start_values = values[
        :-horizon_bins
    ]

    end_values = values[
        horizon_bins:
    ]

    valid = (
        (start_values > 0)
        & (end_values > 0)
    )

    result[
        :-horizon_bins
    ][valid] = (
        np.log(
            end_values[valid]
        )
        - np.log(
            start_values[valid]
        )
    )

    return result


def oos_regression(
    x,
    y,
):
    valid = (
        np.isfinite(x).all(axis=1)
        & np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    if len(x) < 40:
        raise RuntimeError(
            "Too few observations."
        )

    split = int(
        len(x) * 0.70
    )

    if split < 20:
        raise RuntimeError(
            "Too few training observations."
        )

    if len(x) - split < 20:
        raise RuntimeError(
            "Too few testing observations."
        )

    x_train = x[:split]
    y_train = y[:split]

    x_test = x[split:]
    y_test = y[split:]

    x_train_design = np.column_stack(
        [
            np.ones(len(x_train)),
            x_train,
        ]
    )

    coefficients = np.linalg.lstsq(
        x_train_design,
        y_train,
        rcond=None,
    )[0]

    x_test_design = np.column_stack(
        [
            np.ones(len(x_test)),
            x_test,
        ]
    )

    predictions = (
        x_test_design
        @ coefficients
    )

    benchmark = np.full(
        len(y_test),
        y_train.mean(),
    )

    residuals = (
        y_test
        - predictions
    )

    benchmark_residuals = (
        y_test
        - benchmark
    )

    sse = np.sum(
        residuals ** 2
    )

    sse_benchmark = np.sum(
        benchmark_residuals ** 2
    )

    oos_r2 = (
        1.0
        - sse
        / sse_benchmark
    )

    rmse = np.sqrt(
        np.mean(
            residuals ** 2
        )
    )

    if (
        np.std(predictions) == 0
        or np.std(y_test) == 0
    ):
        correlation = np.nan
    else:
        correlation = np.corrcoef(
            predictions,
            y_test,
        )[0, 1]

    return {
        "train_observations": len(
            x_train
        ),
        "test_observations": len(
            x_test
        ),
        "oos_r2": oos_r2,
        "rmse": rmse,
        "prediction_return_correlation": correlation,
    }


def main():
    data = load_data()

    n = len(data)

    print(
        f"Total bins: {n:,}"
    )

    results = []

    for fold_number, (
        train_fraction,
        test_fraction,
    ) in enumerate(
        FOLDS,
        start=1,
    ):
        train_end = int(
            n * train_fraction
        )

        test_end = int(
            n * test_fraction
        )

        train = data.iloc[
            :train_end
        ].copy()

        test = data.iloc[
            train_end:test_end
        ].copy()

        print()
        print(
            f"Fold {fold_number}"
        )

        print(
            f"Training bins: "
            f"{len(train):,}"
        )

        print(
            f"Testing bins: "
            f"{len(test):,}"
        )

        parameters = fit_hawkes(
            train[
                "buy_count"
            ].to_numpy(dtype=float),
            train[
                "sell_count"
            ].to_numpy(dtype=float),
        )

        (
            mu_buy,
            mu_sell,
            beta,
            branching_buy,
            branching_sell,
        ) = parameters

        print(
            f"mu_buy={mu_buy:.6f} "
            f"mu_sell={mu_sell:.6f} "
            f"beta={beta:.6f}"
        )

        print(
            f"branching_buy={branching_buy:.6f} "
            f"branching_sell={branching_sell:.6f}"
        )

        all_buy = data[
            "buy_count"
        ].to_numpy(dtype=float)

        all_sell = data[
            "sell_count"
        ].to_numpy(dtype=float)

        hawkes_buy, hawkes_sell = (
            generate_intensity(
                all_buy,
                all_sell,
                parameters,
            )
        )

        fold_data = data.copy()

        fold_data[
            "hawkes_buy_intensity"
        ] = hawkes_buy

        fold_data[
            "hawkes_sell_intensity"
        ] = hawkes_sell

        fold_data[
            "hawkes_pressure"
        ] = (
            fold_data[
                "hawkes_buy_intensity"
            ]
            - fold_data[
                "hawkes_sell_intensity"
            ]
        )

        train_data = fold_data.iloc[
            :train_end
        ]

        test_data = fold_data.iloc[
            train_end:test_end
        ]

        for horizon_name, horizon_bins in (
            HORIZONS.items()
        ):
            target = future_returns(
                fold_data[
                    "mid_price"
                ],
                horizon_bins,
            )

            target_test = target[
                train_end:test_end
            ]

            models = {
                "OFI_L1": [
                    "ofi_1",
                ],
                "OFI_L10": [
                    f"ofi_{i}"
                    for i in range(1, 11)
                ],
                "OFI_normalized": [
                    "ofi_normalized",
                ],
                "Hawkes_pressure": [
                    "hawkes_pressure",
                ],
                "OFI_plus_Hawkes": [
                    "ofi_1",
                    "hawkes_pressure",
                ],
                "OFI_L10_plus_Hawkes": [
                    *[
                        f"ofi_{i}"
                        for i in range(1, 11)
                    ],
                    "hawkes_pressure",
                ],
            }

            for model_name, columns in (
                models.items()
            ):
                x_train = train_data[
                    columns
                ].to_numpy(dtype=float)

                x_test = test_data[
                    columns
                ].to_numpy(dtype=float)

                y_train = target[
                    :train_end
                ]

                y_test = target_test

                train_valid = (
                    np.isfinite(
                        x_train
                    ).all(axis=1)
                    & np.isfinite(
                        y_train
                    )
                )

                test_valid = (
                    np.isfinite(
                        x_test
                    ).all(axis=1)
                    & np.isfinite(
                        y_test
                    )
                )

                x_train = x_train[
                    train_valid
                ]

                y_train = y_train[
                    train_valid
                ]

                x_test = x_test[
                    test_valid
                ]

                y_test = y_test[
                    test_valid
                ]

                if len(x_train) < 30 or len(x_test) < 20:
                    continue

                train_design = np.column_stack(
                    [
                        np.ones(
                            len(x_train)
                        ),
                        x_train,
                    ]
                )

                coefficients = np.linalg.lstsq(
                    train_design,
                    y_train,
                    rcond=None,
                )[0]

                test_design = np.column_stack(
                    [
                        np.ones(
                            len(x_test)
                        ),
                        x_test,
                    ]
                )

                predictions = (
                    test_design
                    @ coefficients
                )

                benchmark = np.full(
                    len(y_test),
                    y_train.mean(),
                )

                residuals = (
                    y_test
                    - predictions
                )

                benchmark_residuals = (
                    y_test
                    - benchmark
                )

                sse = np.sum(
                    residuals ** 2
                )

                sse_benchmark = np.sum(
                    benchmark_residuals ** 2
                )

                oos_r2 = (
                    1.0
                    - sse
                    / sse_benchmark
                )

                rmse = np.sqrt(
                    np.mean(
                        residuals ** 2
                    )
                )

                if (
                    np.std(predictions) == 0
                    or np.std(y_test) == 0
                ):
                    correlation = np.nan
                else:
                    correlation = np.corrcoef(
                        predictions,
                        y_test,
                    )[0, 1]

                results.append(
                    {
                        "fold": fold_number,
                        "horizon": horizon_name,
                        "model": model_name,
                        "train_observations": len(
                            x_train
                        ),
                        "test_observations": len(
                            x_test
                        ),
                        "oos_r2": oos_r2,
                        "rmse": rmse,
                        "prediction_return_correlation": correlation,
                    }
                )

    results = pd.DataFrame(
        results
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "Walk-forward results:"
    )

    print(
        results.to_string(
            index=False
        )
    )

    print()
    print(
        "Average OOS performance:"
    )

    summary = (
        results
        .groupby(
            [
                "horizon",
                "model",
            ]
        )[
            [
                "oos_r2",
                "rmse",
                "prediction_return_correlation",
            ]
        ]
        .mean()
        .reset_index()
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()