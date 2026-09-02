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
    "data/processed/aligned_hawkes_ofi_results.csv"
)

BIN_SIZE = 0.1
TRAIN_FRACTION = 0.70
STATIONARITY_LIMIT = 0.999


def load_data():
    data = pd.read_parquet(
        INPUT_FILE,
        engine="pyarrow",
    )

    required = [
        "time_bin_ms",
        "mid_price",
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
        "ofi_multilevel",
        "ofi_normalized",
        "ofi_depth_weighted",
        "buy_count",
        "sell_count",
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

    data = (
        data
        .sort_values("time_bin_ms")
        .reset_index(drop=True)
    )

    return data


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


def build_states(counts, decay):
    buy_input = np.empty(
        len(counts),
        dtype=float,
    )

    sell_input = np.empty(
        len(counts),
        dtype=float,
    )

    buy_input[0] = 0.0
    sell_input[0] = 0.0

    if len(counts) > 1:
        buy_input[1:] = counts[:-1, 0]
        sell_input[1:] = counts[:-1, 1]

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

    return buy_state, sell_state


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

    decay = np.exp(
        -beta * BIN_SIZE
    )

    scale = (
        1.0 - decay
    )

    counts = np.column_stack(
        [
            buy,
            sell,
        ]
    )

    buy_state, sell_state = (
        build_states(
            counts,
            decay,
        )
    )

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
        buy.mean() / BIN_SIZE
    )

    sell_rate = (
        sell.mean() / BIN_SIZE
    )

    starts = []

    for beta in [
        1.0,
        5.0,
        10.0,
        25.0,
    ]:
        for branching in [
            0.15,
            0.30,
            0.50,
            0.70,
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

    starts = starts[:4]

    results = []

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

        if np.isfinite(
            result.fun
        ):
            results.append(
                result
            )

    if not results:
        raise RuntimeError(
            "Hawkes optimization failed."
        )

    best = min(
        results,
        key=lambda result: result.fun,
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

    counts = np.column_stack(
        [
            buy,
            sell,
        ]
    )

    decay = np.exp(
        -beta * BIN_SIZE
    )

    scale = (
        1.0 - decay
    )

    buy_state, sell_state = (
        build_states(
            counts,
            decay,
        )
    )

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

    buy_intensity = (
        buy_mean
        / BIN_SIZE
    )

    sell_intensity = (
        sell_mean
        / BIN_SIZE
    )

    return (
        buy_intensity,
        sell_intensity,
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

    valid_start = (
        values[:-horizon_bins] > 0
    )

    valid_end = (
        values[horizon_bins:] > 0
    )

    valid = (
        valid_start
        & valid_end
    )

    result[
        :-horizon_bins
    ][valid] = (
        np.log(
            values[
                horizon_bins:
            ][valid]
        )
        - np.log(
            values[
                :-horizon_bins
            ][valid]
        )
    )

    return result


def fit_oos_regression(
    x,
    y,
    split,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    valid = (
        np.isfinite(x).all(axis=1)
        & np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    split = min(
        split,
        len(x),
    )

    if split < 30:
        raise RuntimeError(
            "Insufficient training observations."
        )

    if (
        len(x) - split
        < 30
    ):
        raise RuntimeError(
            "Insufficient test observations."
        )

    x_train = x[:split]
    y_train = y[:split]

    x_test = x[split:]
    y_test = y[split:]

    x_train_design = np.column_stack(
        [
            np.ones(
                len(x_train)
            ),
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
            np.ones(
                len(x_test)
            ),
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


def build_models():
    return {
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
        "OFI_depth_weighted_plus_Hawkes": [
            "ofi_depth_weighted",
            "hawkes_pressure",
        ],
    }


def run_experiment(
    data,
    split,
):
    models = build_models()

    horizons = {
        "1s": 10,
        "5s": 50,
    }

    results = []

    for horizon_name, horizon_bins in (
        horizons.items()
    ):
        target = future_returns(
            data["mid_price"],
            horizon_bins,
        )

        for model_name, columns in (
            models.items()
        ):
            x = data[
                columns
            ].to_numpy(
                dtype=float
            )

            result = fit_oos_regression(
                x,
                target,
                split,
            )

            result["horizon"] = (
                horizon_name
            )

            result["model"] = (
                model_name
            )

            results.append(
                result
            )

    return pd.DataFrame(
        results
    )


def main():
    data = load_data()

    split = int(
        len(data)
        * TRAIN_FRACTION
    )

    train = data.iloc[
        :split
    ]

    buy_train = train[
        "buy_count"
    ].to_numpy(
        dtype=float
    )

    sell_train = train[
        "sell_count"
    ].to_numpy(
        dtype=float
    )

    print(
        f"Total bins: "
        f"{len(data):,}"
    )

    print(
        f"Training bins: "
        f"{split:,}"
    )

    print(
        f"Test bins: "
        f"{len(data) - split:,}"
    )

    parameters = fit_hawkes(
        buy_train,
        sell_train,
    )

    (
        mu_buy,
        mu_sell,
        beta,
        branching_buy,
        branching_sell,
    ) = parameters

    print()
    print(
        "Hawkes parameters:"
    )

    print(
        f"mu_buy: "
        f"{mu_buy:.8f}"
    )

    print(
        f"mu_sell: "
        f"{mu_sell:.8f}"
    )

    print(
        f"beta: "
        f"{beta:.8f}"
    )

    print(
        f"branching_buy: "
        f"{branching_buy:.8f}"
    )

    print(
        f"branching_sell: "
        f"{branching_sell:.8f}"
    )

    buy_intensity, sell_intensity = (
        generate_intensity(
            data[
                "buy_count"
            ].to_numpy(
                dtype=float
            ),
            data[
                "sell_count"
            ].to_numpy(
                dtype=float
            ),
            parameters,
        )
    )

    data[
        "hawkes_buy_intensity"
    ] = buy_intensity

    data[
        "hawkes_sell_intensity"
    ] = sell_intensity

    data[
        "hawkes_pressure"
    ] = (
        data[
            "hawkes_buy_intensity"
        ]
        - data[
            "hawkes_sell_intensity"
        ]
    )

    data[
        "hawkes_total_intensity"
    ] = (
        data[
            "hawkes_buy_intensity"
        ]
        + data[
            "hawkes_sell_intensity"
        ]
    )

    print()
    print(
        f"Mean Hawkes pressure: "
        f"{data['hawkes_pressure'].mean():.8f}"
    )

    print(
        f"Std Hawkes pressure: "
        f"{data['hawkes_pressure'].std():.8f}"
    )

    results = run_experiment(
        data,
        split,
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = results[
        [
            "horizon",
            "model",
            "train_observations",
            "test_observations",
            "oos_r2",
            "rmse",
            "prediction_return_correlation",
        ]
    ]

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        results.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()