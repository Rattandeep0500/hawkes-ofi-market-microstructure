from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import poisson


TRADE_FILE = Path(
    "data/processed/trade_events.parquet"
)

OFI_FILE = Path(
    "data/live/btc_usdt_multi_level_ofi.parquet"
)

OUTPUT_FILE = Path(
    "data/processed/hawkes_ofi_price_model.csv"
)

BIN_SIZE = 0.1
TRAIN_FRACTION = 0.70
STATIONARITY_LIMIT = 0.999


def load_trade_data():
    data = pd.read_parquet(
        TRADE_FILE,
        columns=[
            "event_time_s",
            "side",
        ],
        engine="pyarrow",
    )

    data = (
        data
        .sort_values("event_time_s")
        .reset_index(drop=True)
    )

    data["time_bin"] = (
        np.floor(
            data["event_time_s"]
            / BIN_SIZE
        )
        .astype(np.int64)
    )

    counts = (
        data
        .pivot_table(
            index="time_bin",
            columns="side",
            values="event_time_s",
            aggfunc="size",
            fill_value=0,
        )
        .reindex(
            columns=["buy", "sell"],
            fill_value=0,
        )
    )

    counts["buy"] = counts[
        "buy"
    ].astype(float)

    counts["sell"] = counts[
        "sell"
    ].astype(float)

    return counts


def load_ofi_data():
    data = pd.read_parquet(
        OFI_FILE,
        columns=[
            "event_time_ms",
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
        ],
        engine="pyarrow",
    )

    data = (
        data
        .sort_values("event_time_ms")
        .reset_index(drop=True)
    )

    first_time = (
        data["event_time_ms"].iloc[0]
    )

    data["relative_time_s"] = (
        data["event_time_ms"]
        - first_time
    ) / 1000.0

    data["time_bin"] = (
        np.floor(
            data["relative_time_s"]
            / BIN_SIZE
        )
        .astype(np.int64)
    )

    ofi_columns = [
        f"ofi_{i}"
        for i in range(1, 11)
    ]

    aggregation = {
        "mid_price": "last",
        "ofi_1": "sum",
        "ofi_multilevel": "sum",
        "ofi_normalized": "sum",
        "ofi_depth_weighted": "sum",
    }

    for column in ofi_columns:
        aggregation[column] = "sum"

    result = (
        data
        .groupby("time_bin")
        .agg(aggregation)
    )

    return result


def fit_parameters(counts):
    train_length = int(
        len(counts)
        * TRAIN_FRACTION
    )

    train = counts.iloc[
        :train_length
    ].to_numpy(dtype=float)

    mean_rates = (
        train.mean(axis=0)
        / BIN_SIZE
    )

    def unpack(x):
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

        n_buy = (
            STATIONARITY_LIMIT
            * expit(x[3])
        )

        n_sell = (
            STATIONARITY_LIMIT
            * expit(x[4])
        )

        return (
            mu,
            beta,
            n_buy,
            n_sell,
        )

    def objective(x):
        mu, beta, n_buy, n_sell = (
            unpack(x)
        )

        decay = np.exp(
            -beta * BIN_SIZE
        )

        scale = (
            1.0 - decay
        )

        state_buy = 0.0
        state_sell = 0.0

        log_likelihood = 0.0

        for i in range(
            len(train)
        ):
            mean_buy = (
                mu[0] * BIN_SIZE
                + n_buy
                * scale
                * state_buy
            )

            mean_sell = (
                mu[1] * BIN_SIZE
                + n_sell
                * scale
                * state_sell
            )

            if (
                mean_buy <= 0
                or mean_sell <= 0
            ):
                return 1e100

            log_likelihood += (
                poisson.logpmf(
                    int(train[i, 0]),
                    mean_buy,
                )
            )

            log_likelihood += (
                poisson.logpmf(
                    int(train[i, 1]),
                    mean_sell,
                )
            )

            state_buy = (
                decay * state_buy
                + train[i, 0]
            )

            state_sell = (
                decay * state_sell
                + train[i, 1]
            )

        if not np.isfinite(
            log_likelihood
        ):
            return 1e100

        return -log_likelihood

    starts = [
        [
            np.log(max(mean_rates[0] * 0.5, 1e-6)),
            np.log(max(mean_rates[1] * 0.5, 1e-6)),
            np.log(1.0),
            np.log(0.30 / (STATIONARITY_LIMIT - 0.30)),
            np.log(0.30 / (STATIONARITY_LIMIT - 0.30)),
        ],
        [
            np.log(max(mean_rates[0] * 0.5, 1e-6)),
            np.log(max(mean_rates[1] * 0.5, 1e-6)),
            np.log(5.0),
            np.log(0.50 / (STATIONARITY_LIMIT - 0.50)),
            np.log(0.50 / (STATIONARITY_LIMIT - 0.50)),
        ],
        [
            np.log(max(mean_rates[0] * 0.5, 1e-6)),
            np.log(max(mean_rates[1] * 0.5, 1e-6)),
            np.log(10.0),
            np.log(0.70 / (STATIONARITY_LIMIT - 0.70)),
            np.log(0.70 / (STATIONARITY_LIMIT - 0.70)),
        ],
        [
            np.log(max(mean_rates[0] * 0.5, 1e-6)),
            np.log(max(mean_rates[1] * 0.5, 1e-6)),
            np.log(25.0),
            np.log(0.15 / (STATIONARITY_LIMIT - 0.15)),
            np.log(0.15 / (STATIONARITY_LIMIT - 0.15)),
        ],
    ]

    results = []

    for x0 in starts:
        fit = minimize(
            objective,
            np.asarray(
                x0,
                dtype=float,
            ),
            method="L-BFGS-B",
            options={
                "maxiter": 3000,
                "ftol": 1e-12,
                "gtol": 1e-8,
                "maxls": 50,
            },
        )

        if np.isfinite(
            fit.fun
        ):
            results.append(
                (
                    fit.fun,
                    fit,
                )
            )

    if not results:
        raise RuntimeError(
            "Training Hawkes optimization failed."
        )

    _, best = min(
        results,
        key=lambda x: x[0],
    )

    return (
        unpack(best.x),
        train_length,
    )


def build_hawkes_intensity(
    counts,
    parameters,
):
    mu, beta, n_buy, n_sell = (
        parameters
    )

    decay = np.exp(
        -beta * BIN_SIZE
    )

    scale = (
        1.0 - decay
    )

    intensity_buy = np.zeros(
        len(counts)
    )

    intensity_sell = np.zeros(
        len(counts)
    )

    state_buy = 0.0
    state_sell = 0.0

    for i in range(
        len(counts)
    ):
        expected_buy = (
            mu[0] * BIN_SIZE
            + n_buy
            * scale
            * state_buy
        )

        expected_sell = (
            mu[1] * BIN_SIZE
            + n_sell
            * scale
            * state_sell
        )

        intensity_buy[i] = (
            expected_buy
            / BIN_SIZE
        )

        intensity_sell[i] = (
            expected_sell
            / BIN_SIZE
        )

        state_buy = (
            decay * state_buy
            + counts[i, 0]
        )

        state_sell = (
            decay * state_sell
            + counts[i, 1]
        )

    return (
        intensity_buy,
        intensity_sell,
    )


def build_dataset():
    trades = load_trade_data()

    ofi = load_ofi_data()

    data = trades.join(
        ofi,
        how="outer",
    ).sort_index()

    data = data.fillna(0.0)

    data["time_seconds"] = (
        data.index
        * BIN_SIZE
    )

    data["hawkes_buy_intensity"] = np.nan
    data["hawkes_sell_intensity"] = np.nan

    counts = data[
        ["buy", "sell"]
    ].to_numpy(dtype=float)

    parameters, train_length = (
        fit_parameters(
            data[
                ["buy", "sell"]
            ]
        )
    )

    intensity_buy, intensity_sell = (
        build_hawkes_intensity(
            counts,
            parameters,
        )
    )

    data[
        "hawkes_buy_intensity"
    ] = intensity_buy

    data[
        "hawkes_sell_intensity"
    ] = intensity_sell

    data["hawkes_pressure"] = (
        data[
            "hawkes_buy_intensity"
        ]
        - data[
            "hawkes_sell_intensity"
        ]
    )

    data["signed_trade_pressure"] = (
        data["buy"]
        - data["sell"]
    )

    data["depth_normalized_ofi"] = (
        data["ofi_normalized"]
    )

    return (
        data,
        parameters,
        train_length,
    )


def future_return(
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

    for i in range(
        len(values)
    ):
        j = (
            i
            + horizon_bins
        )

        if j >= len(values):
            continue

        if (
            values[i] <= 0
            or values[j] <= 0
        ):
            continue

        result[i] = (
            np.log(values[j])
            - np.log(values[i])
        )

    return result


def regression_oos(
    x,
    y,
    train_length,
):
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    split = min(
        train_length,
        len(x),
    )

    if split < 20:
        raise RuntimeError(
            "Too few training observations."
        )

    if len(x) - split < 20:
        raise RuntimeError(
            "Too few test observations."
        )

    x_train = x[:split]
    y_train = y[:split]

    x_test = x[split:]
    y_test = y[split:]

    x_train = np.column_stack(
        [
            np.ones(len(x_train)),
            x_train,
        ]
    )

    beta = np.linalg.lstsq(
        x_train,
        y_train,
        rcond=None,
    )[0]

    x_test_design = np.column_stack(
        [
            np.ones(len(x_test)),
            x_test,
        ]
    )

    prediction = (
        x_test_design
        @ beta
    )

    residual = (
        y_test
        - prediction
    )

    benchmark = np.full(
        len(y_test),
        y_train.mean(),
    )

    sse = np.sum(
        residual ** 2
    )

    sse_benchmark = np.sum(
        (y_test - benchmark) ** 2
    )

    r2 = (
        1.0
        - sse
        / sse_benchmark
    )

    rmse = np.sqrt(
        np.mean(
            residual ** 2
        )
    )

    correlation = np.corrcoef(
        prediction,
        y_test,
    )[0, 1]

    return (
        r2,
        rmse,
        correlation,
    )


def run_model_comparison(
    data,
    train_length,
):
    data = data.copy()

    results = []

    horizons = {
        "1s": 10,
        "5s": 50,
    }

    models = {
        "OFI_L1": [
            "ofi_1",
        ],
        "OFI_L10": [
            *[
                f"ofi_{i}"
                for i in range(1, 11)
            ],
        ],
        "OFI_normalized": [
            "depth_normalized_ofi",
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

    for horizon_name, horizon_bins in (
        horizons.items()
    ):
        data[
            "future_return"
        ] = future_return(
            data["mid_price"],
            horizon_bins,
        )

        valid = (
            data["mid_price"] > 0
        )

        data_h = data.loc[
            valid
        ].copy()

        y = data_h[
            "future_return"
        ].to_numpy(
            dtype=float
        )

        for model_name, columns in (
            models.items()
        ):
            x = data_h[
                columns
            ].to_numpy(
                dtype=float
            )

            valid_rows = (
                np.isfinite(x).all(axis=1)
                & np.isfinite(y)
            )

            x = x[valid_rows]
            y_model = y[valid_rows]

            split = int(
                len(x)
                * TRAIN_FRACTION
            )

            x_train = x[:split]
            y_train = y_model[:split]

            x_test = x[split:]
            y_test = y_model[split:]

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

            residuals = (
                y_test
                - predictions
            )

            benchmark = np.full(
                len(y_test),
                y_train.mean(),
            )

            sse = np.sum(
                residuals ** 2
            )

            sse_benchmark = np.sum(
                (y_test - benchmark) ** 2
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

            correlation = np.corrcoef(
                predictions,
                y_test,
            )[0, 1]

            results.append(
                {
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

    return pd.DataFrame(
        results
    )


def main():
    data, parameters, train_length = (
        build_dataset()
    )

    mu, beta, n_buy, n_sell = (
        parameters
    )

    print(
        f"Training bins: {train_length:,}"
    )

    print(
        f"mu_buy: {mu[0]:.8f}"
    )

    print(
        f"mu_sell: {mu[1]:.8f}"
    )

    print(
        f"beta: {beta:.8f}"
    )

    print(
        f"branching_buy: {n_buy:.8f}"
    )

    print(
        f"branching_sell: {n_sell:.8f}"
    )

    print(
        f"Mean Hawkes pressure: "
        f"{data['hawkes_pressure'].mean():.8f}"
    )

    print(
        f"Std Hawkes pressure: "
        f"{data['hawkes_pressure'].std():.8f}"
    )

    results = run_model_comparison(
        data,
        train_length,
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