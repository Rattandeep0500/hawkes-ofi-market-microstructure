import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import expit
from scipy.stats import poisson


CAPTURE_FILES = {
    "capture_02": Path(
        "data/live/capture_02.jsonl"
    ),
    "capture_03": Path(
        "data/live/capture_03.jsonl"
    ),
}

BOOK_FILE = Path(
    "data/processed/all_capture_book_states.parquet"
)

OUTPUT_FILE = Path(
    "data/processed/cross_capture_hawkes_ofi.csv"
)

BIN_SIZE = 0.1
STATIONARITY_LIMIT = 0.999
HORIZONS = {
    "1s": 10,
    "5s": 50,
}


def load_trade_capture(
    capture_id,
    path,
):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            if record["type"] != "trade":
                continue

            event = record["data"]

            rows.append(
                {
                    "timestamp_ms": int(
                        event["T"]
                    ),
                    "trade_id": int(
                        event["t"]
                    ),
                    "side": (
                        "sell"
                        if bool(event["m"])
                        else "buy"
                    ),
                }
            )

    if not rows:
        raise RuntimeError(
            f"No trades found in {capture_id}."
        )

    trades = pd.DataFrame(rows)

    trades = (
        trades
        .drop_duplicates("trade_id")
        .sort_values(
            [
                "timestamp_ms",
                "trade_id",
            ]
        )
        .reset_index(drop=True)
    )

    trades["capture_id"] = capture_id

    return trades


def load_book_states():
    book = pd.read_parquet(
        BOOK_FILE,
        engine="pyarrow",
    )

    required = [
        "capture_id",
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
        "ofi_normalized",
        "ofi_depth_weighted",
    ]

    missing = [
        column
        for column in required
        if column not in book.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing book columns: {missing}"
        )

    return (
        book[required]
        .sort_values(
            [
                "capture_id",
                "event_time_ms",
            ]
        )
        .reset_index(drop=True)
    )


def aggregate_trades(
    trades,
    bin_size,
):
    data = trades.copy()

    step_ms = int(
        round(
            bin_size * 1000
        )
    )

    start_ms = (
        data["timestamp_ms"].min()
    )

    relative_ms = (
        data["timestamp_ms"]
        - start_ms
    )

    data["bin"] = (
        relative_ms
        // step_ms
    ).astype(np.int64)

    grouped = (
        data
        .groupby(
            [
                "capture_id",
                "bin",
                "side",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reset_index()
    )

    if "buy" not in grouped.columns:
        grouped["buy"] = 0

    if "sell" not in grouped.columns:
        grouped["sell"] = 0

    grouped["buy"] = (
        grouped["buy"]
        .astype(np.int64)
    )

    grouped["sell"] = (
        grouped["sell"]
        .astype(np.int64)
    )

    return grouped[
        [
            "capture_id",
            "bin",
            "buy",
            "sell",
        ]
    ]


def aggregate_book(
    book,
    capture_id,
    bin_size,
):
    data = book.loc[
        book["capture_id"] == capture_id
    ].copy()

    if data.empty:
        raise RuntimeError(
            f"No book states for {capture_id}."
        )

    start_ms = (
        data["event_time_ms"].min()
    )

    step_ms = int(
        round(
            bin_size * 1000
        )
    )

    relative_ms = (
        data["event_time_ms"]
        - start_ms
    )

    data["bin"] = (
        relative_ms
        // step_ms
    ).astype(np.int64)

    aggregation = {
        "mid_price": "last",
        "ofi_1": "sum",
        "ofi_2": "sum",
        "ofi_3": "sum",
        "ofi_4": "sum",
        "ofi_5": "sum",
        "ofi_6": "sum",
        "ofi_7": "sum",
        "ofi_8": "sum",
        "ofi_9": "sum",
        "ofi_10": "sum",
        "ofi_normalized": "sum",
        "ofi_depth_weighted": "sum",
    }

    grouped = (
        data
        .groupby("bin")
        .agg(aggregation)
        .reset_index()
    )

    return grouped


def build_capture_dataset(
    trades,
    book,
    capture_id,
    bin_size,
):
    trade_data = aggregate_trades(
        trades,
        bin_size,
    )

    trade_data = trade_data.loc[
        trade_data["capture_id"]
        == capture_id
    ].copy()

    trade_data = trade_data[
        [
            "bin",
            "buy",
            "sell",
        ]
    ]

    book_data = aggregate_book(
        book,
        capture_id,
        bin_size,
    )

    maximum_bin = max(
        trade_data["bin"].max(),
        book_data["bin"].max(),
    )

    grid = pd.DataFrame(
        {
            "bin": np.arange(
                maximum_bin + 1,
                dtype=np.int64,
            )
        }
    )

    data = (
        grid
        .merge(
            trade_data,
            on="bin",
            how="left",
        )
        .merge(
            book_data,
            on="bin",
            how="left",
        )
        .sort_values("bin")
        .reset_index(drop=True)
    )

    data["buy"] = (
        data["buy"]
        .fillna(0)
        .astype(np.int64)
    )

    data["sell"] = (
        data["sell"]
        .fillna(0)
        .astype(np.int64)
    )

    data["mid_price"] = (
        data["mid_price"]
        .ffill()
    )

    ofi_columns = [
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

    data[ofi_columns] = (
        data[ofi_columns]
        .fillna(0.0)
    )

    data = data.dropna(
        subset=["mid_price"]
    ).reset_index(
        drop=True
    )

    return data


def hawkes_parameters(x):
    mu_buy = np.exp(
        np.clip(
            x[0],
            -20.0,
            20.0,
        )
    )

    mu_sell = np.exp(
        np.clip(
            x[1],
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


def states(
    buy,
    sell,
    beta,
    bin_size,
):
    decay = np.exp(
        -beta * bin_size
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

    if len(sell) > 1:
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
    bin_size,
):
    (
        mu_buy,
        mu_sell,
        beta,
        branching_buy,
        branching_sell,
    ) = hawkes_parameters(x)

    buy_state, sell_state, decay = states(
        buy,
        sell,
        beta,
        bin_size,
    )

    scale = (
        1.0 - decay
    )

    mean_buy = (
        mu_buy * bin_size
        + branching_buy
        * scale
        * buy_state
    )

    mean_sell = (
        mu_sell * bin_size
        + branching_sell
        * scale
        * sell_state
    )

    if (
        not np.isfinite(
            mean_buy
        ).all()
        or not np.isfinite(
            mean_sell
        ).all()
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
    data,
    bin_size,
):
    buy = data[
        "buy"
    ].to_numpy(dtype=float)

    sell = data[
        "sell"
    ].to_numpy(dtype=float)

    buy_rate = (
        buy.mean()
        / bin_size
    )

    sell_rate = (
        sell.mean()
        / bin_size
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
                                buy_rate
                                * 0.5,
                                1e-6,
                            )
                        ),
                        np.log(
                            max(
                                sell_rate
                                * 0.5,
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
                bin_size,
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


def generate_pressure(
    data,
    parameters,
    bin_size,
):
    buy = data[
        "buy"
    ].to_numpy(dtype=float)

    sell = data[
        "sell"
    ].to_numpy(dtype=float)

    (
        mu_buy,
        mu_sell,
        beta,
        branching_buy,
        branching_sell,
    ) = parameters

    buy_state, sell_state, decay = states(
        buy,
        sell,
        beta,
        bin_size,
    )

    scale = (
        1.0 - decay
    )

    buy_intensity = (
        mu_buy
        + branching_buy
        * scale
        * buy_state
        / bin_size
    )

    sell_intensity = (
        mu_sell
        + branching_sell
        * scale
        * sell_state
        / bin_size
    )

    data = data.copy()

    data["hawkes_buy_intensity"] = (
        buy_intensity
    )

    data["hawkes_sell_intensity"] = (
        sell_intensity
    )

    data["hawkes_pressure"] = (
        buy_intensity
        - sell_intensity
    )

    return data


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

    start = values[
        :-horizon_bins
    ]

    end = values[
        horizon_bins:
    ]

    valid = (
        (start > 0)
        & (end > 0)
    )

    result[
        :-horizon_bins
    ][valid] = (
        np.log(
            end[valid]
        )
        - np.log(
            start[valid]
        )
    )

    return result


def evaluate_model(
    train,
    test,
    feature_columns,
    horizon_bins,
):
    train = train.copy()
    test = test.copy()

    train_target = future_returns(
        train["mid_price"],
        horizon_bins,
    )

    test_target = future_returns(
        test["mid_price"],
        horizon_bins,
    )

    x_train = train[
        feature_columns
    ].to_numpy(dtype=float)

    y_train = train_target

    x_test = test[
        feature_columns
    ].to_numpy(dtype=float)

    y_test = test_target

    train_valid = (
        np.isfinite(
            x_train
        ).all(axis=1)
        & np.isfinite(y_train)
    )

    test_valid = (
        np.isfinite(
            x_test
        ).all(axis=1)
        & np.isfinite(y_test)
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

    if (
        len(x_train) < 30
        or len(x_test) < 30
    ):
        raise RuntimeError(
            "Insufficient observations "
            "for OOS evaluation."
        )

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

    prediction = (
        x_test_design
        @ coefficients
    )

    benchmark = np.full(
        len(y_test),
        y_train.mean(),
    )

    residual = (
        y_test
        - prediction
    )

    benchmark_residual = (
        y_test
        - benchmark
    )

    sse = np.sum(
        residual ** 2
    )

    benchmark_sse = np.sum(
        benchmark_residual ** 2
    )

    oos_r2 = (
        1.0
        - sse
        / benchmark_sse
    )

    rmse = np.sqrt(
        np.mean(
            residual ** 2
        )
    )

    if (
        np.std(prediction) == 0
        or np.std(y_test) == 0
    ):
        correlation = np.nan
    else:
        correlation = np.corrcoef(
            prediction,
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
    trades = {}

    for capture_id, path in (
        CAPTURE_FILES.items()
    ):
        trades[capture_id] = (
            load_trade_capture(
                capture_id,
                path,
            )
        )

    book = load_book_states()

    results = []

    for train_id, test_id in [
        (
            "capture_02",
            "capture_03",
        ),
        (
            "capture_03",
            "capture_02",
        ),
    ]:

        print()
        print(
            f"TRAIN: {train_id}"
        )

        print(
            f"TEST:  {test_id}"
        )

        for bin_size in [
            BIN_SIZE
        ]:

            train_data = (
                build_capture_dataset(
                    trades[train_id],
                    book,
                    train_id,
                    bin_size,
                )
            )

            test_data = (
                build_capture_dataset(
                    trades[test_id],
                    book,
                    test_id,
                    bin_size,
                )
            )

            parameters = fit_hawkes(
                train_data,
                bin_size,
            )

            train_data = generate_pressure(
                train_data,
                parameters,
                bin_size,
            )

            test_data = generate_pressure(
                test_data,
                parameters,
                bin_size,
            )

            (
                mu_buy,
                mu_sell,
                beta,
                branching_buy,
                branching_sell,
            ) = parameters

            print(
                f"mu_buy={mu_buy:.8f}"
            )

            print(
                f"mu_sell={mu_sell:.8f}"
            )

            print(
                f"beta={beta:.8f}"
            )

            print(
                f"branching_buy="
                f"{branching_buy:.8f}"
            )

            print(
                f"branching_sell="
                f"{branching_sell:.8f}"
            )

            models = {
                "OFI_L1": [
                    "ofi_1",
                ],
                "OFI_L10": [
                    f"ofi_{i}"
                    for i in range(1, 11)
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

            for horizon_name, horizon_bins in (
                HORIZONS.items()
            ):
                for model_name, columns in (
                    models.items()
                ):
                    metrics = evaluate_model(
                        train_data,
                        test_data,
                        columns,
                        horizon_bins,
                    )

                    results.append(
                        {
                            "train_capture": train_id,
                            "test_capture": test_id,
                            "bin_size_seconds": bin_size,
                            "horizon": horizon_name,
                            "model": model_name,
                            "train_observations": metrics[
                                "train_observations"
                            ],
                            "test_observations": metrics[
                                "test_observations"
                            ],
                            "oos_r2": metrics[
                                "oos_r2"
                            ],
                            "rmse": metrics[
                                "rmse"
                            ],
                            "prediction_return_correlation": metrics[
                                "prediction_return_correlation"
                            ],
                            "mu_buy": mu_buy,
                            "mu_sell": mu_sell,
                            "beta": beta,
                            "branching_buy": branching_buy,
                            "branching_sell": branching_sell,
                        }
                    )

                    print(
                        f"{horizon_name} | "
                        f"{model_name}: "
                        f"R2="
                        f"{metrics['oos_r2']:.6f} | "
                        f"RMSE="
                        f"{metrics['rmse']:.8f} | "
                        f"corr="
                        f"{metrics['prediction_return_correlation']:.6f}"
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
        "Cross-capture results:"
    )

    print(
        results.to_string(
            index=False
        )
    )

    print()
    print(
        "Average by direction:"
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