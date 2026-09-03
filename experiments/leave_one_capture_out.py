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
    "capture_04": Path(
        "data/live/capture_04.jsonl"
    ),
}

BOOK_FILE = Path(
    "data/processed/all_capture_book_states.parquet"
)

OUTPUT_FILE = Path(
    "data/processed/leave_one_capture_out.csv"
)

BIN_SIZE = 0.1
STATIONARITY_LIMIT = 0.999

HORIZONS = {
    "1s": 10,
    "5s": 50,
}

RIDGE_ALPHA = 1.0


def load_trades(path):
    rows = []

    if not path.exists():
        raise FileNotFoundError(
            f"Missing capture file: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            if record.get("type") != "trade":
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
            f"No trade events found in {path}"
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

    return trades


def load_book():
    book = pd.read_parquet(
        BOOK_FILE,
        engine="pyarrow",
    )

    required = [
        "capture_id",
        "event_time_ms",
        "mid_price",
        "queue_imbalance",
        "spread_bps",
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

    book = book[
        required
    ].copy()

    return (
        book
        .sort_values(
            [
                "capture_id",
                "event_time_ms",
            ]
        )
        .reset_index(drop=True)
    )


def build_capture_data(
    trades,
    book,
    capture_id,
):
    local_book = book.loc[
        book["capture_id"] == capture_id
    ].copy()

    if local_book.empty:
        raise RuntimeError(
            f"No book states for {capture_id}"
        )

    step_ms = int(
        round(
            BIN_SIZE * 1000
        )
    )

    start_ms = min(
        int(
            trades[
                "timestamp_ms"
            ].min()
        ),
        int(
            local_book[
                "event_time_ms"
            ].min()
        ),
    )

    trade_data = trades.copy()

    trade_data["bin"] = (
        (
            trade_data[
                "timestamp_ms"
            ]
            - start_ms
        )
        // step_ms
    ).astype(np.int64)

    local_book["bin"] = (
        (
            local_book[
                "event_time_ms"
            ]
            - start_ms
        )
        // step_ms
    ).astype(np.int64)

    trade_counts = (
        trade_data
        .groupby(
            [
                "bin",
                "side",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reindex(
            columns=[
                "buy",
                "sell",
            ],
            fill_value=0,
        )
        .reset_index()
    )

    book_aggregation = {
        "mid_price": "last",
        "queue_imbalance": "last",
        "spread_bps": "last",
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
        "ofi_multilevel": "sum",
        "ofi_normalized": "sum",
        "ofi_depth_weighted": "sum",
    }

    book_data = (
        local_book
        .groupby("bin")
        .agg(book_aggregation)
        .reset_index()
    )

    maximum_bin = max(
        int(
            trade_counts[
                "bin"
            ].max()
        ),
        int(
            book_data[
                "bin"
            ].max()
        ),
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
            trade_counts,
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

    state_columns = [
        "mid_price",
        "queue_imbalance",
        "spread_bps",
    ]

    data[state_columns] = (
        data[state_columns]
        .ffill()
    )

    flow_columns = [
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
    ]

    data[flow_columns] = (
        data[flow_columns]
        .fillna(0.0)
    )

    data = data.dropna(
        subset=[
            "mid_price",
            "queue_imbalance",
            "spread_bps",
        ]
    ).reset_index(
        drop=True
    )

    data["timestamp"] = pd.to_datetime(
        start_ms
        + data["bin"] * step_ms,
        unit="ms",
        utc=True,
    )

    data["relative_time_s"] = (
        data["bin"]
        * BIN_SIZE
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


def hawkes_nll(
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

    scale = (
        1.0 - decay
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

    value = (
        poisson.logpmf(
            buy.astype(np.int64),
            mean_buy,
        ).sum()
        + poisson.logpmf(
            sell.astype(np.int64),
            mean_sell,
        ).sum()
    )

    if not np.isfinite(value):
        return 1e100

    return -value


def fit_hawkes_from_multiple_captures(
    training_data,
):
    buy_arrays = []
    sell_arrays = []

    for frame in training_data:
        buy_arrays.append(
            frame[
                "buy"
            ].to_numpy(
                dtype=float
            )
        )

        sell_arrays.append(
            frame[
                "sell"
            ].to_numpy(
                dtype=float
            )
        )

    buy_rates = np.array(
        [
            values.mean() / BIN_SIZE
            for values in buy_arrays
        ]
    )

    sell_rates = np.array(
        [
            values.mean() / BIN_SIZE
            for values in sell_arrays
        ]
    )

    buy_rate = (
        buy_rates.mean()
    )

    sell_rate = (
        sell_rates.mean()
    )

    def objective(x):
        total = 0.0

        for buy, sell in zip(
            buy_arrays,
            sell_arrays,
        ):
            value = hawkes_nll(
                x,
                buy,
                sell,
            )

            if not np.isfinite(value):
                return 1e100

            total += value

        return total

    starts = []

    for beta in [
        1.0,
        3.0,
        5.0,
        10.0,
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
            objective,
            x0,
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
            "Multi-capture Hawkes "
            "optimization failed."
        )

    return hawkes_parameters(
        best.x
    )


def add_hawkes_pressure(
    data,
    parameters,
):
    buy = data[
        "buy"
    ].to_numpy(
        dtype=float
    )

    sell = data[
        "sell"
    ].to_numpy(
        dtype=float
    )

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

    scale = (
        1.0 - decay
    )

    buy_intensity = (
        mu_buy
        + branching_buy
        * scale
        * buy_state
        / BIN_SIZE
    )

    sell_intensity = (
        mu_sell
        + branching_sell
        * scale
        * sell_state
        / BIN_SIZE
    )

    result = data.copy()

    result[
        "hawkes_buy_intensity"
    ] = buy_intensity

    result[
        "hawkes_sell_intensity"
    ] = sell_intensity

    result[
        "hawkes_pressure"
    ] = (
        buy_intensity
        - sell_intensity
    )

    return result


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
        dtype=float,
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

    valid_indices = np.flatnonzero(
        valid
    )

    result[
        valid_indices
    ] = (
        np.log(
            end[valid]
        )
        - np.log(
            start[valid]
        )
    )

    return result


def standardize_train_test(
    train_x,
    test_x,
):
    mean = train_x.mean(
        axis=0
    )

    std = train_x.std(
        axis=0,
        ddof=1,
    )

    std = np.where(
        std > 0,
        std,
        1.0,
    )

    train_scaled = (
        train_x - mean
    ) / std

    test_scaled = (
        test_x - mean
    ) / std

    return (
        train_scaled,
        test_scaled,
    )


def fit_ols(
    x,
    y,
):
    design = np.column_stack(
        [
            np.ones(
                len(x)
            ),
            x,
        ]
    )

    coefficients = np.linalg.lstsq(
        design,
        y,
        rcond=None,
    )[0]

    return coefficients


def fit_ridge_closed_form(
    x,
    y,
    alpha,
):
    design = np.column_stack(
        [
            np.ones(
                len(x)
            ),
            x,
        ]
    )

    penalty = np.eye(
        design.shape[1]
    )

    # Never penalize the intercept.
    penalty[0, 0] = 0.0

    matrix = (
        design.T @ design
        + alpha * penalty
    )

    coefficients = np.linalg.solve(
        matrix,
        design.T @ y,
    )

    return coefficients


def predict(
    coefficients,
    x,
):
    design = np.column_stack(
        [
            np.ones(
                len(x)
            ),
            x,
        ]
    )

    return (
        design
        @ coefficients
    )


def metrics(
    prediction,
    actual,
    benchmark,
):
    residual = (
        actual
        - prediction
    )

    benchmark_residual = (
        actual
        - benchmark
    )

    sse = np.sum(
        residual ** 2
    )

    benchmark_sse = np.sum(
        benchmark_residual ** 2
    )

    if benchmark_sse <= 0:
        return (
            np.nan,
            np.nan,
            np.nan,
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
        np.std(prediction) <= 0
        or np.std(actual) <= 0
    ):
        correlation = np.nan
    else:
        correlation = np.corrcoef(
            prediction,
            actual,
        )[0, 1]

    return (
        oos_r2,
        rmse,
        correlation,
    )


def evaluate_fold(
    training_data,
    test_data,
    train_ids,
    test_id,
):
    parameters = (
        fit_hawkes_from_multiple_captures(
            training_data
        )
    )

    processed_train = [
        add_hawkes_pressure(
            frame,
            parameters,
        )
        for frame in training_data
    ]

    processed_test = (
        add_hawkes_pressure(
            test_data,
            parameters,
        )
    )

    print()
    print(
        "========================================"
    )

    print(
        f"Train captures: "
        f"{', '.join(train_ids)}"
    )

    print(
        f"Test capture:   "
        f"{test_id}"
    )

    print(
        f"mu_buy: "
        f"{parameters[0]:.8f}"
    )

    print(
        f"mu_sell: "
        f"{parameters[1]:.8f}"
    )

    print(
        f"beta: "
        f"{parameters[2]:.8f}"
    )

    print(
        f"branching_buy: "
        f"{parameters[3]:.8f}"
    )

    print(
        f"branching_sell: "
        f"{parameters[4]:.8f}"
    )

    rows = []

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
        train_rows = []

        for capture_frame in processed_train:
            target = future_return(
                capture_frame[
                    "mid_price"
                ],
                horizon_bins,
            )

            for model_name, columns in (
                models.items()
            ):
                feature_values = (
                    capture_frame[
                        columns
                    ].to_numpy(
                        dtype=float
                    )
                )

                valid = (
                    np.isfinite(
                        feature_values
                    ).all(axis=1)
                    & np.isfinite(
                        target
                    )
                )

                train_rows.append(
                    {
                        "model": model_name,
                        "x": feature_values[
                            valid
                        ],
                        "y": target[
                            valid
                        ],
                    }
                )

        test_target = future_return(
            processed_test[
                "mid_price"
            ],
            horizon_bins,
        )

        for model_name, columns in (
            models.items()
        ):
            train_feature_parts = []
            train_target_parts = []

            for row in train_rows:
                if row["model"] != model_name:
                    continue

                train_feature_parts.append(
                    row["x"]
                )

                train_target_parts.append(
                    row["y"]
                )

            x_train = np.vstack(
                train_feature_parts
            )

            y_train = np.concatenate(
                train_target_parts
            )

            x_test = processed_test[
                columns
            ].to_numpy(
                dtype=float
            )

            test_valid = (
                np.isfinite(
                    x_test
                ).all(axis=1)
                & np.isfinite(
                    test_target
                )
            )

            x_test = x_test[
                test_valid
            ]

            y_test = test_target[
                test_valid
            ]

            if (
                len(x_train) < 100
                or len(x_test) < 100
            ):
                raise RuntimeError(
                    f"Insufficient observations "
                    f"for {model_name}, "
                    f"{horizon_name}."
                )

            x_train_scaled, x_test_scaled = (
                standardize_train_test(
                    x_train,
                    x_test,
                )
            )

            # Use OLS for the single-variable models.
            # Use ridge for multivariate models to reduce
            # coefficient instability across regimes.
            if x_train_scaled.shape[1] == 1:
                coefficients = fit_ols(
                    x_train_scaled,
                    y_train,
                )
            else:
                coefficients = (
                    fit_ridge_closed_form(
                        x_train_scaled,
                        y_train,
                        RIDGE_ALPHA,
                    )
                )

            prediction = predict(
                coefficients,
                x_test_scaled,
            )

            benchmark = np.full(
                len(y_test),
                y_train.mean(),
            )

            (
                oos_r2,
                rmse,
                correlation,
            ) = metrics(
                prediction,
                y_test,
                benchmark,
            )

            rows.append(
                {
                    "train_captures": (
                        "+".join(train_ids)
                    ),
                    "test_capture": test_id,
                    "horizon": horizon_name,
                    "model": model_name,
                    "train_observations": len(
                        y_train
                    ),
                    "test_observations": len(
                        y_test
                    ),
                    "oos_r2": oos_r2,
                    "rmse": rmse,
                    "prediction_return_correlation": (
                        correlation
                    ),
                    "hawkes_mu_buy": parameters[0],
                    "hawkes_mu_sell": parameters[1],
                    "hawkes_beta": parameters[2],
                    "branching_buy": parameters[3],
                    "branching_sell": parameters[4],
                }
            )

            print(
                f"{horizon_name} | "
                f"{model_name}: "
                f"R2={oos_r2:.6f} | "
                f"RMSE={rmse:.8f} | "
                f"corr={correlation:.6f}"
            )

    return rows


def main():
    print(
        "Loading book states..."
    )

    book = load_book()

    trades = {}

    for capture_id, path in (
        CAPTURE_FILES.items()
    ):
        print(
            f"Loading {capture_id}..."
        )

        trades[capture_id] = (
            load_trades(path)
        )

    data = {}

    for capture_id in CAPTURE_FILES:
        print(
            f"Building {capture_id}..."
        )

        data[capture_id] = (
            build_capture_data(
                trades[capture_id],
                book,
                capture_id,
            )
        )

        print(
            f"{capture_id}: "
            f"{len(data[capture_id]):,} bins"
        )

    capture_ids = list(
        CAPTURE_FILES.keys()
    )

    all_results = []

    for held_out in capture_ids:
        training_ids = [
            capture_id
            for capture_id in capture_ids
            if capture_id != held_out
        ]

        training_data = [
            data[capture_id]
            for capture_id in training_ids
        ]

        test_data = data[
            held_out
        ]

        fold_results = evaluate_fold(
            training_data,
            test_data,
            training_ids,
            held_out,
        )

        all_results.extend(
            fold_results
        )

    results = pd.DataFrame(
        all_results
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
        "========================================"
    )

    print(
        "LEAVE-ONE-CAPTURE-OUT RESULTS"
    )

    print(
        results[
            [
                "train_captures",
                "test_capture",
                "horizon",
                "model",
                "train_observations",
                "test_observations",
                "oos_r2",
                "rmse",
                "prediction_return_correlation",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Average across held-out captures:"
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
        .agg(
            [
                "mean",
                "std",
            ]
        )
    )

    print(
        summary.to_string()
    )

    print()
    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()