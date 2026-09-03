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
    "data/processed/final_statistical_test.csv"
)

BIN_SIZE = 0.1
STATIONARITY_LIMIT = 0.999

HORIZONS = {
    "1s": 10,
    "5s": 50,
}

BLOCK_SIZE = 50
N_BOOTSTRAP = 2000
RANDOM_SEED = 20260903


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
            f"No trades found in {path}"
        )

    trades = pd.DataFrame(rows)

    trades = (
        trades
        .drop_duplicates(
            "trade_id"
        )
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

    data["mid_price"] = (
        data["mid_price"]
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
        "ofi_normalized",
        "ofi_depth_weighted",
    ]

    data[flow_columns] = (
        data[flow_columns]
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


def fit_hawkes(
    training_frames,
):
    buy_arrays = [
        frame[
            "buy"
        ].to_numpy(
            dtype=float
        )
        for frame in training_frames
    ]

    sell_arrays = [
        frame[
            "sell"
        ].to_numpy(
            dtype=float
        )
        for frame in training_frames
    ]

    buy_rate = np.mean(
        [
            values.mean()
            / BIN_SIZE
            for values in buy_arrays
        ]
    )

    sell_rate = np.mean(
        [
            values.mean()
            / BIN_SIZE
            for values in sell_arrays
        ]
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
            "Hawkes fitting failed."
        )

    return hawkes_parameters(
        best.x
    )


def add_hawkes_pressure(
    data,
    parameters,
):
    result = data.copy()

    buy = result[
        "buy"
    ].to_numpy(
        dtype=float
    )

    sell = result[
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

    indices = np.flatnonzero(
        valid
    )

    result[
        indices
    ] = (
        np.log(
            end[valid]
        )
        - np.log(
            start[valid]
        )
    )

    return result


def fit_ols(
    x,
    y,
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

    design = np.column_stack(
        [
            np.ones(
                len(x)
            ),
            x,
        ]
    )

    return np.linalg.lstsq(
        design,
        y,
        rcond=None,
    )[0]


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


def bootstrap_indices(
    n,
    block_size,
    rng,
):
    if n <= 0:
        raise ValueError(
            "n must be positive."
        )

    block_size = min(
        block_size,
        n,
    )

    starts = np.arange(
        0,
        n - block_size + 1,
    )

    indices = []

    while len(indices) < n:
        start = int(
            rng.choice(
                starts
            )
        )

        block = np.arange(
            start,
            min(
                start + block_size,
                n,
            ),
        )

        indices.extend(
            block.tolist()
        )

    return np.asarray(
        indices[:n],
        dtype=np.int64,
    )


def paired_bootstrap_loss_difference(
    hawkes_errors,
    ofi_errors,
    rng,
):
    loss_difference = (
        hawkes_errors ** 2
        - ofi_errors ** 2
    )

    observed = (
        -np.mean(
            loss_difference
        )
    )

    bootstrap_values = np.empty(
        N_BOOTSTRAP,
        dtype=float,
    )

    for i in range(
        N_BOOTSTRAP
    ):
        indices = bootstrap_indices(
            len(loss_difference),
            BLOCK_SIZE,
            rng,
        )

        bootstrap_values[i] = (
            -np.mean(
                loss_difference[
                    indices
                ]
            )
        )

    ci_low = np.percentile(
        bootstrap_values,
        2.5,
    )

    ci_high = np.percentile(
        bootstrap_values,
        97.5,
    )

    probability_positive = (
        np.mean(
            bootstrap_values > 0
        )
    )

    return (
        observed,
        ci_low,
        ci_high,
        probability_positive,
    )


def run_fold(
    train_frames,
    test_frame,
    train_ids,
    test_id,
    rng,
):
    parameters = fit_hawkes(
        train_frames
    )

    processed_train = [
        add_hawkes_pressure(
            frame,
            parameters,
        )
        for frame in train_frames
    ]

    processed_test = (
        add_hawkes_pressure(
            test_frame,
            parameters,
        )
    )

    print()
    print(
        "========================================"
    )

    print(
        f"Train: "
        f"{' + '.join(train_ids)}"
    )

    print(
        f"Test:  {test_id}"
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
        f"hawkes_beta: "
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

    for horizon_name, horizon_bins in (
        HORIZONS.items()
    ):
        train_y_parts = []

        train_hawkes_parts = []
        train_ofi_parts = []

        for frame in processed_train:
            target = future_return(
                frame[
                    "mid_price"
                ],
                horizon_bins,
            )

            train_hawkes_parts.append(
                frame[
                    "hawkes_pressure"
                ].to_numpy(
                    dtype=float
                )
            )

            train_ofi_parts.append(
                frame[
                    "ofi_1"
                ].to_numpy(
                    dtype=float
                )
            )

            train_y_parts.append(
                target
            )

        x_hawkes_train = np.concatenate(
            train_hawkes_parts
        )

        x_ofi_train = np.concatenate(
            train_ofi_parts
        )

        y_train = np.concatenate(
            train_y_parts
        )

        test_y = future_return(
            processed_test[
                "mid_price"
            ],
            horizon_bins,
        )

        x_hawkes_test = (
            processed_test[
                "hawkes_pressure"
            ].to_numpy(
                dtype=float
            )
        )

        x_ofi_test = (
            processed_test[
                "ofi_1"
            ].to_numpy(
                dtype=float
            )
        )

        train_valid_hawkes = (
            np.isfinite(
                x_hawkes_train
            )
            & np.isfinite(y_train)
        )

        train_valid_ofi = (
            np.isfinite(
                x_ofi_train
            )
            & np.isfinite(y_train)
        )

        test_valid = (
            np.isfinite(
                x_hawkes_test
            )
            & np.isfinite(
                x_ofi_test
            )
            & np.isfinite(test_y)
        )

        hawkes_x_train = (
            x_hawkes_train[
                train_valid_hawkes
            ][:, None]
        )

        ofi_x_train = (
            x_ofi_train[
                train_valid_ofi
            ][:, None]
        )

        hawkes_y_train = (
            y_train[
                train_valid_hawkes
            ]
        )

        ofi_y_train = (
            y_train[
                train_valid_ofi
            ]
        )

        hawkes_x_test = (
            x_hawkes_test[
                test_valid
            ][:, None]
        )

        ofi_x_test = (
            x_ofi_test[
                test_valid
            ][:, None]
        )

        y_test = (
            test_y[
                test_valid
            ]
        )

        hawkes_coefficients = fit_ols(
            hawkes_x_train,
            hawkes_y_train,
        )

        ofi_coefficients = fit_ols(
            ofi_x_train,
            ofi_y_train,
        )

        hawkes_prediction = predict(
            hawkes_coefficients,
            hawkes_x_test,
        )

        ofi_prediction = predict(
            ofi_coefficients,
            ofi_x_test,
        )

        benchmark = np.full(
            len(y_test),
            hawkes_y_train.mean(),
        )

        hawkes_error = (
            y_test
            - hawkes_prediction
        )

        ofi_error = (
            y_test
            - ofi_prediction
        )

        benchmark_error = (
            y_test
            - benchmark
        )

        hawkes_sse = np.sum(
            hawkes_error ** 2
        )

        ofi_sse = np.sum(
            ofi_error ** 2
        )

        benchmark_sse = np.sum(
            benchmark_error ** 2
        )

        hawkes_r2 = (
            1.0
            - hawkes_sse
            / benchmark_sse
        )

        ofi_r2 = (
            1.0
            - ofi_sse
            / benchmark_sse
        )

        hawkes_rmse = np.sqrt(
            np.mean(
                hawkes_error ** 2
            )
        )

        ofi_rmse = np.sqrt(
            np.mean(
                ofi_error ** 2
            )
        )

        if (
            np.std(
                hawkes_prediction
            ) > 0
            and np.std(y_test) > 0
        ):
            hawkes_corr = np.corrcoef(
                hawkes_prediction,
                y_test,
            )[0, 1]
        else:
            hawkes_corr = np.nan

        if (
            np.std(
                ofi_prediction
            ) > 0
            and np.std(y_test) > 0
        ):
            ofi_corr = np.corrcoef(
                ofi_prediction,
                y_test,
            )[0, 1]
        else:
            ofi_corr = np.nan

        (
            observed_delta_mse,
            delta_ci_low,
            delta_ci_high,
            delta_positive_probability,
        ) = paired_bootstrap_loss_difference(
            hawkes_error,
            ofi_error,
            rng,
        )

        row = {
            "train_captures": (
                "+".join(train_ids)
            ),
            "test_capture": test_id,
            "horizon": horizon_name,
            "train_observations_hawkes": len(
                hawkes_y_train
            ),
            "train_observations_ofi": len(
                ofi_y_train
            ),
            "test_observations": len(
                y_test
            ),
            "hawkes_oos_r2": hawkes_r2,
            "ofi_oos_r2": ofi_r2,
            "delta_oos_r2": (
                hawkes_r2
                - ofi_r2
            ),
            "hawkes_rmse": hawkes_rmse,
            "ofi_rmse": ofi_rmse,
            "hawkes_correlation": hawkes_corr,
            "ofi_correlation": ofi_corr,
            "hawkes_better_mse": (
                observed_delta_mse
            ),
            "delta_mse_ci_low": (
                delta_ci_low
            ),
            "delta_mse_ci_high": (
                delta_ci_high
            ),
            "probability_hawkes_better": (
                delta_positive_probability
            ),
            "block_size_seconds": (
                BLOCK_SIZE
                * BIN_SIZE
            ),
            "bootstrap_repetitions": (
                N_BOOTSTRAP
            ),
            "hawkes_mu_buy": parameters[0],
            "hawkes_mu_sell": parameters[1],
            "hawkes_beta": parameters[2],
            "branching_buy": parameters[3],
            "branching_sell": parameters[4],
        }

        rows.append(row)

        print()
        print(
            f"{horizon_name}:"
        )

        print(
            f"Hawkes OOS R2: "
            f"{hawkes_r2:.6f}"
        )

        print(
            f"OFI OOS R2: "
            f"{ofi_r2:.6f}"
        )

        print(
            f"Delta OOS R2: "
            f"{hawkes_r2 - ofi_r2:.6f}"
        )

        print(
            f"Hawkes RMSE: "
            f"{hawkes_rmse:.8f}"
        )

        print(
            f"OFI RMSE: "
            f"{ofi_rmse:.8f}"
        )

        print(
            f"Paired MSE improvement: "
            f"{observed_delta_mse:.10e}"
        )

        print(
            f"95% bootstrap CI: "
            f"[{delta_ci_low:.10e}, "
            f"{delta_ci_high:.10e}]"
        )

        print(
            f"P(Hawkes better): "
            f"{delta_positive_probability:.4f}"
        )

    return rows


def main():
    rng = np.random.default_rng(
        RANDOM_SEED
    )

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
        train_ids = [
            capture_id
            for capture_id in capture_ids
            if capture_id != held_out
        ]

        train_frames = [
            data[capture_id]
            for capture_id in train_ids
        ]

        test_frame = data[
            held_out
        ]

        fold_results = run_fold(
            train_frames,
            test_frame,
            train_ids,
            held_out,
            rng,
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
        "FINAL STATISTICAL TEST"
    )

    print(
        results[
            [
                "train_captures",
                "test_capture",
                "horizon",
                "hawkes_oos_r2",
                "ofi_oos_r2",
                "delta_oos_r2",
                "hawkes_rmse",
                "ofi_rmse",
                "hawkes_correlation",
                "ofi_correlation",
                "hawkes_better_mse",
                "delta_mse_ci_low",
                "delta_mse_ci_high",
                "probability_hawkes_better",
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
        .groupby("horizon")[
            [
                "hawkes_oos_r2",
                "ofi_oos_r2",
                "delta_oos_r2",
                "hawkes_rmse",
                "ofi_rmse",
                "hawkes_correlation",
                "ofi_correlation",
                "probability_hawkes_better",
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