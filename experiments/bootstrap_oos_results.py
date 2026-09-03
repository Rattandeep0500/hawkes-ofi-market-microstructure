import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import expit
from scipy.stats import poisson
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
    "data/processed/bootstrap_oos_results.csv"
)

BIN_SIZE = 0.1
TRAIN_FRACTION = 0.70
STATIONARITY_LIMIT = 0.999

HORIZONS = {
    "1s": 10,
    "5s": 50,
}

# 50 bins = 5 seconds of contiguous observations.
BLOCK_SIZE = 50

# Number of bootstrap replications.
N_BOOTSTRAP = 2000

RANDOM_SEED = 20260903

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
        "queue_imbalance",
        "spread_bps",
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
        book[
            required
        ]
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
            f"No book data for {capture_id}"
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

    book_data = (
        local_book
        .sort_values(
            "event_time_ms"
        )
        .groupby("bin")
        .agg(
            mid_price=(
                "mid_price",
                "last",
            ),
            queue_imbalance=(
                "queue_imbalance",
                "last",
            ),
            spread_bps=(
                "spread_bps",
                "last",
            ),
        )
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
        .sort_values(
            "bin"
        )
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

    data["queue_imbalance"] = (
        data["queue_imbalance"]
        .ffill()
    )

    data["spread_bps"] = (
        data["spread_bps"]
        .ffill()
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


def fit_hawkes(data):
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
            hawkes_nll,
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

    result["hawkes_buy_intensity"] = (
        buy_intensity
    )

    result["hawkes_sell_intensity"] = (
        sell_intensity
    )

    result["hawkes_pressure"] = (
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

    result_indices = (
        valid_indices
    )

    result[
        result_indices
    ] = (
        np.log(
            end[
                valid
            ]
        )
        - np.log(
            start[
                valid
            ]
        )
    )

    return result


def fit_prediction_model(
    x_train,
    y_train,
):
    valid = (
        np.isfinite(
            x_train
        ).all(axis=1)
        & np.isfinite(
            y_train
        )
    )

    x_train = x_train[
        valid
    ]

    y_train = y_train[
        valid
    ]

    model = Pipeline(
        [
            (
                "scale",
                StandardScaler(),
            ),
            (
                "ridge",
                Ridge(
                    alpha=RIDGE_ALPHA,
                    fit_intercept=True,
                ),
            ),
        ]
    )

    model.fit(
        x_train,
        y_train,
    )

    return model


def predict_oos(
    model,
    x_test,
    y_test,
):
    valid = (
        np.isfinite(
            x_test
        ).all(axis=1)
        & np.isfinite(
            y_test
        )
    )

    x_test = x_test[
        valid
    ]

    y_test = y_test[
        valid
    ]

    predictions = model.predict(
        x_test
    )

    return (
        predictions,
        y_test,
    )


def calculate_metrics(
    prediction,
    actual,
    benchmark_mean,
):
    prediction = np.asarray(
        prediction,
        dtype=float,
    )

    actual = np.asarray(
        actual,
        dtype=float,
    )

    residual = (
        actual
        - prediction
    )

    benchmark = np.full(
        len(actual),
        benchmark_mean,
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


def block_bootstrap_indices(
    n,
    block_size,
    rng,
):
    if n <= 0:
        raise ValueError(
            "n must be positive."
        )

    if block_size <= 0:
        raise ValueError(
            "block_size must be positive."
        )

    starts = np.arange(
        0,
        max(
            n
            - block_size
            + 1,
            1,
        ),
    )

    output = []

    while len(output) < n:
        start = int(
            rng.choice(
                starts
            )
        )

        block_end = min(
            start + block_size,
            n,
        )

        output.extend(
            range(
                start,
                block_end,
            )
        )

    return np.asarray(
        output[:n],
        dtype=np.int64,
    )


def bootstrap_metrics(
    prediction,
    actual,
    benchmark_mean,
    n_bootstrap,
    block_size,
    rng,
):
    n = len(actual)

    if n < block_size:
        block_size = max(
            1,
            n // 10,
        )

    bootstrap_r2 = np.empty(
        n_bootstrap,
        dtype=float,
    )

    bootstrap_corr = np.empty(
        n_bootstrap,
        dtype=float,
    )

    bootstrap_rmse = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for i in range(
        n_bootstrap
    ):
        indices = block_bootstrap_indices(
            n,
            block_size,
            rng,
        )

        pred_sample = (
            prediction[
                indices
            ]
        )

        actual_sample = (
            actual[
                indices
            ]
        )

        (
            r2,
            rmse,
            correlation,
        ) = calculate_metrics(
            pred_sample,
            actual_sample,
            benchmark_mean,
        )

        bootstrap_r2[i] = r2
        bootstrap_rmse[i] = rmse
        bootstrap_corr[i] = (
            correlation
        )

    return (
        bootstrap_r2,
        bootstrap_rmse,
        bootstrap_corr,
    )


def percentile_interval(
    values,
):
    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return (
            np.nan,
            np.nan,
        )

    return (
        np.percentile(
            values,
            2.5,
        ),
        np.percentile(
            values,
            97.5,
        ),
    )


def build_features(data):
    hawkes = data[
        "hawkes_pressure"
    ].to_numpy(
        dtype=float
    )

    qi = data[
        "queue_imbalance"
    ].to_numpy(
        dtype=float
    )

    spread = data[
        "spread_bps"
    ].to_numpy(
        dtype=float
    )

    return np.column_stack(
        [
            hawkes,
            qi,
            spread,
        ]
    )


def run_direction(
    train_data,
    test_data,
    train_capture,
    test_capture,
    rng,
):
    parameters = fit_hawkes(
        train_data
    )

    train_data = add_hawkes_pressure(
        train_data,
        parameters,
    )

    test_data = add_hawkes_pressure(
        test_data,
        parameters,
    )

    print()
    print(
        f"Train: {train_capture}"
    )

    print(
        f"Test:  {test_capture}"
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
        train_y = future_return(
            train_data[
                "mid_price"
            ],
            horizon_bins,
        )

        test_y = future_return(
            test_data[
                "mid_price"
            ],
            horizon_bins,
        )

        features_train = (
            build_features(
                train_data
            )
        )

        features_test = (
            build_features(
                test_data
            )
        )

        valid_train = (
            np.isfinite(
                features_train
            ).all(axis=1)
            & np.isfinite(
                train_y
            )
        )

        valid_test = (
            np.isfinite(
                features_test
            ).all(axis=1)
            & np.isfinite(
                test_y
            )
        )

        x_train = (
            features_train[
                valid_train
            ]
        )

        y_train = (
            train_y[
                valid_train
            ]
        )

        x_test = (
            features_test[
                valid_test
            ]
        )

        y_test = (
            test_y[
                valid_test
            ]
        )

        model = fit_prediction_model(
            x_train,
            y_train,
        )

        prediction, actual = (
            predict_oos(
                model,
                x_test,
                y_test,
            )
        )

        benchmark_mean = (
            y_train.mean()
        )

        (
            point_r2,
            point_rmse,
            point_corr,
        ) = calculate_metrics(
            prediction,
            actual,
            benchmark_mean,
        )

        (
            boot_r2,
            boot_rmse,
            boot_corr,
        ) = bootstrap_metrics(
            prediction,
            actual,
            benchmark_mean,
            N_BOOTSTRAP,
            BLOCK_SIZE,
            rng,
        )

        (
            r2_low,
            r2_high,
        ) = percentile_interval(
            boot_r2
        )

        (
            rmse_low,
            rmse_high,
        ) = percentile_interval(
            boot_rmse
        )

        (
            corr_low,
            corr_high,
        ) = percentile_interval(
            boot_corr
        )

        positive_r2_probability = (
            np.mean(
                boot_r2 > 0
            )
        )

        rows.append(
            {
                "train_capture": train_capture,
                "test_capture": test_capture,
                "horizon": horizon_name,
                "train_observations": len(
                    y_train
                ),
                "test_observations": len(
                    actual
                ),
                "point_oos_r2": point_r2,
                "r2_ci_low": r2_low,
                "r2_ci_high": r2_high,
                "probability_r2_positive": (
                    positive_r2_probability
                ),
                "point_rmse": point_rmse,
                "rmse_ci_low": rmse_low,
                "rmse_ci_high": rmse_high,
                "point_correlation": point_corr,
                "correlation_ci_low": corr_low,
                "correlation_ci_high": corr_high,
                "block_size_bins": BLOCK_SIZE,
                "block_size_seconds": (
                    BLOCK_SIZE
                    * BIN_SIZE
                ),
                "bootstrap_repetitions": (
                    N_BOOTSTRAP
                ),
                "mu_buy": parameters[0],
                "mu_sell": parameters[1],
                "hawkes_beta": parameters[2],
                "branching_buy": parameters[3],
                "branching_sell": parameters[4],
            }
        )

        print()
        print(
            f"{horizon_name}:"
        )

        print(
            f"Point OOS R2: "
            f"{point_r2:.6f}"
        )

        print(
            f"95% block-bootstrap R2 CI: "
            f"[{r2_low:.6f}, "
            f"{r2_high:.6f}]"
        )

        print(
            f"P(R2 > 0): "
            f"{positive_r2_probability:.4f}"
        )

        print(
            f"Point correlation: "
            f"{point_corr:.6f}"
        )

        print(
            f"95% correlation CI: "
            f"[{corr_low:.6f}, "
            f"{corr_high:.6f}]"
        )

    return rows


def main():
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    book = load_book()

    trades = {}

    for capture_id, path in (
        CAPTURE_FILES.items()
    ):
        trades[capture_id] = (
            load_trades(path)
        )

    data = {}

    for capture_id in (
        CAPTURE_FILES
    ):
        data[capture_id] = (
            build_capture_data(
                trades[capture_id],
                book,
                capture_id,
            )
        )

    all_results = []

    all_results.extend(
        run_direction(
            data["capture_02"],
            data["capture_03"],
            "capture_02",
            "capture_03",
            rng,
        )
    )

    all_results.extend(
        run_direction(
            data["capture_03"],
            data["capture_02"],
            "capture_03",
            "capture_02",
            rng,
        )
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
        "Bootstrap results:"
    )

    print(
        results[
            [
                "train_capture",
                "test_capture",
                "horizon",
                "point_oos_r2",
                "r2_ci_low",
                "r2_ci_high",
                "probability_r2_positive",
                "point_rmse",
                "rmse_ci_low",
                "rmse_ci_high",
                "point_correlation",
                "correlation_ci_low",
                "correlation_ci_high",
                "block_size_seconds",
                "bootstrap_repetitions",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()