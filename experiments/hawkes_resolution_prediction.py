import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import expit
from scipy.stats import poisson


CAPTURE_FILE = Path(
    "data/live/btc_usdt_research_capture.jsonl"
)

BOOK_FILE = Path(
    "data/live/btc_usdt_book_states.parquet"
)

OUTPUT_FILE = Path(
    "data/processed/hawkes_resolution_prediction.csv"
)

BIN_SIZES = [
    0.05,
    0.10,
    0.25,
    0.50,
]

HORIZONS = {
    "1s": 1.0,
    "5s": 5.0,
}

TRAIN_FRACTION = 0.70
STATIONARITY_LIMIT = 0.999


def load_trades():
    rows = []

    with CAPTURE_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            record = json.loads(line)

            if record["type"] != "trade":
                continue

            event = record["data"]

            rows.append(
                {
                    "timestamp_ms": int(event["T"]),
                    "trade_id": int(event["t"]),
                    "side": (
                        "sell"
                        if bool(event["m"])
                        else "buy"
                    ),
                }
            )

    if not rows:
        raise RuntimeError(
            "No trades found."
        )

    trades = pd.DataFrame(rows)

    return (
        trades
        .drop_duplicates("trade_id")
        .sort_values(
            ["timestamp_ms", "trade_id"]
        )
        .reset_index(drop=True)
    )


def load_book():
    book = pd.read_parquet(
        BOOK_FILE,
        columns=[
            "event_time_ms",
            "mid_price",
        ],
        engine="pyarrow",
    )

    if book.empty:
        raise RuntimeError(
            "No book states found."
        )

    return (
        book
        .sort_values("event_time_ms")
        .drop_duplicates(
            "event_time_ms",
            keep="last",
        )
        .reset_index(drop=True)
    )


def build_grid(
    trades,
    book,
    bin_size,
):
    start_ms = min(
        trades["timestamp_ms"].min(),
        book["event_time_ms"].min(),
    )

    end_ms = max(
        trades["timestamp_ms"].max(),
        book["event_time_ms"].max(),
    )

    start_ms = (
        start_ms // int(bin_size * 1000)
    ) * int(bin_size * 1000)

    step_ms = int(
        round(bin_size * 1000)
    )

    grid = pd.DataFrame(
        {
            "time_bin_ms": np.arange(
                start_ms,
                end_ms + step_ms,
                step_ms,
                dtype=np.int64,
            )
        }
    )

    return grid


def aggregate_trades(
    trades,
    bin_size,
):
    step_ms = int(
        round(bin_size * 1000)
    )

    data = trades.copy()

    data["time_bin_ms"] = (
        data["timestamp_ms"]
        // step_ms
    ) * step_ms

    grouped = (
        data
        .groupby(
            [
                "time_bin_ms",
                "side",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reindex(
            columns=["buy", "sell"],
            fill_value=0,
        )
        .reset_index()
    )

    grouped["buy"] = (
        grouped["buy"]
        .astype(np.int64)
    )

    grouped["sell"] = (
        grouped["sell"]
        .astype(np.int64)
    )

    return grouped


def aggregate_book(
    book,
    bin_size,
):
    step_ms = int(
        round(bin_size * 1000)
    )

    data = book.copy()

    data["time_bin_ms"] = (
        data["event_time_ms"]
        // step_ms
    ) * step_ms

    grouped = (
        data
        .sort_values("event_time_ms")
        .groupby("time_bin_ms")
        .agg(
            mid_price=(
                "mid_price",
                "last",
            )
        )
        .reset_index()
    )

    return grouped


def build_dataset(
    trades,
    book,
    bin_size,
):
    grid = build_grid(
        trades,
        book,
        bin_size,
    )

    trade_data = aggregate_trades(
        trades,
        bin_size,
    )

    book_data = aggregate_book(
        book,
        bin_size,
    )

    data = (
        grid
        .merge(
            trade_data,
            on="time_bin_ms",
            how="left",
        )
        .merge(
            book_data,
            on="time_bin_ms",
            how="left",
        )
        .sort_values("time_bin_ms")
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

    data = data.dropna(
        subset=["mid_price"]
    ).reset_index(
        drop=True
    )

    data["relative_time_s"] = (
        data["time_bin_ms"]
        - data["time_bin_ms"].iloc[0]
    ) / 1000.0

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

    buy_state, sell_state, decay = (
        hawkes_states(
            buy,
            sell,
            beta,
            bin_size,
        )
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
    buy,
    sell,
    bin_size,
):
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
    buy,
    sell,
    parameters,
    bin_size,
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
            bin_size,
        )
    )

    scale = (
        1.0 - decay
    )

    buy_intensity = (
        mu_buy
        + (
            branching_buy
            * scale
            * buy_state
            / bin_size
        )
    )

    sell_intensity = (
        mu_sell
        + (
            branching_sell
            * scale
            * sell_state
            / bin_size
        )
    )

    pressure = (
        buy_intensity
        - sell_intensity
    )

    return pressure


def build_future_return(
    prices,
    horizon_seconds,
    bin_size,
):
    horizon_bins = int(
        round(
            horizon_seconds
            / bin_size
        )
    )

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


def evaluate_oos(
    pressure,
    target,
    split,
):
    pressure = np.asarray(
        pressure,
        dtype=float,
    )

    target = np.asarray(
        target,
        dtype=float,
    )

    train_valid = (
        np.isfinite(
            pressure[:split]
        )
        & np.isfinite(
            target[:split]
        )
    )

    test_valid = (
        np.isfinite(
            pressure[split:]
        )
        & np.isfinite(
            target[split:]
        )
    )

    x_train = pressure[
        :split
    ][train_valid]

    y_train = target[
        :split
    ][train_valid]

    x_test = pressure[
        split:
    ][test_valid]

    y_test = target[
        split:
    ][test_valid]

    if len(x_train) < 30:
        raise RuntimeError(
            "Insufficient training data."
        )

    if len(x_test) < 20:
        raise RuntimeError(
            "Insufficient test data."
        )

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

    prediction = (
        test_design
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

    return (
        oos_r2,
        rmse,
        correlation,
        len(x_train),
        len(x_test),
    )


def main():
    trades = load_trades()
    book = load_book()

    results = []

    for bin_size in BIN_SIZES:
        data = build_dataset(
            trades,
            book,
            bin_size,
        )

        split = int(
            len(data)
            * TRAIN_FRACTION
        )

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

        parameters = fit_hawkes(
            buy[:split],
            sell[:split],
            bin_size,
        )

        pressure = generate_pressure(
            buy,
            sell,
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

        print()
        print(
            f"Bin size: "
            f"{bin_size:.3f}s"
        )

        print(
            f"Bins: "
            f"{len(data):,}"
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

        for horizon_name, horizon_seconds in (
            HORIZONS.items()
        ):
            target = build_future_return(
                data["mid_price"],
                horizon_seconds,
                bin_size,
            )

            (
                oos_r2,
                rmse,
                correlation,
                train_obs,
                test_obs,
            ) = evaluate_oos(
                pressure,
                target,
                split,
            )

            results.append(
                {
                    "bin_size_seconds": bin_size,
                    "horizon": horizon_name,
                    "train_observations": train_obs,
                    "test_observations": test_obs,
                    "mu_buy": mu_buy,
                    "mu_sell": mu_sell,
                    "beta": beta,
                    "branching_buy": branching_buy,
                    "branching_sell": branching_sell,
                    "oos_r2": oos_r2,
                    "rmse": rmse,
                    "prediction_return_correlation": correlation,
                }
            )

            print(
                f"{horizon_name}: "
                f"OOS R2={oos_r2:.6f} "
                f"RMSE={rmse:.8f} "
                f"corr={correlation:.6f}"
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
        "Resolution prediction results:"
    )

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