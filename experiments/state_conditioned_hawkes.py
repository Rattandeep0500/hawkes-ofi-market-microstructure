from pathlib import Path

import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import expit
from scipy.stats import t


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
    "data/processed/state_conditioned_hawkes.csv"
)

BIN_SIZE = 0.1
STATIONARITY_LIMIT = 0.999

HORIZONS = {
    "1s": 10,
    "5s": 50,
}


def load_trades(path):
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
            f"No trades found in {path}"
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
    trades = trades.copy()
    book = book.loc[
        book["capture_id"] == capture_id
    ].copy()

    if book.empty:
        raise RuntimeError(
            f"No book data for {capture_id}"
        )

    step_ms = int(
        BIN_SIZE * 1000
    )

    trade_start = (
        trades["timestamp_ms"].min()
    )

    book_start = (
        book["event_time_ms"].min()
    )

    start_ms = min(
        trade_start,
        book_start,
    )

    trade_relative = (
        trades["timestamp_ms"]
        - start_ms
    )

    book_relative = (
        book["event_time_ms"]
        - start_ms
    )

    trades["bin"] = (
        trade_relative
        // step_ms
    ).astype(np.int64)

    book["bin"] = (
        book_relative
        // step_ms
    ).astype(np.int64)

    trade_counts = (
        trades
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
        book
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

    max_bin = max(
        int(trade_counts["bin"].max()),
        int(book_data["bin"].max()),
    )

    grid = pd.DataFrame(
        {
            "bin": np.arange(
                max_bin + 1,
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

    from scipy.stats import poisson

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
    data,
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


def make_regression(
    x,
    y,
):
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    if len(x) < 30:
        raise RuntimeError(
            "Too few observations."
        )

    design = np.column_stack(
        [
            np.ones(len(x)),
            x,
        ]
    )

    coefficients = np.linalg.lstsq(
        design,
        y,
        rcond=None,
    )[0]

    predictions = (
        design
        @ coefficients
    )

    residuals = (
        y
        - predictions
    )

    n = len(y)
    p = design.shape[1]

    sse = np.sum(
        residuals ** 2
    )

    mse = (
        sse
        / (
            n - p
        )
    )

    covariance = (
        np.linalg.inv(
            design.T
            @ design
        )
        * mse
    )

    standard_errors = np.sqrt(
        np.diag(
            covariance
        )
    )

    t_statistics = (
        coefficients
        / standard_errors
    )

    p_values = (
        2.0
        * t.sf(
            np.abs(t_statistics),
            df=n - p,
        )
    )

    return {
        "alpha": coefficients[0],
        "beta": coefficients[1],
        "beta_se": standard_errors[1],
        "beta_t": t_statistics[1],
        "beta_p": p_values[1],
        "observations": n,
    }


def main():
    book = load_book()

    results = []

    for capture_id, path in (
        CAPTURE_FILES.items()
    ):
        print(
            f"Processing {capture_id}..."
        )

        trades = load_trades(
            path
        )

        data = build_capture_data(
            trades,
            book,
            capture_id,
        )

        split = int(
            len(data)
            * 0.70
        )

        train = data.iloc[
            :split
        ].copy()

        test = data.iloc[
            split:
        ].copy()

        parameters = fit_hawkes(
            train
        )

        data = add_hawkes_pressure(
            data,
            parameters,
        )

        test = data.iloc[
            split:
        ].copy()

        qi_values = train[
            "queue_imbalance"
        ].to_numpy(
            dtype=float
        )

        low_qi = np.quantile(
            qi_values,
            1.0 / 3.0,
        )

        high_qi = np.quantile(
            qi_values,
            2.0 / 3.0,
        )

        test["qi_regime"] = np.select(
            [
                test[
                    "queue_imbalance"
                ] <= low_qi,
                test[
                    "queue_imbalance"
                ] >= high_qi,
            ],
            [
                "low_QI",
                "high_QI",
            ],
            default="middle_QI",
        )

        print(
            f"Training bins: "
            f"{len(train):,}"
        )

        print(
            f"Testing bins: "
            f"{len(test):,}"
        )

        print(
            f"Low QI threshold: "
            f"{low_qi:.8f}"
        )

        print(
            f"High QI threshold: "
            f"{high_qi:.8f}"
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

        for horizon_name, horizon_bins in (
            HORIZONS.items()
        ):
            target = future_return(
                test["mid_price"],
                horizon_bins,
            )

            for regime in [
                "low_QI",
                "middle_QI",
                "high_QI",
            ]:
                mask = (
                    test["qi_regime"]
                    == regime
                )

                subset = test.loc[
                    mask
                ]

                regime_target = target[
                    mask.to_numpy()
                ]

                pressure = subset[
                    "hawkes_pressure"
                ].to_numpy(
                    dtype=float
                )

                metrics = make_regression(
                    pressure,
                    regime_target,
                )

                results.append(
                    {
                        "capture_id": capture_id,
                        "horizon": horizon_name,
                        "qi_regime": regime,
                        "observations": metrics[
                            "observations"
                        ],
                        "qi_mean": subset[
                            "queue_imbalance"
                        ].mean(),
                        "spread_bps_mean": subset[
                            "spread_bps"
                        ].mean(),
                        "hawkes_pressure_mean": subset[
                            "hawkes_pressure"
                        ].mean(),
                        "hawkes_pressure_std": subset[
                            "hawkes_pressure"
                        ].std(),
                        "alpha": metrics[
                            "alpha"
                        ],
                        "beta": metrics[
                            "beta"
                        ],
                        "beta_se": metrics[
                            "beta_se"
                        ],
                        "beta_t": metrics[
                            "beta_t"
                        ],
                        "beta_p": metrics[
                            "beta_p"
                        ],
                        "mu_buy": parameters[0],
                        "mu_sell": parameters[1],
                        "hawkes_beta": parameters[2],
                        "branching_buy": parameters[3],
                        "branching_sell": parameters[4],
                    }
                )

                print(
                    f"{horizon_name} | "
                    f"{regime} | "
                    f"n={metrics['observations']:,} | "
                    f"beta={metrics['beta']:.10f} | "
                    f"p={metrics['beta_p']:.6e}"
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
        "State-conditioned results:"
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