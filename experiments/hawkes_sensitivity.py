from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import expit
from scipy.stats import poisson


INPUT_FILE = Path(
    "data/live/btc_usdt_research_capture.jsonl"
)

OUTPUT_FILE = Path(
    "data/processed/hawkes_sensitivity.csv"
)

BIN_SIZES = [
    0.05,
    0.10,
    0.25,
    0.50,
]

TRAIN_FRACTION = 0.70
STATIONARITY_LIMIT = 0.999


def load_trades():
    import json

    rows = []

    with INPUT_FILE.open(
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


def build_counts(
    trades,
    bin_size,
):
    start = (
        trades["timestamp_ms"].min()
    )

    relative_seconds = (
        trades["timestamp_ms"]
        - start
    ) / 1000.0

    trades = trades.copy()

    trades["bin"] = np.floor(
        relative_seconds / bin_size
    ).astype(np.int64)

    grouped = (
        trades
        .groupby(["bin", "side"])
        .size()
        .unstack(fill_value=0)
    )

    grouped = grouped.reindex(
        columns=["buy", "sell"],
        fill_value=0,
    )

    last_bin = int(
        grouped.index.max()
    )

    grid = pd.DataFrame(
        {
            "bin": np.arange(
                last_bin + 1,
                dtype=np.int64,
            )
        }
    )

    grouped = (
        grid
        .merge(
            grouped.reset_index(),
            on="bin",
            how="left",
        )
        .fillna(0)
    )

    return grouped[
        ["buy", "sell"]
    ].to_numpy(
        dtype=float
    )


def parameters(x):
    mu_buy = np.exp(
        np.clip(
            x[0],
            -20,
            20,
        )
    )

    mu_sell = np.exp(
        np.clip(
            x[1],
            -20,
            20,
        )
    )

    beta = np.exp(
        np.clip(
            x[2],
            -10,
            10,
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
        mu_buy,
        mu_sell,
        beta,
        n_buy,
        n_sell,
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
        n_buy,
        n_sell,
    ) = parameters(x)

    buy_state, sell_state, decay = states(
        buy,
        sell,
        beta,
        bin_size,
    )

    scale = 1.0 - decay

    mean_buy = (
        mu_buy * bin_size
        + n_buy
        * scale
        * buy_state
    )

    mean_sell = (
        mu_sell * bin_size
        + n_sell
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

    return parameters(
        best.x
    )


def generate_pressure(
    buy,
    sell,
    fitted,
    bin_size,
):
    (
        mu_buy,
        mu_sell,
        beta,
        n_buy,
        n_sell,
    ) = fitted

    buy_state, sell_state, decay = states(
        buy,
        sell,
        beta,
        bin_size,
    )

    scale = 1.0 - decay

    buy_intensity = (
        mu_buy
        + (
            n_buy
            * scale
            * buy_state
            / bin_size
        )
    )

    sell_intensity = (
        mu_sell
        + (
            n_sell
            * scale
            * sell_state
            / bin_size
        )
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


def load_mid_prices():
    data = pd.read_parquet(
        Path(
            "data/live/btc_usdt_book_states.parquet"
        ),
        columns=[
            "event_time_ms",
            "mid_price",
        ],
        engine="pyarrow",
    )

    return data.sort_values(
        "event_time_ms"
    ).reset_index(
        drop=True
    )


def evaluate_pressure(
    pressure,
    mid_prices,
    bin_size,
    horizon_seconds,
):
    horizon_bins = int(
        round(
            horizon_seconds
            / bin_size
        )
    )

    prices = (
        mid_prices
        .set_index("event_time_ms")
    )

    return horizon_bins


def main():
    trades = load_trades()

    results = []

    for bin_size in BIN_SIZES:
        counts = build_counts(
            trades,
            bin_size,
        )

        split = int(
            len(counts)
            * TRAIN_FRACTION
        )

        train = counts[
            :split
        ]

        test = counts[
            split:
        ]

        buy_train = train[
            :, 0
        ]

        sell_train = train[
            :, 1
        ]

        fitted = fit_hawkes(
            buy_train,
            sell_train,
            bin_size,
        )

        (
            mu_buy,
            mu_sell,
            beta,
            branching_buy,
            branching_sell,
        ) = fitted

        train_duration = (
            len(train)
            * bin_size
        )

        test_duration = (
            len(test)
            * bin_size
        )

        results.append(
            {
                "bin_size_seconds": bin_size,
                "train_duration_seconds": train_duration,
                "test_duration_seconds": test_duration,
                "mu_buy": mu_buy,
                "mu_sell": mu_sell,
                "beta": beta,
                "branching_buy": branching_buy,
                "branching_sell": branching_sell,
            }
        )

        print()
        print(
            f"Bin size: {bin_size:.3f}s"
        )

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
            f"branching_buy={branching_buy:.8f}"
        )

        print(
            f"branching_sell={branching_sell:.8f}"
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