from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import expit
from scipy.stats import poisson


BOOK_FILE = Path(
    "data/processed/all_capture_book_states.parquet"
)

TRADES_FILE = Path(
    "data/processed/all_capture_trade_events.parquet"
)

CAPTURE_ID = "capture_04"

BIN_SIZE = 0.1
STATIONARITY_LIMIT = 0.999


def build_capture_data():
    book = pd.read_parquet(
        BOOK_FILE,
        engine="pyarrow",
    )

    trades = pd.read_parquet(
        TRADES_FILE,
        engine="pyarrow",
    )

    book = book.loc[
        book["capture_id"] == CAPTURE_ID
    ].copy()

    trades = trades.loc[
        trades["capture_id"] == CAPTURE_ID
    ].copy()

    if book.empty:
        raise RuntimeError(
            "No Capture 04 book states found."
        )

    if trades.empty:
        raise RuntimeError(
            "No Capture 04 trades found."
        )

    start_ms = min(
        int(book["event_time_ms"].min()),
        int(trades["trade_time_ms"].min()),
    )

    step_ms = int(
        BIN_SIZE * 1000
    )

    book["bin"] = (
        (
            book["event_time_ms"]
            - start_ms
        )
        // step_ms
    ).astype(np.int64)

    trades["bin"] = (
        (
            trades["trade_time_ms"]
            - start_ms
        )
        // step_ms
    ).astype(np.int64)

    trade_counts = (
        trades
        .groupby(
            ["bin", "side"]
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

    max_bin = max(
        int(book["bin"].max()),
        int(trades["bin"].max()),
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
            book[
                [
                    "bin",
                    "mid_price",
                ]
            ]
            .groupby("bin")
            .last()
            .reset_index(),
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

    data = data.dropna(
        subset=["mid_price"]
    ).reset_index(drop=True)

    return data


def decode(x):
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


def nll(
    x,
    buy,
    sell,
):
    (
        mu_buy,
        mu_sell,
        beta,
        n_buy,
        n_sell,
    ) = decode(x)

    decay = np.exp(
        -beta * BIN_SIZE
    )

    scale = 1.0 - decay

    buy_input = np.zeros_like(buy)

    sell_input = np.zeros_like(sell)

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

    mean_buy = (
        mu_buy * BIN_SIZE
        + n_buy * scale * buy_state
    )

    mean_sell = (
        mu_sell * BIN_SIZE
        + n_sell * scale * sell_state
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
            buy,
            mean_buy,
        ).sum()
        + poisson.logpmf(
            sell,
            mean_sell,
        ).sum()
    )

    if not np.isfinite(value):
        return 1e100

    return -value


def fit(data):
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

    for beta in [1, 3, 5, 10]:
        for branching in [
            0.20,
            0.40,
            0.60,
            0.75,
        ]:

            z = np.log(
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
                        z,
                        z,
                    ]
                )
            )

    best = None

    for x0 in starts:

        result = minimize(
            nll,
            x0,
            args=(buy, sell),
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
            "Python optimization failed."
        )

    return (
        decode(best.x),
        best.fun,
        len(buy),
        len(sell),
    )


def main():
    data = build_capture_data()

    (
        parameters,
        negative_log_likelihood,
        n_buy_obs,
        n_sell_obs,
    ) = fit(data)

    (
        mu_buy,
        mu_sell,
        beta,
        branching_buy,
        branching_sell,
    ) = parameters

    print()
    print(
        "PYTHON CAPTURE 04 EXACT-GRID VERIFICATION"
    )
    print(
        "=========================================="
    )

    print(
        f"Bins: {len(data):,}"
    )

    print(
        f"Buy events: "
        f"{int(data['buy'].sum()):,}"
    )

    print(
        f"Sell events: "
        f"{int(data['sell'].sum()):,}"
    )

    print(
        f"mu_buy: "
        f"{mu_buy:.10f}"
    )

    print(
        f"mu_sell: "
        f"{mu_sell:.10f}"
    )

    print(
        f"beta: "
        f"{beta:.10f}"
    )

    print(
        f"branching_buy: "
        f"{branching_buy:.10f}"
    )

    print(
        f"branching_sell: "
        f"{branching_sell:.10f}"
    )

    print(
        f"spectral_radius: "
        f"{max(branching_buy, branching_sell):.10f}"
    )

    print(
        f"negative_log_likelihood: "
        f"{negative_log_likelihood:.10f}"
    )


if __name__ == "__main__":
    main()