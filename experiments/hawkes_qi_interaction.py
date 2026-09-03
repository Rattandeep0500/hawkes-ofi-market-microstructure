import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import expit
from scipy.stats import t
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
    "data/processed/hawkes_qi_interaction.csv"
)

BIN_SIZE = 0.1
TRAIN_FRACTION = 0.70
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
            f"No trades found: {path}"
        )

    trades = pd.DataFrame(rows)

    return (
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
    local_book = book.loc[
        book["capture_id"] == capture_id
    ].copy()

    if local_book.empty:
        raise RuntimeError(
            f"No book data for {capture_id}"
        )

    step_ms = int(
        BIN_SIZE * 1000
    )

    start_ms = min(
        trades["timestamp_ms"].min(),
        local_book["event_time_ms"].min(),
    )

    trade_data = trades.copy()

    trade_data["bin"] = (
        (
            trade_data["timestamp_ms"]
            - start_ms
        )
        // step_ms
    ).astype(np.int64)

    local_book["bin"] = (
        (
            local_book["event_time_ms"]
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
        .sort_values("event_time_ms")
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
    ).reset_index(drop=True)

    data["timestamp"] = pd.to_datetime(
        start_ms
        + data["bin"] * step_ms,
        unit="ms",
        utc=True,
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


def standardize_train_test(
    train,
    test,
):
    means = train.mean(
        axis=0
    )

    stds = train.std(
        axis=0,
        ddof=1,
    )

    stds = np.where(
        stds > 0,
        stds,
        1.0,
    )

    return (
        (train - means) / stds,
        (test - means) / stds,
        means,
        stds,
    )


def hac_covariance(
    x,
    residuals,
    lag,
):
    n = len(residuals)

    xtx_inv = np.linalg.inv(
        x.T @ x
    )

    meat = np.zeros(
        (
            x.shape[1],
            x.shape[1],
        ),
        dtype=float,
    )

    scores = (
        x
        * residuals[:, None]
    )

    meat += (
        scores.T
        @ scores
    )

    max_lag = min(
        lag,
        n - 1,
    )

    for h in range(
        1,
        max_lag + 1,
    ):
        weight = (
            1.0
            - h
            / (
                max_lag + 1.0
            )
        )

        cross = (
            scores[h:].T
            @ scores[:-h]
        )

        meat += (
            weight
            * (
                cross
                + cross.T
            )
        )

    covariance = (
        xtx_inv
        @ meat
        @ xtx_inv
    )

    covariance *= (
        n
        / (
            n
            - x.shape[1]
        )
    )

    return covariance


def fit_interaction_model(
    x,
    y,
    hac_lag,
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

    fitted = (
        design
        @ coefficients
    )

    residuals = (
        y
        - fitted
    )

    covariance = hac_covariance(
        design,
        residuals,
        hac_lag,
    )

    standard_errors = np.sqrt(
        np.maximum(
            np.diag(
                covariance
            ),
            0.0,
        )
    )

    t_statistics = np.divide(
        coefficients,
        standard_errors,
        out=np.full_like(
            coefficients,
            np.nan,
        ),
        where=standard_errors > 0,
    )

    p_values = (
        2.0
        * t.sf(
            np.abs(t_statistics),
            df=max(
                len(y)
                - design.shape[1],
                1,
            ),
        )
    )

    return (
        coefficients,
        standard_errors,
        t_statistics,
        p_values,
    )


def oos_metrics(
    coefficients,
    x_test,
    y_test,
    y_train,
):
    design = np.column_stack(
        [
            np.ones(
                len(x_test)
            ),
            x_test,
        ]
    )

    predictions = (
        design
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

    benchmark_sse = np.sum(
        (
            y_test
            - benchmark
        ) ** 2
    )

    r2 = (
        1.0
        - sse
        / benchmark_sse
    )

    rmse = np.sqrt(
        np.mean(
            residuals ** 2
        )
    )

    correlation = np.nan

    if (
        np.std(predictions) > 0
        and np.std(y_test) > 0
    ):
        correlation = np.corrcoef(
            predictions,
            y_test,
        )[0, 1]

    return (
        r2,
        rmse,
        correlation,
    )


def run_capture(
    data,
    capture_id,
):
    n = len(data)

    split = int(
        n * TRAIN_FRACTION
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

    full = add_hawkes_pressure(
        data,
        parameters,
    )

    train = full.iloc[
        :split
    ].copy()

    test = full.iloc[
        split:
    ].copy()

    print()
    print(
        f"{capture_id}"
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

    for horizon_name, horizon_bins in (
        HORIZONS.items()
    ):
        train_target = future_return(
            train["mid_price"],
            horizon_bins,
        )

        test_target = future_return(
            test["mid_price"],
            horizon_bins,
        )

        feature_base = [
            "hawkes_pressure",
            "queue_imbalance",
            "spread_bps",
        ]

        train_features = train[
            feature_base
        ].to_numpy(
            dtype=float
        )

        test_features = test[
            feature_base
        ].to_numpy(
            dtype=float
        )

        train_features, test_features, _, _ = (
            standardize_train_test(
                train_features,
                test_features,
            )
        )

        hawkes_train = (
            train_features[:, 0]
        )

        qi_train = (
            train_features[:, 1]
        )

        spread_train = (
            train_features[:, 2]
        )

        hawkes_test = (
            test_features[:, 0]
        )

        qi_test = (
            test_features[:, 1]
        )

        spread_test = (
            test_features[:, 2]
        )

        interaction_train = (
            hawkes_train
            * qi_train
        )

        interaction_test = (
            hawkes_test
            * qi_test
        )

        x_train = np.column_stack(
            [
                hawkes_train,
                qi_train,
                spread_train,
                interaction_train,
            ]
        )

        x_test = np.column_stack(
            [
                hawkes_test,
                qi_test,
                spread_test,
                interaction_test,
            ]
        )

        train_valid = (
            np.isfinite(
                x_train
            ).all(axis=1)
            & np.isfinite(
                train_target
            )
        )

        test_valid = (
            np.isfinite(
                x_test
            ).all(axis=1)
            & np.isfinite(
                test_target
            )
        )

        x_train = x_train[
            train_valid
        ]

        y_train = train_target[
            train_valid
        ]

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
                f"{capture_id}, "
                f"{horizon_name}: "
                "insufficient observations."
            )

        hac_lag = max(
            horizon_bins - 1,
            1,
        )

        (
            coefficients,
            standard_errors,
            t_statistics,
            p_values,
        ) = fit_interaction_model(
            x_train,
            y_train,
            hac_lag,
        )

        (
            oos_r2,
            rmse,
            correlation,
        ) = oos_metrics(
            coefficients,
            x_test,
            y_test,
            y_train,
        )

        results = {
            "capture_id": capture_id,
            "horizon": horizon_name,
            "train_observations": len(
                x_train
            ),
            "test_observations": len(
                x_test
            ),
            "beta_hawkes": coefficients[1],
            "beta_qi": coefficients[2],
            "beta_spread": coefficients[3],
            "beta_hawkes_qi": coefficients[4],
            "se_hawkes": standard_errors[1],
            "se_qi": standard_errors[2],
            "se_spread": standard_errors[3],
            "se_hawkes_qi": standard_errors[4],
            "t_hawkes": t_statistics[1],
            "t_qi": t_statistics[2],
            "t_spread": t_statistics[3],
            "t_hawkes_qi": t_statistics[4],
            "p_hawkes": p_values[1],
            "p_qi": p_values[2],
            "p_spread": p_values[3],
            "p_hawkes_qi": p_values[4],
            "oos_r2": oos_r2,
            "rmse": rmse,
            "prediction_return_correlation": correlation,
            "hac_lag": hac_lag,
            "mu_buy": parameters[0],
            "mu_sell": parameters[1],
            "hawkes_beta": parameters[2],
            "branching_buy": parameters[3],
            "branching_sell": parameters[4],
        }

        print()
        print(
            f"{horizon_name}"
        )

        print(
            f"beta_H: "
            f"{coefficients[1]:.8f}"
        )

        print(
            f"beta_QI: "
            f"{coefficients[2]:.8f}"
        )

        print(
            f"beta_Spread: "
            f"{coefficients[3]:.8f}"
        )

        print(
            f"beta_HxQI: "
            f"{coefficients[4]:.8f}"
        )

        print(
            f"p_HxQI: "
            f"{p_values[4]:.8e}"
        )

        print(
            f"OOS R2: "
            f"{oos_r2:.6f}"
        )

        print(
            f"OOS correlation: "
            f"{correlation:.6f}"
        )

        yield results


def main():
    book = load_book()

    all_results = []

    for capture_id, path in (
        CAPTURE_FILES.items()
    ):
        trades = load_trades(
            path
        )

        data = build_capture_data(
            trades,
            book,
            capture_id,
        )

        all_results.extend(
            list(
                run_capture(
                    data,
                    capture_id,
                )
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
        "Formal Hawkes-QI interaction results:"
    )

    print(
        results[
            [
                "capture_id",
                "horizon",
                "beta_hawkes",
                "beta_qi",
                "beta_spread",
                "beta_hawkes_qi",
                "p_hawkes_qi",
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
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()