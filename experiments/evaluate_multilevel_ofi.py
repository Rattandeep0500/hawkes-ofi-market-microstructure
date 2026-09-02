from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path(
    "data/live/btc_usdt_multi_level_ofi.parquet"
)

OUTPUT_FILE = Path(
    "data/processed/multilevel_ofi_oos.csv"
)

HORIZONS_MS = [
    1000,
    5000,
]

LEVEL_SETS = {
    "L1": [1],
    "L2": [1, 2],
    "L5": [1, 2, 3, 4, 5],
    "L10": list(range(1, 11)),
}

TRAIN_FRACTION = 0.70


def future_returns(data, horizon_ms):
    timestamps = data[
        "event_time_ms"
    ].to_numpy(dtype=np.int64)

    log_mid = np.log(
        data["mid_price"].to_numpy(dtype=float)
    )

    future_index = np.searchsorted(
        timestamps,
        timestamps + horizon_ms,
        side="left",
    )

    returns = np.full(
        len(data),
        np.nan,
    )

    valid = future_index < len(data)

    returns[valid] = (
        log_mid[future_index[valid]]
        - log_mid[valid]
    )

    return returns


def fit_ols(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = np.column_stack(
        [
            np.ones(len(x)),
            x,
        ]
    )

    beta = np.linalg.lstsq(
        x,
        y,
        rcond=None,
    )[0]

    return beta


def evaluate_oos(x_train, y_train, x_test, y_test):
    beta = fit_ols(
        x_train,
        y_train,
    )

    x_test_design = np.column_stack(
        [
            np.ones(len(x_test)),
            x_test,
        ]
    )

    predictions = (
        x_test_design @ beta
    )

    residuals = (
        y_test - predictions
    )

    sse = np.sum(
        residuals ** 2
    )

    sst = np.sum(
        (y_test - y_train.mean()) ** 2
    )

    oos_r2 = (
        1.0 - sse / sst
        if sst > 0
        else np.nan
    )

    mae = np.mean(
        np.abs(residuals)
    )

    rmse = np.sqrt(
        np.mean(
            residuals ** 2
        )
    )

    directional_accuracy = np.mean(
        np.sign(predictions)
        == np.sign(y_test)
    )

    return {
        "oos_r2": oos_r2,
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": directional_accuracy,
    }


def evaluate_level_set(
    data,
    levels,
    horizon_ms,
):
    columns = [
        f"ofi_{level}"
        for level in levels
    ]

    data = data.copy()

    data["future_return"] = future_returns(
        data,
        horizon_ms,
    )

    required = columns + [
        "future_return"
    ]

    data = data.dropna(
        subset=required
    )

    x = data[
        columns
    ].to_numpy(dtype=float)

    y = data[
        "future_return"
    ].to_numpy(dtype=float)

    split = int(
        len(data)
        * TRAIN_FRACTION
    )

    if split <= 10 or len(data) - split <= 10:
        raise RuntimeError(
            "Insufficient observations "
            "for train/test split."
        )

    x_train = x[:split]
    y_train = y[:split]

    x_test = x[split:]
    y_test = y[split:]

    result = evaluate_oos(
        x_train,
        y_train,
        x_test,
        y_test,
    )

    result["horizon_ms"] = horizon_ms
    result["levels"] = ",".join(
        map(str, levels)
    )
    result["model"] = (
        f"L{len(levels)}"
    )
    result["train_observations"] = len(
        x_train
    )
    result["test_observations"] = len(
        x_test
    )

    return result


def evaluate_normalized_features(
    data,
    horizon_ms,
):
    data = data.copy()

    data["future_return"] = future_returns(
        data,
        horizon_ms,
    )

    columns = [
        "ofi_multilevel",
        "ofi_normalized",
        "ofi_depth_weighted",
    ]

    data = data.dropna(
        subset=[
            "future_return"
        ] + columns
    )

    results = []

    for column in columns:
        x = data[
            [column]
        ].to_numpy(dtype=float)

        y = data[
            "future_return"
        ].to_numpy(dtype=float)

        split = int(
            len(data)
            * TRAIN_FRACTION
        )

        result = evaluate_oos(
            x[:split],
            y[:split],
            x[split:],
            y[split:],
        )

        result["horizon_ms"] = horizon_ms
        result["levels"] = "10"
        result["model"] = column
        result["train_observations"] = split
        result["test_observations"] = (
            len(data) - split
        )

        results.append(result)

    return results


def main():
    data = pd.read_parquet(
        INPUT_FILE,
        engine="pyarrow",
    )

    data = (
        data
        .sort_values("final_update_id")
        .reset_index(drop=True)
    )

    results = []

    for horizon in HORIZONS_MS:
        for model, levels in LEVEL_SETS.items():
            result = evaluate_level_set(
                data,
                levels,
                horizon,
            )

            results.append(result)

        results.extend(
            evaluate_normalized_features(
                data,
                horizon,
            )
        )

    results = pd.DataFrame(
        results
    )

    results = results[
        [
            "horizon_ms",
            "model",
            "levels",
            "train_observations",
            "test_observations",
            "oos_r2",
            "mae",
            "rmse",
            "directional_accuracy",
        ]
    ]

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
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