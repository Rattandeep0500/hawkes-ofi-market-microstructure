from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


INPUT_FILE = Path(
    "data/live/btc_usdt_ofi.parquet"
)

OUTPUT_FILE = Path(
    "data/processed/ofi_price_response.csv"
)

HORIZONS_MS = [
    100,
    500,
    1000,
    5000,
]


def prepare_data():
    data = pd.read_parquet(
        INPUT_FILE,
        engine="pyarrow",
    )

    data = data.sort_values(
        "event_time_ms"
    ).reset_index(drop=True)

    required = [
        "event_time_ms",
        "mid_price",
        "ofi",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    data["log_mid"] = np.log(
        data["mid_price"]
    )

    return data


def calculate_future_returns(data, horizon_ms):
    timestamps = data[
        "event_time_ms"
    ].to_numpy(dtype=np.int64)

    log_mid = data[
        "log_mid"
    ].to_numpy(dtype=float)

    future_positions = np.searchsorted(
        timestamps,
        timestamps + horizon_ms,
        side="left",
    )

    returns = np.full(
        len(data),
        np.nan,
    )

    valid = (
        future_positions < len(data)
    )

    returns[valid] = (
        log_mid[future_positions[valid]]
        - log_mid[valid]
    )

    return returns


def analyze_horizon(data, horizon_ms):
    returns = calculate_future_returns(
        data,
        horizon_ms,
    )

    ofi = data["ofi"].to_numpy(
        dtype=float
    )

    valid = (
        np.isfinite(ofi)
        & np.isfinite(returns)
    )

    x = ofi[valid]
    y = returns[valid]

    if len(x) < 10:
        raise RuntimeError(
            f"Insufficient observations for "
            f"{horizon_ms} ms horizon."
        )

    correlation, correlation_p = (
        stats.pearsonr(x, y)
    )

    x_centered = (
        x - x.mean()
    )

    y_centered = (
        y - y.mean()
    )

    beta = (
        np.dot(
            x_centered,
            y_centered,
        )
        / np.dot(
            x_centered,
            x_centered,
        )
    )

    alpha = (
        y.mean()
        - beta * x.mean()
    )

    fitted = (
        alpha
        + beta * x
    )

    residuals = (
        y - fitted
    )

    ss_res = np.sum(
        residuals ** 2
    )

    ss_tot = np.sum(
        (y - y.mean()) ** 2
    )

    r_squared = (
        1
        - ss_res / ss_tot
    )

    standard_error = np.sqrt(
        np.sum(residuals ** 2)
        / (
            len(x) - 2
        )
        / np.sum(
            x_centered ** 2
        )
    )

    t_stat = (
        beta
        / standard_error
    )

    p_value = (
        2
        * stats.t.sf(
            abs(t_stat),
            df=len(x) - 2,
        )
    )

    return {
        "horizon_ms": horizon_ms,
        "observations": len(x),
        "alpha": alpha,
        "beta_ofi": beta,
        "r_squared": r_squared,
        "pearson_correlation": correlation,
        "correlation_p_value": correlation_p,
        "t_statistic": t_stat,
        "beta_p_value": p_value,
        "return_mean": y.mean(),
        "return_std": y.std(ddof=1),
    }


def main():
    data = prepare_data()

    print(
        f"Observations: {len(data):,}"
    )

    results = []

    for horizon in HORIZONS_MS:
        result = analyze_horizon(
            data,
            horizon,
        )

        results.append(result)

        print(
            f"\nHorizon: {horizon} ms"
        )

        print(
            f"Observations: "
            f"{result['observations']:,}"
        )

        print(
            f"Beta(OFI): "
            f"{result['beta_ofi']:.10f}"
        )

        print(
            f"R²: "
            f"{result['r_squared']:.8f}"
        )

        print(
            f"Correlation: "
            f"{result['pearson_correlation']:.8f}"
        )

        print(
            f"Correlation p-value: "
            f"{result['correlation_p_value']:.8e}"
        )

        print(
            f"Beta p-value: "
            f"{result['beta_p_value']:.8e}"
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

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()