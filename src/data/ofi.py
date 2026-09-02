from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path(
    "data/live/btc_usdt_book_states.parquet"
)

OUTPUT_FILE = Path(
    "data/live/btc_usdt_ofi.parquet"
)


def compute_best_level_ofi(book):
    book = book.sort_values(
        ["event_time_ms", "final_update_id"]
    ).reset_index(drop=True)

    bid_price = book["best_bid"].to_numpy(
        dtype=float
    )

    ask_price = book["best_ask"].to_numpy(
        dtype=float
    )

    bid_size = book["bid_size_1"].to_numpy(
        dtype=float
    )

    ask_size = book["ask_size_1"].to_numpy(
        dtype=float
    )

    n = len(book)

    ofi = np.zeros(n)

    for i in range(1, n):

        if bid_price[i] > bid_price[i - 1]:
            bid_component = bid_size[i]

        elif bid_price[i] < bid_price[i - 1]:
            bid_component = -bid_size[i - 1]

        else:
            bid_component = (
                bid_size[i] - bid_size[i - 1]
            )

        if ask_price[i] < ask_price[i - 1]:
            ask_component = ask_size[i]

        elif ask_price[i] > ask_price[i - 1]:
            ask_component = -ask_size[i - 1]

        else:
            ask_component = (
                ask_size[i - 1] - ask_size[i]
            )

        ofi[i] = (
            bid_component
            + ask_component
        )

    result = book.copy()

    result["ofi"] = ofi

    result["ofi_abs"] = np.abs(ofi)

    result["ofi_cumulative"] = (
        result["ofi"].cumsum()
    )

    return result


def add_aggregated_ofi(features):
    result = features.copy()

    result["timestamp"] = pd.to_datetime(
        result["event_time_ms"],
        unit="ms",
        utc=True,
    )

    result["second"] = (
        result["event_time_ms"] // 1000
    )

    result["ofi_1s"] = (
        result.groupby("second")["ofi"]
        .transform("sum")
    )

    result["ofi_100ms"] = (
        result["event_time_ms"] // 100
    )

    result["ofi_100ms_sum"] = (
        result.groupby("ofi_100ms")["ofi"]
        .transform("sum")
    )

    return result


def validate(result):
    if result.empty:
        raise RuntimeError(
            "No OFI observations were produced."
        )

    if result["ofi"].isna().any():
        raise RuntimeError(
            "OFI contains missing values."
        )

    if not np.isfinite(
        result["ofi"].to_numpy()
    ).all():
        raise RuntimeError(
            "OFI contains non-finite values."
        )

    if result["ofi"].iloc[0] != 0:
        raise RuntimeError(
            "First OFI observation must be zero "
            "because there is no previous book state."
        )


def main():

    book = pd.read_parquet(
        INPUT_FILE,
        engine="pyarrow",
    )

    print(
        f"Book states: {len(book):,}"
    )

    result = compute_best_level_ofi(
        book
    )

    result = add_aggregated_ofi(
        result
    )

    validate(result)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        OUTPUT_FILE,
        engine="pyarrow",
        index=False,
    )

    print()
    print(
        f"OFI observations: {len(result):,}"
    )

    print(
        f"Mean OFI: "
        f"{result['ofi'].mean():.6f}"
    )

    print(
        f"Std OFI: "
        f"{result['ofi'].std():.6f}"
    )

    print(
        f"Min OFI: "
        f"{result['ofi'].min():.6f}"
    )

    print(
        f"Max OFI: "
        f"{result['ofi'].max():.6f}"
    )

    print()
    print(
        "OFI quantiles:"
    )

    print(
        result["ofi"]
        .quantile(
            [
                0.01,
                0.05,
                0.25,
                0.50,
                0.75,
                0.95,
                0.99,
            ]
        )
        .to_string()
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()