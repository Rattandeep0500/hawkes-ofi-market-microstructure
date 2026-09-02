from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path(
    "data/live/btc_usdt_book_states.parquet"
)

OUTPUT_FILE = Path(
    "data/live/btc_usdt_multi_level_ofi.parquet"
)

LEVELS = 10


def compute_level_ofi(
    bid_price,
    bid_size,
    ask_price,
    ask_size,
):
    n = len(bid_price)
    result = np.zeros(n, dtype=float)

    for i in range(1, n):
        if bid_price[i] > bid_price[i - 1]:
            bid_component = bid_size[i]
        elif bid_price[i] < bid_price[i - 1]:
            bid_component = -bid_size[i - 1]
        else:
            bid_component = bid_size[i] - bid_size[i - 1]

        if ask_price[i] < ask_price[i - 1]:
            ask_component = ask_size[i]
        elif ask_price[i] > ask_price[i - 1]:
            ask_component = -ask_size[i - 1]
        else:
            ask_component = ask_size[i - 1] - ask_size[i]

        result[i] = bid_component + ask_component

    return result


def compute_features(book):
    book = (
        book.sort_values("final_update_id")
        .reset_index(drop=True)
        .copy()
    )

    for level in range(1, LEVELS + 1):
        required = [
            f"bid_price_{level}",
            f"bid_size_{level}",
            f"ask_price_{level}",
            f"ask_size_{level}",
        ]

        missing = [
            column
            for column in required
            if column not in book.columns
        ]

        if missing:
            raise RuntimeError(
                f"Missing level-{level} columns: {missing}"
            )

    for level in range(1, LEVELS + 1):
        book[f"ofi_{level}"] = compute_level_ofi(
            book[f"bid_price_{level}"].to_numpy(dtype=float),
            book[f"bid_size_{level}"].to_numpy(dtype=float),
            book[f"ask_price_{level}"].to_numpy(dtype=float),
            book[f"ask_size_{level}"].to_numpy(dtype=float),
        )

    ofi_columns = [
        f"ofi_{level}"
        for level in range(1, LEVELS + 1)
    ]

    depth_columns = []

    for level in range(1, LEVELS + 1):
        depth_columns.extend(
            [
                f"bid_size_{level}",
                f"ask_size_{level}",
            ]
        )

    book["depth_10"] = book[
        depth_columns
    ].sum(axis=1)

    book["ofi_multilevel"] = book[
        ofi_columns
    ].sum(axis=1)

    book["ofi_normalized"] = np.where(
        book["depth_10"] > 0,
        book["ofi_multilevel"] / book["depth_10"],
        0.0,
    )

    weights = np.array(
        [
            1.0 / level
            for level in range(1, LEVELS + 1)
        ],
        dtype=float,
    )

    ofi_matrix = book[
        ofi_columns
    ].to_numpy(dtype=float)

    book["ofi_depth_weighted"] = (
        ofi_matrix @ weights
    ) / weights.sum()

    for level in range(1, LEVELS + 1):
        book[f"ofi_share_{level}"] = np.where(
            book["ofi_multilevel"].abs() > 0,
            book[f"ofi_{level}"]
            / book["ofi_multilevel"].abs(),
            0.0,
        )

    book["timestamp"] = pd.to_datetime(
        book["event_time_ms"],
        unit="ms",
        utc=True,
    )

    return book


def validate(result):
    if result.empty:
        raise RuntimeError(
            "No OFI observations were produced."
        )

    numeric_columns = [
        f"ofi_{level}"
        for level in range(1, LEVELS + 1)
    ]

    numeric_columns.extend(
        [
            "depth_10",
            "ofi_multilevel",
            "ofi_normalized",
            "ofi_depth_weighted",
        ]
    )

    values = result[
        numeric_columns
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise RuntimeError(
            "Non-finite OFI values detected."
        )

    for level in range(1, LEVELS + 1):
        if result[f"ofi_{level}"].iloc[0] != 0:
            raise RuntimeError(
                f"Initial level-{level} OFI is not zero."
            )


def main():
    book = pd.read_parquet(
        INPUT_FILE,
        engine="pyarrow",
    )

    print(
        f"Book states: {len(book):,}"
    )

    result = compute_features(book)

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

    print(
        f"OFI observations: {len(result):,}"
    )

    print()
    print(
        result[
            [
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
                "ofi_multilevel",
                "ofi_normalized",
                "ofi_depth_weighted",
            ]
        ]
        .describe()
        .T
        .to_string()
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()