from pathlib import Path

import pandas as pd


WINDOWS_PATH = Path("data/processed/research_windows.csv")
DEPTH_PATH = Path("data/raw/depth/binance/BTCUSDT/2026-06.parquet")
TRADES_PATH = Path("data/raw/trades/binance/BTCUSDT/2026-06.parquet")

DEPTH_OUTPUT = Path("data/processed/research_depth.parquet")
TRADES_OUTPUT = Path("data/processed/research_trades.parquet")


def load_windows():
    windows = pd.read_csv(WINDOWS_PATH)

    windows["start_time"] = pd.to_datetime(
        windows["start_time"],
        utc=True,
    )

    windows["end_time"] = pd.to_datetime(
        windows["end_time"],
        utc=True,
    )

    return windows


def extract_depth(windows):
    columns = [
        "timestamp_ms",
        "asset",
        "side",
        "price",
        "quantity",
        "first_update_id",
        "last_update_id",
        "exchange",
    ]

    depth = pd.read_parquet(
        DEPTH_PATH,
        columns=columns,
        engine="pyarrow",
    )

    frames = []

    for _, window in windows.iterrows():
        start_id = int(window["start_update_id"])
        end_id = int(window["end_update_id"])

        mask = (
            (depth["first_update_id"] >= start_id)
            & (depth["last_update_id"] <= end_id)
        )

        frame = depth.loc[mask].copy()
        frame["window_id"] = window["window_id"]

        frames.append(frame)

        print(
            f'{window["window_id"]}: '
            f'{len(frame):,} depth rows'
        )

    if not frames:
        raise RuntimeError("No depth windows extracted.")

    return pd.concat(
        frames,
        ignore_index=True,
    )


def extract_trades(windows):
    columns = [
        "timestamp_ms",
        "asset",
        "trade_id",
        "price",
        "quantity",
        "buyer_maker",
        "exchange",
    ]

    trades = pd.read_parquet(
        TRADES_PATH,
        columns=columns,
        engine="pyarrow",
    )

    frames = []

    for _, window in windows.iterrows():
        start_time = window["start_time"]
        end_time = window["end_time"]

        start_ms = int(start_time.value // 1_000_000)
        end_ms = int(end_time.value // 1_000_000)

        mask = (
            (trades["timestamp_ms"] >= start_ms)
            & (trades["timestamp_ms"] <= end_ms)
        )

        frame = trades.loc[mask].copy()
        frame["window_id"] = window["window_id"]

        frames.append(frame)

        print(
            f'{window["window_id"]}: '
            f'{len(frame):,} trades'
        )

    if not frames:
        raise RuntimeError("No trade windows extracted.")

    return pd.concat(
        frames,
        ignore_index=True,
    )


def validate_depth(windows, depth):
    for _, window in windows.iterrows():
        window_id = window["window_id"]

        frame = depth[
            depth["window_id"] == window_id
        ]

        if frame.empty:
            raise RuntimeError(
                f"{window_id} contains no depth rows."
            )

        minimum = frame["first_update_id"].min()
        maximum = frame["last_update_id"].max()

        expected_min = int(window["start_update_id"])
        expected_max = int(window["end_update_id"])

        if minimum != expected_min:
            raise RuntimeError(
                f"{window_id} starts at {minimum}, "
                f"expected {expected_min}."
            )

        if maximum != expected_max:
            raise RuntimeError(
                f"{window_id} ends at {maximum}, "
                f"expected {expected_max}."
            )


def validate_trades(windows, trades):
    for _, window in windows.iterrows():
        window_id = window["window_id"]

        frame = trades[
            trades["window_id"] == window_id
        ]

        if frame.empty:
            raise RuntimeError(
                f"{window_id} contains no trades."
            )


def validate_update_continuity(windows, depth):
    for _, window in windows.iterrows():
        frame = depth[
            depth["window_id"] == window["window_id"]
        ]

        updates = (
            frame[
                [
                    "first_update_id",
                    "last_update_id",
                ]
            ]
            .drop_duplicates()
            .sort_values("first_update_id")
            .reset_index(drop=True)
        )

        first_ids = updates[
            "first_update_id"
        ].to_numpy()

        last_ids = updates[
            "last_update_id"
        ].to_numpy()

        if len(updates) <= 1:
            continue

        continuous = (
            first_ids[1:]
            == last_ids[:-1] + 1
        )

        if not continuous.all():
            raise RuntimeError(
                f"{window['window_id']} contains "
                "an internal update-ID gap."
            )


def main():
    windows = load_windows()

    print(
        f"Research windows loaded: "
        f"{len(windows)}\n"
    )

    depth = extract_depth(windows)

    print(
        f"\nTotal depth rows: "
        f"{len(depth):,}"
    )

    trades = extract_trades(windows)

    print(
        f"\nTotal trades: "
        f"{len(trades):,}"
    )

    print("\nValidating depth...")
    validate_depth(
        windows,
        depth,
    )

    print("Validating trade windows...")
    validate_trades(
        windows,
        trades,
    )

    print("Validating update continuity...")
    validate_update_continuity(
        windows,
        depth,
    )

    print("\nValidation passed.")

    DEPTH_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    depth.to_parquet(
        DEPTH_OUTPUT,
        engine="pyarrow",
        index=False,
    )

    trades.to_parquet(
        TRADES_OUTPUT,
        engine="pyarrow",
        index=False,
    )

    print(
        f"\nSaved depth: {DEPTH_OUTPUT}"
    )

    print(
        f"Saved trades: {TRADES_OUTPUT}"
    )


if __name__ == "__main__":
    main()