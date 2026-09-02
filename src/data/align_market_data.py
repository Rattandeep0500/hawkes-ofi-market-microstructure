import json
from pathlib import Path

import numpy as np
import pandas as pd


CAPTURE_FILE = Path(
    "data/live/btc_usdt_research_capture.jsonl"
)

OFI_FILE = Path(
    "data/live/btc_usdt_multi_level_ofi.parquet"
)

OUTPUT_FILE = Path(
    "data/processed/aligned_market_data.parquet"
)

BIN_SIZE_MS = 100


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
                    "price": float(event["p"]),
                    "quantity": float(event["q"]),
                    "buyer_maker": bool(event["m"]),
                }
            )

    if not rows:
        raise RuntimeError(
            "No trade events found."
        )

    trades = pd.DataFrame(rows)

    trades = (
        trades
        .drop_duplicates("trade_id")
        .sort_values(
            ["timestamp_ms", "trade_id"]
        )
        .reset_index(drop=True)
    )

    trades["side"] = np.where(
        trades["buyer_maker"],
        "sell",
        "buy",
    )

    trades["signed_quantity"] = np.where(
        trades["side"] == "buy",
        trades["quantity"],
        -trades["quantity"],
    )

    trades["notional"] = (
        trades["price"]
        * trades["quantity"]
    )

    trades["signed_notional"] = np.where(
        trades["side"] == "buy",
        trades["notional"],
        -trades["notional"],
    )

    return trades


def aggregate_trades(trades):
    data = trades.copy()

    data["time_bin_ms"] = (
        data["timestamp_ms"]
        // BIN_SIZE_MS
    ) * BIN_SIZE_MS

    data["buy_quantity"] = np.where(
        data["side"] == "buy",
        data["quantity"],
        0.0,
    )

    data["sell_quantity"] = np.where(
        data["side"] == "sell",
        data["quantity"],
        0.0,
    )

    grouped = (
        data
        .groupby("time_bin_ms")
        .agg(
            trade_count=(
                "trade_id",
                "count",
            ),
            buy_count=(
                "side",
                lambda x: int(
                    (x == "buy").sum()
                ),
            ),
            sell_count=(
                "side",
                lambda x: int(
                    (x == "sell").sum()
                ),
            ),
            buy_quantity=(
                "buy_quantity",
                "sum",
            ),
            sell_quantity=(
                "sell_quantity",
                "sum",
            ),
            signed_quantity=(
                "signed_quantity",
                "sum",
            ),
            total_notional=(
                "notional",
                "sum",
            ),
            signed_notional=(
                "signed_notional",
                "sum",
            ),
        )
        .reset_index()
    )

    return grouped


def load_ofi():
    data = pd.read_parquet(
        OFI_FILE,
        engine="pyarrow",
    )

    required = [
        "event_time_ms",
        "mid_price",
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

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing OFI columns: {missing}"
        )

    data = data[
        required
    ].copy()

    data["time_bin_ms"] = (
        data["event_time_ms"]
        // BIN_SIZE_MS
    ) * BIN_SIZE_MS

    aggregation = {
        "mid_price": "last",
        "ofi_1": "sum",
        "ofi_2": "sum",
        "ofi_3": "sum",
        "ofi_4": "sum",
        "ofi_5": "sum",
        "ofi_6": "sum",
        "ofi_7": "sum",
        "ofi_8": "sum",
        "ofi_9": "sum",
        "ofi_10": "sum",
        "ofi_multilevel": "sum",
        "ofi_normalized": "sum",
        "ofi_depth_weighted": "sum",
    }

    grouped = (
        data
        .groupby("time_bin_ms")
        .agg(aggregation)
        .reset_index()
        .sort_values("time_bin_ms")
        .reset_index(drop=True)
    )

    return grouped


def build_time_grid(trades, ofi):
    start = min(
        trades["time_bin_ms"].min(),
        ofi["time_bin_ms"].min(),
    )

    end = max(
        trades["time_bin_ms"].max(),
        ofi["time_bin_ms"].max(),
    )

    return pd.DataFrame(
        {
            "time_bin_ms": np.arange(
                start,
                end + BIN_SIZE_MS,
                BIN_SIZE_MS,
                dtype=np.int64,
            )
        }
    )


def align_data(trades, ofi):
    trade_data = aggregate_trades(
        trades
    )

    grid = build_time_grid(
        trade_data,
        ofi,
    )

    aligned = (
        grid
        .merge(
            trade_data,
            on="time_bin_ms",
            how="left",
        )
        .merge(
            ofi,
            on="time_bin_ms",
            how="left",
        )
        .sort_values("time_bin_ms")
        .reset_index(drop=True)
    )

    count_columns = [
        "trade_count",
        "buy_count",
        "sell_count",
    ]

    flow_columns = [
        "buy_quantity",
        "sell_quantity",
        "signed_quantity",
        "total_notional",
        "signed_notional",
    ]

    ofi_columns = [
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

    for column in count_columns:
        aligned[column] = (
            aligned[column]
            .fillna(0)
            .astype(np.int64)
        )

    for column in flow_columns:
        aligned[column] = (
            aligned[column]
            .fillna(0.0)
        )

    for column in ofi_columns:
        aligned[column] = (
            aligned[column]
            .fillna(0.0)
        )

    aligned["mid_price"] = (
        aligned["mid_price"]
        .ffill()
    )

    aligned = aligned.dropna(
        subset=["mid_price"]
    ).reset_index(
        drop=True
    )

    aligned["timestamp"] = pd.to_datetime(
        aligned["time_bin_ms"],
        unit="ms",
        utc=True,
    )

    aligned["relative_time_s"] = (
        aligned["time_bin_ms"]
        - aligned["time_bin_ms"].iloc[0]
    ) / 1000.0

    aligned["trade_pressure"] = (
        aligned["buy_count"]
        - aligned["sell_count"]
    )

    aligned["quantity_pressure"] = (
        aligned["buy_quantity"]
        - aligned["sell_quantity"]
    )

    aligned["notional_pressure"] = (
        aligned["signed_notional"]
    )

    return aligned


def validate(aligned):
    if aligned.empty:
        raise RuntimeError(
            "Aligned dataset is empty."
        )

    timestamps = aligned[
        "time_bin_ms"
    ].to_numpy(
        dtype=np.int64
    )

    if len(timestamps) > 1:
        steps = np.diff(
            timestamps
        )

        if not np.all(
            steps == BIN_SIZE_MS
        ):
            raise RuntimeError(
                "Time grid is not regular."
            )

    if aligned[
        "mid_price"
    ].isna().any():
        raise RuntimeError(
            "Mid-price still contains missing values."
        )

    numeric = aligned.select_dtypes(
        include=np.number
    )

    if not np.isfinite(
        numeric.to_numpy()
    ).all():
        raise RuntimeError(
            "Non-finite numerical values detected."
        )

    if not aligned[
        "timestamp"
    ].is_monotonic_increasing:
        raise RuntimeError(
            "Timestamps are not increasing."
        )


def print_summary(aligned):
    print(
        f"Aligned bins: "
        f"{len(aligned):,}"
    )

    print(
        f"Start: "
        f"{aligned['timestamp'].iloc[0]}"
    )

    print(
        f"End: "
        f"{aligned['timestamp'].iloc[-1]}"
    )

    print(
        f"Duration: "
        f"{aligned['relative_time_s'].iloc[-1]:.3f} s"
    )

    print()

    print(
        f"Trade events: "
        f"{aligned['trade_count'].sum():,}"
    )

    print(
        f"Buy events: "
        f"{aligned['buy_count'].sum():,}"
    )

    print(
        f"Sell events: "
        f"{aligned['sell_count'].sum():,}"
    )

    print(
        f"Non-empty trade bins: "
        f"{(aligned['trade_count'] > 0).sum():,}"
    )

    print(
        f"Non-zero L1 OFI bins: "
        f"{(aligned['ofi_1'] != 0).sum():,}"
    )

    print()

    print(
        "Mean values:"
    )

    print(
        aligned[
            [
                "trade_count",
                "buy_count",
                "sell_count",
                "ofi_1",
                "ofi_multilevel",
                "ofi_normalized",
                "trade_pressure",
                "quantity_pressure",
            ]
        ]
        .mean()
        .to_string()
    )


def main():
    print("Loading trades...")

    trades = load_trades()

    print(
        f"Trades loaded: "
        f"{len(trades):,}"
    )

    print("Loading OFI...")

    ofi = load_ofi()

    print(
        f"OFI states loaded: "
        f"{len(ofi):,}"
    )

    print("Aligning data...")

    aligned = align_data(
        trades,
        ofi,
    )

    validate(
        aligned
    )

    print()
    print("Alignment validation: OK")
    print()

    print_summary(
        aligned
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    aligned.to_parquet(
        OUTPUT_FILE,
        engine="pyarrow",
        index=False,
    )

    print()
    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()