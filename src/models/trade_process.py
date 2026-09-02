import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path(
    "data/live/btc_usdt_research_capture.jsonl"
)

OUTPUT_FILE = Path(
    "data/processed/trade_events.parquet"
)


def load_trades():
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
                    "event_time_ms": int(event["T"]),
                    "trade_id": int(event["t"]),
                    "price": float(event["p"]),
                    "quantity": float(event["q"]),
                    "buyer_maker": bool(event["m"]),
                    "received_at_ms": int(
                        record["received_at_ms"]
                    ),
                }
            )

    if not rows:
        raise RuntimeError(
            "No trade events found."
        )

    trades = pd.DataFrame(rows)

    trades = (
        trades
        .sort_values(
            ["event_time_ms", "trade_id"]
        )
        .drop_duplicates(
            subset="trade_id"
        )
        .reset_index(drop=True)
    )

    return trades


def build_event_process(trades):
    result = trades.copy()

    result["side"] = np.where(
        result["buyer_maker"],
        "sell",
        "buy",
    )

    result["event_time_s"] = (
        result["event_time_ms"]
        - result["event_time_ms"].iloc[0]
    ) / 1000.0

    result["signed_quantity"] = np.where(
        result["side"] == "buy",
        result["quantity"],
        -result["quantity"],
    )

    result["notional"] = (
        result["price"]
        * result["quantity"]
    )

    result["signed_notional"] = np.where(
        result["side"] == "buy",
        result["notional"],
        -result["notional"],
    )

    result["interarrival_ms"] = (
        result["event_time_ms"].diff()
    )

    result["same_timestamp"] = (
        result["event_time_ms"]
        .duplicated(
            keep=False
        )
    )

    result["event_index"] = np.arange(
        len(result),
        dtype=np.int64,
    )

    result["buy_event"] = (
        result["side"] == "buy"
    ).astype(np.int8)

    result["sell_event"] = (
        result["side"] == "sell"
    ).astype(np.int8)

    return result


def validate_events(events):
    if events.empty:
        raise RuntimeError(
            "Empty event process."
        )

    if not events[
        "event_time_s"
    ].is_monotonic_increasing:
        raise RuntimeError(
            "Event times are not monotonic."
        )

    if not events[
        "trade_id"
    ].is_unique:
        raise RuntimeError(
            "Duplicate trade IDs detected."
        )

    if (
        events["quantity"] <= 0
    ).any():
        raise RuntimeError(
            "Non-positive trade quantity detected."
        )

    if (
        events["price"] <= 0
    ).any():
        raise RuntimeError(
            "Non-positive trade price detected."
        )

    if not np.isfinite(
        events[
            "event_time_s"
        ].to_numpy()
    ).all():
        raise RuntimeError(
            "Non-finite event times detected."
        )

    if not np.isfinite(
        events[
            "quantity"
        ].to_numpy()
    ).all():
        raise RuntimeError(
            "Non-finite quantities detected."
        )

    if not np.isfinite(
        events[
            "price"
        ].to_numpy()
    ).all():
        raise RuntimeError(
            "Non-finite prices detected."
        )


def print_summary(events):
    total = len(events)

    buys = (
        events["side"] == "buy"
    ).sum()

    sells = (
        events["side"] == "sell"
    ).sum()

    tied = (
        events["same_timestamp"]
    ).sum()

    unique_timestamps = (
        events["event_time_ms"]
        .nunique()
    )

    duration = (
        events["event_time_s"].iloc[-1]
        - events["event_time_s"].iloc[0]
    )

    print(
        f"Total events: {total:,}"
    )

    print(
        f"Buy events: {buys:,}"
    )

    print(
        f"Sell events: {sells:,}"
    )

    print(
        f"Buy fraction: {buys / total:.6f}"
    )

    print(
        f"Sell fraction: {sells / total:.6f}"
    )

    print(
        f"Unique timestamps: "
        f"{unique_timestamps:,}"
    )

    print(
        f"Events sharing timestamp: "
        f"{tied:,}"
    )

    print(
        f"Timestamp duration: "
        f"{duration:.6f} seconds"
    )

    print(
        f"Average event rate: "
        f"{total / duration:.4f} events/s"
    )

    print(
        f"Buy event rate: "
        f"{buys / duration:.4f} events/s"
    )

    print(
        f"Sell event rate: "
        f"{sells / duration:.4f} events/s"
    )


def main():
    trades = load_trades()

    events = build_event_process(
        trades
    )

    validate_events(
        events
    )

    print_summary(
        events
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    events.to_parquet(
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