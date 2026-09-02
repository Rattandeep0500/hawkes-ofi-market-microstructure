import json
from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/live/btc_usdt_research_capture.jsonl"
)

OUTPUT_FILE = Path(
    "data/live/btc_usdt_book_states.parquet"
)


def load_capture():
    snapshot = None
    depth_events = []

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            record = json.loads(line)

            if record["type"] == "snapshot":
                snapshot = record["data"]

            elif record["type"] == "depth":
                depth_events.append(
                    (
                        record["received_at_ms"],
                        record["data"],
                    )
                )

    if snapshot is None:
        raise RuntimeError(
            "No snapshot found."
        )

    if not depth_events:
        raise RuntimeError(
            "No depth events found."
        )

    return snapshot, depth_events


def initialize_book(snapshot):
    bids = {
        float(price): float(quantity)
        for price, quantity in snapshot["bids"]
        if float(quantity) > 0
    }

    asks = {
        float(price): float(quantity)
        for price, quantity in snapshot["asks"]
        if float(quantity) > 0
    }

    return bids, asks


def apply_depth_event(bids, asks, event):
    for price, quantity in event["b"]:
        price = float(price)
        quantity = float(quantity)

        if quantity == 0:
            bids.pop(price, None)
        else:
            bids[price] = quantity

    for price, quantity in event["a"]:
        price = float(price)
        quantity = float(quantity)

        if quantity == 0:
            asks.pop(price, None)
        else:
            asks[price] = quantity


def build_book_states(snapshot, depth_events):
    bids, asks = initialize_book(snapshot)

    rows = []

    for received_at_ms, event in depth_events:
        apply_depth_event(
            bids,
            asks,
            event,
        )

        if not bids or not asks:
            continue

        best_bid = max(bids)
        best_ask = min(asks)

        bid_prices = sorted(
            bids,
            reverse=True,
        )[:10]

        ask_prices = sorted(
            asks
        )[:10]

        bid_size_1 = bids[best_bid]
        ask_size_1 = asks[best_ask]

        mid_price = (
            best_bid + best_ask
        ) / 2

        spread = (
            best_ask - best_bid
        )

        queue_imbalance = (
            bid_size_1 - ask_size_1
        ) / (
            bid_size_1 + ask_size_1
        )

        row = {
            "received_at_ms": received_at_ms,
            "event_time_ms": event["E"],
            "first_update_id": event["U"],
            "final_update_id": event["u"],
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid_price,
            "spread": spread,
            "bid_size_1": bid_size_1,
            "ask_size_1": ask_size_1,
            "queue_imbalance": queue_imbalance,
        }

        for level, price in enumerate(
            bid_prices,
            start=1,
        ):
            row[f"bid_price_{level}"] = price
            row[f"bid_size_{level}"] = bids[price]

        for level, price in enumerate(
            ask_prices,
            start=1,
        ):
            row[f"ask_price_{level}"] = price
            row[f"ask_size_{level}"] = asks[price]

        rows.append(row)

    return pd.DataFrame(rows)


def validate_states(states):
    if states.empty:
        raise RuntimeError(
            "No book states were produced."
        )

    if (
        states["best_bid"]
        >= states["best_ask"]
    ).any():
        raise RuntimeError(
            "Crossed book detected."
        )

    if (
        states["spread"] <= 0
    ).any():
        raise RuntimeError(
            "Non-positive spread detected."
        )

    if (
        states["queue_imbalance"].isna()
    ).any():
        raise RuntimeError(
            "Invalid queue imbalance detected."
        )

    update_ids = states[
        "final_update_id"
    ].to_numpy()

    if len(update_ids) > 1:
        if (
            update_ids[1:]
            <= update_ids[:-1]
        ).any():
            raise RuntimeError(
                "Update IDs are not strictly increasing."
            )


def main():
    snapshot, depth_events = (
        load_capture()
    )

    print(
        f"Snapshot update ID: "
        f"{snapshot['lastUpdateId']:,}"
    )

    print(
        f"Depth events: "
        f"{len(depth_events):,}"
    )

    states = build_book_states(
        snapshot,
        depth_events,
    )

    validate_states(states)

    states["timestamp"] = pd.to_datetime(
        states["event_time_ms"],
        unit="ms",
        utc=True,
    )

    states["spread_bps"] = (
        states["spread"]
        / states["mid_price"]
        * 10_000
    )

    states["mid_return"] = (
        states["mid_price"]
        .pct_change()
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    states.to_parquet(
        OUTPUT_FILE,
        engine="pyarrow",
        index=False,
    )

    print()
    print(
        f"Book states: {len(states):,}"
    )

    print(
        f"First mid-price: "
        f"{states['mid_price'].iloc[0]:,.2f}"
    )

    print(
        f"Last mid-price: "
        f"{states['mid_price'].iloc[-1]:,.2f}"
    )

    print(
        f"Mean spread: "
        f"{states['spread'].mean():.6f}"
    )

    print(
        f"Mean spread (bps): "
        f"{states['spread_bps'].mean():.4f}"
    )

    print(
        f"Mean queue imbalance: "
        f"{states['queue_imbalance'].mean():.6f}"
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()