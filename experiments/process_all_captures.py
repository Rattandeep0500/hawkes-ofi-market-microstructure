import json
from pathlib import Path

import numpy as np
import pandas as pd


CAPTURE_FILES = [
    Path("data/live/capture_02.jsonl"),
    Path("data/live/capture_03.jsonl"),
]

OUTPUT_FILE = Path(
    "data/processed/all_capture_book_states.parquet"
)

LEVELS = 10


def load_capture(path):
    snapshot = None
    depth_events = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            if record["type"] == "snapshot":
                snapshot = record["data"]

            elif record["type"] == "depth":
                depth_events.append(
                    (
                        int(record["received_at_ms"]),
                        record["data"],
                    )
                )

    if snapshot is None:
        raise RuntimeError(
            f"No snapshot found in {path}"
        )

    if not depth_events:
        raise RuntimeError(
            f"No depth events found in {path}"
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


def apply_depth_event(
    bids,
    asks,
    event,
):
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


def build_states(
    snapshot,
    depth_events,
    capture_id,
):
    bids, asks = initialize_book(
        snapshot
    )

    rows = []

    for (
        received_at_ms,
        event,
    ) in depth_events:

        apply_depth_event(
            bids,
            asks,
            event,
        )

        if not bids or not asks:
            continue

        bid_prices = sorted(
            bids.keys(),
            reverse=True,
        )[:LEVELS]

        ask_prices = sorted(
            asks.keys()
        )[:LEVELS]

        if (
            len(bid_prices) < LEVELS
            or len(ask_prices) < LEVELS
        ):
            continue

        best_bid = bid_prices[0]
        best_ask = ask_prices[0]

        bid_size_1 = bids[
            best_bid
        ]

        ask_size_1 = asks[
            best_ask
        ]

        mid_price = (
            best_bid
            + best_ask
        ) / 2.0

        spread = (
            best_ask
            - best_bid
        )

        denominator = (
            bid_size_1
            + ask_size_1
        )

        if denominator > 0:
            queue_imbalance = (
                bid_size_1
                - ask_size_1
            ) / denominator
        else:
            queue_imbalance = 0.0

        row = {
            "capture_id": capture_id,
            "received_at_ms": received_at_ms,
            "event_time_ms": int(
                event["E"]
            ),
            "first_update_id": int(
                event["U"]
            ),
            "final_update_id": int(
                event["u"]
            ),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid_price,
            "spread": spread,
            "spread_bps": (
                spread
                / mid_price
                * 10000.0
            ),
            "bid_size_1": bid_size_1,
            "ask_size_1": ask_size_1,
            "queue_imbalance": queue_imbalance,
        }

        for level in range(
            LEVELS
        ):
            bid_price = (
                bid_prices[level]
            )

            ask_price = (
                ask_prices[level]
            )

            row[
                f"bid_price_{level + 1}"
            ] = bid_price

            row[
                f"bid_size_{level + 1}"
            ] = bids[bid_price]

            row[
                f"ask_price_{level + 1}"
            ] = ask_price

            row[
                f"ask_size_{level + 1}"
            ] = asks[ask_price]

        rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No book states produced "
            f"for {capture_id}"
        )

    return pd.DataFrame(rows)


def validate_states(
    states,
    capture_id,
):
    if states.empty:
        raise RuntimeError(
            f"Empty state set: {capture_id}"
        )

    if not states[
        "mid_price"
    ].gt(0).all():
        raise RuntimeError(
            f"Invalid mid-price in {capture_id}"
        )

    if not (
        states["best_bid"]
        < states["best_ask"]
    ).all():
        raise RuntimeError(
            f"Crossed book detected in "
            f"{capture_id}"
        )

    if not states[
        "spread"
    ].gt(0).all():
        raise RuntimeError(
            f"Non-positive spread in "
            f"{capture_id}"
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
                f"Update IDs not increasing "
                f"in {capture_id}"
            )


def add_ofi(
    states,
):
    result = states.copy()

    for level in range(
        1,
        LEVELS + 1,
    ):
        bid_price = result[
            f"bid_price_{level}"
        ].to_numpy(
            dtype=float
        )

        bid_size = result[
            f"bid_size_{level}"
        ].to_numpy(
            dtype=float
        )

        ask_price = result[
            f"ask_price_{level}"
        ].to_numpy(
            dtype=float
        )

        ask_size = result[
            f"ask_size_{level}"
        ].to_numpy(
            dtype=float
        )

        ofi = np.zeros(
            len(result),
            dtype=float,
        )

        for i in range(
            1,
            len(result),
        ):
            if (
                bid_price[i]
                > bid_price[i - 1]
            ):
                bid_component = (
                    bid_size[i]
                )
            elif (
                bid_price[i]
                < bid_price[i - 1]
            ):
                bid_component = (
                    -bid_size[i - 1]
                )
            else:
                bid_component = (
                    bid_size[i]
                    - bid_size[i - 1]
                )

            if (
                ask_price[i]
                < ask_price[i - 1]
            ):
                ask_component = (
                    ask_size[i]
                )
            elif (
                ask_price[i]
                > ask_price[i - 1]
            ):
                ask_component = (
                    -ask_size[i - 1]
                )
            else:
                ask_component = (
                    ask_size[i - 1]
                    - ask_size[i]
                )

            ofi[i] = (
                bid_component
                + ask_component
            )

        result[
            f"ofi_{level}"
        ] = ofi

    ofi_columns = [
        f"ofi_{level}"
        for level in range(
            1,
            LEVELS + 1,
        )
    ]

    depth_columns = []

    for level in range(
        1,
        LEVELS + 1,
    ):
        depth_columns.extend(
            [
                f"bid_size_{level}",
                f"ask_size_{level}",
            ]
        )

    result["ofi_multilevel"] = (
        result[ofi_columns]
        .sum(axis=1)
    )

    result["depth_10"] = (
        result[depth_columns]
        .sum(axis=1)
    )

    result["ofi_normalized"] = np.where(
        result["depth_10"] > 0,
        result["ofi_multilevel"]
        / result["depth_10"],
        0.0,
    )

    weights = np.array(
        [
            1.0 / level
            for level in range(
                1,
                LEVELS + 1,
            )
        ],
        dtype=float,
    )

    result["ofi_depth_weighted"] = (
        result[
            ofi_columns
        ].to_numpy(
            dtype=float
        )
        @ weights
        / weights.sum()
    )

    return result


def process_capture(
    path,
    capture_id,
):
    print(
        f"Processing {capture_id}..."
    )

    snapshot, depth_events = (
        load_capture(path)
    )

    states = build_states(
        snapshot,
        depth_events,
        capture_id,
    )

    validate_states(
        states,
        capture_id,
    )

    states = add_ofi(
        states
    )

    print(
        f"{capture_id}: "
        f"{len(states):,} book states"
    )

    print(
        f"{capture_id}: "
        f"{len(depth_events):,} depth events"
    )

    return states


def main():
    parts = []

    for index, path in enumerate(
        CAPTURE_FILES,
        start=2,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing capture: {path}"
            )

        capture_id = (
            f"capture_{index:02d}"
        )

        states = process_capture(
            path,
            capture_id,
        )

        parts.append(states)

    combined = pd.concat(
        parts,
        ignore_index=True,
    )

    combined["timestamp"] = (
        pd.to_datetime(
            combined[
                "event_time_ms"
            ],
            unit="ms",
            utc=True,
        )
    )

    combined = combined.sort_values(
        [
            "capture_id",
            "event_time_ms",
            "final_update_id",
        ]
    ).reset_index(
        drop=True
    )

    numeric_columns = (
        combined
        .select_dtypes(
            include=np.number
        )
        .columns
    )

    if not np.isfinite(
        combined[numeric_columns]
        .to_numpy()
    ).all():
        raise RuntimeError(
            "Non-finite values detected."
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_parquet(
        OUTPUT_FILE,
        engine="pyarrow",
        index=False,
    )

    print()
    print(
        "Combined capture summary:"
    )

    print(
        combined.groupby(
            "capture_id"
        ).agg(
            states=(
                "final_update_id",
                "size",
            ),
            start=(
                "timestamp",
                "min",
            ),
            end=(
                "timestamp",
                "max",
            ),
            mean_mid=(
                "mid_price",
                "mean",
            ),
            mean_spread_bps=(
                "spread_bps",
                "mean",
            ),
            mean_qi=(
                "queue_imbalance",
                "mean",
            ),
            mean_ofi=(
                "ofi_1",
                "mean",
            ),
        ).to_string()
    )

    print()
    print(
        f"Total book states: "
        f"{len(combined):,}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()