import json
from pathlib import Path

import numpy as np
import pandas as pd


CAPTURE_FILES = [
    (
        "capture_02",
        Path("data/live/capture_02.jsonl"),
    ),
    (
        "capture_03",
        Path("data/live/capture_03.jsonl"),
    ),
    (
        "capture_04",
        Path("data/live/capture_04.jsonl"),
    ),
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
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"{path}: invalid JSON on line "
                    f"{line_number}: {error}"
                ) from error

            record_type = record.get(
                "type"
            )

            if record_type == "snapshot":
                if snapshot is not None:
                    raise RuntimeError(
                        f"{path}: multiple snapshots."
                    )

                snapshot = record["data"]

            elif record_type == "depth":
                if "received_at_ms" not in record:
                    raise RuntimeError(
                        f"{path}: depth record on "
                        f"line {line_number} is missing "
                        "received_at_ms."
                    )

                depth_events.append(
                    (
                        int(
                            record[
                                "received_at_ms"
                            ]
                        ),
                        record["data"],
                    )
                )

    if snapshot is None:
        raise RuntimeError(
            f"No snapshot found in {path}."
        )

    if not depth_events:
        raise RuntimeError(
            f"No depth events found in {path}."
        )

    return (
        snapshot,
        depth_events,
    )


def initialize_book(snapshot):
    if "bids" not in snapshot:
        raise RuntimeError(
            "Snapshot has no bids."
        )

    if "asks" not in snapshot:
        raise RuntimeError(
            "Snapshot has no asks."
        )

    bids = {}

    for price, quantity in snapshot["bids"]:
        price = float(price)
        quantity = float(quantity)

        if price > 0 and quantity > 0:
            bids[price] = quantity

    asks = {}

    for price, quantity in snapshot["asks"]:
        price = float(price)
        quantity = float(quantity)

        if price > 0 and quantity > 0:
            asks[price] = quantity

    if not bids or not asks:
        raise RuntimeError(
            "Snapshot contains an empty side."
        )

    return bids, asks


def apply_depth_event(
    bids,
    asks,
    event,
):
    if "b" not in event or "a" not in event:
        raise RuntimeError(
            "Depth event missing bid/ask updates."
        )

    for price, quantity in event["b"]:
        price = float(price)
        quantity = float(quantity)

        if quantity == 0:
            bids.pop(
                price,
                None,
            )
        elif quantity > 0:
            bids[price] = quantity

    for price, quantity in event["a"]:
        price = float(price)
        quantity = float(quantity)

        if quantity == 0:
            asks.pop(
                price,
                None,
            )
        elif quantity > 0:
            asks[price] = quantity


def build_states(
    snapshot,
    depth_events,
    capture_id,
):
    bids, asks = initialize_book(
        snapshot
    )

    snapshot_update_id = int(
        snapshot["lastUpdateId"]
    )

    previous_update_id = (
        snapshot_update_id
    )

    rows = []

    for (
        received_at_ms,
        event,
    ) in depth_events:

        first_update_id = int(
            event["U"]
        )

        final_update_id = int(
            event["u"]
        )

        if final_update_id <= (
            previous_update_id
        ):
            continue

        if first_update_id > (
            previous_update_id + 1
        ):
            raise RuntimeError(
                f"{capture_id}: update-ID gap: "
                f"previous={previous_update_id}, "
                f"current_U={first_update_id}"
            )

        apply_depth_event(
            bids,
            asks,
            event,
        )

        previous_update_id = (
            final_update_id
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

        mid_price = (
            best_bid
            + best_ask
        ) / 2.0

        spread = (
            best_ask
            - best_bid
        )

        bid_size_1 = bids[
            best_bid
        ]

        ask_size_1 = asks[
            best_ask
        ]

        queue_denominator = (
            bid_size_1
            + ask_size_1
        )

        if queue_denominator > 0:
            queue_imbalance = (
                bid_size_1
                - ask_size_1
            ) / queue_denominator
        else:
            queue_imbalance = 0.0

        row = {
            "capture_id": capture_id,
            "received_at_ms": received_at_ms,
            "event_time_ms": int(
                event["E"]
            ),
            "first_update_id": (
                first_update_id
            ),
            "final_update_id": (
                final_update_id
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

        for level in range(LEVELS):
            row[
                f"bid_price_{level + 1}"
            ] = bid_prices[level]

            row[
                f"bid_size_{level + 1}"
            ] = bids[
                bid_prices[level]
            ]

            row[
                f"ask_price_{level + 1}"
            ] = ask_prices[level]

            row[
                f"ask_size_{level + 1}"
            ] = asks[
                ask_prices[level]
            ]

        rows.append(row)

    if not rows:
        raise RuntimeError(
            f"{capture_id}: no book states created."
        )

    return pd.DataFrame(rows)


def validate_states(
    states,
    capture_id,
):
    if states.empty:
        raise RuntimeError(
            f"{capture_id}: empty states."
        )

    required_numeric = [
        "best_bid",
        "best_ask",
        "mid_price",
        "spread",
        "spread_bps",
        "queue_imbalance",
    ]

    numeric = states[
        required_numeric
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        numeric
    ).all():
        raise RuntimeError(
            f"{capture_id}: non-finite "
            "book-state values."
        )

    if not (
        states["best_bid"]
        < states["best_ask"]
    ).all():
        raise RuntimeError(
            f"{capture_id}: crossed book."
        )

    if not (
        states["spread"] > 0
    ).all():
        raise RuntimeError(
            f"{capture_id}: non-positive spread."
        )

    update_ids = states[
        "final_update_id"
    ].to_numpy(
        dtype=np.int64
    )

    if len(update_ids) > 1:
        if (
            update_ids[1:]
            <= update_ids[:-1]
        ).any():
            raise RuntimeError(
                f"{capture_id}: update IDs "
                "not strictly increasing."
            )

    timestamps = states[
        "event_time_ms"
    ].to_numpy(
        dtype=np.int64
    )

    if len(timestamps) > 1:
        if (
            timestamps[1:]
            < timestamps[:-1]
        ).any():
            raise RuntimeError(
                f"{capture_id}: event times "
                "not monotonic."
            )


def calculate_ofi(
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

        if len(result) > 1:
            bid_up = (
                bid_price[1:]
                > bid_price[:-1]
            )

            bid_down = (
                bid_price[1:]
                < bid_price[:-1]
            )

            bid_same = (
                ~bid_up
                & ~bid_down
            )

            bid_component = np.zeros(
                len(result) - 1,
                dtype=float,
            )

            bid_component[
                bid_up
            ] = bid_size[1:][
                bid_up
            ]

            bid_component[
                bid_down
            ] = -bid_size[:-1][
                bid_down
            ]

            bid_component[
                bid_same
            ] = (
                bid_size[1:][
                    bid_same
                ]
                - bid_size[:-1][
                    bid_same
                ]
            )

            ask_down = (
                ask_price[1:]
                < ask_price[:-1]
            )

            ask_up = (
                ask_price[1:]
                > ask_price[:-1]
            )

            ask_same = (
                ~ask_down
                & ~ask_up
            )

            ask_component = np.zeros(
                len(result) - 1,
                dtype=float,
            )

            ask_component[
                ask_down
            ] = ask_size[1:][
                ask_down
            ]

            ask_component[
                ask_up
            ] = -ask_size[:-1][
                ask_up
            ]

            ask_component[
                ask_same
            ] = (
                ask_size[:-1][
                    ask_same
                ]
                - ask_size[1:][
                    ask_same
                ]
            )

            ofi[1:] = (
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
        result[
            ofi_columns
        ].sum(axis=1)
    )

    result["depth_10"] = (
        result[
            depth_columns
        ].sum(axis=1)
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
    capture_id,
    path,
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

    states = calculate_ofi(
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

    print(
        f"{capture_id}: "
        f"snapshot ID="
        f"{snapshot['lastUpdateId']:,}"
    )

    print(
        f"{capture_id}: "
        f"final update ID="
        f"{states['final_update_id'].iloc[-1]:,}"
    )

    return states


def validate_combined(
    combined,
):
    if combined.empty:
        raise RuntimeError(
            "Combined book-state dataset is empty."
        )

    capture_count = (
        combined["capture_id"]
        .nunique()
    )

    if capture_count != len(
        CAPTURE_FILES
    ):
        raise RuntimeError(
            "Not all expected captures "
            "are present."
        )

    numeric_columns = (
        combined
        .select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    if not np.isfinite(
        combined[
            numeric_columns
        ].to_numpy(
            dtype=float
        )
    ).all():
        raise RuntimeError(
            "Combined dataset contains "
            "non-finite numeric values."
        )

    for capture_id, frame in (
        combined.groupby(
            "capture_id",
            sort=False,
        )
    ):
        validate_states(
            frame,
            capture_id,
        )


def main():
    parts = []

    for capture_id, path in (
        CAPTURE_FILES
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing capture file: {path}"
            )

        states = process_capture(
            capture_id,
            path,
        )

        parts.append(states)

    combined = pd.concat(
        parts,
        ignore_index=True,
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

    combined["timestamp"] = (
        pd.to_datetime(
            combined[
                "event_time_ms"
            ],
            unit="ms",
            utc=True,
        )
    )

    combined["relative_time_s"] = (
        combined
        .groupby(
            "capture_id"
        )[
            "event_time_ms"
        ]
        .transform(
            lambda x:
                (
                    x - x.iloc[0]
                ) / 1000.0
        )
    )

    validate_combined(
        combined
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

    summary = (
        combined
        .groupby("capture_id")
        .agg(
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
            mean_queue_imbalance=(
                "queue_imbalance",
                "mean",
            ),
            mean_ofi_1=(
                "ofi_1",
                "mean",
            ),
            mean_ofi_multilevel=(
                "ofi_multilevel",
                "mean",
            ),
        )
    )

    print()
    print(
        "Combined capture summary:"
    )

    print(
        summary.to_string()
    )

    print()

    print(
        f"Total book states: "
        f"{len(combined):,}"
    )

    print(
        f"Captures: "
        f"{combined['capture_id'].nunique()}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()