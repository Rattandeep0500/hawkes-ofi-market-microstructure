import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "data/live/btc_usdt_research_capture.jsonl"
)


def get_input_file():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])

    return DEFAULT_INPUT


def load_records(input_file):
    records = []

    with input_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                records.append(
                    json.loads(line)
                )
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON on line "
                    f"{line_number}: {error}"
                )

    if not records:
        raise RuntimeError(
            "Capture file is empty."
        )

    return records


def validate_snapshot(records):
    snapshots = [
        record
        for record in records
        if record.get("type") == "snapshot"
    ]

    if len(snapshots) != 1:
        raise RuntimeError(
            f"Expected exactly one snapshot, "
            f"found {len(snapshots)}."
        )

    snapshot = snapshots[0]["data"]

    required = {
        "lastUpdateId",
        "bids",
        "asks",
    }

    missing = required.difference(
        snapshot
    )

    if missing:
        raise RuntimeError(
            f"Snapshot missing fields: "
            f"{sorted(missing)}"
        )

    if not snapshot["bids"]:
        raise RuntimeError(
            "Snapshot contains no bids."
        )

    if not snapshot["asks"]:
        raise RuntimeError(
            "Snapshot contains no asks."
        )

    return snapshot


def validate_depth(
    records,
    snapshot,
):
    depths = [
        record
        for record in records
        if record.get("type") == "depth"
    ]

    if not depths:
        raise RuntimeError(
            "No depth events found."
        )

    snapshot_id = int(
        snapshot["lastUpdateId"]
    )

    previous_u = snapshot_id

    first_depth = None

    for index, record in enumerate(
        depths,
        start=1,
    ):
        event = record["data"]

        required = {
            "U",
            "u",
            "b",
            "a",
        }

        missing = required.difference(
            event
        )

        if missing:
            raise RuntimeError(
                f"Depth event {index} missing "
                f"fields: {sorted(missing)}"
            )

        first_u = int(
            event["U"]
        )

        last_u = int(
            event["u"]
        )

        if first_depth is None:
            first_depth = event

            if not (
                first_u
                <= snapshot_id + 1
                <= last_u
            ):
                raise RuntimeError(
                    "First depth event does not "
                    "bridge snapshot ID."
                )

        if last_u < first_u:
            raise RuntimeError(
                f"Depth event {index} has "
                "u < U."
            )

        if last_u <= previous_u:
            raise RuntimeError(
                f"Depth event {index} does not "
                "advance the update ID."
            )

        if first_u > previous_u + 1:
            raise RuntimeError(
                f"Sequence gap at depth event "
                f"{index}: "
                f"previous_u={previous_u}, "
                f"current_U={first_u}"
            )

        previous_u = last_u

    return depths, first_depth, previous_u


def validate_trades(records):
    trades = [
        record
        for record in records
        if record.get("type") == "trade"
    ]

    if not trades:
        raise RuntimeError(
            "No trade events found."
        )

    required = {
        "e",
        "E",
        "s",
        "t",
        "p",
        "q",
        "T",
        "m",
    }

    trade_ids = []

    for index, record in enumerate(
        trades,
        start=1,
    ):
        event = record["data"]

        missing = required.difference(
            event
        )

        if missing:
            raise RuntimeError(
                f"Trade event {index} missing "
                f"fields: {sorted(missing)}"
            )

        if event["e"] != "trade":
            raise RuntimeError(
                f"Unexpected event type at "
                f"trade {index}."
            )

        price = float(
            event["p"]
        )

        quantity = float(
            event["q"]
        )

        if price <= 0:
            raise RuntimeError(
                f"Invalid trade price at "
                f"trade {index}."
            )

        if quantity <= 0:
            raise RuntimeError(
                f"Invalid trade quantity at "
                f"trade {index}."
            )

        trade_ids.append(
            int(event["t"])
        )

    if len(trade_ids) != len(
        set(trade_ids)
    ):
        raise RuntimeError(
            "Duplicate trade IDs detected."
        )

    return trades


def validate_timestamps(records):
    rows = []

    for record in records:
        if "received_at_ms" not in record:
            raise RuntimeError(
                "Record missing received_at_ms."
            )

        rows.append(
            {
                "received_at_ms": int(
                    record["received_at_ms"]
                )
            }
        )

    frame = pd.DataFrame(rows)

    timestamps = pd.to_datetime(
        frame["received_at_ms"],
        unit="ms",
        utc=True,
    )

    if not timestamps.is_monotonic_increasing:
        raise RuntimeError(
            "Reception timestamps are not "
            "monotonically increasing."
        )

    return timestamps


def main():
    input_file = get_input_file()

    if not input_file.exists():
        raise FileNotFoundError(
            f"Capture file not found: "
            f"{input_file}"
        )

    print(
        f"Validating: {input_file}"
    )

    records = load_records(
        input_file
    )

    print(
        f"Total records: "
        f"{len(records):,}"
    )

    counts = pd.Series(
        record["type"]
        for record in records
    ).value_counts()

    print()
    print(
        "Record types:"
    )
    print(
        counts.to_string()
    )

    snapshot = validate_snapshot(
        records
    )

    print()
    print(
        "Snapshot validation: OK"
    )

    snapshot_id = int(
        snapshot["lastUpdateId"]
    )

    print(
        f"Snapshot update ID: "
        f"{snapshot_id:,}"
    )

    print(
        f"Snapshot bids: "
        f"{len(snapshot['bids']):,}"
    )

    print(
        f"Snapshot asks: "
        f"{len(snapshot['asks']):,}"
    )

    (
        depths,
        first_depth,
        final_update_id,
    ) = validate_depth(
        records,
        snapshot,
    )

    print(
        "Depth sequence validation: OK"
    )

    print(
        f"Depth events: "
        f"{len(depths):,}"
    )

    print(
        f"First depth range: "
        f"{first_depth['U']:,} -> "
        f"{first_depth['u']:,}"
    )

    print(
        f"Final update ID: "
        f"{final_update_id:,}"
    )

    trades = validate_trades(
        records
    )

    print(
        "Trade validation: OK"
    )

    print(
        f"Trade events: "
        f"{len(trades):,}"
    )

    timestamps = validate_timestamps(
        records
    )

    print(
        "Reception timestamp validation: OK"
    )

    print(
        f"First received: "
        f"{timestamps.min()}"
    )

    print(
        f"Last received: "
        f"{timestamps.max()}"
    )

    print()
    print(
        "LIVE CAPTURE VALIDATION PASSED"
    )


if __name__ == "__main__":
    main()