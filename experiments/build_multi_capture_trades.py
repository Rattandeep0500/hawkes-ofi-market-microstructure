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
]

OUTPUT_FILE = Path(
    "data/processed/all_capture_trade_events.parquet"
)


def load_capture(
    capture_id,
    path,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Capture file not found: {path}"
        )

    rows = []

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
                    f"{capture_id}: invalid JSON "
                    f"on line {line_number}: {error}"
                ) from error

            if record.get("type") != "trade":
                continue

            event = record.get("data")

            if not isinstance(event, dict):
                raise RuntimeError(
                    f"{capture_id}: invalid trade "
                    f"data on line {line_number}"
                )

            required_fields = [
                "e",
                "E",
                "s",
                "t",
                "p",
                "q",
                "T",
                "m",
            ]

            missing = [
                field
                for field in required_fields
                if field not in event
            ]

            if missing:
                raise RuntimeError(
                    f"{capture_id}: trade on line "
                    f"{line_number} is missing "
                    f"{missing}"
                )

            if "received_at_ms" not in record:
                raise RuntimeError(
                    f"{capture_id}: trade on line "
                    f"{line_number} is missing "
                    "received_at_ms"
                )

            rows.append(
                {
                    "capture_id": capture_id,
                    "event_type": str(event["e"]),
                    "symbol": str(event["s"]),
                    "trade_id": int(event["t"]),
                    "event_time_ms": int(event["E"]),
                    "trade_time_ms": int(event["T"]),
                    "received_at_ms": int(
                        record["received_at_ms"]
                    ),
                    "price": float(event["p"]),
                    "quantity": float(event["q"]),
                    "buyer_maker": bool(event["m"]),
                }
            )

    if not rows:
        raise RuntimeError(
            f"{capture_id}: no trade events found."
        )

    return pd.DataFrame(rows)


def classify_trades(trades):
    result = trades.copy()

    result = result.sort_values(
        [
            "trade_time_ms",
            "trade_id",
        ]
    ).reset_index(
        drop=True
    )

    result["side"] = np.where(
        result["buyer_maker"],
        "sell",
        "buy",
    )

    result["buy_event"] = (
        result["side"] == "buy"
    ).astype(np.int8)

    result["sell_event"] = (
        result["side"] == "sell"
    ).astype(np.int8)

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

    first_time = result[
        "trade_time_ms"
    ].iloc[0]

    result["relative_time_s"] = (
        result["trade_time_ms"]
        - first_time
    ) / 1000.0

    result["interarrival_ms"] = (
        result[
            "trade_time_ms"
        ].diff()
    )

    result["same_timestamp"] = (
        result[
            "trade_time_ms"
        ].duplicated(
            keep=False
        )
    )

    result["event_index"] = np.arange(
        len(result),
        dtype=np.int64,
    )

    return result


def validate_capture(
    trades,
    capture_id,
):
    if trades.empty:
        raise RuntimeError(
            f"{capture_id}: empty trade dataset."
        )

    if not trades[
        "trade_id"
    ].is_unique:
        raise RuntimeError(
            f"{capture_id}: duplicate trade IDs."
        )

    if not trades[
        "trade_time_ms"
    ].is_monotonic_increasing:
        raise RuntimeError(
            f"{capture_id}: trade timestamps "
            "are not monotonic."
        )

    if not trades[
        "received_at_ms"
    ].is_monotonic_increasing:
        raise RuntimeError(
            f"{capture_id}: reception timestamps "
            "are not monotonic."
        )

    if (
        trades["price"] <= 0
    ).any():
        raise RuntimeError(
            f"{capture_id}: non-positive "
            "trade price detected."
        )

    if (
        trades["quantity"] <= 0
    ).any():
        raise RuntimeError(
            f"{capture_id}: non-positive "
            "trade quantity detected."
        )

    valid_numeric_columns = [
        "trade_id",
        "event_time_ms",
        "trade_time_ms",
        "received_at_ms",
        "price",
        "quantity",
        "relative_time_s",
        "signed_quantity",
        "notional",
        "signed_notional",
    ]

    values = trades[
        valid_numeric_columns
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():
        raise RuntimeError(
            f"{capture_id}: non-finite values "
            "detected in required fields."
        )

    first_interarrival = trades[
        "interarrival_ms"
    ].iloc[0]

    if not pd.isna(
        first_interarrival
    ):
        raise RuntimeError(
            f"{capture_id}: first interarrival "
            "must be NaN."
        )

    if len(trades) > 1:
        interarrivals = trades[
            "interarrival_ms"
        ].iloc[1:].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            interarrivals
        ).all():
            raise RuntimeError(
                f"{capture_id}: invalid "
                "interarrival values detected."
            )

        if (
            interarrivals < 0
        ).any():
            raise RuntimeError(
                f"{capture_id}: negative "
                "interarrival time detected."
            )


def summarize_capture(
    trades,
):
    total = len(trades)

    buy_count = int(
        (
            trades["side"] == "buy"
        ).sum()
    )

    sell_count = int(
        (
            trades["side"] == "sell"
        ).sum()
    )

    duration = (
        trades[
            "relative_time_s"
        ].iloc[-1]
        - trades[
            "relative_time_s"
        ].iloc[0]
    )

    unique_timestamps = int(
        trades[
            "trade_time_ms"
        ].nunique()
    )

    events_with_ties = int(
        trades[
            "same_timestamp"
        ].sum()
    )

    positive_interarrivals = (
        trades[
            "interarrival_ms"
        ]
        .iloc[1:]
        .to_numpy(
            dtype=float
        )
    )

    positive_interarrivals = (
        positive_interarrivals[
            positive_interarrivals > 0
        ]
    )

    summary = {
        "capture_id": trades[
            "capture_id"
        ].iloc[0],
        "trade_count": total,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "buy_fraction": (
            buy_count / total
        ),
        "sell_fraction": (
            sell_count / total
        ),
        "unique_timestamps": (
            unique_timestamps
        ),
        "events_sharing_timestamp": (
            events_with_ties
        ),
        "timestamp_tie_fraction": (
            events_with_ties / total
        ),
        "duration_seconds": duration,
        "event_rate_per_second": (
            total / duration
        ),
        "buy_rate_per_second": (
            buy_count / duration
        ),
        "sell_rate_per_second": (
            sell_count / duration
        ),
        "total_quantity": trades[
            "quantity"
        ].sum(),
        "buy_quantity": trades.loc[
            trades["side"] == "buy",
            "quantity",
        ].sum(),
        "sell_quantity": trades.loc[
            trades["side"] == "sell",
            "quantity",
        ].sum(),
        "signed_quantity": trades[
            "signed_quantity"
        ].sum(),
        "total_notional": trades[
            "notional"
        ].sum(),
        "buy_notional": trades.loc[
            trades["side"] == "buy",
            "notional",
        ].sum(),
        "sell_notional": trades.loc[
            trades["side"] == "sell",
            "notional",
        ].sum(),
        "signed_notional": trades[
            "signed_notional"
        ].sum(),
        "mean_trade_quantity": trades[
            "quantity"
        ].mean(),
        "median_trade_quantity": trades[
            "quantity"
        ].median(),
        "mean_interarrival_ms": (
            trades[
                "interarrival_ms"
            ].iloc[1:].mean()
        ),
        "median_interarrival_ms": (
            trades[
                "interarrival_ms"
            ].iloc[1:].median()
        ),
        "positive_interarrival_count": (
            len(positive_interarrivals)
        ),
        "start_time": pd.to_datetime(
            trades[
                "trade_time_ms"
            ].min(),
            unit="ms",
            utc=True,
        ),
        "end_time": pd.to_datetime(
            trades[
                "trade_time_ms"
            ].max(),
            unit="ms",
            utc=True,
        ),
    }

    return summary


def validate_combined(
    trades,
):
    if trades.empty:
        raise RuntimeError(
            "Combined trade dataset is empty."
        )

    duplicate_pairs = trades.duplicated(
        subset=[
            "capture_id",
            "trade_id",
        ]
    )

    if duplicate_pairs.any():
        raise RuntimeError(
            "Duplicate capture_id/trade_id "
            "pairs detected."
        )

    for capture_id, frame in trades.groupby(
        "capture_id",
        sort=False,
    ):
        validate_capture(
            frame,
            capture_id,
        )


def main():
    frames = []
    summaries = []

    for capture_id, path in CAPTURE_FILES:
        print(
            f"Processing {capture_id}..."
        )

        trades = load_capture(
            capture_id,
            path,
        )

        trades = classify_trades(
            trades
        )

        validate_capture(
            trades,
            capture_id,
        )

        summary = summarize_capture(
            trades
        )

        frames.append(
            trades
        )

        summaries.append(
            summary
        )

        print(
            f"{capture_id}: "
            f"{len(trades):,} trades"
        )

        print(
            f"{capture_id}: "
            f"{summary['buy_count']:,} buys | "
            f"{summary['sell_count']:,} sells"
        )

        print(
            f"{capture_id}: "
            f"{summary['duration_seconds']:.3f} s"
        )

        print(
            f"{capture_id}: "
            f"unique timestamps="
            f"{summary['unique_timestamps']:,}"
        )

        print()

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    validate_combined(
        combined
    )

    summaries = pd.DataFrame(
        summaries
    )

    combined = combined.sort_values(
        [
            "capture_id",
            "trade_time_ms",
            "trade_id",
        ]
    ).reset_index(
        drop=True
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

    print(
        "Capture summary:"
    )

    print(
        summaries[
            [
                "capture_id",
                "trade_count",
                "buy_count",
                "sell_count",
                "buy_fraction",
                "sell_fraction",
                "unique_timestamps",
                "events_sharing_timestamp",
                "timestamp_tie_fraction",
                "duration_seconds",
                "event_rate_per_second",
                "buy_rate_per_second",
                "sell_rate_per_second",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        f"Total trade events: "
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